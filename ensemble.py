import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy
import scipy.sparse as sp
import torch

import xclib.data.data_utils as data_utils
import xclib.evaluation.xc_metrics as xc_metrics

from dl_base import PrecEvaluator  # uses printacc/_filter internally


class DummyIndexDataset(torch.utils.data.Dataset):
    """Dataset that only stores ground-truth labels and yields indices."""
    def __init__(self, labels: sp.spmatrix):
        self.labels = labels.tocsr()

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> int:
        return int(idx)


def _collate_indices(batch: List[int]) -> Dict[str, object]:
    """Collate function that only provides batch_size and ids."""
    return {
        "batch_size": len(batch),
        "ids": np.asarray(batch, dtype=np.int64),
    }


@dataclass
class DummyModel:
    """Minimal model interface required by PrecEvaluator."""
    numy: int
    epochs: int = 1
    rank: int = 0
    world_size: int = 1
    device: str = "cpu"

    def __post_init__(self) -> None:
        self._target_device = torch.device(self.device)

    def save(self, out_dir: str) -> None:
        return


class EnsembleNPZPredictor:
    """Load preds_<batch>.npz from multiple model directories and ensemble them."""

    def __init__(
        self,
        results_dir: str,
        model_ids: List[int],
        K: int = 5,
        apply_sigmoid: bool = False,
    ):
        self.results_dir = results_dir
        self.model_ids = model_ids
        self.K = K
        self.apply_sigmoid = apply_sigmoid

        self._preds_dirs = {}
        self._label_indices = {}

        for mid in self.model_ids:
            mdir = os.path.join(self.results_dir, str(mid))
            preds_dir = os.path.join(mdir, "preds")
            if not os.path.isdir(preds_dir):
                print(f"[warn] Missing preds/ for model {mid}: {preds_dir}")
                continue
            self._preds_dirs[mid] = preds_dir
            label_path = os.path.exists(os.path.join(mdir, "label_indices.json"))
            self._label_indices[mid] = None if not label_path else np.asarray(json.load(open(label_path, "r")), dtype=np.int32) 

        if len(self._preds_dirs) == 0:
            raise ValueError(f"No valid model preds/ found under: {results_dir}")

        p0 = os.path.join(self._preds_dirs[next(iter(self._preds_dirs.keys()))], "preds_0.npz")
        data = np.load(p0)
        self.inferred_batch_size = int(data["indices"].shape[0]) if "indices" in data else int(data["values"].shape[0])

    def _load_batch_pred(
        self,
        preds_dir: str,
        batch_idx: int,
    ) -> Optional[Tuple[Optional[np.ndarray], np.ndarray]]:
        """Return (local_indices or None, values) for a given batch file."""
        batch_file = os.path.join(preds_dir, f"preds_{batch_idx}.npz")
        if not os.path.exists(batch_file):
            return None
        data = np.load(batch_file)
        if "indices" in data:
            return data["indices"].astype(np.int32, copy=False), data["values"].astype(np.float32, copy=False)
        return None, data["values"].astype(np.float32, copy=False)

    def _ensemble_one_batch_dense(
        self,
        batch_idx: int,
        bsz: int,
        num_labels: int,
        topk: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dense accumulator for faster ensembling when bsz * num_labels is manageable."""
        preds_sum = np.zeros((bsz, num_labels), dtype=np.float32)
        preds_cnt = np.zeros((bsz, num_labels), dtype=np.int16)

        for mid, preds_dir in self._preds_dirs.items():
            loaded = self._load_batch_pred(preds_dir, batch_idx)
            if loaded is None:
                continue
            values = scipy.special.expit(values)

            label_idx = self._label_indices[mid]

            preds_sum[:, label_idx] += values
            preds_cnt[:, label_idx] += 1

        # Average predictions per (sample,label) where covered; push uncovered labels down.
        preds_avg = np.divide(
            preds_sum,
            preds_cnt,
            out=np.full_like(preds_sum, -1e9, dtype=np.float32),
            where=preds_cnt != 0,
        )

        # Top-k per row.
        k = min(topk, num_labels)
        top_idx = np.argpartition(preds_avg, -k, axis=1)[:, -k:]
        top_val = preds_avg[np.arange(bsz)[:, None], top_idx]
        order = np.argsort(-top_val, axis=1)
        top_idx = top_idx[np.arange(bsz)[:, None], order].astype(np.int32, copy=False)
        top_val = top_val[np.arange(bsz)[:, None], order].astype(np.float32, copy=False)
        return top_val, top_idx

    def __call__(self, loss_model, model: DummyModel, dataloader: torch.utils.data.DataLoader):
        """Return score_mat as CSR with exactly self.K entries per row."""
        num_labels = int(dataloader.dataset.labels.shape[1])
        datalen = int(dataloader.dataset.labels.shape[0])

        data = np.zeros((datalen, self.K), dtype=np.float32)
        inds = np.zeros((datalen, self.K), dtype=np.int32)
        indptr = np.arange(0, datalen * self.K + 1, self.K, dtype=np.int64)

        ctr = 0
        for batch_idx, batch in enumerate(dataloader):
            bsz = int(batch["batch_size"])

            top_val, top_idx = self._ensemble_one_batch_dense(
                batch_idx=batch_idx,
                bsz=bsz,
                num_labels=num_labels,
                topk=self.K,
            )

            # Write into CSR buffers.
            data[ctr : ctr + bsz, : top_val.shape[1]] = top_val
            inds[ctr : ctr + bsz, : top_idx.shape[1]] = top_idx
            ctr += bsz

        return sp.csr_matrix((data.ravel(), inds.ravel(), indptr), shape=(datalen, num_labels))


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble preds_*.npz and evaluate with PrecEvaluator")

    # Data
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset directory containing trn_X_Y.txt and tst_X_Y.txt")

    # Results
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory like Results/Bert-XC/<DATASET_NAME> containing per-model subdirs 0/,1/,...",
    )
    parser.add_argument("--min-model", type=int, default=0, help="Minimum model id (inclusive)")
    parser.add_argument("--max-model", type=int, default=100, help="Maximum model id (inclusive)")

    # Eval
    parser.add_argument("--k", type=int, default=5, help="Evaluate at K (e.g., 5 for P@5)")

    # Output
    parser.add_argument("--out-dir", type=str, default="", help="If set, PrecEvaluator will write score_mat.npz + evaluation.tsv here")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tst_X_Y = data_utils.read_sparse_file(os.path.join(args.data_dir, "tst_X_Y.txt"))

    tst_filter_mat = None
    filter_path = os.path.join(args.data_dir, "tst_filter_labels.txt")
    if os.path.exists(filter_path):
        temp = np.fromfile(filter_path, sep=" ").astype(int).reshape(-1, 2).T
        tst_filter_mat = sp.coo_matrix(
            (np.ones(temp.shape[1]), (temp[0], temp[1])),
            shape=tst_X_Y.shape,
        ).tocsr()
        print(f"Loaded filter matrix from: {filter_path}")

    model_ids = list(range(args.min_model, args.max_model + 1))
    predictor = EnsembleNPZPredictor(
        results_dir=args.results_dir,
        model_ids=model_ids,
        K=100,
        apply_sigmoid=args.apply_sigmoid,
        verbose=True,
    )

    dataset = DummyIndexDataset(tst_X_Y)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=predictor.inferred_batch_size,
        shuffle=False,
        collate_fn=_collate_indices,
    )

    dummy_model = DummyModel(numy=tst_X_Y.shape[1], epochs=1, device="cpu")

    evaluator = PrecEvaluator(
        model=dummy_model,
        dataloader=dataloader,
        predictor=predictor,
        filter_mat=tst_filter_mat,
        K=args.k,
        metric="P",
        inv_prop=None,
    )

    out_dir = args.out_dir if args.out_dir.strip() != "" else None
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    evaluator(loss_model=None, epoch=-1, loss=-1.0, out_dir=out_dir, name="ensemble")


if __name__ == "__main__":
    main()
