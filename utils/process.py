#!/usr/bin/env python3
"""
Convert a JSONL training file (trn.json) into:
  1) trn_X.txt    : one text per line (title + content)
  2) trn_X_Y.txt  : label:weight pairs per line (xclib/libxmlc style)

Now writes a header line to *_X_Y.txt:
  num_instances num_classes

Input JSONL format (one JSON object per line), e.g.:
{"uid": "...", "title": "...", "content": "...", "target_ind": [...], "target_rel": [...]}

Usage:
  python jsonl_to_xclib.py --input trn.json --out-dir data/Amazon-131K --prefix trn

Sampling labels:
  --sample-rate 0.1   will keep ~10% of labels (e.g., Amazon-670K -> Amazon-67K),  
  --sample-rate 67000 will keep exactly 67000 labels (absolute count),              
  sampled labels are remapped to a compact id space, and instances that end up with
  0 labels are dropped.
"""

import argparse
import json
import os
import random
from typing import Dict, Optional, Tuple


def _clean_text(s: str) -> str:
    """Make sure it's one line; replace tabs/newlines with spaces."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split())
    return s


def _scan_counts(path: str, labels_are_1based: bool = False) -> Tuple[int, int]:
    """
    First pass: count instances and determine num_classes from label indices.

    For 0-based label ids:
      num_classes = max_label + 1
    For 1-based label ids (labels_are_1based=True):
      num_classes = max_label

    If no labels appear, num_classes = 0.
    """
    n_instances = 0
    max_label = -1

    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            inds = ex.get("target_ind", []) or []
            for i in inds:
                li = int(i)
                if li > max_label:
                    max_label = li
            n_instances += 1

    if max_label < 0:
        return n_instances, 0

    num_classes = max_label if labels_are_1based else (max_label + 1)
    return n_instances, num_classes


def _make_label_mapping(
    num_classes: int,
    sample_rate: float,
    labels_are_1based: bool,
    seed: int,
) -> Tuple[Optional[Dict[int, int]], int]:
    """
    Build an (old_label -> new_label) mapping for label sampling.

    Modes:  
    - If sample_rate == 1.0 or num_classes == 0: return (None, num_classes) meaning "no sampling / no remap".  
    - If 0 < sample_rate < 1.0: sample k = floor(num_classes * sample_rate) labels (at least 1 if num_classes > 0).  
    - If sample_rate > 1.0: sample k = int(sample_rate) labels (absolute count; must be integer-like).  

    If sampling is active, kept labels are remapped to a compact id space:
      0-based: new labels in [0, k)
      1-based: new labels in [1, k]
    """
    if num_classes == 0:
        return None, 0  

    if sample_rate <= 0.0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")  

    # if sample_rate == 1.0:
    #     return None, num_classes  

    if 0.0 < sample_rate < 1.0:
        k = int(num_classes * sample_rate)  
        if k < 1 and num_classes > 0:
            k = 1  
    else:
        # sample_rate > 1.0 => treat as absolute number of labels to keep  
        if abs(sample_rate - round(sample_rate)) > 1e-9:
            raise ValueError(  
                f"sample_rate > 1 is treated as an integer label count; got non-integer {sample_rate}"
            )
        k = int(round(sample_rate))  

    if k >= num_classes:
        return None, num_classes

    rng = random.Random(seed)
    universe = range(1, num_classes + 1) if labels_are_1based else range(num_classes)
    kept = rng.sample(universe, k)
    kept_sorted = sorted(kept)

    base = 1 if labels_are_1based else 0
    mapping = {old: (idx + base) for idx, old in enumerate(kept_sorted)}
    return mapping, k


def _normalize_rels(inds, rels, line_no: int):
    """Ensure we have a weight for every label."""
    if rels is None or len(rels) == 0:
        rels = [1.0] * len(inds)
    if len(rels) != len(inds):
        raise ValueError(
            f"Line {line_no}: len(target_ind)={len(inds)} but len(target_rel)={len(rels)}"
        )
    return rels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to trn.json (JSONL)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--prefix", default="trn", help="Output file prefix (default: trn)")
    ap.add_argument(
        "--title-only",
        action="store_true",
        help="If set, trn_X.txt contains only title (no content).",
    )
    ap.add_argument(
        "--labels-are-1based",
        action="store_true",
        help="Set this if target_ind labels are 1-based.",
    )

    ap.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help=(
            "If 0 < sample_rate <= 1, keep this fraction of labels (default: 1.0). "  
            "If sample_rate > 1, keep exactly int(sample_rate) labels (absolute). "  
            "If sampling is active, labels are remapped to a compact id space and instances with "
            "0 remaining labels are dropped."
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for label sampling (default: 0).",
    )
    ap.add_argument(
        "--drop-empty-instances",
        action="store_true",
        help="Drop instances with 0 labels even when --sample-rate=1.0 (default: off).",
    )

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_x = os.path.join(args.out_dir, f"{args.prefix}_X.txt")
    out_xy = os.path.join(args.out_dir, f"{args.prefix}_X_Y.txt")

    # ---- Pass 1: compute original header ----
    orig_num_instances, orig_num_classes = _scan_counts(
        args.input, labels_are_1based=args.labels_are_1based
    )

    label_map, num_classes = _make_label_mapping(
        num_classes=orig_num_classes,
        sample_rate=args.sample_rate,
        labels_are_1based=args.labels_are_1based,
        seed=args.seed,
    )

    drop_empty = args.drop_empty_instances or (label_map is not None)

    # ---- Pass 2: count output instances (after filtering) ----
    num_instances = 0
    with open(args.input, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            inds = ex.get("target_ind", []) or []
            rels = ex.get("target_rel", None)

            if label_map is None:
                if drop_empty and len(inds) == 0:
                    continue
                num_instances += 1
                continue

            rels = _normalize_rels(inds, rels, line_no)
            kept_any = False
            for i in inds:
                if int(i) in label_map:
                    kept_any = True
                    break
            if kept_any:
                num_instances += 1

    # ---- Pass 3: write outputs ----
    n = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(out_x, "w", encoding="utf-8") as fx, \
         open(out_xy, "w", encoding="utf-8") as fxy:

        fxy.write(f"{num_instances} {num_classes}\n")

        for line_no, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)

            title = _clean_text(ex.get("title", ""))
            content = _clean_text(ex.get("content", ""))

            inds = ex.get("target_ind", []) or []
            rels = ex.get("target_rel", None)

            if label_map is None:
                if drop_empty and len(inds) == 0:
                    continue
                rels = _normalize_rels(inds, rels, line_no)
                pairs = sorted(zip(inds, rels), key=lambda x: int(x[0]))
            else:
                rels = _normalize_rels(inds, rels, line_no)
                pairs = []
                for i, v in zip(inds, rels):
                    oi = int(i)
                    if oi in label_map:
                        pairs.append((label_map[oi], float(v)))
                if drop_empty and len(pairs) == 0:
                    continue
                pairs.sort(key=lambda x: int(x[0]))

            if args.title_only or not content:
                text = title
            else:
                text = f"{title} {content}"
            fx.write(text + "\n")

            xy_line = " ".join(f"{int(i)}:{float(v):.5f}" for i, v in pairs)
            fxy.write(xy_line + "\n")

            n += 1

    print(f"Done. Wrote {n} lines (header says {num_instances}).")
    print(f"  X  -> {out_x}")
    print(f"  X_Y-> {out_xy}")
    if label_map is not None:
        print(f"Label sampling: sample_rate={args.sample_rate}, seed={args.seed}")
        print(f"  Original: instances={orig_num_instances}, classes={orig_num_classes}")
        print(f"  Output:   instances={num_instances}, classes={num_classes}")
    else:
        print(f"Header in X_Y: {num_instances} {num_classes}")


if __name__ == "__main__":
    main()
