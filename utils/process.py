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
"""

import argparse
import json
import os
from typing import Tuple


def _clean_text(s: str) -> str:
    """Make sure it's one line; replace tabs/newlines with spaces."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split())
    return s


def _scan_counts(path: str) -> Tuple[int, int]:
    """
    First pass: count instances and determine num_classes from label indices.
    Assumes label ids are 0-based -> num_classes = max_label + 1.
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

    num_classes = max_label + 1 if max_label >= 0 else 0
    return n_instances, num_classes


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
        help="Set this if target_ind labels are 1-based. Then num_classes = max_label.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_x = os.path.join(args.out_dir, f"{args.prefix}_X.txt")
    out_xy = os.path.join(args.out_dir, f"{args.prefix}_X_Y.txt")

    # ---- Pass 1: compute header ----
    num_instances, num_classes = _scan_counts(args.input)
    if args.labels_are_1based:
        # If labels start at 1, classes count is max_label (not max_label+1).
        # Example: labels {1..K} => max_label=K => num_classes=K
        # If no labels, keep 0.
        with open(args.input, "r", encoding="utf-8") as fin:
            max_label = -1
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
        num_classes = max_label if max_label >= 0 else 0

    # ---- Pass 2: write outputs ----
    n = 0
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(out_x, "w", encoding="utf-8") as fx, \
         open(out_xy, "w", encoding="utf-8") as fxy:

        # Header required by readsparsefile
        fxy.write(f"{num_instances} {num_classes}\n")

        for line in fin:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)

            title = _clean_text(ex.get("title", ""))
            content = _clean_text(ex.get("content", ""))

            # ----- write *_X.txt -----
            if args.title_only or not content:
                text = title
            else:
                text = f"{title} {content}"
            fx.write(text + "\n")

            # ----- write *_X_Y.txt -----
            inds = ex.get("target_ind", []) or []
            rels = ex.get("target_rel", None)

            if rels is None or len(rels) == 0:
                rels = [1.0] * len(inds)
            if len(rels) != len(inds):
                raise ValueError(
                    f"Line {n}: len(target_ind)={len(inds)} but len(target_rel)={len(rels)}"
                )

            pairs = sorted(zip(inds, rels), key=lambda x: int(x[0]))
            xy_line = " ".join(f"{int(i)}:{float(v):.5f}" for i, v in pairs)
            fxy.write(xy_line + "\n")

            n += 1

    print(f"Done. Wrote {n} lines (expected {num_instances}):")
    print(f"  X  -> {out_x}")
    print(f"  X_Y-> {out_xy}")
    print(f"Header in X_Y: {num_instances} {num_classes}")


if __name__ == "__main__":
    main()
