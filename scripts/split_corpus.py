#!/usr/bin/env python3
"""Split a .sbcsr corpus into train/heldout by row indices.

Usage:
    python3 scripts/split_corpus.py data/corpus.full.sbcsr data/ --seed 42 --train-frac 0.9
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Split .sbcsr into train/heldout")
    parser.add_argument("input", type=Path, help="Full .sbcsr corpus")
    parser.add_argument("dest", type=Path, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.9)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found")
        sys.exit(1)

    with open(args.input, "rb") as f:
        magic = f.read(8)
        n_rows, n_cols, nnz_total, has_values = struct.unpack("<IIII", f.read(16))
        row_ptr = np.frombuffer(f.read((n_rows + 1) * 4), dtype=np.uint32).copy()
        col_idx = np.frombuffer(f.read(nnz_total * 2), dtype=np.uint16).copy()

    print(f"Full corpus: {n_rows:,} rows, {n_cols:,} cols, {nnz_total:,} nnz")

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n_rows)
    n_train = int(n_rows * args.train_frac)

    train_idx = np.sort(perm[:n_train])
    held_idx = np.sort(perm[n_train:])

    args.dest.mkdir(parents=True, exist_ok=True)

    _write_subset(magic, n_cols, row_ptr, col_idx, train_idx,
                  args.dest / "corpus.train.sbcsr", "train")
    _write_subset(magic, n_cols, row_ptr, col_idx, held_idx,
                  args.dest / "corpus.heldout.sbcsr", "heldout")


def _write_subset(magic, n_cols, row_ptr, col_idx, indices, out_path, label):
    n = len(indices)
    new_row_ptr = np.zeros(n + 1, dtype=np.uint32)
    nnz_counts = np.diff(row_ptr)[indices]
    np.cumsum(nnz_counts, out=new_row_ptr[1:])
    new_nnz = int(new_row_ptr[-1])

    new_col_idx = np.empty(new_nnz, dtype=np.uint16)
    pos = 0
    for i in indices:
        start, end = int(row_ptr[i]), int(row_ptr[i + 1])
        length = end - start
        new_col_idx[pos:pos + length] = col_idx[start:end]
        pos += length

    with open(out_path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<IIII", n, n_cols, new_nnz, 0))
        f.write(new_row_ptr.tobytes())
        f.write(new_col_idx.tobytes())

    print(f"  {label}: {n:,} rows, {new_nnz:,} nnz → {out_path.name} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
