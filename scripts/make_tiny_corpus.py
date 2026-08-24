#!/usr/bin/env python3
"""Create a tiny ~50k-article corpus for smoke testing and the bring-up gate.

Reads the first N articles from the full .sbcsr and writes a new .sbcsr.
This preserves the binary format so all implementations can consume it.

Usage:
    python3 scripts/make_tiny_corpus.py data/corpus.train.sbcsr data/corpus.tiny.sbcsr [50000]
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Create a tiny .sbcsr corpus")
    parser.add_argument("input", type=Path, help="Input .sbcsr file")
    parser.add_argument("output", type=Path, help="Output tiny .sbcsr file")
    parser.add_argument("n_articles", type=int, nargs="?", default=50000,
                        help="Number of articles to keep (default: 50000)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found")
        sys.exit(1)

    with open(args.input, "rb") as f:
        magic = f.read(8)
        n_rows, n_cols, nnz_total, has_values = struct.unpack("<IIII", f.read(16))
        row_ptr = np.frombuffer(f.read((n_rows + 1) * 4), dtype=np.uint32)
        col_idx = np.frombuffer(f.read(nnz_total * 2), dtype=np.uint16)

    n_keep = min(args.n_articles, n_rows)
    new_row_ptr = row_ptr[:n_keep + 1]
    new_nnz = int(new_row_ptr[-1])
    new_col_idx = col_idx[:new_nnz]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<IIII", n_keep, n_cols, new_nnz, 0))
        f.write(new_row_ptr.tobytes())
        f.write(new_col_idx.tobytes())

    print(f"Tiny corpus: {n_keep:,} articles, {n_cols:,} features, {new_nnz:,} nnz")
    print(f"  {args.input} ({args.input.stat().st_size / 1e6:.1f} MB) → "
          f"{args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
