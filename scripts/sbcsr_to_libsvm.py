#!/usr/bin/env python3
"""Convert an .sbcsr corpus to somoclu's libsvm-style sparse text format.

somoclu (sparse CPU kernel, -k 2) reads one vector per line as
space-separated `index:value` pairs, 1-based column indices, optional
leading label (skipped). Binary corpus → all values 1.

Usage:
    sbcsr_to_libsvm.py CORPUS.sbcsr OUT.txt [--max-rows N]
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def read_sbcsr(path: Path):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic not in (b"SBCSR1\x00\x00", b"SCSR1\x00\x00\x00"):
            raise ValueError(f"Bad .sbcsr magic: {magic!r}")
        n_rows, n_cols, nnz, has_values = struct.unpack("<IIII", f.read(16))
        row_ptr = np.frombuffer(f.read((n_rows + 1) * 4), dtype=np.uint32)
        col = np.frombuffer(f.read(nnz * 2), dtype=np.uint16)
    return n_rows, n_cols, nnz, row_ptr, col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("out")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="cap rows written (0 = all)")
    args = ap.parse_args()

    n_rows, n_cols, nnz, row_ptr, col = read_sbcsr(Path(args.corpus))
    limit = n_rows if args.max_rows <= 0 else min(args.max_rows, n_rows)
    print(f"{args.corpus}: {n_rows} rows x {n_cols} cols, {nnz} nnz "
          f"-> writing {limit} rows", file=sys.stderr)

    # 1-based indices; binary values are all 1
    with open(args.out, "w", buffering=1 << 20) as out:
        for i in range(limit):
            a, b = row_ptr[i], row_ptr[i + 1]
            cols = col[a:b].astype(np.int64) + 1
            out.write(" ".join(f"{c}:1" for c in cols))
            out.write("\n")
            if (i & 0x3FFFF) == 0 and i:
                print(f"  {i}/{limit}", file=sys.stderr)

    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
