#!/usr/bin/env python3
"""Convert an .sbcsr corpus to MedSOM's expected directory format.

MedSOM expects:
    <dir>/bin/mesh_offsets.bin   — header + row_ptr (uint32[n+1])
    <dir>/bin/articles.bin      — header + {pmid: u32, year: u16, pad: u16}[n]
    <dir>/csr/csr_col_u16.bin   — header + col_indices (uint16[nnz])
    <dir>/csr/vocab.bin         — header + vocab IDs (uint32[n_vocab])

The .sbcsr format (from SparseBinarySOM dataset.cpp):
    Header (24 bytes): magic[8] + n_samples[u32] + n_features[u32] + n_nonzeros[u32] + has_values[u32]
    row_ptr: uint32[n_samples + 1]
    col_idx: uint16[n_nonzeros]

Since the anonymised corpus has no PMIDs, articles.bin is filled with
dummy values (pmid=index, year=2026). MedSOM only uses PMIDs for labelling,
not for computation.

Usage:
    python3 scripts/sbcsr_to_medsom.py data/corpus.train.sbcsr data/medsom_input
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def read_sbcsr(path: Path) -> tuple[int, int, int, np.ndarray, np.ndarray]:
    """Read .sbcsr and return (n_rows, n_cols, nnz, row_ptr_u32, col_idx_u16)."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic not in (b"SBCSR1\x00\x00", b"SCSR1\x00\x00\x00"):
            raise ValueError(f"Bad .sbcsr magic: {magic!r}")

        n_rows, n_cols, nnz, has_values = struct.unpack("<IIII", f.read(16))
        row_ptr = np.frombuffer(f.read((n_rows + 1) * 4), dtype=np.uint32)
        col_idx = np.frombuffer(f.read(nnz * 2), dtype=np.uint16)

    return n_rows, n_cols, nnz, row_ptr, col_idx


def _preamble(magic: bytes = b"MEDSM\x00\x00\x00") -> bytes:
    """16-byte FilePreamble: magic[8] + version[1] + schema_crc[4] + reserved[3]."""
    return struct.pack("<8sBI3s", magic, 1, 0, b"\x00\x00\x00")


def write_medsom_dir(out_dir: Path, n_rows: int, n_cols: int, nnz: int,
                     row_ptr: np.ndarray, col_idx: np.ndarray):
    """Write MedSOM-format binary directory."""
    bin_dir = out_dir / "bin"
    csr_dir = out_dir / "csr"
    bin_dir.mkdir(parents=True, exist_ok=True)
    csr_dir.mkdir(parents=True, exist_ok=True)

    pre = _preamble()

    # mesh_offsets.bin: OffHeader{preamble(16) + n_articles: u32, n_mesh_total: u32} + row_ptr[n+1]
    with open(bin_dir / "mesh_offsets.bin", "wb") as f:
        f.write(pre)
        f.write(struct.pack("<II", n_rows, nnz))
        f.write(row_ptr.tobytes())

    # csr_col_u16.bin: ColHeader{preamble(16) + n_nonzeros: u32, reserved: u32} + col_idx[nnz]
    with open(csr_dir / "csr_col_u16.bin", "wb") as f:
        f.write(pre)
        f.write(struct.pack("<II", nnz, 0))
        f.write(col_idx.tobytes())

    # vocab.bin: VocHeader{preamble(16) + n_vocab: u32, reserved: u32} + vocab_ids[n_vocab]
    with open(csr_dir / "vocab.bin", "wb") as f:
        f.write(pre)
        f.write(struct.pack("<II", n_cols, 0))
        vocab_ids = np.arange(n_cols, dtype=np.uint32)
        f.write(vocab_ids.tobytes())

    # articles.bin: ArtHeader{preamble(16) + n_articles: u32, reserved: u32} + ArtRecord[n]
    with open(bin_dir / "articles.bin", "wb") as f:
        f.write(pre)
        f.write(struct.pack("<II", n_rows, 0))
        for i in range(n_rows):
            f.write(struct.pack("<IHH", i, 2026, 0))

    print(f"MedSOM directory written to {out_dir}/")
    print(f"  n_articles={n_rows:,}  n_features={n_cols:,}  nnz={nnz:,}")
    for sub in [bin_dir, csr_dir]:
        for p in sorted(sub.iterdir()):
            print(f"  {p.relative_to(out_dir)}: {p.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert .sbcsr corpus to MedSOM directory format")
    parser.add_argument("input", type=Path, help="Input .sbcsr file")
    parser.add_argument("output", type=Path, help="Output directory for MedSOM")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found")
        sys.exit(1)

    print(f"Reading {args.input} ...")
    n_rows, n_cols, nnz, row_ptr, col_idx = read_sbcsr(args.input)

    print(f"Writing MedSOM format to {args.output} ...")
    write_medsom_dir(args.output, n_rows, n_cols, nnz, row_ptr, col_idx)


if __name__ == "__main__":
    main()
