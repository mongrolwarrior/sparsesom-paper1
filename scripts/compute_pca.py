#!/usr/bin/env python3
"""Compute top-k principal components of a .sbcsr corpus and write a .sompca file.

The .sompca file is tiny (~360 KB for k=2, V=30766) and contains the mean vector,
principal component directions, and singular values.  sparsesom --init-pca reads it
and generates the full codebook directly in GPU memory — no host buffer, no disk.

  ./pca_init.py corpus.train.sbcsr --out pca_components.sompca
"""

import argparse
import struct
import sys
from array import array

import numpy as np

_SBCSR_MAGIC = b"SBCSR1\x00\x00"
_SOMPCA_MAGIC = b"SOMPCA1\x00"


def load_sbcsr_sparse(path):
    """Load a .sbcsr file into a scipy CSR matrix."""
    from scipy.sparse import csr_matrix

    with open(path, "rb") as f:
        if f.read(8) != _SBCSR_MAGIC:
            raise SystemExit(f"not a .sbcsr file: {path}")
        n, V, nnz, _ = struct.unpack("<IIII", f.read(16))
        row_ptr = np.frombuffer(f.read(4 * (n + 1)), dtype=np.uint32).copy()
        col_idx = np.frombuffer(f.read(2 * nnz), dtype=np.uint16).copy()

    data = np.ones(nnz, dtype=np.float32)
    return csr_matrix((data, col_idx.astype(np.int32), row_ptr.astype(np.int64)),
                      shape=(n, V)), V


def compute_pca(sbcsr_path, n_components=2):
    """Compute mean + top-k SVD of a .sbcsr corpus. Returns (mean, components, svs, V)."""
    from scipy.sparse.linalg import svds

    X, V = load_sbcsr_sparse(sbcsr_path)
    n = X.shape[0]
    print(f"loaded {n:,} x {V} sparse matrix ({X.nnz:,} nnz)", flush=True)

    mean = np.asarray(X.mean(axis=0)).ravel().astype(np.float32)

    print(f"computing top-{n_components} SVD...", flush=True)
    U, s, Vt = svds(X, k=n_components)

    idx = np.argsort(-s)
    s = s[idx]
    Vt = Vt[idx]

    components = Vt.astype(np.float32)
    svs = s.astype(np.float32)

    for i in range(n_components):
        print(f"  PC{i+1}: sv={svs[i]:.2f}, explained_var={svs[i]**2/(n-1):.6f}")

    return mean, components, svs, V, n


def write_sompca(path, n_features, n_samples, mean, components, singular_values):
    """Write a .sompca file: magic + header + data.

    Format: SOMPCA1\\0 | uint32 n_features | uint32 n_components | uint32 n_samples | uint32 reserved |
            float32[n_components] svs | float32[n_features] mean | float32[n_components * n_features] comps
    """
    n_components = len(singular_values)
    with open(path, "wb") as f:
        f.write(_SOMPCA_MAGIC)
        f.write(struct.pack("<IIII", n_features, n_components, n_samples, 0))
        f.write(singular_values.astype(np.float32).tobytes())
        f.write(mean.astype(np.float32).tobytes())
        f.write(components.astype(np.float32).tobytes())
    size = 8 + 16 + n_components * 4 + n_features * 4 + n_components * n_features * 4
    print(f"wrote {path} ({size:,} bytes, {n_components} components, {n_features} features, "
          f"{n_samples:,} samples)")


def load_sompca(path):
    """Read a .sompca file. Returns (n_features, n_components, n_samples, svs, mean, components)."""
    with open(path, "rb") as f:
        if f.read(8) != _SOMPCA_MAGIC:
            raise SystemExit(f"not a .sompca file: {path}")
        n_features, n_components, n_samples, _ = struct.unpack("<IIII", f.read(16))
        svs = np.frombuffer(f.read(4 * n_components), dtype=np.float32).copy()
        mean = np.frombuffer(f.read(4 * n_features), dtype=np.float32).copy()
        components = np.frombuffer(f.read(4 * n_components * n_features),
                                   dtype=np.float32).copy().reshape(n_components, n_features)
    return n_features, n_components, n_samples, svs, mean, components


def main():
    p = argparse.ArgumentParser(description="Compute PCA components from a .sbcsr corpus")
    p.add_argument("corpus", help=".sbcsr corpus to compute PCA from")
    p.add_argument("--out", required=True, help="output .sompca path")
    p.add_argument("--components", type=int, default=2, help="number of PCs (default 2)")
    a = p.parse_args()

    mean, components, svs, V, n = compute_pca(a.corpus, a.components)
    write_sompca(a.out, V, n, mean, components, svs)


if __name__ == "__main__":
    main()
