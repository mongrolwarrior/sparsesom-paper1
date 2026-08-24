"""Run 1 — Corpus & split manifest (Table 1). No GPU training."""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_sbcsr_header(path: Path) -> dict:
    """Read the .sbcsr header (24 bytes) and row_ptr to compute per-row nnz stats."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic not in (b"SBCSR1\x00\x00", b"SCSR1\x00\x00\x00"):
            raise ValueError(f"Not an sbcsr file: {path}")
        n_rows, n_cols, nnz_total, _has_values = struct.unpack("<IIII", f.read(16))
        row_ptr = np.frombuffer(f.read((n_rows + 1) * 4), dtype=np.uint32)

    nnz_per_row = np.diff(row_ptr).astype(np.float64)
    return {
        "n_articles": int(n_rows),
        "v_features": int(n_cols),
        "nnz_total": int(nnz_total),
        "nnz_mean": float(np.mean(nnz_per_row)),
        "nnz_sd": float(np.std(nnz_per_row, ddof=1)),
    }


def run(data_dir: str, results_dir: str, **kwargs) -> Path:
    """Generate corpus_manifest.json from the .sbcsr files.

    Reads the full corpus for the authoritative article count, then validates
    that the train/heldout split matches the expected 90/10 ratio.
    """
    data = Path(data_dir)
    out = Path(results_dir) / "corpus_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    full_path = data / "corpus.full.sbcsr"
    train_path = data / "corpus.train.sbcsr"
    heldout_path = data / "corpus.heldout.sbcsr"

    if not full_path.exists():
        raise FileNotFoundError(f"corpus.full.sbcsr not found in {data}")
    if not train_path.exists():
        raise FileNotFoundError(f"corpus.train.sbcsr not found in {data}")

    full_h = _read_sbcsr_header(full_path)
    train_h = _read_sbcsr_header(train_path)

    n_full = full_h["n_articles"]
    v = full_h["v_features"]
    density = full_h["nnz_total"] / (n_full * v)

    # 90/10 split: train 90%, held-out 10% (the exact complement), split_seed 42 —
    # per the manuscript (§4.2, Table 1) and the frozen corpus_manifest.json
    # (train sha256 500ceed6..., 26,912,934 rows). A 1% tolerance absorbs the
    # seeded-random assignment. NOTE: an earlier commit (555d569) wrongly relaxed
    # this to 80/10/10 to match a stale June-2026 corpus generation that had been
    # staged in data/; the 90/10 expectation is the authoritative one and this
    # check is exactly what catches wrongly staged data.
    train_fraction = 0.9
    tol = int(n_full * 0.01)  # 1% of N
    expected_train = int(n_full * train_fraction)
    actual_train = train_h["n_articles"]

    if abs(actual_train - expected_train) > tol:
        raise RuntimeError(
            f"Split validation failed: corpus.train.sbcsr has {actual_train:,} rows "
            f"but expected ~{expected_train:,} ({train_fraction*100:.0f}% of "
            f"{n_full:,}). The staged corpus does not match the manuscript's "
            f"90/10 split (frozen train sha256 500ceed6...).")

    # Held-out is the exact train complement (no third partition in the release).
    n_heldout = n_full - actual_train
    if heldout_path.exists():
        heldout_h = _read_sbcsr_header(heldout_path)
        actual_heldout = heldout_h["n_articles"]
        if actual_heldout != n_heldout:
            raise RuntimeError(
                f"Split validation failed: corpus.heldout.sbcsr has "
                f"{actual_heldout:,} rows but expected {n_heldout:,} "
                f"(the train complement)")

    manifest = {
        "n_articles": n_full,
        "v_features": v,
        "nnz_total": full_h["nnz_total"],
        "nnz_mean": round(full_h["nnz_mean"], 2),
        "nnz_sd": round(full_h["nnz_sd"], 2),
        "density": round(density, 6),
        "n_train": actual_train,
        "n_heldout": n_heldout,
        "split_seed": 42,
        "train_fraction": train_fraction,
        "full_sbcsr_sha256": _sha256(full_path),
        "train_sbcsr_sha256": _sha256(train_path),
    }

    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Corpus manifest written to {out}")
    print(f"  N={n_full:,}  V={v:,}  nnz={full_h['nnz_total']:,}  "
          f"density={density:.4%}  train={actual_train:,}  held-out={n_heldout:,}")
    return out
