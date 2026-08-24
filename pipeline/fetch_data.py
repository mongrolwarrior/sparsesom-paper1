"""Fetch the anonymised .sbcsr corpus from Zenodo (or local path) and verify checksums."""

import argparse
import hashlib
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen, Request
import json

import yaml


ZENODO_RECORD_API = "https://zenodo.org/api/records/{record_id}"

FILE_MAP = {
    "corpus.sbcsr": "corpus.full.sbcsr",
    "summary.json": "summary.json",
    "vocab.json": "vocab.json",
}

# Pre-split files to copy only from the Zenodo tarball (where we control the split).
# Local sources may have splits from different parameters, so _copy_local skips these
# and lets _ensure_split derive them from the full corpus.
_TARBALL_EXTRA_FILES = {
    "corpus.train.sbcsr": "corpus.train.sbcsr",
    "corpus.val.sbcsr": "corpus.heldout.sbcsr",
    "pca_components.sompca": "corpus.sompca",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_zenodo_doi(doi: str) -> tuple[str, str]:
    """Resolve a Zenodo DOI to the record ID and file download URL."""
    record_id = doi.split(".")[-1]

    api_url = ZENODO_RECORD_API.format(record_id=record_id)
    try:
        with urlopen(api_url) as resp:
            data = json.loads(resp.read())
    except Exception:
        api_url_v2 = f"https://zenodo.org/api/records/{record_id}"
        with urlopen(api_url_v2) as resp:
            data = json.loads(resp.read())

    files = data.get("files", [])
    for f in files:
        if f["key"].endswith(".tar.gz"):
            return record_id, f["links"]["self"]

    for f in files:
        link = f.get("links", {}).get("self", "")
        if link:
            base = link.rsplit("/files/", 1)[0]
            return record_id, f"{base}/files/medline-mesh-pubmed26-ge5.tar.gz/content"

    raise RuntimeError(f"No tar.gz file found in Zenodo record {record_id}")


def _download_with_resume(url: str, dest: Path, max_retries: int = 5) -> Path:
    """Download a file with resume support and retries."""
    print(f"  Downloading: {url}")
    print(f"  Destination: {dest}")

    for attempt in range(max_retries):
        existing_size = dest.stat().st_size if dest.exists() else 0

        req = Request(url)
        if existing_size > 0:
            req.add_header("Range", f"bytes={existing_size}-")
            print(f"  Resuming from {existing_size / 1e6:.0f} MB (attempt {attempt + 1}) ...")

        try:
            with urlopen(req) as resp:
                total_size = int(resp.headers.get("Content-Length", 0))
                if resp.status == 206:
                    total_size += existing_size
                elif resp.status == 200 and existing_size > 0:
                    existing_size = 0
                    total_size = int(resp.headers.get("Content-Length", 0))

                mode = "ab" if existing_size > 0 and resp.status == 206 else "wb"
                downloaded = existing_size if mode == "ab" else 0

                with open(dest, mode) as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = downloaded * 100 // max(total_size, 1)
                        print(f"\r  {downloaded / 1e6:.0f} MB / {total_size / 1e6:.0f} MB ({pct}%)",
                              end="", flush=True)

                print()

                if total_size > 0 and downloaded < total_size:
                    raise IOError(f"Incomplete: got {downloaded} of {total_size} bytes")

                return dest

        except (IOError, OSError) as e:
            print(f"\n  Download interrupted: {e}")
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Retrying in {wait}s ...")
                time.sleep(wait)
            else:
                print(f"  Failed after {max_retries} attempts.")
                raise

    return dest


SPLIT_SEED = 42
TRAIN_FRACTION = 0.9


def _read_sbcsr_n_rows(path: Path) -> int:
    """Read just the row count from an .sbcsr header."""
    with open(path, "rb") as f:
        f.read(8)  # magic
        n_rows = struct.unpack("<I", f.read(4))[0]
    return n_rows


def _split_is_valid(dest: Path) -> bool:
    """Check whether the existing train/heldout split matches expected parameters.

    Returns False (triggering a re-split) if:
    - .split_params.json is missing (split was done by an unknown tool)
    - The recorded parameters don't match the current config
    - The actual row counts don't match the recorded ones
    """
    manifest = dest / ".split_params.json"
    full = dest / "corpus.full.sbcsr"
    train = dest / "corpus.train.sbcsr"
    heldout = dest / "corpus.heldout.sbcsr"

    if not all(p.exists() for p in (manifest, full, train, heldout)):
        return False

    try:
        with open(manifest) as f:
            params = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    if params.get("seed") != SPLIT_SEED:
        return False
    if abs(params.get("train_fraction", 0) - TRAIN_FRACTION) > 1e-6:
        return False

    n_full = _read_sbcsr_n_rows(full)
    n_train = _read_sbcsr_n_rows(train)
    n_heldout = _read_sbcsr_n_rows(heldout)
    expected_train = int(n_full * TRAIN_FRACTION)

    if n_train != params.get("train_n") or n_heldout != params.get("heldout_n"):
        return False
    if abs(n_train - expected_train) > 100:
        return False
    if n_train + n_heldout != n_full:
        return False

    return True


def _write_split_manifest(dest: Path):
    """Record the split parameters so future runs can verify them."""
    full = dest / "corpus.full.sbcsr"
    train = dest / "corpus.train.sbcsr"
    heldout = dest / "corpus.heldout.sbcsr"

    params = {
        "seed": SPLIT_SEED,
        "train_fraction": TRAIN_FRACTION,
        "full_n": _read_sbcsr_n_rows(full),
        "train_n": _read_sbcsr_n_rows(train),
        "heldout_n": _read_sbcsr_n_rows(heldout),
    }
    with open(dest / ".split_params.json", "w") as f:
        json.dump(params, f, indent=2)


def _do_split(dest: Path):
    """Run the split and write the manifest."""
    full = dest / "corpus.full.sbcsr"
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    subprocess.run(
        [sys.executable, str(scripts_dir / "split_corpus.py"),
         str(full), str(dest),
         "--seed", str(SPLIT_SEED), "--train-frac", str(TRAIN_FRACTION)],
        check=True,
    )
    _write_split_manifest(dest)


def _ensure_split(dest: Path):
    """Split full corpus into train/heldout, verifying any existing split is correct."""
    full = dest / "corpus.full.sbcsr"
    if not full.exists():
        return

    train = dest / "corpus.train.sbcsr"
    if not train.exists():
        print("\nSplitting corpus (seed=42, 90/10 train/heldout) ...")
        _do_split(dest)
        return

    if _split_is_valid(dest):
        return

    n_full = _read_sbcsr_n_rows(full)
    n_train = _read_sbcsr_n_rows(train)
    expected = int(n_full * TRAIN_FRACTION)
    print(f"\nWARNING: existing corpus.train.sbcsr has {n_train:,} rows, "
          f"expected ~{expected:,} ({TRAIN_FRACTION*100:.0f}% of {n_full:,})")
    print("Re-splitting corpus ...")
    _do_split(dest)
    # PCA and MedSOM data depend on the train corpus — invalidate them
    for stale in [dest / "corpus.sompca", dest / "medsom_input"]:
        if stale.exists():
            print(f"  Removing stale derived data: {stale.name}")
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()


def _ensure_pca(dest: Path):
    """Compute PCA components from training corpus if not already done."""
    train = dest / "corpus.train.sbcsr"
    pca = dest / "corpus.sompca"
    if train.exists() and not pca.exists():
        print("\nComputing PCA components (top-2 SVD) ...")
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        subprocess.run(
            [sys.executable, str(scripts_dir / "compute_pca.py"),
             str(train), "--out", str(pca)],
            check=True,
        )


def _ensure_medsom(dest: Path):
    """Convert corpus to MedSOM directory format if not already done."""
    medsom_dir = dest / "medsom_input"
    train_sbcsr = dest / "corpus.train.sbcsr"
    if train_sbcsr.exists() and not (medsom_dir / "bin" / "mesh_offsets.bin").exists():
        print("\nConverting corpus to MedSOM format ...")
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        subprocess.run(
            [sys.executable, str(scripts_dir / "sbcsr_to_medsom.py"),
             str(train_sbcsr), str(medsom_dir)],
            check=True,
        )


def fetch_and_prepare(dest: Path, local_source: Path | None = None,
                      skip_verify: bool = False):
    """Download corpus from Zenodo (or local source) and prepare splits.

    Called programmatically by the CLI; also usable as a library function.
    """
    dest.mkdir(parents=True, exist_ok=True)

    config_dir = Path(__file__).resolve().parent.parent / "configs"
    with open(config_dir / "defaults.yaml") as f:
        defaults = yaml.safe_load(f)

    corpus_cfg = defaults.get("corpus", {})
    zenodo_doi = corpus_cfg.get("zenodo_doi")
    expected_sha256 = corpus_cfg.get("tarball_sha256")

    required_outputs = ["corpus.train.sbcsr", "corpus.heldout.sbcsr", "corpus.sompca"]
    existing = [f for f in required_outputs if (dest / f).exists()]
    if len(existing) == len(required_outputs):
        print("All required files already present:")
        for f in required_outputs:
            sz = (dest / f).stat().st_size / 1e6
            print(f"  {f:30s} {sz:8.1f} MB")
        _ensure_split(dest)
        _ensure_pca(dest)
        _ensure_medsom(dest)
        return

    if local_source:
        _copy_local(local_source, dest)
    elif zenodo_doi:
        _download_zenodo(zenodo_doi, dest, expected_sha256,
                         skip_verify=skip_verify)
    else:
        raise RuntimeError(
            "No Zenodo DOI configured and no local source given. "
            "Set corpus.zenodo_doi in configs/defaults.yaml or pass local_source.")

    _ensure_split(dest)
    _ensure_pca(dest)
    _ensure_medsom(dest)


def main():
    parser = argparse.ArgumentParser(description="Fetch corpus data")
    parser.add_argument("--dest", type=Path, default=Path("data"))
    parser.add_argument("--local", type=Path, default=None,
                        help="Copy from a local directory instead of Zenodo")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip SHA256 verification of the tarball")
    args = parser.parse_args()

    fetch_and_prepare(args.dest, local_source=args.local,
                      skip_verify=args.skip_verify)
    print("\nAll required files present. Ready for: repro gate")


def _copy_local(src_dir: Path, dest: Path):
    """Copy only source corpus files from a local directory.

    Copies only the full (unsplit) corpus, summary, and vocab — NOT pre-split
    train/heldout files, which may have been created with different parameters.
    The split is derived from the full corpus by _ensure_split().
    """
    print(f"Copying corpus from local path: {src_dir}")

    for src_name, dst_name in FILE_MAP.items():
        src = src_dir / src_name
        if not src.exists():
            for candidate in src_dir.rglob(src_name):
                src = candidate
                break

        dst = dest / dst_name
        if src.exists():
            print(f"  {src_name} → {dst_name} ...", end=" ", flush=True)
            shutil.copy2(src, dst)
            print(f"OK ({dst.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  WARNING: {src_name} not found")


def _download_zenodo(doi: str, dest: Path, expected_sha256: str | None,
                     skip_verify: bool = False):
    """Download from Zenodo, verify checksum, extract."""
    print(f"Resolving Zenodo DOI: {doi}")
    record_id, download_url = _resolve_zenodo_doi(doi)
    print(f"  Record: {record_id}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tarball = Path(tmpdir) / "corpus.tar.gz"

        _download_with_resume(download_url, tarball)

        # Verify tarball checksum
        if not skip_verify and expected_sha256:
            print("  Verifying SHA256 ...", end=" ", flush=True)
            actual = sha256_file(tarball)
            if actual != expected_sha256:
                print(f"MISMATCH!")
                print(f"    Expected: {expected_sha256}")
                print(f"    Got:      {actual}")
                print("    Use --skip-verify to proceed anyway.")
                sys.exit(1)
            print("OK")
        else:
            print("  Skipping checksum verification.")

        # Extract
        print("  Extracting ...", end=" ", flush=True)
        extract_dir = Path(tmpdir) / "extracted"
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(extract_dir)
        print("OK")

        # Find and copy files using the mapping (source files + any pre-split extras)
        all_files = {**FILE_MAP, **_TARBALL_EXTRA_FILES}
        for src_name, dst_name in all_files.items():
            matches = list(extract_dir.rglob(src_name))
            if matches:
                src = matches[0]
                dst = dest / dst_name
                shutil.copy2(src, dst)
                print(f"  {src_name} → {dst_name} ({dst.stat().st_size / 1e6:.1f} MB)")
            else:
                print(f"  WARNING: {src_name} not found in archive")


if __name__ == "__main__":
    main()
