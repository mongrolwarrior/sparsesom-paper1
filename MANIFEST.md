# MANIFEST — the frozen Phase 1 result set

The manuscript's tables and figures are derived from one frozen set of result files,
produced by experiment folder `SBS-230726-0827` and frozen on 2026-08-02. Their SHA-256
hashes are recorded in
[`manifests/DATA_MANIFEST_2026-08-02.sha256`](manifests/DATA_MANIFEST_2026-08-02.sha256);
verify any copy with

```bash
sha256sum -c manifests/DATA_MANIFEST_2026-08-02.sha256
```

run from the directory holding the files. The reproduction pipeline regenerates every one of
them (`repro --profile full`, then `repro verify` scores `configs/claims.yaml` against them).

| File | Produced by | Feeds |
|------|-------------|-------|
| `corpus_manifest.json` | run 1 (corpus manifest) | Table 1; ties the results to the corpus by SHA-256 |
| `bringup_gate.json` | run 2 (bring-up gate) | §5.1 bit-exactness claim |
| `size_sweep.parquet`, `size_sweep_epochs.parquet` | run 5 (size sweep) | Tables 7–8, Figs 4 and 6 |
| `sigma0_sweep.parquet`, `sigma0_sweep_epochs.parquet` | run 9 (σ₀ × PCA) | Table 5 |
| `impl_compare.parquet` | run 6 (implementation comparison) | Tables 3 and 4 |
| `efficiency_sweep.parquet` (+ `.csv`) | runs 8 and 11 (efficiency sweep, MedSOM ladder, somoclu anchor) | Table 6, Fig 5 |
| `roofline_phase_summary.csv`, `roofline_phase_breakdown.csv` | run 10 (Nsight Compute capture + `scripts/aggregate_phase.py`) | Tables 8–9, Fig 7 |

Result folders under `experiments/` are git-ignored; the frozen copies distributed with the
release live in `frozen/2026-08-02/` (same files, same hashes).

## Submodule versions

Pinned by the superproject; list them with `git submodule status`.
