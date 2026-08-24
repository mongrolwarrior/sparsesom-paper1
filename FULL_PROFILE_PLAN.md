# Making `--profile full` reproduce the whole paper (P1), then sections (P2)

Goal: `repro --profile full` runs every simulation the manuscript's statistics
rest on; the named sections partition that set (run all sections = full);
`--profile demonstrate` stays the ~3h pattern set.

## Current gap
`repro all` runs experiments 1,2,3,5,6,7,8. It does NOT run:
- **ex09 sigma0_pca** — in the registry but in no section and not in `all`
  (it produced the σ₀×PCA finding that replaced pca_halves_epochs).
- **roofline** (§5.7) — only in `scripts/ncu_full_epoch.sh` + `aggregate_phase.py`.
- **somoclu anchor** (§5.5 CPU baseline) — only in `scripts/run_somoclu_anchor.py`.
- **MedSOM ladder** (§5.4) — the working version is `scripts/run_medsom_ladder.py`;
  ex08's inline MedSOM crashed on the sbcsr CLI.
- **equal-quality ssom** (§5.3) — the `--epochs 200` KL rerun is
  `scripts/rerun_ssom_kl_arms.py`, not ex06 (ex06 left ssom at the 10-epoch cap).
- **ex04 microbench** — dead code, retired; delete.

## P1 steps — fold standalone scripts into pipeline experiments
1. Delete `pipeline/experiments/ex04_microbench.py`; drop from registry.
2. New `ex10_roofline.py` wrapping ncu_full_epoch + aggregate_phase (edges 128
   exact, 256 sampled). Out-of-CI (needs ncu); guarded/skippable.
3. New `ex11_somoclu.py` wrapping run_somoclu_anchor (libsvm convert + 4-edge).
4. Fix ex08 to call the corrected MedSOM path (run_medsom_ladder logic) and to
   run the ssom arms to convergence (fold rerun_ssom_kl_arms) so ex06/ex08 match
   what produced the numbers.
5. Add 9,10,11 to the registry; make `all = [1,2,3,5,6,7,8,9,10,11]`.
6. `repro verify` maps every claim in configs/claims.yaml to a produced artifact;
   add the missing ones (dram_ratio→roofline, sigma0 finding, somoclu ratio).

## P2 steps — sections partition full
- gate=[1,2,3]  sweep=[5,9]  bench=[7]  compare=[6,8,10,11]
  (choose the split so every experiment is in exactly one section; verify
  gate∪sweep∪bench∪compare == all, no overlap).
- `--profile sections` prints this exact list; add a `repro sections --check`
  that asserts the partition is exhaustive.

## Validation
- `--profile demonstrate` end-to-end (~3h) — P3, already coded.
- After P1, a dry `repro verify` on the existing SBS-230726-0827 results should
  pass every claim with an artifact present.

Provenance, seeds (5 for statistical arms), and the 24GB/edge-≤512 limit are
unchanged. H200 1024 stays excluded (documented).
