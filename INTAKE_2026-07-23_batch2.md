# Phase 1 results intake — 2026-07-23 (batch 2: runs 4 & 7)

**Delivered:** BMU kernel microbenchmark (run 4 → Table 2 / Fig 3) and capacity frontier (run 7 → E1). Archived to `results/phase1_repro/2026-07-23_batch2/`. σ₀ experiment still running; size sweep / impl-compare / efficiency / roofline still pending.

**Verdict:** the capacity frontier is clean and consistent. **The microbenchmark does not reproduce the paper's headline number and must be treated as a STOP-and-investigate before the manuscript is touched.** I have not edited the manuscript.

---

## 1. Microbenchmark (run 4) — CRITICAL: the 281× headline does not reproduce

Kernels are computing correctly (bmu_mismatch_count = 0 everywhere), but the timings bear no resemblance to Table 2.

**Speed-up vs node-major:**

| kernel | pass | reproduced | manuscript |
|---|---|---|---|
| spmm-tile | Pass 1 | **7.07×** | 40.8× |
| spmm-tile | Pass 2 | **7.32×** | **281×** |
| feature-major | Pass 1 | 4.08× | 24.9× |
| feature-major | Pass 2 | 3.67× | 31.3× |

The abstract, §5.1, and the conclusion all lead with "**up to 281×**." The reproduction gives **7×** — a factor of ~40 lower.

**The scaling story also fails to reproduce.** The manuscript's mechanism is that node-major degrades super-linearly (9.76× time for 4× work) while spmm-tile scales near-linearly (1.4×). In the new data *all three* kernels scale ~2× for 4× work (spmm 1.97×, feature 2.27×, node 2.04×) — node-major shows no super-linear degradation at all. The qualitative argument of §5.1, not just the number, is absent.

**Diagnosis.** This is not a wrong-answer bug (0 mismatches — the kernels agree). It is a *performance* discrepancy, and the pattern points to configuration, not noise (seed-to-seed variance is <2%):

- Absolute times are 30–180× larger than the manuscript's, and unevenly so: spmm-tile slowed 178× (115.7 ms → 20,580 ms) while node-major slowed only 31× (4,717 ms → 145,557 ms). spmm-tile's *relative* advantage is what collapsed — its optimisation isn't paying off in this configuration.
- The manuscript microbench was D = 32,768, nnz = 10, TA = 8, 50-iteration mean. The repo's config for this run (TA, D, nnz, iteration count, warm-up handling, and whether the timed region includes H2D transfer) is unknown and is the first thing to establish.

**Required of Claude Code (before any manuscript change):**
1. Report the exact microbench config actually run (TA, D, nnz, iterations, warm-up, timed region) and diff it against TA = 8 / D = 32,768 / nnz = 10.
2. Confirm the spmm-tile kernel is taking its optimised tiled path, not a fallback, at this problem size — the 178× slowdown of spmm-tile specifically is the smoking gun.
3. Check whether the timed region now includes something the original excluded (e.g. transfers, allocation, or a per-call sync) that would compress the ratios.
4. Re-run at the documented TA = 8 configuration and report whether the 281× / super-linear-node behaviour returns.

Until this is resolved we cannot know whether (a) the repo benchmark is mis-configured, or (b) the original 281× was an artefact — and those have opposite consequences for the paper. **Do not update Table 2 or the headline either way yet.**

---

## 2. Capacity frontier (run 7 / E1) — clean, consistent, with one expected shift

| method | max edge | max neurons | mem (GB) | limit |
|---|---|---|---|---|
| somoclu-gpu | — | — | 3680 (analytic) | infeasible: dense input alone needs 3680 GB |
| **sbsom-bin** | **559** | 312,481 | 19.2 | OOM (largest on 4090, FP16) |
| sbsom-float | 512 | 262,144 | 16.1 | OOM |
| ssom-feat (cuSPARSE) | 256 | 65,536 | 6.0 | OOM beyond (score-tile) |
| ssom-node (cuSPARSE) | 256 | 65,536 | 6.0 | OOM beyond |
| medsom | 128 | 16,384 | 2.0 | wall-time impractical |
| somoclu-cpu | 128 | 16,384 | 5.0 (64 GB RAM) | wall-time |

**Consistency checks all pass:**
- sbsom-bin ceiling **559 / 312,481 neurons** matches the manuscript's stated 4090 memory limit (Table 8) exactly.
- somoclu-gpu infeasible (3680 GB = V·N·4 for the full corpus) matches the §2.2 dense-infeasibility argument. *(Note: the analysis spec's E1 figure of "736 GB" was wrong; 3680 GB is the correct V·N·4 — fix the spec, not the paper.)*
- The cuSPARSE ceiling is now **edge 256**, up from the manuscript's FP32 "OOM by ~257 / feasible ~182." This is the **expected consequence of the FP16-matched cuSPARSE change** — halving its codebook raises its ceiling. It's a to-be-updated number (Table 6 "both cuSPARSE OOM at 257" becomes "feasible to 256"), not a problem.

**Status:** E1 is usable and will feed the new feasibility table that replaces part of Table 7 — but that table needs the efficiency sweep (E2), which is still pending, so I've held it rather than build half of it.

---

## 3. What I changed

Nothing in the manuscript. Batch 2 delivers one clean-but-partial result (E1, waiting on E2) and one result that actively contradicts the paper's headline and needs investigation first. Neither is ready to write in.

## 4. Recommended next actions

1. **Claude Code — microbench investigation (highest priority):** the four steps in §1. This gates the single most important claim in the paper.
2. **Analysis spec fix:** correct the E1 somoclu-gpu figure 736 GB → 3680 GB.
3. **Hold** the Table 6 cuSPARSE-ceiling update (182→256) until it can be made together with the FP16-matched efficiency numbers from E2.
4. Continue awaiting σ₀ experiment, size sweep, impl-compare, efficiency, roofline.
