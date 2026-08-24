# `--profile outline` — fast pattern-reproduction spec

**Purpose.** A ~3-hour single-GPU run that demonstrates every qualitative result in the Phase 1 manuscript. It is **not** a statistical reproduction: it shows each effect's *direction* and rough *order of magnitude*, and deliberately does not support any published numeric value, range, confidence interval, or hypothesis test. Those come from `--profile full`.

**Contrast with the other profiles**

| profile | purpose | cost |
|---|---|---|
| `outline` | show the patterns; verify the pipeline reproduces every shape | ~3 h |
| `minimal` | smallest set that still supports the paper's tests (5-rung elbow test, 2-point ranges) | ~13.5 h |
| `full` | published values: 5 seeds, all edges, all implementations | 100 h+ |

**Global settings.** Single seed (seed 0) throughout. One canonical full-corpus 90/10 split (seed 42), as in all profiles. FP16 codebook / FP32 accumulation for every GPU implementation except MedSOM and somoclu (FP32 by construction). Deterministic exponential σ schedule, rate 0.3, σ₀ = 0.5·E. PCA init. Kaski–Lagus stop where marked `KL`; otherwise the fixed epoch count given. **Nothing in this profile requires more than 24 GB of device memory; edge 1024 and above are excluded entirely.**

**Why per-epoch claims get 1–3 epochs.** MedSOM, somoclu, binary-vs-float and all DRAM figures are per-epoch quantities. Running them to 20 epochs multiplies cost without adding evidence, so the profile runs the fewest epochs that reach steady state (2–3; epoch 0 is discarded as warm-up where more than one epoch is run).

---

## Run list

| # | Run | Impl(s) | Edges | Epochs | Est. cost |
|---|---|---|---|---|---|
| 1 | `corpus_manifest` | — | — | — | secs |
| 2 | `bitexact_gate` | sbsom-bin, FP16 vs FP32 | 32 (tiny ~50k corpus) | 5 | 1 min |
| 3 | `determinism_check` | sbsom-bin, seeds 0 and 1 | 32 | KL | 2 min |
| 4 | `size_sweep` | sbsom-bin | 32, 64, 128, 256, 512 | KL | **1.9 h** |
| 5 | `impl_crossover` | ssom-feat | 32, 64, 128 | KL | 8 min |
| 6 | `layout_gap` | ssom-node | 32, 64 | KL | 10 min |
| 7 | `update_rule` | ssom-feat, variants {gaussian, box} | 128 | 10 fixed | 3 min |
| 8 | `fp16_asymmetry` | ssom-feat, ssom-node at **FP32** | 32 | KL | 2 min |
| 9 | `binary_vs_float` | sbsom-float | 128 | 3 | 1 min |
| 10 | `medsom_ref` | MedSOM | 64 | 2 | 7 min |
| 11 | `somoclu_ref` | somoclu (28 threads) | 32 | 2 | 14 min |
| 12 | `capacity_probe` | ssom-feat, ssom-node, MedSOM | 512 | — (expect OOM) | secs |
| 13 | `roofline` | ncu phase-total: sbsom-bin, ssom-feat | 128 | 1 | 20 min |
| 14 | `sigma0_interaction` | sbsom-bin: {PCA, random} × σ₀ {0.5·E, 0.25·E} | 64 | KL | 5 min |

**Total ≈ 3.1 h.** Run 4 dominates: its edge-512 rung alone is ~1.5 h, while the other four rungs together are ~24 min. Run 4's sbsom-bin rows also **serve as the sbsom arm for runs 5–7** (identical config — same edges give the same epochs, wall and QE), so they must not be re-run.

## Reuse and ordering

- Runs 1–3 first (cheap gates). If `bitexact_gate` fails, stop: the exactness argument underpins every quality claim.
- Run 4 next; runs 5–9 and 14 read its sbsom-bin rows as the comparison arm.
- Runs 10–12 any time. Run 13 last (out-of-CI, manual ncu).
- Runs 11 (somoclu) and 12 are CPU / instant and may run alongside GPU work.

---

## Expected patterns — assert direction, not value

The runner should print **PASS/FAIL against the direction or order of magnitude**, never against the manuscript's number. Suggested assertions, with the value this profile is expected to produce and the corresponding published figure for reference only:

| # | Assertion (direction) | outline expects | published (`full`) |
|---|---|---|---|
| 2 | BMU assignment identical FP16 vs FP32 | 0 mismatches | 0 mismatches |
| 3 | Both seeds identical (PCA is deterministic) | zero variance | zero variance |
| 4 | held-out QE strictly decreasing in K; no kink on log-log | 0.524 → 0.376 | 0.524 → 0.376 |
| 4 | per-doubling gain roughly constant, well above 0.5% | ~3–5% | 3.3–4.8% |
| 4 | BMU share of wall > 95% and rising with K | ~98.7 → 99.5% | 98.7 → 99.5% |
| 4 | largest map trains on 24 GB | 262,144 neurons | 262,144 |
| 5 | wall ratio ssom-feat/sbsom **crosses 1.0** with increasing edge | 0.36× → 1.05× → 1.47× | 0.36× → 2.60× (to edge 256) |
| 5 | held-out QE agrees between implementations within ~1% | ≤1% | ≤0.5% |
| 6 | feature-major BMU several-fold faster than node-major | ~8× | 4.5–8.5× |
| 7 | box blur cuts TE by ~an order of magnitude vs Gaussian | 0.21 → 0.03 | 0.21 → 0.03 |
| 8 | FP16 helps feature-major much more than node-major | ~1.6× vs ~1.1× | 1.65× vs 1.07× |
| 9 | binary faster than float, order ~2–3× | ~2.8× | 2.1–2.8× |
| 10 | MedSOM slower by ≥1 order of magnitude | ~55× | ~82× (edge 128) |
| 11 | somoclu slower by ≥2 orders of magnitude | ~180× | ~692× (edge 128) |
| 12 | ssom-feat, ssom-node, MedSOM all OOM at 512; sbsom-bin does not | 3 OOM, 1 pass | same |
| 13 | ssom-feat sustains a small fraction of peak DRAM bandwidth | ~20–25% | 15.6% (edge 256) |
| 14 | σ₀ = 0.25·E improves QE under PCA and degrades it under random | signs opposite | same |

**Deliberate divergences from published values.** Runs 10, 11 and 5 are measured at cheaper edges than the paper's, so they will differ systematically and by design: somoclu ~180× rather than ~692×, MedSOM ~55× rather than ~82×, and the crossover ratio topping out at 1.47× rather than 2.60×. These are not failures. The runner must not compare them to the published figures.

**One caveat to carry on run 13.** The DRAM *traffic* comparison may change sign between edge 128 and edge 256 (at 128 the cuSPARSE score-block round trip appears to dominate; at 256 our codebook re-reads do). This profile profiles edge 128 only, so it should report the **bandwidth** pattern — that cuSPARSE runs far below peak and is therefore not bandwidth-limited — and must **not** assert a direction for which implementation moves more DRAM. That comparison belongs to the paper's headline edge (256) and to `--profile full`.

---

## Required summary output

`repro --profile outline` must end with a block to this effect, so results are not mistaken for the published values:

> **Profile: outline (pattern reproduction).** 14 checks, N passed. This profile demonstrates the qualitative results of the Phase 1 manuscript — the direction and approximate scale of every reported effect. It uses a single seed, reduced epoch budgets and the cheapest map size at which each effect is visible, so **the numeric values here differ from the published figures by design** and support none of the paper's ranges, exponents, confidence intervals or hypothesis tests. In particular the somoclu, MedSOM and crossover ratios are measured at smaller maps and are correspondingly smaller. For the published values run `--profile full`; for the smallest set that still supports the paper's statistical tests, run `--profile minimal`.

Per-run rows should print: run name, implementations, edges, epochs, the asserted direction, the observed value, and PASS/FAIL — with a `published:` column shown greyed or parenthesised for orientation only, explicitly labelled *not a target*.
