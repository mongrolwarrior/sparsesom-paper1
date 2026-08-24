# Experiment spec — PCA initial-σ optimisation (and the keep-or-drop-PCA decision)

**Status:** self-contained side-experiment. Runs on the RTX 4090. Does **not** gate the main size sweep — run the sweep on the σ₀ = 0.5·E baseline regardless, and run this in parallel or after. Consistent with `PHASE1_HANDOVER_ITERATIONS.md` and `PHASE1_ANALYSIS_SPEC.md`; uses the same corpus, split, deterministic rate-0.3 schedule, and KL stop.

## 1. Motivation

Batch-1 results (`INTAKE_2026-07-23.md`) showed that under the deterministic σ schedule, PCA initialisation gives **no epoch advantage** over random init at the shared σ₀ = 0.5·E — both converge in ~16–25 epochs — even though PCA starts from a better-ordered map. The likely reason: a linearly-initialised map already carries its large-scale topology, so the wide-σ *ordering* phase is partly wasted work; a fixed σ₀ = 0.5·E forces PCA through it anyway. The classic benefit of PCA init is precisely that it permits a **smaller σ₀**, skipping straight toward fine-tuning. This experiment tests whether a reduced σ₀ converts PCA's head start into fewer epochs **without changing quality**.

**Critical caveat from the data — the head start is edge-dependent.** PCA's initial topographic error rises steeply with map size, so "already ordered" holds only at small maps:

| edge | PCA epoch-0 TE |
|---|---|
| 32 | 0.000 |
| 64 | 0.017 |
| 128 | 0.225 |
| 1024 | 0.739 |

At large maps a 2-component linear projection is badly disordered and genuinely needs the wide-σ phase. So the safe σ₀ reduction is expected to be **aggressive at small edges and conservative (or nil) at large edges**, and a single constant like σ₀ = edge/4 is unlikely to be uniformly safe. Do not assume it will help the H200 rungs — at edge 1024 the ordering phase is doing real work.

## 2. Questions

- **Q1 (σ₀ tuning).** For PCA init, what is the smallest σ₀ per edge that preserves converged quality, and how many epochs does it save?
- **Q2 (keep or drop PCA).** Does PCA at its best σ₀ actually beat **random init at σ₀ = 0.5·E** on epochs, at equal quality? This is the decision test for whether PCA earns its place in the method at all (see §6).

## 3. Design

**Arms (all on the deterministic rate-0.3 schedule, KL stop, canonical split, full corpus):**

| arm | init | σ₀ | seeds |
|---|---|---|---|
| A (baseline) | PCA | 0.5·E | 1 (deterministic) |
| B | PCA | 0.25·E | 1 |
| C | PCA | 0.125·E | 1 |
| R (reference) | random | 0.5·E | 5 (ensemble) |
| R′ (control, optional) | random | 0.25·E | 3 |

**Edges:** 32, 64, 128, 256, 512 (RTX 4090). No H200 — the caveat in §1 makes a reduced σ₀ unpromising and expensive there; only extend to 1024 if the 4090 trend is strongly positive and safe at 512.

**Seeds.** PCA arms are deterministic (batch-1 confirmed seed-sd = 0), so **1 seed each** — no CI, exact values. The random reference needs its ensemble (5 seeds) because its epoch count and quality vary seed to seed; report random as mean ± 95% CI (t-based, per analysis-spec §0). Control arm R′ only needs enough seeds to show direction.

**Per-run metrics:** `epochs, wall_s, qe_heldout, qe_eucl, te, dead_frac, converged`, plus the per-epoch σ and QE trace so the ordering-phase length is visible.

## 4. Acceptance / decision rules

**Q1 — smallest safe σ₀ per edge.** For each edge, the chosen σ₀ is the smallest of {0.5, 0.25, 0.125}·E for which, versus PCA@0.5·E at that edge:

- ΔQE_heldout ≤ **0.5%** relative, **and**
- ΔTE ≤ **+0.005** absolute (TE must not rise — the primary failure mode of too-small σ₀ is topological folds), **and**
- Δdead_frac ≤ **+0.02** absolute.

Report the epoch saving = epochs(PCA@0.5·E) − epochs(PCA@chosen σ₀). Expect this to shrink toward zero as edge grows.

**Q2 — does PCA earn its place?** At each edge, compare PCA@chosen-σ₀ epochs against random@0.5·E epochs (mean of the ensemble) at matched quality. PCA is worth keeping only if it delivers a **material, quality-neutral epoch reduction** (suggest a ≥ 25% epoch saving at ≥ 2 edges) that random init cannot match. Otherwise Q2 fails (see §6).

## 5. Phase-2 safety spot-check (required before adopting any reduced σ₀)

σ₀ governs how strongly the map relaxes to *global* structure, which is exactly what the Phase-2 recovery metrics measure — so a reduced σ₀ can preserve QE/TE yet shift the layout. This is the one place this experiment deliberately steps outside the "no Phase-2 metrics" rule. At **one small edge** (e.g. 50×50, the Phase-2 operating point) compute cophenetic correlation and AMI against the MeSH tree for PCA@0.5·E versus PCA@chosen-σ₀. Adopt the reduced σ₀ as a default **only if** recovery is unchanged within its own tolerance; otherwise the reduced σ₀ is an efficiency-only option, not a global default, and must be flagged as potentially affecting Phase 2.

## 6. What each outcome means

- **Q1 passes and Q2 passes** → PCA earns its place. Adopt an **edge-dependent σ₀ rule** for PCA, reinstate a *principled* PCA-efficiency claim in the manuscript ("PCA init permits a reduced σ₀, cutting epochs by X% at small-to-mid maps with no quality change"), and update §3.4 accordingly. Keep PCA as the sweep default.
- **Q1 passes but Q2 fails** (savings real but modest, or matched by random) → PCA's benefit does not justify the extra parameter and the init asymmetry with the baselines. Prefer **dropping PCA for a fixed-seed random init** (see the removal analysis below).
- **Q1 fails** (no safe σ₀ reduction) → PCA provides neither quality nor speed under the deterministic schedule; its only remaining property is seed-invariance, which is redundant with fixing the random seed → **drop PCA**.

## 7. Output artifact

`results/sigma0_sweep.parquet` → `edge, init, sigma0_frac, seed, epochs, wall_s, qe_heldout, qe_eucl, te, dead_frac, converged`
`results/sigma0_sweep_epochs.parquet` → `edge, init, sigma0_frac, seed, epoch, sigma, qe_heldout, te` (per-epoch trace)
`results/sigma0_phase2_spotcheck.json` → `{edge, init, sigma0_frac, cophenetic, ami}` for baseline vs chosen σ₀.

Provenance columns as per every other run.
