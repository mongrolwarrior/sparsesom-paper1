# sparsesom-paper1

Reproducibility repository for the Phase 1 SparseSOM manuscript — **every figure,
table, and headline statistic regenerated from the raw corpus**. One command in a
pinned Docker image reproduces the paper at the depth you choose.

## What the paper calls things vs. where the code lives

The repositories keep their development names; the manuscript uses cleaned-up
labels. The correspondence:

| name in the manuscript | repository |
|---|---|
| SparseBin.SOM | [`mongrolwarrior/SparseSOM`](https://github.com/mongrolwarrior/SparseSOM) |
| SparseFloat.SOM | same repository — the real-valued sparse variant of the same design |
| cuSPARSE.SOM | [`mongrolwarrior/StandardSparseSOM`](https://github.com/mongrolwarrior/StandardSparseSOM) |
| MedSOM | [`mongrolwarrior/MedSOM-Naive`](https://github.com/mongrolwarrior/MedSOM-Naive) |
| somoclu | [`peterwittek/somoclu`](https://github.com/peterwittek/somoclu) (third-party, pinned) |
| — | [`mongrolwarrior/SparseBinEval`](https://github.com/mongrolwarrior/SparseBinEval) (shared evaluator) |
| — | `mongrolwarrior/sparsesom-paper1` (this repository — the reproduction pipeline) |

Two names differ from the paper's: **`SparseSOM`** holds what the manuscript calls
SparseBin.SOM (the repository predates the paper's naming), and **`MedSOM-Naive`**
holds what the manuscript calls MedSOM — the suffix is a development-lineage label
distinguishing this original implementation from later experimental offshoots, not
a judgement on the code; it is the exact prior baseline the paper benchmarks.

---

## Prerequisites

- **Docker** with the **NVIDIA Container Toolkit** (`--gpus all` support).
- A **CUDA GPU with ≥ 24 GB** (an RTX 4090 is the reference; the image targets SM 8.6/8.9/9.0).
- The repo cloned **with submodules**: `git clone --recursive <repo-url>`.
- The anonymised MEDLINE corpus. The first run fetches it from Zenodo
  (concept DOI [10.5281/zenodo.20770707](https://doi.org/10.5281/zenodo.20770707));
  or mount a local copy (see *Data* below).

Build the image once (pins CUDA 12.8 and every binary — SparseBinarySOM,
StandardSparseSOM, MedSOM, somoclu, and the evaluator):

```bash
docker build -t sparsesom-paper1 .
```

For convenience, define the run command once (mounts `experiments/` so results and
the fetched corpus persist between runs):

```bash
RUN="docker run --gpus all -v $PWD/experiments:/workspace/experiments sparsesom-paper1"
```

Everything below runs `$RUN <args>`, which is just `repro <args>` inside the image.

---

## Choose your reproduction

There are two depths — `outline` (~3 h, direction and order of magnitude only;
reproduces no published value) and `full` (~6.5 days; regenerates every published
value and scores the 12 claims in `configs/claims.yaml`) — plus `sections`, which is
simply `full` run piecewise. They share the same pipeline and the same
`repro verify` acceptance checks.

### 1. `outline` — see the patterns (~3 hours)

The fastest path. Runs a minimum set at the cheapest map size that makes each
effect visible, and prints **PASS/FAIL against the direction and order of
magnitude** of every result in the paper.

```bash
$RUN --profile outline
```

It demonstrates the *shape* of every claim (QE decreasing in K, the layout
speed-up, the DRAM crossover, binary-vs-float, the somoclu/MedSOM gaps, …) in 18
checks. **It deliberately does not reproduce any published number, range, CI, or
hypothesis test** — single seed, reduced epochs, smallest edge. Use it to confirm
the toolchain works and the qualitative story holds. It ends with a disclaimer to
that effect.

### 2. `sections` — the full set, run in chunks

The same simulations as `full`, exposed as named sections you run one at a time
(useful for a cluster queue or resuming). Running **all** sections equals `full`.

```bash
$RUN --profile sections     # prints the ordered command list, then:
$RUN gate                   # corpus manifest + bring-up + PCA equivalence
$RUN sweep                  # size sweep + σ₀×PCA  (statistics)
$RUN bench                  # capacity frontier
$RUN compare                # impl-compare + efficiency + roofline + somoclu
$RUN verify                 # check accumulated results against configs/claims.yaml
```

The sections are an exhaustive, non-overlapping partition of the full run — each
writes into the same experiment folder.

### 3. `full` — the complete statistical reproduction (~6.5 days)

Every published value: **5 seeds, all edges (32–512), all implementations**, plus
the roofline and the somoclu/MedSOM CPU-vs-GPU baselines. Regenerates the data
behind every figure and table.

```bash
$RUN --profile full
$RUN verify
```

Budget realistically: the CPU baselines and the edge-512 runs dominate — somoclu
at 256² alone is ~23 h, MedSOM ~18 h — a clean-clone run of the whole profile took
**6.5 days** on a single 4090, so budget a week.
The `sections` path is the practical way to run this in stages.

> The edge-1024 (1,050,625-neuron) point in Table 7 was measured on an H200 under
> the earlier adaptive schedule and is disclosed as such (row `1024*`); it is **not**
> part of the 24 GB reproduction here.

---

## Verifying the claims

After any run, `repro verify` evaluates every manuscript claim in
`configs/claims.yaml` against the produced artifacts and prints per-claim PASS/FAIL:

```bash
$RUN verify
```

Each claim maps to a result file, a derivation, and an acceptance check (monotone,
CI-direction, exact, model-comparison, …). Against the frozen reference data this
passes 12/12.

Regenerate figures with `$RUN figures`.

---

## Building / running without Docker

Point the `SPARSESOM` / `STANDARDSPARSESOM` / `MEDSOM` / `SOMOCLU` / `SOM_METRICS`
env vars at locally-built binaries and run `python3 repro …` directly. Each
`external/` submodule builds with CMake; the evaluator is a single `g++` command.

**somoclu build note.** somoclu is the pristine upstream repo
(`peterwittek/somoclu`, pinned at `1.7.6-11-g63895f4`) — we make **no source
changes**. Its normal build is autotools: `./autogen.sh && ./configure --without-mpi
--without-cuda && make`. On a machine **without** autotools (`autoreconf`/`automake`
absent), build the CPU sparse kernel directly with the helper in this repo:
`scripts/somoclu_build/build_cpu.sh` — it compiles the same sources with `g++`,
supplying via a small prelude the `using namespace std;`/`VERSION` that `config.h`
would otherwise provide. Either binary works; set `SOMOCLU` to whichever you built.

---

## Repository structure

```
sparsesom-paper1/
├── external/           # Pinned upstream submodules (unmodified)
│   ├── SparseBinarySOM/  StandardSparseSOM/  SparseBinEval/  MedSOM/  somoclu/
├── configs/claims.yaml # every manuscript claim → artifact → acceptance check
├── pipeline/           # runners, parsers, stats, verification, the profiles
├── figures/            # figure generators (read results/*.parquet)
├── scripts/            # roofline (ncu), somoclu build, data prep
├── manifests/          # sha256 of the frozen reference results (data-frozen tag)
├── experiments/        # per-run data + results (git-ignored)
├── Dockerfile          # pinned CUDA 12.8 multi-stage build
└── repro               # CLI entrypoint (Docker ENTRYPOINT)
```

The ten experiments (`repro run <ID>`): 1 corpus manifest · 2 bring-up gate ·
3 PCA equivalence · 5 size sweep · 6 impl-compare · 7 capacity frontier ·
8 efficiency sweep · 9 σ₀×PCA · 10 roofline (needs `ncu`) · 11 somoclu.

### Reproducing the roofline (experiment 10) in Docker

Experiment 10 needs NVIDIA Nsight Compute (`ncu`) plus GPU performance-counter
access; without them the roofline check is **skipped, not failed**, and every other
experiment runs normally. To include it:

1. **Build the image with the profiler** (adds ~1 GB):

   ```bash
   docker build --build-arg WITH_NCU=1 -t sparsesom-paper1 .
   ```

2. **Grant counter access at run time** — either give the container
   `--cap-add=SYS_ADMIN` (NVIDIA's documented fix for `ERR_NVGPUCTRPERM` in
   containers), or lift the driver restriction host-wide and persistently:

   ```bash
   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | \
     sudo tee /etc/modprobe.d/nvidia-profiling.conf   # then reboot / reload the module
   ```

   Then:

   ```bash
   docker run --gpus all --cap-add=SYS_ADMIN \
     -v $PWD/experiments:/workspace/experiments sparsesom-paper1 run 10
   ```

With the roofline artifacts produced, `repro verify` covers all 12 claims instead
of 10. Budget several hours: edge 128 is captured exactly; edge 256 is sampled
(`NCU_CAP=60`) and scaled, matching the published methodology.

---

## Data

The released `.sbcsr` corpus contains **only MeSH descriptor indices** (integer
feature ids) — **no PMIDs, titles, or abstracts**. The author's contribution to it is
dedicated to the public domain under **CC0 1.0** and deposited on Zenodo
(DOI: [10.5281/zenodo.20822012](https://doi.org/10.5281/zenodo.20822012)). The
software is licensed separately under MIT (see [License](#license)).

**`metadata.sqlite` and `openalex_labels.sqlite` are LOCAL ONLY — they contain
PMIDs and MUST NOT be uploaded to Zenodo or any public repository.**

### NLM notices

The corpus derives from MEDLINE/PubMed, produced by the U.S. National Library of
Medicine (NLM). NLM's Terms and Conditions require the following notices wherever the
corpus is distributed (full text in
[`NLM_NOTICE_and_licence_text.md`](NLM_NOTICE_and_licence_text.md)):

- **Courtesy of the U.S. National Library of Medicine.**
- **Currency.** This is a fixed 2026 snapshot of MEDLINE/PubMed MeSH data. NLM updates
  its data regularly; this static copy does not reflect the most current or accurate
  data available from NLM. For authoritative, current data, consult
  [NLM](https://www.nlm.nih.gov).
- **No endorsement.** NLM has not reviewed, approved, or endorsed this work; the U.S.
  National Library of Medicine, NIH, and HHS do not endorse or recommend it and are
  not responsible for its content.

MEDLINE, PubMed, and MeSH are registered trademarks of the U.S. National Library of
Medicine.

---

## Manuscript précis

<!-- PLACEHOLDER — to be written by Claude Desktop.
     A short plain-language summary of the Phase 1 SparseSOM paper: the problem,
     the method (feature-major fused BMU, FP16 binary codebook), the headline
     results (layout speed-up, DRAM/bandwidth story, no-elbow scaling to 262k
     neurons, ~650× over the best CPU library), and what this repo reproduces. -->

_A plain-language summary of the paper will go here._

---

## License

MIT — see [LICENSE](LICENSE).
