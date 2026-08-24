# sparsesom-paper1 — Phase 1 reproducibility pipeline
# Requires: NVIDIA GPU with >=24 GB device memory (Ampere or newer).
#
# Quick start:
#   make init             Download corpus, create experiment folder
#   make smoke            ~10-min smoke test
#   make repro            Full reproduction (~14-16 h)
#
# Section-by-section:
#   make gate             Prerequisites (manifest + bringup + PCA)
#   make sweep            Master size sweep (~4 h)
#   make bench            Microbenchmark + capacity frontier (~1 h)
#   make compare          Impl comparison + efficiency sweep (~9 h)
#
# Docker workflow:
#   make image            Build the pinned CUDA container
#   make docker-init      Create experiment folder + fetch data in container
#   make docker-repro     Full reproduction in container

SHELL := /bin/bash
.ONESHELL:
.PHONY: image init smoke repro gate sweep bench compare run figures verify \
        docker-init docker-smoke docker-repro docker-run all clean help

IMAGE ?= sparsesom-paper1
RUN   ?=

# Pass --run FOLDER if the user set RUN=SBS-...
RUN_FLAG := $(if $(RUN),--run $(RUN),)

# ── Container ──────────────────────────────────────────────────
CONFIGS_HASH := $(shell find configs/ -type f | sort | xargs cat | sha256sum | cut -c1-16)
image:
	docker build --build-arg CONFIGS_HASH=$(CONFIGS_HASH) -t $(IMAGE) .

# ── Init (create experiment folder + download data) ───────────
init:
	python3 repro init

# ── Smoke test (~10 min) ──────────────────────────────────────
smoke:
	python3 repro --smoke $(RUN_FLAG)

# ── Sections ──────────────────────────────────────────────────
gate:
	python3 repro gate $(RUN_FLAG)

sweep:
	python3 repro sweep $(RUN_FLAG)

bench:
	python3 repro bench $(RUN_FLAG)

compare:
	python3 repro compare $(RUN_FLAG)

# ── Full reproduction ─────────────────────────────────────────
repro:
	python3 repro all $(RUN_FLAG)

# ── Single experiment ─────────────────────────────────────────
run:
	@test -n "$(ID)" || (echo "Usage: make run ID=<experiment_number>"; exit 1)
	python3 repro run $(ID) $(RUN_FLAG)

# ── Figures ────────────────────────────────────────────────────
figures:
	python3 repro figures $(RUN_FLAG)

# ── Acceptance report ─────────────────────────────────────────
verify:
	python3 repro verify $(RUN_FLAG)

# ── Everything ─────────────────────────────────────────────────
all: repro figures verify

# ── Docker workflow ───────────────────────────────────────────
DOCKER_RUN := docker run --rm --gpus all \
	-v $(CURDIR)/experiments:/workspace/experiments \
	$(IMAGE)

docker-init:
	@mkdir -p experiments
	$(DOCKER_RUN) init

docker-smoke:
	@mkdir -p experiments
	$(DOCKER_RUN) --smoke $(RUN_FLAG)

docker-repro:
	@mkdir -p experiments
	$(DOCKER_RUN) all $(RUN_FLAG)

docker-run:
	@test -n "$(ID)" || (echo "Usage: make docker-run ID=<experiment_number>"; exit 1)
	@mkdir -p experiments
	$(DOCKER_RUN) run $(ID) $(RUN_FLAG)

# ── Cleanup ───────────────────────────────────────────────────
clean:
	@echo "This removes results from all experiment folders."
	@echo "Data is preserved (expensive to re-download)."
	@echo "Press Ctrl-C to abort, Enter to continue."
	@read _
	find experiments/SBS-*/results -type f -delete 2>/dev/null || true

help:
	@echo "sparsesom-paper1 — Phase 1 reproducibility pipeline"
	@echo "Requires: NVIDIA GPU with >= 24 GB VRAM (Ampere or newer)"
	@echo ""
	@echo "Experiment folders live in experiments/SBS-ddmmyy-hhmm/."
	@echo "Each folder contains data/ and results/ subdirectories."
	@echo ""
	@echo "Commands:"
	@echo "  make init                   Create experiment folder + download data"
	@echo "  make smoke                  ~10-min smoke test"
	@echo "  make gate                   Prerequisites (ex 1-3, ~40 min)"
	@echo "  make sweep                  Size sweep (ex 5, ~4 h)"
	@echo "  make bench                  Microbench + capacity (ex 4,7, ~1 h)"
	@echo "  make compare                Impl compare + efficiency (ex 6,8, ~9 h)"
	@echo "  make repro                  Full reproduction (ex 1-8, ~14-16 h)"
	@echo "  make run ID=5               Run a single experiment"
	@echo "  make figures                Generate figures and tables"
	@echo "  make verify                 Run acceptance checks"
	@echo "  make all                    repro + figures + verify"
	@echo ""
	@echo "  RUN=SBS-230726-1430         Target a specific experiment folder"
	@echo ""
	@echo "Docker:"
	@echo "  make image                  Build the CUDA container"
	@echo "  make docker-init            Init in container"
	@echo "  make docker-repro           Full reproduction in container"
	@echo "  make docker-run ID=5        Single experiment in container"
