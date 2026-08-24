#!/usr/bin/env bash
# ncu_profile.sh — Nsight Compute profiling for the roofline figure.
# Out-of-CI: run manually, produces results/roofline_dram.csv.
#
# Usage:
#   ./scripts/ncu_profile.sh [EDGE] [CORPUS]
#
# Requirements:
#   - NVIDIA Nsight Compute (ncu) installed
#   - Sufficient permissions (may need sudo or --target-processes all)

set -euo pipefail

EDGE="${1:-128}"
CORPUS="${2:-data/corpus.train.sbcsr}"
OUTDIR="${OUTDIR:-results}"
NCU="${NCU:-ncu}"
SPARSESOM="${SPARSESOM:-sparsesom}"
STANDARDSPARSESOM="${STANDARDSPARSESOM:-standardsparsesom}"
MEDSOM="${MEDSOM:-medsom}"

mkdir -p "$OUTDIR"

echo "=== Roofline ncu profiling at edge $EDGE ==="
echo "Corpus: $CORPUS"
echo "Output: $OUTDIR/roofline_dram.csv"
echo ""

# Common ncu flags: collect DRAM throughput metrics
NCU_METRICS="dram__bytes_read.sum,dram__bytes_write.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed"
# Cap profiled launches: MedSOM launches 3 kernels per neuron per epoch, and
# uncapped replay would take days. The analysis layer scales via scale_factor.
NCU_FLAGS="--metrics $NCU_METRICS -c ${NCU_LAUNCH_COUNT:-500} --csv --log-file"

echo "impl,kernel_name,dram_bytes_read,dram_bytes_write,duration_s,achieved_bw_gbs,flops,samples_profiled,scale_factor" > "$OUTDIR/roofline_dram.csv"

profile_impl() {
    local impl="$1"
    local cmd="$2"
    local csv_file="$OUTDIR/ncu_${impl}_${EDGE}.csv"

    echo ""
    echo "--- Profiling: $impl ---"
    echo "  Command: $cmd"

    # Run ncu
    eval "$NCU $NCU_FLAGS $csv_file $cmd" 2>&1 | tail -5

    if [ -f "$csv_file" ]; then
        echo "  Raw ncu output: $csv_file"
        # Parse will be done by the Python layer
        echo "  (Parse with figures/fig7_roofline.py)"
    else
        echo "  WARNING: No ncu output produced"
    fi
}

# 1. sbsom-bin
profile_impl "sbsom-bin" \
    "$SPARSESOM $CORPUS --rows $EDGE --cols $EDGE --bin --epochs 1 --seed 0 --sigma-init $(echo "$EDGE * 0.5" | bc)"

# 2. sbsom-float
profile_impl "sbsom-float" \
    "$SPARSESOM $CORPUS --rows $EDGE --cols $EDGE --epochs 1 --seed 0 --sigma-init $(echo "$EDGE * 0.5" | bc)"

# 3. ssom-feat (FP16-matched)
profile_impl "ssom-feat" \
    "$STANDARDSPARSESOM $CORPUS --map $EDGE --layout feature --precision fp16 --stop fixed --epochs 1 --seed 0"

# 4. ssom-node (FP16-matched)
profile_impl "ssom-node" \
    "$STANDARDSPARSESOM $CORPUS --map $EDGE --layout node --precision fp16 --stop fixed --epochs 1 --seed 0"

# 5. MedSOM (positional args: corpus epochs dimX)
profile_impl "medsom" \
    "$MEDSOM $CORPUS 1 $EDGE"

echo ""
echo "=== Profiling complete ==="
echo "Raw CSVs in $OUTDIR/ncu_*.csv"
echo "Run 'make figures' to generate the roofline figure from these."
