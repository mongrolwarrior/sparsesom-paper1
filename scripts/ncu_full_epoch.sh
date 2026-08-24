#!/usr/bin/env bash
# ncu_full_epoch.sh — exact per-epoch, WHOLE-BMU-PHASE DRAM capture.
#
# Captures every kernel in one epoch's BMU phase (not csrmm alone), so the
# cuSPARSE score-block round trip is counted:
#   ssom (feature/node): norm(s)_kernel + per-tile { csrmm, argmax_kernel }
#       — argmax_kernel reads the dense score block back for the argmin.
#   sbsom (bin):         norm_fm_kernel + fused bmu_spmm_kernel
#       — the fused kernel scores and argmins in one launch, no score block.
# Comparing sbsom's fused kernel against ssom's csrmm ALONE undercounts ssom;
# this captures the full phase on both sides for a like-for-like total.
#
# Metrics: DRAM bytes (r/w), measured DRAM throughput %, per-kernel exec time
# (gpu__time_duration.sum), and SM/compute occupancy (kept, clearly separate).
#
# Usage: OUTDIR=<dir> ./scripts/ncu_full_epoch.sh EDGE CORPUS
set -euo pipefail

EDGE="${1:-128}"
CORPUS="${2:-data/corpus.train.sbcsr}"
OUTDIR="${OUTDIR:-results}"
NCU="${NCU:-ncu}"
SPARSESOM="${SPARSESOM:-sparsesom}"
STANDARDSPARSESOM="${STANDARDSPARSESOM:-standardsparsesom}"

mkdir -p "$OUTDIR"
METRICS="dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed"
SIGMA=$(python3 -c "print(0.5*$EDGE)")
IMPLS="${IMPLS:-ssom-feat,ssom-node,sbsom-bin}"   # subset to limit cost at edge 256
# NCU_CAP limits profiled launches: at edge 256 the tiled cuSPARSE phase has
# ~13k launches and an uncapped ncu replay takes days. Per-tile DRAM is uniform,
# so capturing the first NCU_CAP launches (norms + a few hundred tiles) and
# scaling to the full tile count (aggregate_phase.py) is accurate to ~7%
# (validated against the exact edge-128 capture). Empty = capture everything.
CAP_ARG=""
[ -n "${NCU_CAP:-}" ] && CAP_ARG="-c $NCU_CAP"

echo "=== full BMU-phase capture at edge $EDGE (impls: $IMPLS) ==="

case ",$IMPLS," in *,ssom-feat,*)
  echo "--- ssom-feat (norms + csrmm + argmax, 1 epoch) ---"
  "$NCU" --metrics "$METRICS" $CAP_ARG \
      -k "regex:norms_kernel|csrmm|argmax_kernel|rebase_rowptr" --csv \
      --log-file "$OUTDIR/ncu_full_ssom-feat_${EDGE}.csv" \
      "$STANDARDSPARSESOM" "$CORPUS" --map "$EDGE" --layout feature \
      --precision fp16 --stop fixed --epochs 1 --seed 0 2>&1 | tail -2 ;;
esac

case ",$IMPLS," in *,ssom-node,*)
  echo "--- ssom-node (norms + csrmm + argmax, 1 epoch) ---"
  "$NCU" --metrics "$METRICS" $CAP_ARG \
      -k "regex:norms_kernel|csrmm|argmax_kernel|rebase_rowptr" --csv \
      --log-file "$OUTDIR/ncu_full_ssom-node_${EDGE}.csv" \
      "$STANDARDSPARSESOM" "$CORPUS" --map "$EDGE" --layout node \
      --precision fp16 --stop fixed --epochs 1 --seed 0 2>&1 | tail -2 ;;
esac

case ",$IMPLS," in *,sbsom-bin,*)
  echo "--- sbsom-bin (norm_fm + bmu_spmm, 1 epoch) ---"
  "$NCU" --metrics "$METRICS" $CAP_ARG \
      -k "regex:norm_fm_kernel|bmu_spmm" --csv \
      --log-file "$OUTDIR/ncu_full_sbsom-bin_${EDGE}.csv" \
      "$SPARSESOM" "$CORPUS" --rows "$EDGE" --cols "$EDGE" --bin \
      --epochs 1 --seed 0 --sigma-init "$SIGMA" 2>&1 | tail -2 ;;
esac

echo "=== done: edge $EDGE ==="
