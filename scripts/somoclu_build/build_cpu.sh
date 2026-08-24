#!/usr/bin/env bash
# Build somoclu's sparse CPU kernel without autotools (autoreconf/automake
# may be absent). Produces external/somoclu/somoclu_cpu.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../../external/somoclu/src"
OUT="$HERE/../../external/somoclu/somoclu_cpu"
g++ -O3 -fopenmp -DHAVE_CONFIG_H=0 -include "$HERE/cpu_prelude.h" -o "$OUT" \
    "$SRC/somoclu.cpp" "$SRC/io.cpp" "$SRC/training.cpp" \
    "$SRC/denseCpuKernels.cpp" "$SRC/sparseCpuKernels.cpp" \
    "$SRC/mapDistanceFunctions.cpp" "$SRC/uMatrix.cpp"
echo "built $OUT"
