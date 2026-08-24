#!/usr/bin/env python3
"""Runtime verification that StandardSparseSOM's --precision fp16 is active.

Runs one node-major epoch at each precision and checks two runtime
signals that cannot come from a label:

  1. The BMU scores-tile allocation, printed with its byte size, must
     halve under fp16 (tile rows x K x 2 B vs x 4 B).
  2. Per-epoch BMU time must be substantially faster under fp16 (the
     kernel is bandwidth-bound; halving operand bytes must show up).

Exit 0 = fp16 verified active; exit 1 = confound present, do not profile.

Usage: verify_ssom_fp16.py CORPUS [EDGE]
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.runner import run_standardsparsesom  # noqa: E402


def one(corpus: str, edge: int, layout: str, precision: str) -> tuple[float, float]:
    r = run_standardsparsesom(
        corpus, edge, 0,
        layout=layout, neighbourhood="box", stop="fixed",
        precision=precision,
        extra_flags=["--epochs", "1"],
    )
    out = r["raw_stdout"]
    tile_mib = 0.0
    m = re.search(r"scores tile (\d+) MiB, (fp16|fp32)", out)
    if m:
        if m.group(2) != precision:
            print(f"FAIL: tile line reports {m.group(2)}, expected {precision}")
            sys.exit(1)
        tile_mib = float(m.group(1))
    done = r["done"] or {}
    return tile_mib, done.get("bmu_s", 0.0)


def main():
    corpus = sys.argv[1]
    edge = int(sys.argv[2]) if len(sys.argv) > 2 else 128

    # Byte check on node-major (the only layout that prints its tile
    # allocation); speed check on feature-major (the only layout whose
    # BMU is operand-bandwidth-bound — node-major is gather-latency
    # bound and shows ~1.07x regardless of precision, measured 2026-07-28).
    print(f"FP16 runtime verification at edge {edge}, 1 epoch per run")
    tile16, _ = one(corpus, edge, "node", "fp16")
    tile32, _ = one(corpus, edge, "node", "fp32")
    _, bmu16 = one(corpus, edge, "feature", "fp16")
    _, bmu32 = one(corpus, edge, "feature", "fp32")

    ratio_tile = tile32 / tile16 if tile16 else 0
    ratio_bmu = bmu32 / bmu16 if bmu16 else 0
    print(f"  node scores tile: fp16={tile16:.0f} MiB  fp32={tile32:.0f} MiB  "
          f"ratio={ratio_tile:.2f} (expect 2.0)")
    print(f"  feature BMU 1 ep: fp16={bmu16:.1f}s  fp32={bmu32:.1f}s  "
          f"ratio={ratio_bmu:.2f} (expect > 1.4)")

    if abs(ratio_tile - 2.0) < 0.05 and ratio_bmu > 1.4:
        print("PASS: fp16 codebook and operands active at runtime")
        sys.exit(0)
    print("FAIL: fp16 not verifiably active — do not run the roofline")
    sys.exit(1)


if __name__ == "__main__":
    main()
