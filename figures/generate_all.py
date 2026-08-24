"""Generate all manuscript figures and tables from results artifacts.

Usage:
    python3 -m figures.generate_all --results-dir results --output-dir figures/out
"""

import argparse
import sys
from pathlib import Path

from . import (
    fig3_microbench,
    fig4_bmu_dominance,
    fig5_efficiency,
    fig6_size_sweep,
    fig7_roofline,
    table_pca_equivalence,
)


ALL_GENERATORS = [
    ("Fig 3 — BMU kernel microbenchmark", fig3_microbench),
    ("Fig 4 — BMU dominance", fig4_bmu_dominance),
    ("Fig 5 — Efficiency sweep", fig5_efficiency),
    ("Fig 6 — Size sweep / power law", fig6_size_sweep),
    ("Fig 7 — Roofline", fig7_roofline),
    ("Table — PCA equivalence", table_pca_equivalence),
]


def main(results_dir: str = "results", output_dir: str = "figures/out"):
    res = Path(results_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, module in ALL_GENERATORS:
        print(f"Generating {name} ...", end=" ", flush=True)
        try:
            module.generate(res, out)
            print("OK")
        except FileNotFoundError as e:
            print(f"SKIPPED (missing: {e})")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nFigures written to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="figures/out")
    args = parser.parse_args()
    main(args.results_dir, args.output_dir)
