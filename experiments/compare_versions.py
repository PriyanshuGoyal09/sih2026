"""Compare all scheduler versions on PDW data and the synthetic SIH scenario."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.engine import compare_schedulers, metrics_row
from data.pdw_loader import load_pdw_environment
from experiments.scenario_1 import create_environment as create_synthetic


NAMES = [
    "sequential",
    "random",
    "ucb",
    "v2",
    "v3",
    "v31",
    "v4",
    "v41",
    "v5",
    "v51",
    "v52",
    "v53",
]


def _print_table(title: str, rows: list[dict]) -> None:
    print()
    print("=" * 108)
    print(title)
    print("=" * 108)
    header = (
        f"{'Scheduler':<12} {'Pd':>7} {'Pfa':>7} {'Prec':>7} "
        f"{'Intercept':>10} {'AvgT':>7} {'Events':>10} {'Dets':>6}"
    )
    print(header)
    print("-" * 108)
    ranked = sorted(rows, key=lambda r: (-r["Intercept rate"], -r["Pd"], r["Avg intercept time (slots)"]))
    for row in ranked:
        print(
            f"{row['Scheduler']:<12} "
            f"{100 * row['Pd']:6.1f}% "
            f"{100 * row['Pfa']:6.1f}% "
            f"{100 * row['Precision']:6.1f}% "
            f"{100 * row['Intercept rate']:9.1f}% "
            f"{row['Avg intercept time (slots)']:7.2f} "
            f"{row['Events intercepted']:4d}/{row['Total events']:<4d}"
        )
    print()
    best = ranked[0]
    print(
        f"Best intercept: {best['Scheduler']} "
        f"({100 * best['Intercept rate']:.1f}%, avg time {best['Avg intercept time (slots)']:.2f})"
    )


def run_suite(environment, title: str, scans_per_slot: int = 4):
    runs = compare_schedulers(
        environment,
        names=NAMES,
        scans_per_slot=scans_per_slot,
        seed=123,
    )
    rows = [metrics_row(run) for run in runs]
    _print_table(title, rows)
    return rows


def main() -> int:
    synthetic = create_synthetic()
    synthetic.reset()
    run_suite(
        synthetic,
        "SYNTHETIC scenario_1  (20 bands, 100 slots, 5 hand-built emitters)",
    )

    pdw, info = load_pdw_environment("config_1.h5", slot_duration_s=0.1)
    run_suite(
        pdw,
        f"PDW {info['name']}  ({pdw.config.num_bands} dwells, "
        f"{pdw.config.num_time_slots} slots, occupancy {100 * info['occupancy']:.2f}%)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
