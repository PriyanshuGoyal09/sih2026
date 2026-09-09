"""CLI entry: load a PDW HDF5 scenario and compare scan schedulers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.engine import compare_schedulers, metrics_row
from data.pdw_loader import list_pdw_scenarios, load_pdw_environment
from metrics.evaluation import print_metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Smart Scan scheduler comparison on PDW HDF5 data"
    )
    parser.add_argument(
        "--scenario",
        default="config_1.h5",
        help="HDF5 file in the project root (default: config_1.h5)",
    )
    parser.add_argument("--slot-duration", type=float, default=0.1)
    parser.add_argument("--scans-per-slot", type=int, default=4)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available PDW scenarios and exit",
    )
    args = parser.parse_args(argv)

    if args.list:
        for item in list_pdw_scenarios(ROOT):
            print(
                f"{item.name:16s}  {item.scan_mode:10s}  "
                f"{item.num_pulses:8d} pulses  {item.num_transmitters:3d} emitters"
            )
        return 0

    environment, info = load_pdw_environment(
        args.scenario,
        slot_duration_s=args.slot_duration,
    )
    print(f"Loaded {info['name']}: {info['mapped_pulses']}/{info['num_pulses']} pulses")
    print(
        f"{environment.config.num_bands} bands, "
        f"{environment.config.num_time_slots} slots, "
        f"occupancy {info['occupancy'] * 100:.2f}%"
    )

    for run in compare_schedulers(
        environment,
        names=["sequential", "ucb", "smartscan"],
        scans_per_slot=args.scans_per_slot,
    ):
        print(f"\n=== {run['name']} ===")
        row = metrics_row(run)
        print(
            f"Pd={row['Pd']:.3f}  Pfa={row['Pfa']:.3f}  "
            f"intercept={row['Intercept rate']:.3f}  "
            f"avg_time={row['Avg intercept time (slots)']:.2f}"
        )
        print_metrics(run["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
