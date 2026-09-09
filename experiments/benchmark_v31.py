import numpy as np

from environment.emitter import Emitter
from environment.rf_environment import (
    RFEnvironment,
    RFEnvironmentConfig,
)

from receiver.receiver import (
    NarrowbandReceiver,
    ReceiverConfig,
)

from receiver.detector import (
    SignalDetector,
    DetectorConfig,
)

from scheduler.sequential import SequentialScanner
from scheduler.random_scan import RandomScanner
from scheduler.smartscan import SmartScanScheduler
from scheduler.smartscan_v31 import SmartScanV31Scheduler

from scheduler.scan_config import ScanConfig

from metrics.evaluation import MetricsEvaluator


def create_emitters():
    return [
        Emitter(
            emitter_id="E01",
            name="Fixed Radar",
            center_frequency=120.0,
            bandwidth=5.0,
            power_dbm=-55.0,
            behavior="fixed",
            start_band=4,
        ),
        Emitter(
            emitter_id="E02",
            name="Intermittent Radar",
            center_frequency=150.0,
            bandwidth=5.0,
            power_dbm=-60.0,
            behavior="intermittent",
            duty_cycle=0.35,
            start_band=10,
        ),
        Emitter(
            emitter_id="E03",
            name="Weak Intermittent",
            center_frequency=175.0,
            bandwidth=5.0,
            power_dbm=-65.0,
            behavior="intermittent",
            duty_cycle=0.20,
            start_band=15,
        ),
        Emitter(
            emitter_id="E04",
            name="Agile Emitter",
            center_frequency=160.0,
            bandwidth=5.0,
            power_dbm=-50.0,
            behavior="agile",
            agility=3,
            start_band=12,
        ),
        Emitter(
            emitter_id="E05",
            name="Bursty Emitter",
            center_frequency=135.0,
            bandwidth=5.0,
            power_dbm=-58.0,
            behavior="bursty",
            duty_cycle=0.15,
            start_band=7,
        ),
    ]


def create_environment(seed):
    config = RFEnvironmentConfig(
        num_bands=20,
        num_time_slots=100,
        band_start_mhz=100.0,
        band_width_mhz=5.0,
        seed=seed,
    )

    env = RFEnvironment(
        emitters=create_emitters(),
        config=config,
    )

    # Generate the truth/power/event realization once.
    # Every algorithm using this environment will therefore
    # be evaluated against the same scenario.
    env.reset()

    return env


def create_receiver(env):
    config = ReceiverConfig(
        noise_floor_dbm=-100.0,
        noise_std_db=2.0,
        receiver_bandwidth_mhz=5.0,
    )

    return NarrowbandReceiver(
        environment=env,
        config=config,
        seed=123,
    )


def create_detector(seed=456):
    config = DetectorConfig(
        threshold_dbm=-85.0,
        false_alarm_probability=0.02,
    )

    return SignalDetector(
        config=config,
        seed=seed,
    )


def run_algorithm(
    algorithm_name,
    env,
    scanner,
    scan_config,
):
    receiver = create_receiver(env)
    detector = create_detector()

    results = []

    # IMPORTANT:
    # Do NOT call env.reset() here.
    #
    # create_environment() already generated the
    # deterministic environment for this seed.

    if hasattr(scanner, "reset"):
        scanner.reset()

    for time_slot in range(
        env.config.num_time_slots
    ):

        # ------------------------------------------
        # SELECT BANDS
        # ------------------------------------------

        if isinstance(
            scanner,
            SmartScanV31Scheduler,
        ):
            bands = scanner.select_bands(
                time_slot=time_slot,
                num_scans=scan_config.scans_per_time_slot,
            )

        else:
            bands = [
                scanner.next_band()
                for _ in range(
                    scan_config.scans_per_time_slot
                )
            ]

        # ------------------------------------------
        # OBSERVE
        # ------------------------------------------

        for scan_index, band in enumerate(bands):

            observation = receiver.observe(
                time_slot=time_slot,
                band=band,
            )

            detection = detector.detect(
                observation
            )

            detection["actual_occupied"] = bool(
                env.truth[
                    time_slot,
                    band,
                ]
            )

            detection["scan_index"] = scan_index

            results.append(detection)

            # --------------------------------------
            # V2
            # --------------------------------------

            if isinstance(
                scanner,
                SmartScanScheduler,
            ):
                scanner.update(
                    band=band,
                    detected=detection["detected"],
                    measured_power_dbm=detection[
                        "measured_power_dbm"
                    ],
                    false_alarm=detection[
                        "false_alarm"
                    ],
                )

            # --------------------------------------
            # V3.1
            # --------------------------------------

            elif isinstance(
                scanner,
                SmartScanV31Scheduler,
            ):
                scanner.update(
                    band=band,
                    detected=detection["detected"],
                    measured_power_dbm=detection[
                        "measured_power_dbm"
                    ],
                    false_alarm=detection[
                        "false_alarm"
                    ],
                    time_slot=time_slot,
                )

    return MetricsEvaluator(
        env
    ).evaluate(results)


def make_scanners(seed):

    return {
        "Sequential": SequentialScanner(20),

        "Random": RandomScanner(
            num_bands=20,
            seed=999 + seed,
        ),

        "SmartScan V2": SmartScanScheduler(
            num_bands=20,
            exploration=2.0,
            prediction_weight=1.5,
            seed=2026 + seed,
        ),

        "SmartScan V3.1": SmartScanV31Scheduler(
            num_bands=20,
            exploration=2.0,
            prediction_weight=1.5,
            exploration_weight=0.8,
            revisit_weight=0.6,
            quality_weight=0.35,
            max_staleness=15,
            seed=3031 + seed,
        ),
    }


def run_seed(seed, scan_config):

    envs = {
        name: create_environment(seed)
        for name in make_scanners(seed)
    }

    metrics = {}

    for name in envs:

        scanner = make_scanners(seed)[name]

        metrics[name] = run_algorithm(
            name,
            envs[name],
            scanner,
            scan_config,
        )

    return metrics


def mean_std(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        float(np.mean(values)),
        float(np.std(values)),
    )


def main():

    print("\n" + "=" * 110)
    print(" SMARTSCAN V3.1 MULTI-SEED BENCHMARK")
    print("=" * 110)

    scan_config = ScanConfig(
        scans_per_time_slot=4,
        dwell_time_ms=1.0,
    )

    seeds = list(
        range(42, 52)
    )

    all_metrics = {}

    for seed in seeds:

        print(
            f"\nRunning seed {seed}..."
        )

        result = run_seed(
            seed,
            scan_config,
        )

        for name, metrics in result.items():

            all_metrics.setdefault(
                name,
                [],
            ).append(metrics)

    print("\n" + "=" * 110)
    print(" MULTI-SEED RESULTS (MEAN ± STD)")
    print("=" * 110)

    print(
        f"{'Algorithm':<20}"
        f"{'Pd':>16}"
        f"{'Pfa':>16}"
        f"{'Precision':>18}"
        f"{'Intercept Rate':>18}"
        f"{'Avg Time':>16}"
    )

    print("-" * 110)

    for name, runs in all_metrics.items():

        pd_mean, pd_std = mean_std([
            m.probability_of_detection
            for m in runs
        ])

        pfa_mean, pfa_std = mean_std([
            m.probability_of_false_alarm
            for m in runs
        ])

        precision_mean, precision_std = mean_std([
            m.precision
            for m in runs
        ])

        rate_mean, rate_std = mean_std([
            m.intercept_rate
            for m in runs
        ])

        time_mean, time_std = mean_std([
            m.average_intercept_time
            for m in runs
        ])

        print(
            f"{name:<20}"
            f"{pd_mean:.4f} ± {pd_std:.4f}"
            f"{pfa_mean:.4f} ± {pfa_std:.4f}"
            f"{precision_mean:.4f} ± {precision_std:.4f}"
            f"{rate_mean:.4f} ± {rate_std:.4f}"
            f"{time_mean:.2f} ± {time_std:.2f}"
        )

    print("=" * 110)

    print(
        "\nDetailed event counts:"
    )

    for name, runs in all_metrics.items():

        event_rates = [
            m.intercept_rate
            for m in runs
        ]

        intercepted = [
            m.intercepted_events
            for m in runs
        ]

        print(
            f"{name:<20}"
            f" mean intercepted = "
            f"{np.mean(intercepted):.2f}/"
            f"{runs[0].total_transmission_events:.0f}"
            f" | event rate = "
            f"{np.mean(event_rates):.4f}"
        )

    print(
        "\nSeeds:",
        seeds,
    )

    print(
        "Benchmark completed."
    )


if __name__ == "__main__":
    main()
