import numpy as np

from environment.emitter import Emitter
from environment.rf_environment import RFEnvironment, RFEnvironmentConfig
from receiver.receiver import NarrowbandReceiver, ReceiverConfig
from receiver.detector import SignalDetector, DetectorConfig

from scheduler.sequential import SequentialScanner
from scheduler.random_scan import RandomScanner
from scheduler.smartscan import SmartScanScheduler
from scheduler.smartscan_v3 import SmartScanV3Scheduler

from scheduler.scan_config import ScanConfig

from metrics.evaluation import (
    MetricsEvaluator,
    print_metrics,
)


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


def create_environment():

    config = RFEnvironmentConfig(
        num_bands=20,
        num_time_slots=100,
        band_start_mhz=100.0,
        band_width_mhz=5.0,
        seed=42,
    )

    return RFEnvironment(
        emitters=create_emitters(),
        config=config,
    )


def create_receiver(env):

    config = ReceiverConfig(
        receiver_bandwidth_mhz=5.0,
    )

    return NarrowbandReceiver(
        environment=env,
        config=config,
        seed=123,
    )


def create_detector():

    config = DetectorConfig(
        threshold_dbm=-85.0,
        false_alarm_probability=0.02,
    )

    return SignalDetector(
        config=config,
        seed=456,
    )


def evaluate_scanner(
    name,
    scanner,
    env,
    receiver,
    detector,
    scan_config,
):
    print(
        "\n" + "#" * 60
    )

    print(
        f" {name}"
    )

    print(
        "#" * 60
    )

    results = []

    env.reset()

    if hasattr(scanner, "reset"):
        scanner.reset()

    for time_slot in range(
        env.config.num_time_slots
    ):

        # ------------------------------------------
        # MULTI-SCAN V3 / SMARTSCAN
        # ------------------------------------------

        if isinstance(
            scanner,
            SmartScanV3Scheduler
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
        # OBSERVE SELECTED BANDS
        # ------------------------------------------

        for scan_index, band in enumerate(
            bands
        ):

            observation = receiver.observe(
                time_slot=time_slot,
                band=band,
            )

            detection = detector.detect(
                observation
            )

            occupied = bool(
                env.truth[
                    time_slot,
                    band
                ]
            )

            detection[
                "actual_occupied"
            ] = occupied

            detection[
                "scan_index"
            ] = scan_index

            results.append(
                detection
            )

            # --------------------------------------
            # UPDATE V2
            # --------------------------------------

            if isinstance(
                scanner,
                SmartScanScheduler
            ):

                scanner.update(
                    band=band,
                    detected=detection[
                        "detected"
                    ],
                    measured_power_dbm=detection[
                        "measured_power_dbm"
                    ],
                    false_alarm=detection[
                        "false_alarm"
                    ],
                )

            # --------------------------------------
            # UPDATE V3
            # --------------------------------------

            elif isinstance(
                scanner,
                SmartScanV3Scheduler
            ):

                scanner.update(
                    band=band,
                    detected=detection[
                        "detected"
                    ],
                    measured_power_dbm=detection[
                        "measured_power_dbm"
                    ],
                    false_alarm=detection[
                        "false_alarm"
                    ],
                    time_slot=time_slot,
                )

    metrics = MetricsEvaluator(
        env
    ).evaluate(
        results
    )

    print_metrics(
        metrics
    )

    return metrics


def main():

    print(
        "\n" + "=" * 60
    )

    print(
        " SMARTSCAN V3 BENCHMARK"
    )

    print(
        "=" * 60
    )

    scan_config = ScanConfig(
        scans_per_time_slot=4,
        dwell_time_ms=1.0,
    )

    results = {}

    # ==========================================
    # SEQUENTIAL
    # ==========================================

    env = create_environment()

    receiver = create_receiver(env)

    detector = create_detector()

    scanner = SequentialScanner(
        num_bands=env.config.num_bands
    )

    results["Sequential"] = evaluate_scanner(
        "SEQUENTIAL",
        scanner,
        env,
        receiver,
        detector,
        scan_config,
    )

    # ==========================================
    # RANDOM
    # ==========================================

    env = create_environment()

    receiver = create_receiver(env)

    detector = create_detector()

    scanner = RandomScanner(
        num_bands=env.config.num_bands,
        seed=999,
    )

    results["Random"] = evaluate_scanner(
        "RANDOM",
        scanner,
        env,
        receiver,
        detector,
        scan_config,
    )

    # ==========================================
    # SMARTSCAN V2
    # ==========================================

    env = create_environment()

    receiver = create_receiver(env)

    detector = create_detector()

    scanner = SmartScanScheduler(
        num_bands=env.config.num_bands,
        exploration=2.0,
        prediction_weight=1.5,
        seed=2026,
    )

    results["SmartScan V2"] = evaluate_scanner(
        "SMARTSCAN V2",
        scanner,
        env,
        receiver,
        detector,
        scan_config,
    )

    # ==========================================
    # SMARTSCAN V3
    # ==========================================

    env = create_environment()

    receiver = create_receiver(env)

    detector = create_detector()

    scanner = SmartScanV3Scheduler(
        num_bands=env.config.num_bands,
        exploration=2.0,
        prediction_weight=1.5,
        exploration_weight=0.8,
        revisit_weight=0.6,
        max_staleness=15,
        seed=3030,
    )

    results["SmartScan V3"] = evaluate_scanner(
        "SMARTSCAN V3",
        scanner,
        env,
        receiver,
        detector,
        scan_config,
    )

    # ==========================================
    # COMPARISON
    # ==========================================

    print(
        "\n\n" + "=" * 105
    )

    print(
        " FINAL COMPARISON"
    )

    print(
        "=" * 105
    )

    print(
        f"{'Algorithm':<20}"
        f"{'Pd':>10}"
        f"{'Pfa':>10}"
        f"{'Precision':>12}"
        f"{'Intercepted':>14}"
        f"{'Events':>10}"
        f"{'Rate':>10}"
        f"{'Avg Time':>12}"
    )

    print(
        "-" * 105
    )

    for name, metrics in results.items():

        print(
            f"{name:<20}"
            f"{metrics.probability_of_detection:>10.4f}"
            f"{metrics.probability_of_false_alarm:>10.4f}"
            f"{metrics.precision:>12.4f}"
            f"{metrics.intercepted_events:>14}"
            f"{metrics.total_transmission_events:>10}"
            f"{metrics.intercept_rate:>10.4f}"
            f"{metrics.average_intercept_time:>12.2f}"
        )

    print(
        "=" * 105
    )

    print(
        "\nBenchmark V3 completed."
    )


if __name__ == "__main__":
    main()
