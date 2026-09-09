from environment.emitter import Emitter
from environment.rf_environment import (
    RFEnvironment,
    RFEnvironmentConfig
)
from environment.noise import NoiseModel

from receiver.receiver import (
    NarrowbandReceiver,
    ReceiverConfig
)
from receiver.detector import (
    SignalDetector,
    DetectorConfig
)

from scheduler.sequential import SequentialScanner
from scheduler.random_scan import RandomScanner
from scheduler.smartscan import SmartScanScheduler
from scheduler.scan_config import ScanConfig

from metrics.evaluation import (
    MetricsEvaluator,
    print_metrics
)


def create_emitters():

    return [
        Emitter(
            "E01",
            "Fixed Threat",
            120,
            5,
            -55,
            "fixed",
            1.0,
            0,
            4
        ),

        Emitter(
            "E02",
            "Intermittent Threat",
            150,
            5,
            -60,
            "intermittent",
            0.35,
            0,
            10
        ),

        Emitter(
            "E03",
            "Weak Intermittent",
            175,
            5,
            -65,
            "intermittent",
            0.20,
            0,
            15
        ),

        Emitter(
            "E04",
            "Frequency Agile",
            160,
            5,
            -50,
            "agile",
            1.0,
            3,
            12
        ),

        Emitter(
            "E05",
            "Bursty",
            135,
            5,
            -58,
            "bursty",
            0.15,
            0,
            7
        )
    ]


def create_environment():

    env = RFEnvironment(
        RFEnvironmentConfig(
            num_bands=20,
            num_time_slots=100,
            seed=42
        ),
        create_emitters(),
        NoiseModel(
            noise_floor_dbm=-100,
            noise_std_db=2
        )
    )

    env.reset()

    return env


def create_receiver(env):

    return NarrowbandReceiver(
        env,
        ReceiverConfig(
            noise_floor_dbm=-100,
            noise_std_db=2,
            receiver_bandwidth_mhz=5
        ),
        seed=123
    )


def create_detector():

    return SignalDetector(
        DetectorConfig(
            threshold_dbm=-85,
            false_alarm_probability=0.02
        ),
        seed=456
    )


def evaluate_scanner(
    name,
    scanner,
    env,
    receiver,
    detector,
    scan_config
):

    results = []

    if hasattr(scanner, "reset"):
        scanner.reset()

    for t in range(
        env.config.num_time_slots
    ):

        for scan_index in range(
            scan_config.scans_per_time_slot
        ):

            # Select frequency band.
            band = scanner.next_band()

            # Receiver observation.
            observation = receiver.observe(
                t,
                band
            )

            # Detector decision.
            detection = detector.detect(
                observation
            )

            # SmartScan needs feedback.
            if isinstance(
                scanner,
                SmartScanScheduler
            ):

                scanner.update(
                    band=band,
                    detected=detection["detected"],
                    measured_power_dbm=
                        detection[
                            "measured_power_dbm"
                        ],
                    false_alarm=
                        detection[
                            "false_alarm"
                        ]
                )

            result = detection.copy()

            # Ground truth is used ONLY for evaluation.
            result["actual_occupied"] = bool(
                env.truth[t, band]
            )

            result["scan_index"] = scan_index

            results.append(result)

    evaluator = MetricsEvaluator(
        env
    )

    metrics = evaluator.evaluate(
        results
    )

    print(
        "\n\n"
        + "#" * 60
    )

    print(
        f" {name.upper()}"
    )

    print(
        "#" * 60
    )

    print_metrics(
        metrics
    )

    return metrics


def main():

    print("\n" + "=" * 60)
    print(" SMARTSCAN FAIR BENCHMARK")
    print("=" * 60)

    scan_config = ScanConfig(
        scans_per_time_slot=4,
        dwell_time_ms=1.0
    )

    # --------------------------------------------------
    # IMPORTANT:
    #
    # Every scanner gets a fresh environment generated
    # with exactly the same seed and emitter configuration.
    #
    # Therefore the RF world is identical.
    # --------------------------------------------------

    results = {}

    # --------------------------------------------------
    # SEQUENTIAL
    # --------------------------------------------------

    env = create_environment()

    receiver = create_receiver(env)
    detector = create_detector()

    scanner = SequentialScanner(
        num_bands=env.config.num_bands
    )

    results["Sequential"] = evaluate_scanner(
        "Sequential",
        scanner,
        env,
        receiver,
        detector,
        scan_config
    )

    # --------------------------------------------------
    # RANDOM
    # --------------------------------------------------

    env = create_environment()

    receiver = create_receiver(env)
    detector = create_detector()

    scanner = RandomScanner(
        num_bands=env.config.num_bands,
        seed=999
    )

    results["Random"] = evaluate_scanner(
        "Random",
        scanner,
        env,
        receiver,
        detector,
        scan_config
    )

    # --------------------------------------------------
    # SMARTSCAN V2
    # --------------------------------------------------

    env = create_environment()

    receiver = create_receiver(env)
    detector = create_detector()

    scanner = SmartScanScheduler(
        num_bands=env.config.num_bands,
        exploration=2.0,
        prediction_weight=1.5,
        seed=2026
    )

    results["SmartScan V2"] = evaluate_scanner(
        "SmartScan V2",
        scanner,
        env,
        receiver,
        detector,
        scan_config
    )

    # --------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------

    print("\n\n" + "=" * 90)
    print(" FINAL COMPARISON")
    print("=" * 90)

    print(
        f"{'Algorithm':<18}"
        f"{'Pd':>10}"
        f"{'Pfa':>10}"
        f"{'Precision':>12}"
        f"{'Intercepted':>14}"
        f"{'Events':>10}"
        f"{'Rate':>10}"
        f"{'Avg Time':>12}"
    )

    print("-" * 90)

    for name, metrics in results.items():

        print(
            f"{name:<18}"
            f"{metrics.probability_of_detection:>10.4f}"
            f"{metrics.probability_of_false_alarm:>10.4f}"
            f"{metrics.precision:>12.4f}"
            f"{metrics.intercepted_events:>14}"
            f"{metrics.total_transmission_events:>10}"
            f"{metrics.intercept_rate:>10.4f}"
            f"{metrics.average_intercept_time:>12.2f}"
        )

    print("=" * 90)

    print(
        "\nBenchmark completed."
    )


if __name__ == "__main__":
    main()
