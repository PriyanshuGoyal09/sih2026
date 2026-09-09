from environment.emitter import Emitter
from environment.rf_environment import RFEnvironment, RFEnvironmentConfig
from environment.noise import NoiseModel

from receiver.receiver import NarrowbandReceiver, ReceiverConfig
from receiver.detector import SignalDetector, DetectorConfig

from scheduler.sequential import SequentialScanner
from scheduler.random_scan import RandomScanner

from scheduler.scan_config import ScanConfig


def create_environment():

    emitters = [
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
        ),
    ]

    config = RFEnvironmentConfig(
        num_bands=20,
        num_time_slots=100,
        seed=42
    )

    noise = NoiseModel(
        noise_floor_dbm=-100.0,
        noise_std_db=2.0
    )

    env = RFEnvironment(
        config,
        emitters,
        noise
    )

    env.reset()

    return env


def run_scanner(name, scanner):

    print("\n" + "=" * 45)
    print(f" {name}")
    print("=" * 45)

    env = create_environment()

    receiver = NarrowbandReceiver(
        env,
        ReceiverConfig(
            noise_floor_dbm=-100.0,
            noise_std_db=2.0,
            receiver_bandwidth_mhz=5.0
        ),
        seed=123
    )

    detector = SignalDetector(
        DetectorConfig(
            threshold_dbm=-85.0,
            false_alarm_probability=0.02
        ),
        seed=456
    )

    scan_config = ScanConfig(
        scans_per_time_slot=4
    )

    scanner.reset()

    results = []

    for t in range(
        env.config.num_time_slots
    ):

        for scan_index in range(
            scan_config.scans_per_time_slot
        ):

            band = scanner.next_band()

            observation = receiver.observe(
                t,
                band
            )

            detection = detector.detect(
                observation
            )

            result = detection.copy()

            # Evaluation-only information.
            # The scheduler itself never receives this.
            result["actual_occupied"] = bool(
                env.truth[t, band]
            )

            result["scan_index"] = scan_index

            results.append(result)

    detections = sum(
        r["detected"]
        for r in results
    )

    true_positives = sum(
        r["detected"]
        and r["actual_occupied"]
        for r in results
    )

    false_alarms = sum(
        r["detected"]
        and not r["actual_occupied"]
        for r in results
    )

    print()
    print(f"Time slots:       {env.config.num_time_slots}")
    print(f"Scans per slot:   {scan_config.scans_per_time_slot}")
    print(f"Total scans:      {len(results)}")
    print(f"Detections:       {detections}")
    print(f"True positives:   {true_positives}")
    print(f"False alarms:     {false_alarms}")

    print("\nFirst 30 scans:")

    for r in results[:30]:

        print(
            f"t={r['time_slot']:02d} "
            f"s={r['scan_index']} "
            f"| band={r['band']:02d} "
            f"| power={r['measured_power_dbm']:7.2f} dBm "
            f"| detected={str(r['detected']):5s} "
            f"| occupied={r['actual_occupied']}"
        )

    return env, results


def main():

    print("\n" + "=" * 45)
    print(" SMARTSCAN MULTI-SCAN EXPERIMENT")
    print("=" * 45)

    sequential = SequentialScanner(20)

    random_scanner = RandomScanner(
        20,
        seed=999
    )

    run_scanner(
        "SEQUENTIAL MULTI-SCAN",
        sequential
    )

    run_scanner(
        "RANDOM MULTI-SCAN",
        random_scanner
    )


if __name__ == "__main__":
    main()
