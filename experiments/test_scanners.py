from environment.emitter import Emitter
from environment.rf_environment import RFEnvironment, RFEnvironmentConfig
from receiver.receiver import NarrowbandReceiver, ReceiverConfig
from receiver.detector import SignalDetector, DetectorConfig
from scheduler.sequential import SequentialScanner
from scheduler.random_scan import RandomScanner
from metrics.evaluation import evaluate, print_metrics


def create_environment():
    config = RFEnvironmentConfig(
        num_bands=20,
        num_time_slots=100,
        seed=42
    )

    emitters = [
        Emitter(
            emitter_id="E01",
            name="Fixed",
            center_frequency=122.5,
            bandwidth=5,
            power_dbm=-55,
            behavior="fixed",
            start_band=4
        ),
        Emitter(
            emitter_id="E02",
            name="Intermittent",
            center_frequency=152.5,
            bandwidth=5,
            power_dbm=-60,
            behavior="intermittent",
            duty_cycle=0.35,
            start_band=10
        ),
        Emitter(
            emitter_id="E03",
            name="Intermittent",
            center_frequency=177.5,
            bandwidth=5,
            power_dbm=-65,
            behavior="intermittent",
            duty_cycle=0.20,
            start_band=15
        ),
        Emitter(
            emitter_id="E04",
            name="Agile",
            center_frequency=162.5,
            bandwidth=5,
            power_dbm=-50,
            behavior="agile",
            agility=3,
            start_band=12
        ),
        Emitter(
            emitter_id="E05",
            name="Bursty",
            center_frequency=135.0,
            bandwidth=5,
            power_dbm=-58,
            behavior="bursty",
            duty_cycle=0.15,
            start_band=7
        ),
    ]

    environment = RFEnvironment(config, emitters)
    environment.reset()

    return environment


def run_scanner(scanner_name, scanner):
    environment = create_environment()

    receiver = NarrowbandReceiver(
        environment,
        ReceiverConfig(
            noise_floor_dbm=-100,
            noise_std_db=2
        ),
        seed=123
    )

    detector = SignalDetector(
        DetectorConfig(
            threshold_dbm=-85,
            false_alarm_probability=0.02
        ),
        seed=456
    )

    results = []

    for time_slot in range(environment.config.num_time_slots):

        band = scanner.next_band()

        observation = receiver.observe(
            time_slot,
            band
        )

        detection = detector.detect(
            observation
        )

        # Ground truth is recorded ONLY for evaluation.
        # The scanner itself never receives this information.
        actual_occupied = bool(
            environment.truth[time_slot, band]
        )

        results.append({
            "time_slot": time_slot,
            "band": band,
            "detected": detection["detected"],
            "actual_occupied": actual_occupied,
            "false_alarm": detection["false_alarm"],
            "measured_power_dbm":
                detection["measured_power_dbm"],
            "snr_db":
                detection["snr_db"],
        })

    print(f"\n=== {scanner_name} ===\n")

    for result in results[:20]:
        print(
            f"t={result['time_slot']:02d} | "
            f"band={result['band']:02d} | "
            f"power={result['measured_power_dbm']:7.2f} dBm | "
            f"detected={str(result['detected']):5s} | "
            f"occupied={str(result['actual_occupied']):5s}"
        )

    detections = sum(
        r["detected"] for r in results
    )

    true_positives = sum(
        r["detected"] and r["actual_occupied"]
        for r in results
    )

    false_alarms = sum(
        r["detected"] and not r["actual_occupied"]
        for r in results
    )

    print("\nSummary:")
    print(f"Total scans:       {len(results)}")
    print(f"Detections:        {detections}")
    print(f"True positives:    {true_positives}")
    print(f"False alarms:      {false_alarms}")

    metrics = evaluate(results)
    print_metrics(scanner_name, metrics)

    return results


def main():
    print("\n========================================")
    print(" SMARTSCAN BASELINE SCANNER EXPERIMENT")
    print("========================================")

    sequential = SequentialScanner(
        num_bands=20
    )

    random = RandomScanner(
        num_bands=20,
        seed=999
    )

    run_scanner(
        "SEQUENTIAL SCANNER",
        sequential
    )

    run_scanner(
        "RANDOM SCANNER",
        random
    )


if __name__ == "__main__":
    main()
