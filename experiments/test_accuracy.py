import sys

from environment.emitter import Emitter
from environment.rf_environment import RFEnvironment, RFEnvironmentConfig
from receiver.receiver import NarrowbandReceiver, ReceiverConfig
from receiver.detector import SignalDetector, DetectorConfig
from scheduler.smartscan_v31 import SmartScanV31Scheduler


def create_environment(seed=42):
    config = RFEnvironmentConfig(
        num_bands=20,
        num_time_slots=100,
        band_start_mhz=100.0,
        band_width_mhz=5.0,
        seed=seed,
    )

    emitters = [
        Emitter(
            emitter_id="E01",
            name="Fixed Emitter",
            center_frequency=120.0,
            bandwidth=5.0,
            power_dbm=-55.0,
            behavior="fixed",
            start_band=4,
        ),
        Emitter(
            emitter_id="E02",
            name="Intermittent Emitter 1",
            center_frequency=150.0,
            bandwidth=5.0,
            power_dbm=-60.0,
            behavior="intermittent",
            duty_cycle=0.35,
            start_band=10,
        ),
        Emitter(
            emitter_id="E03",
            name="Intermittent Emitter 2",
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

    env = RFEnvironment(config, emitters)
    env.reset()
    return env


def calculate_accuracy(seed=42):
    env = create_environment(seed)

    receiver = NarrowbandReceiver(
        env,
        ReceiverConfig(
            noise_floor_dbm=-100.0,
            noise_std_db=2.0,
            receiver_bandwidth_mhz=5.0,
        ),
        seed=123,
    )

    detector = SignalDetector(
        DetectorConfig(
            threshold_dbm=-85.0,
            false_alarm_probability=0.02,
        ),
        seed=456,
    )

    scheduler = SmartScanV31Scheduler(
        num_bands=env.config.num_bands,
        exploration=2.0,
        prediction_weight=1.5,
        exploration_weight=0.8,
        revisit_weight=0.6,
        quality_weight=0.35,
        max_staleness=15,
        seed=3131 + seed,
    )

    scheduler.reset()

    tp = tn = fp = fn = 0
    total_scans = 0

    for time_slot in range(env.config.num_time_slots):

        bands = scheduler.select_bands(
            time_slot=time_slot,
            num_scans=4,
        )

        for band in bands:
            observation = receiver.observe(time_slot, band)
            detection = detector.detect(observation)

            detected = bool(detection["detected"])
            actual = bool(env.truth[time_slot, band] > 0)

            if actual and detected:
                tp += 1
            elif actual and not detected:
                fn += 1
            elif not actual and detected:
                fp += 1
            else:
                tn += 1

            scheduler.update(
                band=band,
                detected=detected,
                measured_power_dbm=observation["measured_power_dbm"],
                false_alarm=detection["false_alarm"],
            )

            total_scans += 1

    accuracy = (tp + tn) / total_scans if total_scans else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    pd = tp / (tp + fn) if (tp + fn) else 0.0
    pfa = fp / (fp + tn) if (fp + tn) else 0.0

    print()
    print("=" * 65)
    print("SMARTSCAN V3.1 ACCURACY TEST")
    print("=" * 65)
    print(f"Seed                   : {seed}")
    print(f"Total scans            : {total_scans}")
    print()
    print("CONFUSION MATRIX")
    print("-" * 65)
    print(f"True Positives (TP)    : {tp}")
    print(f"True Negatives (TN)    : {tn}")
    print(f"False Positives (FP)   : {fp}")
    print(f"False Negatives (FN)   : {fn}")
    print()
    print("METRICS")
    print("-" * 65)
    print(f"Accuracy               : {accuracy * 100:.2f}%")
    print(f"Precision              : {precision * 100:.2f}%")
    print(f"Detection Probability  : {pd * 100:.2f}%")
    print(f"False Alarm Probability: {pfa * 100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    calculate_accuracy(seed)
