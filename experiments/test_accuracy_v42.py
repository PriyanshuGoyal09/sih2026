import sys
import numpy as np

from environment.emitter import Emitter
from environment.rf_environment import RFEnvironment, RFEnvironmentConfig
from receiver.receiver import NarrowbandReceiver, ReceiverConfig
from receiver.adaptive_detector import AdaptiveSignalDetector, AdaptiveDetectorConfig
from scheduler.smartscan_v4 import SmartScanV4Scheduler


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
            "E01", "Fixed Emitter", 120.0, 5.0, -55.0,
            behavior="fixed", start_band=4
        ),
        Emitter(
            "E02", "Intermittent Emitter 1", 150.0, 5.0, -60.0,
            behavior="intermittent", duty_cycle=0.35, start_band=10
        ),
        Emitter(
            "E03", "Intermittent Emitter 2", 175.0, 5.0, -65.0,
            behavior="intermittent", duty_cycle=0.20, start_band=15
        ),
        Emitter(
            "E04", "Agile Emitter", 160.0, 5.0, -50.0,
            behavior="agile", agility=3, start_band=12
        ),
        Emitter(
            "E05", "Bursty Emitter", 135.0, 5.0, -58.0,
            behavior="bursty", duty_cycle=0.15, start_band=7
        ),
    ]

    env = RFEnvironment(config, emitters)
    env.reset()
    return env


def calculate_metrics(seed=42):
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

    detector = AdaptiveSignalDetector(
        num_bands=env.config.num_bands,
        config=AdaptiveDetectorConfig(
            noise_floor_dbm=-100.0,
            initial_noise_std_db=2.0,
            threshold_sigma=5.0,
            noise_alpha=0.05,
            min_threshold_dbm=-92.0,
            max_threshold_dbm=-80.0,
        ),
    )


    scheduler = SmartScanV4Scheduler(
        num_bands=env.config.num_bands,
        scans_per_slot=4,
        exploration_weight=1.0,
        prediction_weight=1.5,
        quality_weight=1.0,
        staleness_weight=1.2,
        revisit_weight=0.8,
        coverage_weight=1.0,
        decay=0.90,
        max_staleness=12,
        global_exploration_period=10,
        seed=3131 + seed,
    )

    scheduler.reset()

    # -1 = not scanned
    #  0 = scanned and no detection
    #  1 = scanned and detection
    prediction = -np.ones(
        (env.config.num_time_slots, env.config.num_bands),
        dtype=int
    )

    scanned = set()

    # Genuine detections only, excluding false alarms.
    genuine_detections = set()

    # ---------------------------------------------------------
    # RUN SCENARIO
    # ---------------------------------------------------------
    for time_slot in range(env.config.num_time_slots):

        bands = scheduler.select_bands(
            time_slot=time_slot,
            num_scans=4,
        )

        for band in bands:

            observation = receiver.observe(
                time_slot,
                band
            )

            detection = detector.detect(
                observation
            )

            detected = bool(
                detection["detected"]
            )

            prediction[time_slot, band] = (
                1 if detected else 0
            )

            scanned.add(
                (time_slot, band)
            )

            # A genuine detection is required for event interception.
            if detected and not detection["false_alarm"]:
                genuine_detections.add(
                    (time_slot, band)
                )

            scheduler.update(
                band=band,
                detected=detected,
                measured_power_dbm=observation[
                    "measured_power_dbm"
                ],
                false_alarm=detection["false_alarm"],
                time_slot=time_slot,
            )

    # ---------------------------------------------------------
    # OBSERVED-CELL CONFUSION MATRIX
    # ---------------------------------------------------------
    tp = 0
    tn = 0
    fp = 0
    fn_observed = 0

    for time_slot, band in scanned:

        actual = bool(
            env.truth[time_slot, band] > 0
        )

        detected = (
            prediction[time_slot, band] == 1
        )

        if actual and detected:
            tp += 1

        elif actual and not detected:
            fn_observed += 1

        elif not actual and detected:
            fp += 1

        else:
            tn += 1

    total_scanned = len(scanned)

    accuracy = (
        (tp + tn) / total_scanned
        if total_scanned else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) else 0.0
    )

    observed_pd = (
        tp / (tp + fn_observed)
        if (tp + fn_observed) else 0.0
    )

    pfa = (
        fp / (fp + tn)
        if (fp + tn) else 0.0
    )

    # ---------------------------------------------------------
    # GLOBAL Pd
    # ---------------------------------------------------------
    total_occupied_cells = int(
        np.sum(env.truth)
    )

    global_tp = tp

    global_fn = (
        total_occupied_cells - global_tp
    )

    global_pd = (
        global_tp / total_occupied_cells
        if total_occupied_cells else 0.0
    )

    # ---------------------------------------------------------
    # EVENT INTERCEPTION
    # ---------------------------------------------------------
    events = env.transmission_events

    intercepted_events = 0
    intercept_delays = []

    for event in events:

        emitter_id = event["emitter_id"]
        start_time = event["start_time"]
        end_time = event["end_time"]

        first_intercept = None

        for time_slot in range(
            start_time,
            end_time + 1
        ):

            actual_band = (
                env.emitter_band_map[
                    time_slot
                ].get(emitter_id)
            )

            if actual_band is None:
                continue

            if (
                time_slot,
                actual_band
            ) in genuine_detections:

                first_intercept = time_slot
                break

        if first_intercept is not None:

            intercepted_events += 1

            intercept_delays.append(
                first_intercept - start_time
            )

    event_rate = (
        intercepted_events / len(events)
        if events else 0.0
    )

    avg_intercept_time = (
        np.mean(intercept_delays)
        if intercept_delays else 0.0
    )

    # ---------------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------------
    print()
    print("=" * 72)
    print("SMARTSCAN V4 — COMPLETE SIH METRIC TEST")
    print("=" * 72)

    print(f"Seed                         : {seed}")
    print(
        f"Total spectrum cells        : "
        f"{env.config.num_time_slots * env.config.num_bands}"
    )
    print(
        f"Total occupied cells        : "
        f"{total_occupied_cells}"
    )
    print(
        f"Scheduler scans             : "
        f"{total_scanned}"
    )
    print(
        f"Transmission events         : "
        f"{len(events)}"
    )

    print()
    print("CONFUSION MATRIX")
    print("-" * 72)
    print(f"True Positives (TP)         : {tp}")
    print(f"True Negatives (TN)         : {tn}")
    print(f"False Positives (FP)        : {fp}")
    print(f"Observed False Negatives    : {fn_observed}")
    print(f"Global False Negatives      : {global_fn}")

    print()
    print("OBSERVED-CELL METRICS")
    print("-" * 72)
    print(f"Accuracy                    : {accuracy * 100:.2f}%")
    print(f"Precision                   : {precision * 100:.2f}%")
    print(f"Observed Pd                 : {observed_pd * 100:.2f}%")
    print(f"Pfa                         : {pfa * 100:.2f}%")

    print()
    print("GLOBAL / SIH-RELEVANT METRICS")
    print("-" * 72)
    print(f"Global Pd                   : {global_pd * 100:.2f}%")
    print(
        f"Event interception rate     : "
        f"{event_rate * 100:.2f}%"
    )
    print(
        f"Events intercepted          : "
        f"{intercepted_events}/{len(events)}"
    )
    print(
        f"Average intercept time      : "
        f"{avg_intercept_time:.2f} slots"
    )

    print("=" * 72)

    return {
        "seed": seed,
        "accuracy": accuracy,
        "precision": precision,
        "observed_pd": observed_pd,
        "global_pd": global_pd,
        "pfa": pfa,
        "event_rate": event_rate,
        "avg_intercept_time": avg_intercept_time,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "global_fn": global_fn,
    }


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    calculate_metrics(seed)
