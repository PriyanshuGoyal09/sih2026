"""Run scan policies on an RFEnvironment without leaking ground truth to the scheduler."""

from __future__ import annotations

import numpy as np

from metrics.evaluation import MetricsEvaluator
from receiver.adaptive_detector import AdaptiveDetectorConfig, AdaptiveSignalDetector
from receiver.receiver import NarrowbandReceiver, ReceiverConfig
from scheduler.bandit import UCBScanner
from scheduler.random_scan import RandomScanner
from scheduler.sequential import SequentialScanner
from scheduler.smartscan_v53 import SmartScanV53Scheduler


def _select_bands(scheduler, time_slot: int, num_scans: int) -> list[int]:
    if hasattr(scheduler, "select_bands"):
        return [int(b) for b in scheduler.select_bands(time_slot=time_slot, num_scans=num_scans)]

    if isinstance(scheduler, SequentialScanner):
        return [scheduler.next_band() for _ in range(num_scans)]

    if isinstance(scheduler, UCBScanner):
        scores = np.array(
            [scheduler._ucb_score(band) for band in range(scheduler.num_bands)],
            dtype=float,
        )
        order = np.argsort(-scores)
        return [int(b) for b in order[:num_scans]]

    selected = []
    seen = set()
    guard = 0
    while len(selected) < num_scans and guard < num_scans * 8:
        band = int(scheduler.next_band())
        if band not in seen:
            selected.append(band)
            seen.add(band)
        guard += 1
    return selected


def make_scheduler(name: str, num_bands: int, scans_per_slot: int, seed: int):
    key = name.lower()
    if key in {"smartscan", "smartscan_v53", "v53"}:
        scheduler = SmartScanV53Scheduler(
            num_bands=num_bands,
            scans_per_slot=scans_per_slot,
            seed=seed,
        )
    elif key in {"sequential", "round_robin"}:
        scheduler = SequentialScanner(num_bands=num_bands)
    elif key in {"ucb", "bandit"}:
        scheduler = UCBScanner(num_bands=num_bands, seed=seed)
    elif key in {"random"}:
        scheduler = RandomScanner(num_bands=num_bands, seed=seed)
    else:
        raise ValueError(f"Unknown scheduler: {name}")
    scheduler.reset()
    return scheduler


def run_scheduler(
    environment,
    scheduler_name: str,
    scans_per_slot: int = 4,
    seed: int = 123,
):
    environment.reset()
    receiver = NarrowbandReceiver(
        environment,
        ReceiverConfig(
            noise_floor_dbm=environment.noise_model.noise_floor_dbm,
            noise_std_db=2.0,
            receiver_bandwidth_mhz=environment.config.band_width_mhz,
        ),
        seed=seed,
    )
    detector = AdaptiveSignalDetector(
        num_bands=environment.config.num_bands,
        config=AdaptiveDetectorConfig(
            noise_floor_dbm=environment.noise_model.noise_floor_dbm,
            initial_noise_std_db=2.0,
            threshold_sigma=5.0,
            min_threshold_dbm=environment.noise_model.noise_floor_dbm + 8.0,
            max_threshold_dbm=environment.noise_model.noise_floor_dbm + 30.0,
        ),
    )
    scheduler = make_scheduler(
        scheduler_name,
        environment.config.num_bands,
        scans_per_slot,
        seed,
    )

    results = []
    scan_mask = np.zeros_like(environment.truth, dtype=int)
    detect_mask = np.zeros_like(environment.truth, dtype=int)

    for time_slot in range(environment.config.num_time_slots):
        bands = _select_bands(scheduler, time_slot, scans_per_slot)
        for band in bands:
            observation = receiver.observe(time_slot, band)
            detection = detector.detect(observation)
            occupied = bool(environment.truth[time_slot, band] > 0)
            detected = bool(detection["detected"])
            false_alarm = bool(detection.get("false_alarm", False))

            scan_mask[time_slot, band] = 1
            if detected:
                detect_mask[time_slot, band] = 1

            result = {
                "time_slot": time_slot,
                "band": int(band),
                "detected": detected,
                "false_alarm": false_alarm,
                "actual_occupied": occupied,
                "measured_power_dbm": float(observation["measured_power_dbm"]),
                "snr_db": float(observation["snr_db"]),
            }
            results.append(result)

            update_kwargs = {
                "band": int(band),
                "detected": detected,
                "measured_power_dbm": float(observation["measured_power_dbm"]),
                "false_alarm": false_alarm,
            }
            if hasattr(scheduler, "update"):
                try:
                    scheduler.update(time_slot=time_slot, **update_kwargs)
                except TypeError:
                    scheduler.update(**update_kwargs)

    metrics = MetricsEvaluator(environment).evaluate(results)
    return {
        "name": scheduler_name,
        "results": results,
        "metrics": metrics,
        "scan_mask": scan_mask,
        "detect_mask": detect_mask,
    }


def compare_schedulers(
    environment,
    names: list[str] | None = None,
    scans_per_slot: int = 4,
    seed: int = 123,
):
    names = names or ["sequential", "ucb", "smartscan"]
    return [
        run_scheduler(environment, name, scans_per_slot=scans_per_slot, seed=seed)
        for name in names
    ]


def metrics_row(run: dict) -> dict:
    metrics = run["metrics"]
    return {
        "Scheduler": run["name"],
        "Pd": metrics.probability_of_detection,
        "Pfa": metrics.probability_of_false_alarm,
        "Precision": metrics.precision,
        "Accuracy": metrics.accuracy,
        "Intercept rate": metrics.intercept_rate,
        "Avg intercept time (slots)": metrics.average_intercept_time,
        "Events intercepted": metrics.intercepted_events,
        "Total events": metrics.total_transmission_events,
        "Scans": metrics.total_scans,
    }
