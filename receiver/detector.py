from dataclasses import dataclass
import numpy as np


@dataclass
class DetectorConfig:
    threshold_dbm: float = -85.0
    false_alarm_probability: float = 0.02


class SignalDetector:
    """
    Threshold-based signal detector.

    The detector uses the receiver's measured power, not RF ground truth.
    """

    def __init__(self, config=None, seed=456):
        self.config = config or DetectorConfig()
        self.rng = np.random.default_rng(seed)

    def detect(self, observation: dict) -> dict:
        measured_power = observation["measured_power_dbm"]

        # Basic energy/threshold detection.
        threshold_detection = measured_power >= self.config.threshold_dbm

        # Small stochastic false-alarm mechanism when the measured
        # signal is below threshold.
        if not threshold_detection:
            false_alarm = (
                self.rng.random() < self.config.false_alarm_probability
            )
        else:
            false_alarm = False

        detected = threshold_detection or false_alarm

        result = observation.copy()

        result["detected"] = bool(detected)
        result["threshold_dbm"] = self.config.threshold_dbm
        result["false_alarm"] = bool(
            false_alarm and not threshold_detection
        )

        return result
