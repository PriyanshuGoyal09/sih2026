from dataclasses import dataclass
import numpy as np


@dataclass
class AdaptiveDetectorConfig:
    noise_floor_dbm: float = -100.0
    initial_noise_std_db: float = 2.0

    # Number of standard deviations above estimated noise.
    threshold_sigma: float = 5.0

    # How quickly the noise estimate adapts.
    noise_alpha: float = 0.05

    # Minimum allowed threshold.
    min_threshold_dbm: float = -92.0

    # Maximum allowed threshold.
    max_threshold_dbm: float = -80.0


class AdaptiveSignalDetector:
    """
    Lightweight adaptive detector.

    Estimates the local noise level for each frequency band and
    places the detection threshold above that estimate.

    No artificial random false alarms are injected.
    """

    def __init__(self, num_bands, config=None):
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")

        self.num_bands = num_bands
        self.config = config or AdaptiveDetectorConfig()

        self.reset()

    def reset(self):
        self.noise_mean = np.full(
            self.num_bands,
            self.config.noise_floor_dbm,
            dtype=float,
        )

        self.noise_std = np.full(
            self.num_bands,
            self.config.initial_noise_std_db,
            dtype=float,
        )

        self.samples = np.zeros(
            self.num_bands,
            dtype=int,
        )

    def _threshold(self, band):
        threshold = (
            self.noise_mean[band]
            + self.config.threshold_sigma
            * self.noise_std[band]
        )

        return float(
            np.clip(
                threshold,
                self.config.min_threshold_dbm,
                self.config.max_threshold_dbm,
            )
        )

    def detect(self, observation):
        band = int(observation["band"])
        measured_power = float(
            observation["measured_power_dbm"]
        )

        if not 0 <= band < self.num_bands:
            raise ValueError("Invalid band.")

        threshold = self._threshold(band)

        detected = measured_power >= threshold

        # Learn noise only from observations that are currently
        # considered below the signal threshold.
        #
        # This prevents strong emitters from contaminating the
        # noise estimate.
        if not detected:

            alpha = self.config.noise_alpha

            old_mean = self.noise_mean[band]

            new_mean = (
                (1.0 - alpha) * old_mean
                + alpha * measured_power
            )

            deviation = abs(
                measured_power - old_mean
            )

            old_std = self.noise_std[band]

            new_std = (
                (1.0 - alpha) * old_std
                + alpha * max(
                    deviation,
                    0.25,
                )
            )

            self.noise_mean[band] = new_mean

            self.noise_std[band] = np.clip(
                new_std,
                0.5,
                6.0,
            )

        self.samples[band] += 1

        return {
            **observation,
            "detected": bool(detected),
            "false_alarm": False,
            "threshold_dbm": threshold,
            "estimated_noise_dbm": float(
                self.noise_mean[band]
            ),
            "estimated_noise_std_db": float(
                self.noise_std[band]
            ),
        }
