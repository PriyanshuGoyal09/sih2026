from dataclasses import dataclass
import numpy as np


@dataclass
class NoiseModel:
    noise_floor_dbm: float = -100.0
    noise_std_db: float = 2.0
    false_alarm_probability: float = 0.02

    def generate_noise(
        self,
        num_bands: int,
        rng: np.random.Generator
    ) -> np.ndarray:

        return rng.normal(
            self.noise_floor_dbm,
            self.noise_std_db,
            num_bands
        )

    def generate_false_alarms(
        self,
        num_bands: int,
        rng: np.random.Generator
    ) -> np.ndarray:

        return rng.random(num_bands) < self.false_alarm_probability
