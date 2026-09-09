from dataclasses import dataclass, field
import numpy as np


@dataclass
class Emitter:
    emitter_id: str
    name: str
    center_frequency: float
    bandwidth: float
    power_dbm: float
    behavior: str = "fixed"
    duty_cycle: float = 1.0
    agility: int = 0
    start_band: int = 0

    _current_band: int = field(init=False, default=-1)

    def reset(self):
        self._current_band = self.start_band

    def is_active(self, time_slot: int, rng: np.random.Generator) -> bool:

        if self.behavior == "fixed":
            return True

        if self.behavior == "intermittent":
            return rng.random() < self.duty_cycle

        if self.behavior == "periodic":
            period = max(
                1,
                int(round(1 / max(self.duty_cycle, 0.01)))
            )
            return time_slot % period == 0

        if self.behavior == "bursty":
            return rng.random() < self.duty_cycle

        if self.behavior == "agile":
            return True

        return False

    def get_band(
        self,
        time_slot: int,
        num_bands: int,
        rng: np.random.Generator
    ) -> int:

        if self.behavior != "agile":
            return self.start_band

        if self._current_band == -1:
            self._current_band = self.start_band
            return self._current_band

        movement = rng.integers(
            -self.agility,
            self.agility + 1
        )

        self._current_band = int(
            np.clip(
                self._current_band + movement,
                0,
                num_bands - 1
            )
        )

        return self._current_band
