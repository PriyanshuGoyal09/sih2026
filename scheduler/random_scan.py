import numpy as np


class RandomScanner:
    """
    Random baseline scanner.

    Selects a band uniformly at random at every scan.
    """

    def __init__(self, num_bands: int, seed: int = 999):
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")

        self.num_bands = num_bands
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def next_band(self) -> int:
        return int(self.rng.integers(0, self.num_bands))
