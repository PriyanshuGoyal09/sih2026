class SequentialScanner:
    """
    Conventional sequential wideband scanner.

    Scans bands in ascending order and wraps around after the
    last band.
    """

    def __init__(self, num_bands: int):
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")

        self.num_bands = num_bands
        self.current_band = 0

    def reset(self):
        self.current_band = 0

    def next_band(self) -> int:
        band = self.current_band
        self.current_band = (self.current_band + 1) % self.num_bands
        return band
