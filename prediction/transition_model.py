import numpy as np


class BandTransitionModel:
    """
    First-order Markov model for frequency-band transitions.

    Learns:
        P(next_band | current_band)

    The model is updated only from receiver observations.
    Ground-truth emitter information is never required.
    """

    def __init__(
        self,
        num_bands: int,
        smoothing: float = 1.0
    ):

        if num_bands <= 0:
            raise ValueError(
                "num_bands must be positive"
            )

        self.num_bands = num_bands
        self.smoothing = smoothing

        self.reset()

    def reset(self):

        self.transition_counts = np.zeros(
            (
                self.num_bands,
                self.num_bands
            ),
            dtype=float
        )

        self.last_detected_band = None

    def update(
        self,
        band: int,
        detected: bool
    ):
        """
        Learn a transition only when a signal is detected.

        Consecutive detections on the same band are also useful:
        they indicate persistence.
        """

        if not detected:
            return

        if not 0 <= band < self.num_bands:
            raise ValueError(
                "Invalid band."
            )

        if self.last_detected_band is not None:

            previous = self.last_detected_band

            self.transition_counts[
                previous,
                band
            ] += 1.0

        self.last_detected_band = band

    def predict_next(
        self,
        current_band: int
    ) -> np.ndarray:

        if not 0 <= current_band < self.num_bands:
            raise ValueError(
                "Invalid band."
            )

        counts = (
            self.transition_counts[
                current_band
            ]
            + self.smoothing
        )

        probabilities = (
            counts
            / np.sum(counts)
        )

        return probabilities

    def get_transition_matrix(self):

        matrix = np.zeros_like(
            self.transition_counts
        )

        for band in range(
            self.num_bands
        ):

            counts = (
                self.transition_counts[band]
                + self.smoothing
            )

            matrix[band] = (
                counts
                / np.sum(counts)
            )

        return matrix

    def get_most_likely_next(
        self,
        current_band: int,
        top_k: int = 5
    ):

        probabilities = self.predict_next(
            current_band
        )

        indices = np.argsort(
            probabilities
        )[::-1]

        return [
            (
                int(index),
                float(probabilities[index])
            )
            for index in indices[:top_k]
        ]
