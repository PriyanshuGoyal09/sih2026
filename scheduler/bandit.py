import numpy as np


class UCBScanner:
    """
    SmartScan UCB scheduler.

    Learns which bands are useful from receiver observations.

    The scheduler does NOT receive RF ground truth.
    """

    def __init__(
        self,
        num_bands: int,
        exploration: float = 2.0,
        seed: int = 2026
    ):

        if num_bands <= 0:
            raise ValueError(
                "num_bands must be positive"
            )

        self.num_bands = num_bands
        self.exploration = exploration

        self.rng = np.random.default_rng(seed)

        self.reset()

    def reset(self):

        self.counts = np.zeros(
            self.num_bands,
            dtype=int
        )

        self.rewards = np.zeros(
            self.num_bands,
            dtype=float
        )

        self.total_scans = 0

        self.last_band = None

        # Tracks whether a band was recently detected.
        self.recent_detection = np.zeros(
            self.num_bands,
            dtype=int
        )

    def update(
        self,
        band: int,
        detected: bool,
        measured_power_dbm: float,
        false_alarm: bool = False
    ):
        """
        Update scheduler using receiver information only.
        """

        if not 0 <= band < self.num_bands:
            raise ValueError(
                "Invalid band."
            )

        self.counts[band] += 1
        self.total_scans += 1

        # Base reward.
        if false_alarm:

            # False alarms waste receiver resources.
            reward = -0.75

            self.recent_detection[band] = 0

        elif detected:

            # A new detection is more valuable than repeatedly
            # observing the same already-active signal.
            if self.recent_detection[band] > 0:

                reward = 0.20

            else:

                reward = 1.0

            self.recent_detection[band] = 1

        else:

            reward = -0.05

            self.recent_detection[band] = 0

        # Weak signal evidence still contains information.
        if (
            not detected
            and measured_power_dbm > -90
        ):

            reward += 0.10

        # False-alarm penalty is handled through detector output
        # in the experiment layer.

        self.rewards[band] += reward

        self.last_band = band

    def _ucb_score(self, band: int) -> float:

        count = self.counts[band]

        # Explore every band at least once.
        if count == 0:
            return float("inf")

        mean_reward = (
            self.rewards[band]
            / count
        )

        exploration_bonus = (
            self.exploration
            * np.sqrt(
                np.log(
                    max(
                        1,
                        self.total_scans
                    )
                )
                / count
            )
        )

        return (
            mean_reward
            + exploration_bonus
        )

    def next_band(self) -> int:

        scores = np.array([
            self._ucb_score(band)
            for band in range(
                self.num_bands
            )
        ])

        best_score = np.max(scores)

        candidates = np.flatnonzero(
            np.isclose(
                scores,
                best_score
            )
        )

        return int(
            self.rng.choice(
                candidates
            )
        )

    def get_statistics(self):

        statistics = []

        for band in range(
            self.num_bands
        ):

            count = int(
                self.counts[band]
            )

            if count > 0:

                average_reward = (
                    self.rewards[band]
                    / count
                )

            else:

                average_reward = 0.0

            statistics.append({
                "band": band,
                "scans": count,
                "reward": float(
                    self.rewards[band]
                ),
                "average_reward": float(
                    average_reward
                ),
                "ucb_score": float(
                    self._ucb_score(band)
                )
            })

        return statistics
