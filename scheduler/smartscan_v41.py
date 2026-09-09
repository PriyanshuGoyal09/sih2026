import numpy as np


class SmartScanV41Scheduler:
    """
    SmartScan V4.1

    Lightweight adaptive scheduler for simulation/research.

    Design goals:
    - No GPU
    - No deep neural network
    - No additional RF hardware assumptions
    - Incremental O(N) updates
    - Adaptive exploration
    - Confidence-aware exploitation
    - Temporal prediction
    - Coverage protection
    - Staleness recovery
    - Revisit suppression
    - Distinct bands per scan cycle
    """

    def __init__(
        self,
        num_bands: int,
        scans_per_slot: int = 4,
        exploration_weight: float = 1.0,
        prediction_weight: float = 1.5,
        quality_weight: float = 1.0,
        staleness_weight: float = 1.2,
        revisit_weight: float = 0.8,
        coverage_weight: float = 1.0,
        confidence_weight: float = 0.8,
        decay: float = 0.90,
        max_staleness: int = 12,
        global_exploration_period: int = 10,
        seed: int = 4141,
    ):
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")

        if scans_per_slot <= 0:
            raise ValueError("scans_per_slot must be positive")

        self.num_bands = num_bands
        self.scans_per_slot = min(scans_per_slot, num_bands)

        self.exploration_weight = exploration_weight
        self.prediction_weight = prediction_weight
        self.quality_weight = quality_weight
        self.staleness_weight = staleness_weight
        self.revisit_weight = revisit_weight
        self.coverage_weight = coverage_weight
        self.confidence_weight = confidence_weight

        self.decay = decay
        self.max_staleness = max_staleness
        self.global_exploration_period = global_exploration_period

        self.rng = np.random.default_rng(seed)

        self.reset()

    def reset(self):
        self.scan_count = np.zeros(self.num_bands, dtype=float)

        self.detect_count = np.zeros(self.num_bands, dtype=float)

        self.false_alarm_count = np.zeros(
            self.num_bands,
            dtype=float,
        )

        self.recent_detection = np.zeros(
            self.num_bands,
            dtype=float,
        )

        self.recent_miss = np.zeros(
            self.num_bands,
            dtype=float,
        )

        self.signal_strength = np.zeros(
            self.num_bands,
            dtype=float,
        )

        self.signal_confidence = np.zeros(
            self.num_bands,
            dtype=float,
        )

        self.last_scanned = np.full(
            self.num_bands,
            -1,
            dtype=int,
        )

        self.transition_counts = np.zeros(
            (self.num_bands, self.num_bands),
            dtype=float,
        )

        self.last_detection_band = None
        self.last_detection_time = -1
        self.previous_detected_band = None

        self.total_scans = 0
        self.current_time = 0

    # ---------------------------------------------------------
    # LEARNING
    # ---------------------------------------------------------

    def update(
        self,
        band: int,
        detected: bool,
        measured_power_dbm: float,
        false_alarm: bool = False,
        time_slot: int | None = None,
    ):
        if not 0 <= band < self.num_bands:
            raise ValueError("Invalid band.")

        if time_slot is not None:
            self.current_time = time_slot

        # Gradually forget old information.
        self.recent_detection *= self.decay
        self.recent_miss *= self.decay
        self.signal_confidence *= self.decay

        self.scan_count[band] += 1.0
        self.last_scanned[band] = self.current_time
        self.total_scans += 1

        # Normalize signal strength into a useful bounded evidence value.
        # -100 dBm ≈ 0 evidence
        # -50 dBm  ≈ 1 evidence
        strength = np.clip(
            (measured_power_dbm + 100.0) / 50.0,
            0.0,
            1.0,
        )

        self.signal_strength[band] = (
            0.85 * self.signal_strength[band]
            + 0.15 * strength
        )

        if false_alarm:
            self.false_alarm_count[band] += 1.0

            # False alarms reduce confidence.
            self.signal_confidence[band] -= 0.50

            self.recent_miss[band] += 1.0
            return

        if detected:
            self.detect_count[band] += 1.0
            self.recent_detection[band] += 1.0

            # Detection confidence rises with signal strength.
            confidence_gain = 0.5 + strength

            self.signal_confidence[band] += (
                confidence_gain
            )

            if self.previous_detected_band is not None:
                self.transition_counts[
                    self.previous_detected_band,
                    band
                ] += 1.0

            self.previous_detected_band = band
            self.last_detection_band = band
            self.last_detection_time = self.current_time

        else:
            self.recent_miss[band] += 0.20

            # A relatively strong undetected observation can still
            # indicate uncertainty rather than a clean negative.
            if measured_power_dbm > -90.0:
                self.signal_confidence[band] += 0.05

    # ---------------------------------------------------------
    # QUALITY
    # ---------------------------------------------------------

    def _quality_score(self):
        """
        Smoothed detection probability adjusted for false alarms.
        """

        alpha = 1.0
        beta = 2.0

        detection_probability = (
            self.detect_count + alpha
        ) / (
            self.scan_count + alpha + beta
        )

        false_alarm_rate = (
            self.false_alarm_count
            / np.maximum(self.scan_count, 1.0)
        )

        quality = (
            detection_probability
            - 0.75 * false_alarm_rate
        )

        return np.clip(quality, 0.0, 1.0)

    # ---------------------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------------------

    def _confidence_score(self):
        """
        Confidence is high when observations are consistent.

        Uncertain bands receive exploration priority.
        """

        confidence = np.maximum(
            self.signal_confidence,
            0.0,
        )

        maximum = max(
            float(np.max(confidence)),
            1.0,
        )

        confidence /= maximum

        # Uncertainty = 1 - confidence
        uncertainty = 1.0 - confidence

        return uncertainty

    # ---------------------------------------------------------
    # TEMPORAL PREDICTION
    # ---------------------------------------------------------

    def _prediction_score(self):
        if self.last_detection_band is None:
            return np.zeros(
                self.num_bands,
                dtype=float,
            )

        row = self.transition_counts[
            self.last_detection_band
        ]

        # Lightweight Laplace smoothing.
        probability = row + 1.0
        probability /= probability.sum()

        return probability

    # ---------------------------------------------------------
    # STALENESS
    # ---------------------------------------------------------

    def _staleness_score(self, time_slot):
        age = np.where(
            self.last_scanned < 0,
            self.max_staleness,
            time_slot - self.last_scanned,
        )

        age = np.maximum(age, 0)

        return np.clip(
            age / max(self.max_staleness, 1),
            0.0,
            1.0,
        )

    # ---------------------------------------------------------
    # COVERAGE
    # ---------------------------------------------------------

    def _coverage_score(self):
        if self.total_scans == 0:
            return np.ones(
                self.num_bands,
                dtype=float,
            )

        maximum = max(
            float(np.max(self.scan_count)),
            1.0,
        )

        return 1.0 - (
            self.scan_count / maximum
        )

    # ---------------------------------------------------------
    # REVISIT
    # ---------------------------------------------------------

    def _revisit_penalty(self):
        return np.clip(
            self.recent_detection,
            0.0,
            2.0,
        )

    # ---------------------------------------------------------
    # ADAPTIVE EXPLORATION
    # ---------------------------------------------------------

    def _exploration_score(self, time_slot):
        progress = min(
            1.0,
            self.total_scans
            / max(self.num_bands * 2, 1),
        )

        base_exploration = (
            1.0 - 0.60 * progress
        )

        if self.last_detection_time < 0:
            silence = 1.0
        else:
            silence = min(
                1.0,
                max(
                    0.0,
                    time_slot
                    - self.last_detection_time,
                )
                / 10.0,
            )

        uncertainty = self._confidence_score()

        return (
            base_exploration
            + 0.50 * silence
            + 0.75 * uncertainty
        )

    # ---------------------------------------------------------
    # MAIN SCORE
    # ---------------------------------------------------------

    def get_scores(self, time_slot):
        quality = self._quality_score()

        prediction = self._prediction_score()

        staleness = self._staleness_score(
            time_slot
        )

        coverage = self._coverage_score()

        revisit = self._revisit_penalty()

        exploration = self._exploration_score(
            time_slot
        )

        uncertainty = self._confidence_score()

        score = (
            self.quality_weight
            * quality

            + self.prediction_weight
            * prediction

            + self.staleness_weight
            * staleness

            + self.coverage_weight
            * coverage

            + self.exploration_weight
            * exploration

            + self.confidence_weight
            * uncertainty

            - self.revisit_weight
            * revisit
        )

        return {
            "quality": quality,
            "prediction": prediction,
            "staleness": staleness,
            "coverage": coverage,
            "exploration": exploration,
            "uncertainty": uncertainty,
            "revisit": revisit,
            "score": score,
        }

    # ---------------------------------------------------------
    # BAND SELECTION
    # ---------------------------------------------------------

    def select_bands(
        self,
        time_slot: int,
        num_scans=None,
    ):
        if num_scans is None:
            num_scans = self.scans_per_slot

        num_scans = min(
            max(1, num_scans),
            self.num_bands,
        )

        self.current_time = time_slot

        scores = self.get_scores(
            time_slot
        )["score"].copy()

        # Periodic coverage sweep.
        if (
            self.global_exploration_period > 0
            and time_slot > 0
            and time_slot
            % self.global_exploration_period == 0
        ):
            scores += (
                1.25
                * self._coverage_score()
            )

        # Tiny jitter only for tie-breaking.
        scores += self.rng.normal(
            0.0,
            1e-8,
            self.num_bands,
        )

        selected = []

        for _ in range(num_scans):

            band = int(
                np.argmax(scores)
            )

            selected.append(band)

            # Never select the same band twice
            # during the same time slot.
            scores[band] = -np.inf

        return selected

    def next_band(self, time_slot=None):
        if time_slot is None:
            time_slot = self.current_time

        return self.select_bands(
            time_slot=time_slot,
            num_scans=1,
        )[0]

    # ---------------------------------------------------------
    # DIAGNOSTICS
    # ---------------------------------------------------------

    def get_statistics(self):
        return {
            "scan_count":
                self.scan_count.copy(),

            "detect_count":
                self.detect_count.copy(),

            "false_alarm_count":
                self.false_alarm_count.copy(),

            "quality":
                self._quality_score().copy(),

            "signal_strength":
                self.signal_strength.copy(),

            "signal_confidence":
                self.signal_confidence.copy(),

            "transition_matrix":
                self.get_transition_matrix(),

            "last_detection_band":
                self.last_detection_band,

            "last_detection_time":
                self.last_detection_time,

            "total_scans":
                self.total_scans,
        }

    def get_transition_matrix(self):
        matrix = np.zeros_like(
            self.transition_counts
        )

        for band in range(self.num_bands):
            row = (
                self.transition_counts[band]
                + 1.0
            )

            matrix[band] = (
                row / row.sum()
            )

        return matrix
