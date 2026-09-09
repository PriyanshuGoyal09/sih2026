import numpy as np


class SmartScanV53Scheduler:
    """
    SmartScan V5.3

    Discovery-balanced scheduler with diminishing returns.

    Design:
      - exploit historically useful bands
      - predict likely next bands
      - preserve useful persistence
      - maintain spectrum coverage
      - recover stale bands
      - reduce marginal value of repeated recent detections
      - keep computation lightweight

    No neural network or additional hardware is assumed.
    """

    def __init__(
        self,
        num_bands: int,
        scans_per_slot: int = 4,
        prediction_weight: float = 2.0,
        quality_weight: float = 1.0,
        persistence_weight: float = 0.8,
        coverage_weight: float = 0.8,
        staleness_weight: float = 0.8,
        exploration_weight: float = 0.8,
        revisit_weight: float = 0.5,
        diversity_weight: float = 1.0,
        decay: float = 0.90,
        max_staleness: int = 12,
        diversity_window: int = 6,
        global_exploration_period: int = 10,
        seed: int = 5353,
    ):
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")

        if scans_per_slot <= 0:
            raise ValueError("scans_per_slot must be positive")

        self.num_bands = num_bands
        self.scans_per_slot = min(
            scans_per_slot,
            num_bands,
        )

        self.prediction_weight = prediction_weight
        self.quality_weight = quality_weight
        self.persistence_weight = persistence_weight
        self.coverage_weight = coverage_weight
        self.staleness_weight = staleness_weight
        self.exploration_weight = exploration_weight
        self.revisit_weight = revisit_weight
        self.diversity_weight = diversity_weight

        self.decay = decay
        self.max_staleness = max_staleness
        self.diversity_window = diversity_window
        self.global_exploration_period = global_exploration_period

        self.rng = np.random.default_rng(seed)

        self.reset()

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------

    def reset(self):
        self.scan_count = np.zeros(
            self.num_bands,
            dtype=float,
        )

        self.detect_count = np.zeros(
            self.num_bands,
            dtype=float,
        )

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

        self.last_scanned = np.full(
            self.num_bands,
            -1,
            dtype=int,
        )

        self.last_detected = np.full(
            self.num_bands,
            -1,
            dtype=int,
        )

        self.transition_counts = np.zeros(
            (self.num_bands, self.num_bands),
            dtype=float,
        )

        # Recent genuine detections:
        # (time_slot, band)
        self.detection_history = []

        self.last_detection_band = None
        self.last_detection_time = -1
        self.previous_detection_band = None

        self.total_scans = 0
        self.current_time = 0

    # ---------------------------------------------------------
    # UPDATE
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

        self.recent_detection *= self.decay
        self.recent_miss *= self.decay

        self.scan_count[band] += 1.0
        self.last_scanned[band] = self.current_time
        self.total_scans += 1

        if false_alarm:
            self.false_alarm_count[band] += 1.0

            # False alarms should reduce short-term priority.
            self.recent_miss[band] += 1.0
            return

        if detected:
            self.detect_count[band] += 1.0
            self.recent_detection[band] += 1.0
            self.last_detected[band] = self.current_time

            if self.previous_detection_band is not None:
                self.transition_counts[
                    self.previous_detection_band,
                    band,
                ] += 1.0

            self.previous_detection_band = band
            self.last_detection_band = band
            self.last_detection_time = self.current_time

            self.detection_history.append(
                (
                    self.current_time,
                    band,
                )
            )

            # Limit memory size.
            if len(self.detection_history) > 200:
                self.detection_history.pop(0)

        else:
            # Mild penalty because intermittent emitters can disappear.
            self.recent_miss[band] += 0.20

    # ---------------------------------------------------------
    # HISTORICAL QUALITY
    # ---------------------------------------------------------

    def _quality(self):
        alpha = 1.0
        beta = 2.0

        detection_rate = (
            self.detect_count + alpha
        ) / (
            self.scan_count + alpha + beta
        )

        false_alarm_rate = (
            self.false_alarm_count
            / np.maximum(
                self.scan_count,
                1.0,
            )
        )

        return np.clip(
            detection_rate
            - 0.75 * false_alarm_rate,
            0.0,
            1.0,
        )

    # ---------------------------------------------------------
    # TEMPORAL PREDICTION
    # ---------------------------------------------------------

    def _prediction(self):
        if self.last_detection_band is None:
            return np.zeros(
                self.num_bands,
                dtype=float,
            )

        row = (
            self.transition_counts[
                self.last_detection_band
            ]
            + 1.0
        )

        return row / row.sum()

    # ---------------------------------------------------------
    # PERSISTENCE
    # ---------------------------------------------------------

    def _persistence(self, time_slot):
        score = self.recent_detection.copy()

        if self.last_detection_time < 0:
            return score

        age = (
            time_slot
            - self.last_detection_time
        )

        if age > 10:
            score *= 0.20

        elif age > 6:
            score *= 0.45

        elif age > 3:
            score *= 0.70

        return score

    # ---------------------------------------------------------
    # COVERAGE
    # ---------------------------------------------------------

    def _coverage(self):
        if self.total_scans == 0:
            return np.ones(
                self.num_bands,
                dtype=float,
            )

        maximum = max(
            float(
                np.max(self.scan_count)
            ),
            1.0,
        )

        return 1.0 - (
            self.scan_count / maximum
        )

    # ---------------------------------------------------------
    # STALENESS
    # ---------------------------------------------------------

    def _staleness(self, time_slot):
        age = np.where(
            self.last_scanned < 0,
            self.max_staleness,
            time_slot - self.last_scanned,
        )

        age = np.maximum(
            age,
            0,
        )

        return np.clip(
            age / max(
                self.max_staleness,
                1,
            ),
            0.0,
            1.0,
        )

    # ---------------------------------------------------------
    # EXPLORATION
    # ---------------------------------------------------------

    def _exploration(self, time_slot):
        progress = min(
            1.0,
            self.total_scans
            / max(
                self.num_bands * 2,
                1,
            ),
        )

        base = (
            1.0
            - 0.60 * progress
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

        return (
            base
            + 0.50 * silence
        )

    # ---------------------------------------------------------
    # DIMINISHING RETURNS
    # ---------------------------------------------------------

    def _diversity_score(self, time_slot):
        """
        Higher score means the band has greater marginal value.

        Repeated recent detections reduce the score according to
        diminishing returns.

        This is deliberately bounded so that useful persistent
        emitters are not completely discarded.
        """

        recent_hits = np.zeros(
            self.num_bands,
            dtype=float,
        )

        for t, band in self.detection_history:
            age = time_slot - t

            if 0 <= age <= self.diversity_window:
                recent_hits[band] += (
                    1.0
                    / (1.0 + age)
                )

        # Saturating diminishing return:
        #
        # 0 recent hits → 1.0
        # 1 recent hit  → 0.50
        # 2 hits        → 0.33
        # ...
        #
        # This prevents a persistent emitter from dominating
        # the entire scan budget.
        marginal_value = (
            1.0
            / (1.0 + recent_hits)
        )

        return np.clip(
            marginal_value,
            0.0,
            1.0,
        )

    # ---------------------------------------------------------
    # REVISIT
    # ---------------------------------------------------------

    def _revisit(self):
        return np.clip(
            self.recent_detection,
            0.0,
            2.0,
        )

    # ---------------------------------------------------------
    # SCORES
    # ---------------------------------------------------------

    def _informed_score(self, time_slot):
        return (
            self.quality_weight
            * self._quality()

            + self.prediction_weight
            * self._prediction()

            + self.persistence_weight
            * self._persistence(time_slot)

            + self.staleness_weight
            * self._staleness(time_slot)

            + self.diversity_weight
            * self._diversity_score(time_slot)

            - self.revisit_weight
            * self._revisit()
        )

    def _exploration_score(self, time_slot):
        return (
            self.coverage_weight
            * self._coverage()

            + self.exploration_weight
            * self._exploration(time_slot)

            + self.staleness_weight
            * self._staleness(time_slot)

            + self.diversity_weight
            * self._diversity_score(time_slot)
        )

    # ---------------------------------------------------------
    # SELECTION
    # ---------------------------------------------------------

    def select_bands(
        self,
        time_slot: int,
        num_scans=None,
    ):
        if num_scans is None:
            num_scans = self.scans_per_slot

        num_scans = min(
            max(
                1,
                num_scans,
            ),
            self.num_bands,
        )

        self.current_time = time_slot

        # Start with normal informed ranking.
        informed_scores = (
            self._informed_score(time_slot)
        )

        exploration_scores = (
            self._exploration_score(time_slot)
        )

        informed_scores = informed_scores.copy()
        exploration_scores = exploration_scores.copy()

        informed_scores += self.rng.normal(
            0.0,
            1e-8,
            self.num_bands,
        )

        exploration_scores += self.rng.normal(
            0.0,
            1e-8,
            self.num_bands,
        )

        selected = []
        available = np.ones(
            self.num_bands,
            dtype=bool,
        )

        # -----------------------------------------------------
        # Adaptive number of exploration slots
        # -----------------------------------------------------

        # Normally reserve one slot for discovery.
        exploration_count = 1

        # After a prolonged silence, use two exploration slots.
        if self.last_detection_time >= 0:
            silence = (
                time_slot
                - self.last_detection_time
            )

            if silence >= 6:
                exploration_count = min(
                    2,
                    num_scans,
                )

        elif time_slot >= 4:
            exploration_count = min(
                2,
                num_scans,
            )

        informed_count = (
            num_scans
            - exploration_count
        )

        # -----------------------------------------------------
        # Informed selections
        # -----------------------------------------------------

        for _ in range(informed_count):
            masked = np.where(
                available,
                informed_scores,
                -np.inf,
            )

            band = int(
                np.argmax(masked)
            )

            selected.append(band)
            available[band] = False

        # -----------------------------------------------------
        # Exploration selections
        # -----------------------------------------------------

        for _ in range(exploration_count):
            masked = np.where(
                available,
                exploration_scores,
                -np.inf,
            )

            band = int(
                np.argmax(masked)
            )

            selected.append(band)
            available[band] = False

        # -----------------------------------------------------
        # Periodic global coverage safeguard
        # -----------------------------------------------------

        if (
            self.global_exploration_period > 0
            and time_slot > 0
            and time_slot
            % self.global_exploration_period == 0
        ):
            coverage = self._coverage()

            candidates = np.where(
                available,
                coverage,
                -np.inf,
            )

            replacement = int(
                np.argmax(candidates)
            )

            if np.isfinite(
                candidates[replacement]
            ):
                selected[-1] = replacement

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

    def get_statistics(self):
        return {
            "scan_count":
                self.scan_count.copy(),

            "detect_count":
                self.detect_count.copy(),

            "false_alarm_count":
                self.false_alarm_count.copy(),

            "quality":
                self._quality().copy(),

            "recent_detection":
                self.recent_detection.copy(),

            "last_scanned":
                self.last_scanned.copy(),

            "last_detected":
                self.last_detected.copy(),

            "transition_matrix":
                self.get_transition_matrix(),

            "detection_history":
                list(self.detection_history),

            "last_detection_band":
                self.last_detection_band,

            "last_detection_time":
                self.last_detection_time,

            "total_scans":
                self.total_scans,
        }
