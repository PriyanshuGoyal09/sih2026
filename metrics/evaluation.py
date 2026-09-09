from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class MetricsResult:

    # Cell-level metrics
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    probability_of_detection: float
    probability_of_false_alarm: float
    sensitivity: float
    precision: float
    accuracy: float

    # Scan statistics
    total_scans: int
    detection_count: int
    false_alarm_count: int

    # Event-level metrics
    total_transmission_events: int
    intercepted_events: int
    missed_events: int

    intercept_rate: float

    average_intercept_time: float
    maximum_intercept_time: float

    average_intercept_time_error: float


class MetricsEvaluator:
    """
    Evaluation engine for SmartScan.

    CELL LEVEL
    ----------
    Probability of Detection:
        detected occupied cells / all occupied cells

    Probability of False Alarm:
        false positives / scanned unoccupied cells

    EVENT LEVEL
    -----------
    A transmission event is considered intercepted when
    the first valid detection occurs while that emitter
    is transmitting on the scanned band.

    Ground truth is used ONLY by this evaluator.
    """

    def __init__(self, environment):

        self.environment = environment

    def evaluate(
        self,
        results: List[Dict[str, Any]]
    ) -> MetricsResult:

        if results is None:
            results = []

        # -------------------------------------------------
        # BUILD SCANNED CELL SET
        # -------------------------------------------------

        scanned_cells = set()

        for result in results:

            scanned_cells.add(
                (
                    int(result["time_slot"]),
                    int(result["band"])
                )
            )

        # -------------------------------------------------
        # CELL-LEVEL METRICS
        # -------------------------------------------------

        true_positives = 0
        false_positives = 0
        true_negatives = 0

        detection_count = 0
        false_alarm_count = 0

        # Detection statistics are based on actual
        # receiver observations.

        for result in results:

            detected = bool(
                result.get(
                    "detected",
                    False
                )
            )

            occupied = bool(
                result.get(
                    "actual_occupied",
                    False
                )
            )

            if detected:

                detection_count += 1

            if result.get(
                "false_alarm",
                False
            ):

                false_alarm_count += 1

            if detected and occupied:

                true_positives += 1

            elif detected and not occupied:

                false_positives += 1

            elif not detected and not occupied:

                true_negatives += 1

        # -------------------------------------------------
        # TOTAL OCCUPIED CELLS
        # -------------------------------------------------

        total_occupied_cells = int(
            self.environment.truth.sum()
        )

        # Every occupied cell that was not successfully
        # detected is a missed detection.

        false_negatives = (
            total_occupied_cells
            - true_positives
        )

        if false_negatives < 0:
            false_negatives = 0

        # -------------------------------------------------
        # PROBABILITY OF DETECTION
        # -------------------------------------------------

        if total_occupied_cells > 0:

            probability_of_detection = (
                true_positives
                / total_occupied_cells
            )

        else:

            probability_of_detection = 0.0

        # Sensitivity is equivalent to Pd.

        sensitivity = (
            probability_of_detection
        )

        # -------------------------------------------------
        # PROBABILITY OF FALSE ALARM
        # -------------------------------------------------

        scanned_unoccupied_cells = sum(
            1
            for result in results
            if not result.get(
                "actual_occupied",
                False
            )
        )

        if scanned_unoccupied_cells > 0:

            probability_of_false_alarm = (
                false_positives
                / scanned_unoccupied_cells
            )

        else:

            probability_of_false_alarm = 0.0

        # -------------------------------------------------
        # PRECISION
        # -------------------------------------------------

        if (
            true_positives
            + false_positives
            > 0
        ):

            precision = (
                true_positives
                / (
                    true_positives
                    + false_positives
                )
            )

        else:

            precision = 0.0

        # -------------------------------------------------
        # ACCURACY
        #
        # Accuracy is deliberately based on observed
        # cells, because unobserved cells cannot be
        # classified by the receiver.
        # -------------------------------------------------

        observed_cells = len(
            results
        )

        if observed_cells > 0:

            accuracy = (
                true_positives
                + true_negatives
            ) / observed_cells

        else:

            accuracy = 0.0

        # -------------------------------------------------
        # EVENT-LEVEL METRICS
        # -------------------------------------------------

        events = getattr(
            self.environment,
            "transmission_events",
            []
        )

        total_events = len(events)

        intercepted_events = 0

        intercept_times = []

        for event in events:

            emitter_id = event[
                "emitter_id"
            ]

            start_time = int(
                event["start_time"]
            )

            end_time = int(
                event["end_time"]
            )

            first_intercept_time = None

            for result in results:

                if not result.get(
                    "detected",
                    False
                ):
                    continue

                # False alarms cannot constitute
                # emitter interception.

                if result.get(
                    "false_alarm",
                    False
                ):
                    continue

                time_slot = int(
                    result["time_slot"]
                )

                band = int(
                    result["band"]
                )

                if not (
                    start_time
                    <= time_slot
                    <= end_time
                ):
                    continue

                emitter_bands = (
                    self.environment
                    .emitter_band_map
                    .get(
                        time_slot,
                        {}
                    )
                )

                actual_band = (
                    emitter_bands.get(
                        emitter_id
                    )
                )

                if actual_band != band:
                    continue

                if (
                    first_intercept_time
                    is None
                    or time_slot
                    < first_intercept_time
                ):

                    first_intercept_time = (
                        time_slot
                    )

            if first_intercept_time is not None:

                intercepted_events += 1

                intercept_delay = (
                    first_intercept_time
                    - start_time
                )

                intercept_times.append(
                    intercept_delay
                )

        missed_events = (
            total_events
            - intercepted_events
        )

        # -------------------------------------------------
        # INTERCEPT RATE
        # -------------------------------------------------

        if total_events > 0:

            intercept_rate = (
                intercepted_events
                / total_events
            )

        else:

            intercept_rate = 0.0

        # -------------------------------------------------
        # INTERCEPT TIME
        # -------------------------------------------------

        if intercept_times:

            average_intercept_time = (
                sum(intercept_times)
                / len(intercept_times)
            )

            maximum_intercept_time = max(
                intercept_times
            )

            average_intercept_time_error = (
                average_intercept_time
            )

        else:

            average_intercept_time = 0.0
            maximum_intercept_time = 0.0
            average_intercept_time_error = 0.0

        return MetricsResult(

            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives,

            probability_of_detection=
                probability_of_detection,

            probability_of_false_alarm=
                probability_of_false_alarm,

            sensitivity=sensitivity,
            precision=precision,
            accuracy=accuracy,

            total_scans=len(results),

            detection_count=detection_count,

            false_alarm_count=
                false_alarm_count,

            total_transmission_events=
                total_events,

            intercepted_events=
                intercepted_events,

            missed_events=
                missed_events,

            intercept_rate=
                intercept_rate,

            average_intercept_time=
                average_intercept_time,

            maximum_intercept_time=
                maximum_intercept_time,

            average_intercept_time_error=
                average_intercept_time_error
        )


def print_metrics(
    metrics: MetricsResult
):

    print(
        "\n" + "=" * 50
    )

    print(
        " SMARTSCAN EVALUATION"
    )

    print(
        "=" * 50
    )

    print(
        "\nCELL-LEVEL METRICS"
    )

    print(
        "-" * 50
    )

    print(
        f"True positives:       "
        f"{metrics.true_positives}"
    )

    print(
        f"False positives:      "
        f"{metrics.false_positives}"
    )

    print(
        f"True negatives:       "
        f"{metrics.true_negatives}"
    )

    print(
        f"False negatives:      "
        f"{metrics.false_negatives}"
    )

    print(
        f"Probability Detection: "
        f"{metrics.probability_of_detection:.4f}"
    )

    print(
        f"Probability False Alarm:"
        f" {metrics.probability_of_false_alarm:.4f}"
    )

    print(
        f"Sensitivity:           "
        f"{metrics.sensitivity:.4f}"
    )

    print(
        f"Precision:             "
        f"{metrics.precision:.4f}"
    )

    print(
        f"Accuracy:              "
        f"{metrics.accuracy:.4f}"
    )

    print(
        "\nEVENT-LEVEL METRICS"
    )

    print(
        "-" * 50
    )

    print(
        f"Transmission events:   "
        f"{metrics.total_transmission_events}"
    )

    print(
        f"Intercepted events:    "
        f"{metrics.intercepted_events}"
    )

    print(
        f"Missed events:         "
        f"{metrics.missed_events}"
    )

    print(
        f"Intercept rate:        "
        f"{metrics.intercept_rate:.4f}"
    )

    print(
        f"Average intercept time:"
        f" {metrics.average_intercept_time:.2f} slots"
    )

    print(
        f"Maximum intercept time:"
        f" {metrics.maximum_intercept_time:.2f} slots"
    )

    print(
        f"Average intercept time error:"
        f" {metrics.average_intercept_time_error:.2f} slots"
    )

    print(
        "\nSCAN STATISTICS"
    )

    print(
        "-" * 50
    )

    print(
        f"Total scans:           "
        f"{metrics.total_scans}"
    )

    print(
        f"Detections:            "
        f"{metrics.detection_count}"
    )

    print(
        f"False alarms:          "
        f"{metrics.false_alarm_count}"
    )

    print(
        "=" * 50
    )
