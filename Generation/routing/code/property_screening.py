from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class PropertyDecision(str, Enum):
    PASS = "pass"
    THRESHOLD_REVIEW = "review"
    REJECT = "reject"
    REFERENCE_ONLY = "reference_only"


class ThresholdDirection(str, Enum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"


class ThresholdVerdict(str, Enum):
    """Outcome of the hard-property thresholds for one molecule.

    ``PASS`` - every enabled hard threshold is met; ``REVIEW`` - the values sit
    inside the uncertainty band and need a cheap look; ``REJECT`` - below the
    thresholds. The *values* stay as written here because they are the internal
    keys of :data:`ROUTE_TO_SCREENING_STATUS` and never reach the run tables.
    """

    PASS = "pass"
    REVIEW = "review"
    REJECT = "reject"


#: internal screening route -> neutral status written into the run tables. The
#: two-axis channel split decides the pool; this status only records why a row is
#: is (thresholds met, needs a cheap review, below thresholds, or a data gap).
ROUTE_TO_SCREENING_STATUS = {
    ThresholdVerdict.PASS.value: "pass",
    ThresholdVerdict.REVIEW.value: "review",
    ThresholdVerdict.REJECT.value: "below_threshold",
    "missing_properties": "missing_properties",
    "property_failed": "property_failed",
    "reject_pre_property": "pre_screen_reject",
}

GROUP_SUMMARY_STATUSES = (
    "pass",
    "review",
    "below_threshold",
    "missing_properties",
    "property_failed",
)


def screening_status(route: object) -> str:
    """Neutral screening status for an internal route value."""
    value = str(route or "").strip().lower()
    return ROUTE_TO_SCREENING_STATUS.get(value, value)


@dataclass(frozen=True)
class FuelPropertyThresholds:
    density_min_g_cm3: float = 0.85
    heating_value_min_mj_kg: float = 38.0
    heating_value_possible_min_mj_kg: float = 37.0
    melting_point_max_k: float = 273.15
    boiling_point_min_k: float = 450.0
    flash_point_min_k: float = 310.0
    volumetric_heating_value_min_mj_l: float = 32.3


@dataclass(frozen=True)
class PropertyThresholdDecision:
    property_name: str
    predicted_value: float
    prediction_uncertainty: float | None
    threshold: float
    direction: ThresholdDirection
    relative_error_margin: float
    uncertainty_scale: float | None
    decision: PropertyDecision


def classify_property_threshold_with_uncertainty(
    property_name: str,
    predicted_value: float,
    prediction_uncertainty: float,
    threshold: float,
    direction: ThresholdDirection,
    uncertainty_scale: float = 1.0,
) -> PropertyThresholdDecision:
    """Classify one predicted property against a hard threshold.

    A value inside the uncertainty band is marked for review instead of being
    rejected, so a prediction that could still cross the threshold is kept.
    """

    if not all(math.isfinite(v) for v in (predicted_value, prediction_uncertainty, threshold, uncertainty_scale)):
        raise ValueError("property value, uncertainty, threshold and scale must be finite")
    if uncertainty_scale < 0:
        raise ValueError("uncertainty_scale must be non-negative")
    if prediction_uncertainty < 0:
        raise ValueError("prediction_uncertainty must be non-negative")

    margin = uncertainty_scale * prediction_uncertainty
    if direction is ThresholdDirection.MINIMUM:
        if predicted_value >= threshold:
            decision = PropertyDecision.PASS
        elif predicted_value + margin >= threshold:
            decision = PropertyDecision.THRESHOLD_REVIEW
        else:
            decision = PropertyDecision.REJECT
    else:
        if predicted_value <= threshold:
            decision = PropertyDecision.PASS
        elif predicted_value - margin <= threshold:
            decision = PropertyDecision.THRESHOLD_REVIEW
        else:
            decision = PropertyDecision.REJECT

    return PropertyThresholdDecision(
        property_name=property_name,
        predicted_value=predicted_value,
        prediction_uncertainty=prediction_uncertainty,
        threshold=threshold,
        direction=direction,
        relative_error_margin=0.0,
        uncertainty_scale=uncertainty_scale,
        decision=decision,
    )
