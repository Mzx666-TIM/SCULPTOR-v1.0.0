from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoveltyAdjustedUncertainty:
    uncertainty: float
    novelty_score: float | None
    source: str


def novelty_adjusted_uncertainty(
    low_novelty_uncertainty: float,
    high_novelty_uncertainty: float,
    novelty_score: float | None = None,
    high_novelty: bool = False,
) -> NoveltyAdjustedUncertainty:
    """Scale an external-model uncertainty by fingerprint novelty.

    A novelty score of 0 uses the low-novelty baseline; a score of 1 uses the
    high-novelty uncertainty. If no score is available, ``high_novelty`` keeps
    the old two-level behavior for backward compatibility.
    """

    if low_novelty_uncertainty < 0 or high_novelty_uncertainty < 0:
        raise ValueError("uncertainty values must be non-negative")
    if high_novelty_uncertainty < low_novelty_uncertainty:
        raise ValueError("high novelty uncertainty must be >= low novelty uncertainty")

    if novelty_score is None:
        uncertainty = high_novelty_uncertainty if high_novelty else low_novelty_uncertainty
        mode = "high_novelty_flag" if high_novelty else "low_novelty_default"
        return NoveltyAdjustedUncertainty(
            uncertainty=float(uncertainty),
            novelty_score=None,
            source=(
                f"{mode}: low={low_novelty_uncertainty:.6g}, "
                f"high={high_novelty_uncertainty:.6g}"
            ),
        )

    clamped = max(0.0, min(1.0, float(novelty_score)))
    uncertainty = low_novelty_uncertainty + clamped * (
        high_novelty_uncertainty - low_novelty_uncertainty
    )
    return NoveltyAdjustedUncertainty(
        uncertainty=float(uncertainty),
        novelty_score=clamped,
        source=(
            f"novelty_scaled: novelty={clamped:.6f}, "
            f"low={low_novelty_uncertainty:.6g}, "
            f"high={high_novelty_uncertainty:.6g}"
        ),
    )
