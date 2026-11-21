from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class GenerationSchedule:
    round_index: int
    temperature: float
    novelty_threshold: float
    base_diversity: float
    target_novelty_rate: float
    min_valid_rate: float
    mode: str

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def generation_schedule(round_index: int) -> GenerationSchedule:
    """Default curriculum schedule for molecule generation rounds."""

    if round_index < 1:
        raise ValueError("round_index must start at 1")

    if round_index <= 20:
        return GenerationSchedule(
            round_index=round_index,
            temperature=0.70,
            novelty_threshold=0.30,
            base_diversity=0.20,
            target_novelty_rate=0.05,
            min_valid_rate=0.80,
            mode="warmup",
        )
    if round_index <= 50:
        progress = (round_index - 21) / 29
        return GenerationSchedule(
            round_index=round_index,
            temperature=_lerp(0.80, 0.95, progress),
            novelty_threshold=_lerp(0.35, 0.45, progress),
            base_diversity=_lerp(0.30, 0.45, progress),
            target_novelty_rate=0.08,
            min_valid_rate=0.70,
            mode="ramp",
        )
    if round_index <= 80:
        progress = (round_index - 51) / 29
        return GenerationSchedule(
            round_index=round_index,
            temperature=_lerp(1.00, 1.15, progress),
            novelty_threshold=_lerp(0.48, 0.55, progress),
            base_diversity=_lerp(0.50, 0.65, progress),
            target_novelty_rate=0.12,
            min_valid_rate=0.60,
            mode="explore",
        )

    progress = min((round_index - 81) / 39, 1.0)
    return GenerationSchedule(
        round_index=round_index,
        temperature=_lerp(1.20, 1.35, progress),
        novelty_threshold=_lerp(0.58, 0.65, progress),
        base_diversity=_lerp(0.70, 0.85, progress),
        target_novelty_rate=0.15,
        min_valid_rate=0.50,
        mode="high_explore",
    )


def adjust_schedule_with_feedback(
    schedule: GenerationSchedule,
    novelty_rate: float,
    valid_rate: float,
) -> GenerationSchedule:
    """Adjust a scheduled round using observed novelty and validity rates."""

    temperature = schedule.temperature
    novelty_threshold = schedule.novelty_threshold
    base_diversity = schedule.base_diversity
    mode = schedule.mode

    if valid_rate < schedule.min_valid_rate:
        temperature -= 0.08
        base_diversity -= 0.08
        mode = f"{mode}_validity_guard"
    elif novelty_rate < schedule.target_novelty_rate and schedule.round_index > 20:
        temperature += 0.08
        novelty_threshold += 0.03
        base_diversity += 0.08
        mode = f"{mode}_novelty_boost"

    return GenerationSchedule(
        round_index=schedule.round_index,
        temperature=round(_clamp(temperature, 0.50, 1.60), 4),
        novelty_threshold=round(_clamp(novelty_threshold, 0.20, 0.80), 4),
        base_diversity=round(_clamp(base_diversity, 0.10, 1.00), 4),
        target_novelty_rate=schedule.target_novelty_rate,
        min_valid_rate=schedule.min_valid_rate,
        mode=mode,
    )


def schedule_to_json(schedule: GenerationSchedule) -> str:
    return json.dumps(schedule.as_dict(), ensure_ascii=False)


def _lerp(start: float, end: float, progress: float) -> float:
    progress = _clamp(progress, 0.0, 1.0)
    return round(start + (end - start) * progress, 4)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
