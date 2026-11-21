from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ThresholdCalibrationRow:
    threshold: float
    e3nn_count: int
    passed_count: int
    e3nn_fraction: float
    mae: float
    quantile_error: float
    passes: bool


@dataclass(frozen=True)
class ThresholdCalibrationResult:
    recommended_threshold: float | None
    max_mae: float
    max_quantile_error: float
    quantile: float
    min_e3nn_fraction: float
    error_mode: str
    total_rows: int
    candidate_rows: tuple[ThresholdCalibrationRow, ...]


def calibrate_novelty_threshold(
    records: list[dict[str, str]],
    novelty_column: str = "fingerprint_novelty",
    actual_column: str | None = None,
    predicted_column: str | None = None,
    abs_error_column: str | None = None,
    max_mae: float = 0.05,
    max_quantile_error: float = 0.10,
    quantile: float = 0.90,
    min_e3nn_fraction: float = 0.20,
    error_mode: str = "relative",
    relative_epsilon: float = 1e-12,
) -> ThresholdCalibrationResult:
    """Pick the largest novelty threshold where e3nn validation error is acceptable."""

    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    if not 0.0 <= min_e3nn_fraction <= 1.0:
        raise ValueError("min_e3nn_fraction must be in [0, 1]")
    if error_mode not in {"relative", "absolute"}:
        raise ValueError("error_mode must be 'relative' or 'absolute'")

    points = sorted(
        _validation_points(
            records,
            novelty_column=novelty_column,
            actual_column=actual_column,
            predicted_column=predicted_column,
            abs_error_column=abs_error_column,
            error_mode=error_mode,
            relative_epsilon=relative_epsilon,
        ),
        key=lambda point: point[0],
    )
    if not points:
        raise ValueError("no usable calibration rows")

    thresholds = sorted({round(point[0], 6) for point in points})
    candidate_rows: list[ThresholdCalibrationRow] = []
    total = len(points)
    for threshold in thresholds:
        e3nn_errors = [error for novelty, error in points if novelty < threshold]
        if not e3nn_errors:
            continue
        mae = sum(e3nn_errors) / len(e3nn_errors)
        quantile_error = _quantile(e3nn_errors, quantile)
        e3nn_fraction = len(e3nn_errors) / total
        passes = (
            mae <= max_mae
            and quantile_error <= max_quantile_error
            and e3nn_fraction >= min_e3nn_fraction
        )
        candidate_rows.append(
            ThresholdCalibrationRow(
                threshold=round(threshold, 6),
                e3nn_count=len(e3nn_errors),
                passed_count=total - len(e3nn_errors),
                e3nn_fraction=round(e3nn_fraction, 6),
                mae=round(mae, 6),
                quantile_error=round(quantile_error, 6),
                passes=passes,
            )
        )

    passing = [row for row in candidate_rows if row.passes]
    recommended = max((row.threshold for row in passing), default=None)
    return ThresholdCalibrationResult(
        recommended_threshold=recommended,
        max_mae=max_mae,
        max_quantile_error=max_quantile_error,
        quantile=quantile,
        min_e3nn_fraction=min_e3nn_fraction,
        error_mode=error_mode,
        total_rows=total,
        candidate_rows=tuple(candidate_rows),
    )


def calibrate_novelty_threshold_from_csv(
    csv_path: str | Path,
    novelty_column: str = "fingerprint_novelty",
    actual_column: str | None = None,
    predicted_column: str | None = None,
    abs_error_column: str | None = None,
    max_mae: float = 0.05,
    max_quantile_error: float = 0.10,
    quantile: float = 0.90,
    min_e3nn_fraction: float = 0.20,
    error_mode: str = "relative",
    relative_epsilon: float = 1e-12,
) -> ThresholdCalibrationResult:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    return calibrate_novelty_threshold(
        records,
        novelty_column=novelty_column,
        actual_column=actual_column,
        predicted_column=predicted_column,
        abs_error_column=abs_error_column,
        max_mae=max_mae,
        max_quantile_error=max_quantile_error,
        quantile=quantile,
        min_e3nn_fraction=min_e3nn_fraction,
        error_mode=error_mode,
        relative_epsilon=relative_epsilon,
    )


def threshold_calibration_to_json(result: ThresholdCalibrationResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False)


def _validation_points(
    records: list[dict[str, str]],
    novelty_column: str,
    actual_column: str | None,
    predicted_column: str | None,
    abs_error_column: str | None,
    error_mode: str,
    relative_epsilon: float,
) -> list[tuple[float, float]]:
    if abs_error_column is None and (actual_column is None or predicted_column is None):
        raise ValueError(
            "provide either abs_error_column or both actual_column and predicted_column"
        )

    points: list[tuple[float, float]] = []
    for record in records:
        try:
            novelty = float(record[novelty_column])
            if abs_error_column is not None:
                error = abs(float(record[abs_error_column]))
            else:
                assert actual_column is not None
                assert predicted_column is not None
                actual = float(record[actual_column])
                predicted = float(record[predicted_column])
                error = _prediction_error(
                    actual,
                    predicted,
                    error_mode=error_mode,
                    relative_epsilon=relative_epsilon,
                )
        except (KeyError, TypeError, ValueError):
            continue
        points.append((novelty, error))
    return points


def _prediction_error(
    actual: float,
    predicted: float,
    error_mode: str,
    relative_epsilon: float,
) -> float:
    absolute_error = abs(actual - predicted)
    if error_mode == "absolute":
        return absolute_error
    denominator = max(abs(actual), relative_epsilon)
    return absolute_error / denominator


def _quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
