"""Channel definitions of the two-axis split (Tier 1-4, Potential 1-3)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROUTING_DIR = Path(__file__).resolve().parents[1] / "Generation" / "routing" / "code"
if str(ROUTING_DIR) not in sys.path:
    sys.path.insert(0, str(ROUTING_DIR))

from tier_routing import (  # noqa: E402
    POTENTIAL1,
    POTENTIAL2,
    POTENTIAL3,
    TIER1,
    TIER2,
    TIER3,
    TIER4,
    JP10Benchmark,
    apply_tier_fields,
    channel_for,
    hard_fail_count,
)

BENCHMARK = JP10Benchmark.from_values(
    density=0.962674,
    mass_heating_value=42.082709,
    melting_point=230.126,
    boiling_point=440.0,
    flash_point=330.0,
)


@pytest.mark.parametrize(
    "missed, beats, expected",
    [
        (0, 3, TIER1),
        (0, 2, TIER2),
        (0, 1, TIER3),
        (0, 0, TIER4),
        (1, 2, POTENTIAL1),
        (1, 1, POTENTIAL2),
        (2, 1, POTENTIAL3),
        (1, 0, ""),
        (2, 0, ""),
        (3, 0, ""),
    ],
)
def test_channel_profiles(missed: int, beats: int, expected: str) -> None:
    channel, reason = channel_for(beats, missed_count=missed)
    assert channel == expected
    assert reason


def _row(*, density: float, heating: float, melting: float) -> dict[str, object]:
    def decide(value: float, threshold: float, *, lower_is_better: bool) -> str:
        if lower_is_better:
            return "pass" if value <= threshold else "reject"
        return "pass" if value >= threshold else "reject"

    return {
        "external_density_g_cm3": density,
        "external_heating_value_mj_kg": heating,
        "external_melting_point_k": melting,
        "density_decision": decide(density, 0.85, lower_is_better=False),
        "heating_value_decision": decide(heating, 38.0, lower_is_better=False),
        "melting_point_decision": decide(melting, 273.15, lower_is_better=True),
    }


def test_potential3_is_one_beaten_with_two_thresholds_missed() -> None:
    """High density beats JP-10 while both other thresholds are missed."""
    row = _row(density=1.05, heating=36.0, melting=300.0)
    decision = apply_tier_fields(row, BENCHMARK)
    assert row["hard_fail_count"] == 2
    assert decision.tier == POTENTIAL3
    assert decision.beats_jp10 == ("density",)


def test_potential2_is_one_beaten_with_one_threshold_missed() -> None:
    row = _row(density=1.05, heating=36.0, melting=250.0)
    decision = apply_tier_fields(row, BENCHMARK)
    assert row["hard_fail_count"] == 1
    assert decision.tier == POTENTIAL2


def test_potential1_beats_two_and_misses_one() -> None:
    row = _row(density=1.05, heating=45.0, melting=300.0)
    decision = apply_tier_fields(row, BENCHMARK)
    assert row["hard_fail_count"] == 1
    assert decision.tier == POTENTIAL1
    assert decision.beats_jp10 == ("density", "mass_heating_value")


def test_tier1_beats_all_three() -> None:
    row = _row(density=1.05, heating=45.0, melting=200.0)
    decision = apply_tier_fields(row, BENCHMARK)
    assert row["hard_fail_count"] == 0
    assert decision.tier == TIER1


def test_unstylish_profile_gets_no_channel() -> None:
    """Below a threshold and better than JP-10 on nothing: not a candidate."""
    row = _row(density=0.80, heating=39.0, melting=250.0)
    decision = apply_tier_fields(row, BENCHMARK)
    assert row["hard_fail_count"] == 1
    assert decision.tier == ""
    assert row["potential_candidate"] is False
    assert "no salvage pool" in decision.reason


def test_hard_fail_count_reads_per_property_decisions() -> None:
    assert hard_fail_count({"density_decision": "pass", "heating_value_decision": "pass",
                            "melting_point_decision": "review"}) == 1
    assert hard_fail_count({}) is None
