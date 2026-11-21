"""The 'auto' device decision must be taken once, not once per lookup."""

from __future__ import annotations

import os

import pytest

import fuel_design  # noqa: F401  (adds the pipeline code folders to sys.path)

import device_policy
from device_policy import (
    RESOLVED_ENV_VAR, announce_device, describe_device, reset_device_cache,
    resolve_device,
)


@pytest.fixture(autouse=True)
def clean_device_state(monkeypatch):
    reset_device_cache()
    monkeypatch.delenv(RESOLVED_ENV_VAR, raising=False)
    monkeypatch.delenv(device_policy.NO_CACHE_ENV_VAR, raising=False)
    yield
    reset_device_cache()


def test_auto_probes_only_once_per_process(monkeypatch):
    calls = {"n": 0}

    def fake_smoke(index: int = 0):
        calls["n"] += 1
        return False, "stub"

    monkeypatch.setattr(device_policy, "_cuda_smoke_test", fake_smoke)

    reports = [describe_device("auto") for _ in range(5)]

    assert calls["n"] == 1, "the CUDA smoke test ran more than once"
    assert {report.device for report in reports} == {"cpu"}


def test_resolution_is_exported_for_child_processes(monkeypatch):
    monkeypatch.setattr(device_policy, "_cuda_smoke_test",
                        lambda index=0: (False, "stub"))
    describe_device("auto")
    assert os.environ.get(RESOLVED_ENV_VAR) == "cpu"

    # a fresh process (cache cleared) takes the parent decision without probing
    reset_device_cache()
    calls = {"n": 0}

    def counting_smoke(index: int = 0):
        calls["n"] += 1
        return False, "stub"

    monkeypatch.setattr(device_policy, "_cuda_smoke_test", counting_smoke)
    report = describe_device("auto")
    assert report.device == "cpu"
    assert report.smoke_test.startswith("skipped")
    assert calls["n"] == 0, "the inherited decision was probed again"


def test_device_line_is_printed_once(monkeypatch):
    monkeypatch.setattr(device_policy, "_cuda_smoke_test",
                        lambda index=0: (False, "stub"))
    printed = [announce_device(describe_device("auto")) for _ in range(4)]
    assert printed == [True, False, False, False]


def test_gpu_mode_keeps_failing_loudly(monkeypatch):
    monkeypatch.setattr(device_policy, "_cuda_smoke_test",
                        lambda index=0: (False, "stub"))
    for _ in range(3):
        with pytest.raises(RuntimeError):
            resolve_device("gpu")


def test_no_cache_switch_restores_the_old_behaviour(monkeypatch):
    calls = {"n": 0}

    def counting_smoke(index: int = 0):
        calls["n"] += 1
        return False, "stub"

    monkeypatch.setattr(device_policy, "_cuda_smoke_test", counting_smoke)
    monkeypatch.setenv(device_policy.NO_CACHE_ENV_VAR, "1")
    for _ in range(3):
        describe_device("auto")
    assert calls["n"] == 3
