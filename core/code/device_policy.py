"""Explicit ``cpu`` / ``gpu`` / ``auto`` device policy for SCULPTOR.

Only a small part of the workflow can use the GPU (the property heads); the
generation, screening, perturbation and analysis stages run on the CPU.  Three
modes keep that decision in the hands of the person reproducing the run:

``cpu``   never touch CUDA - the whole workflow runs on the CPU.  Use this on
          machines without a GPU, or to check that a CPU-only run reproduces.
``gpu``   use the GPU for the stages that support it.  If CUDA cannot actually
          be used the run stops with an explanatory error instead of silently
          falling back, so a GPU run is never mistaken for a CPU run.
``auto``  probe CUDA, run a short smoke test on the device and take the GPU
          path only when the probe succeeds; otherwise run the CPU path.

The mode can be set per process with :func:`set_device_mode` or through the
environment variable ``SCULPTOR_DEVICE_MODE``, which the CLI and the control
panel also use.

Resolution is done **once**: the ``auto`` smoke test is executed at the first
lookup, the result is cached for the rest of the process and written to
``SCULPTOR_RESOLVED_DEVICE`` so that child processes inherit the decision
instead of probing CUDA again.  A production run therefore probes the GPU at
start-up only, not once per property-prediction call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

MODES = ("auto", "cpu", "gpu")
DEFAULT_MODE = "auto"
ENV_VAR = "SCULPTOR_DEVICE_MODE"
RESOLVED_ENV_VAR = "SCULPTOR_RESOLVED_DEVICE"
# Diagnostic switch: set to 1 to probe CUDA on every lookup again.  Only used
# to measure how often the old behaviour probed; leave it unset in a run.
NO_CACHE_ENV_VAR = "SCULPTOR_DEVICE_NO_CACHE"

# mode/index/smoke -> report, and the keys already printed in this process
_REPORT_CACHE: dict[tuple[str, int, bool], "DeviceReport"] = {}
_ANNOUNCED: set[tuple[str, int, bool]] = set()


def reset_device_cache() -> None:
    """Forget everything resolved in this process (used by the tests)."""
    _REPORT_CACHE.clear()
    _ANNOUNCED.clear()


def _cache_disabled() -> bool:
    return (os.environ.get(NO_CACHE_ENV_VAR) or "").strip() not in ("", "0",
                                                                   "false")


def _remember(key: tuple[str, int, bool], report: "DeviceReport") -> None:
    """Cache the report and hand the decision to child processes."""
    _REPORT_CACHE[key] = report
    os.environ[RESOLVED_ENV_VAR] = report.device


def resolved_device() -> str | None:
    """The device already decided by this process or by a parent process."""
    value = (os.environ.get(RESOLVED_ENV_VAR) or "").strip()
    if value == "cpu" or value.startswith("cuda"):
        return value
    return None


def normalise_mode(mode: str | None) -> str:
    value = (mode or "").strip().lower()
    if value in MODES:
        return value
    return DEFAULT_MODE


def current_mode() -> str:
    return normalise_mode(os.environ.get(ENV_VAR))


def set_device_mode(mode: str | None) -> str:
    """Set the process-wide mode; child processes inherit it via the env var."""
    value = normalise_mode(mode)
    os.environ[ENV_VAR] = value
    return value


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _cuda_smoke_test(index: int = 0) -> tuple[bool, str]:
    """Try a tiny operation on the device.

    A GPU can be reported as available and still be unusable - older torch
    builds do not contain kernels for newer architectures, which otherwise
    only fails much later inside a model call.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency
        return False, f"torch unavailable: {exc}"
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    try:
        device = torch.device(f"cuda:{index}")
        a = torch.ones(256, 256, device=device)
        b = torch.ones(256, 256, device=device)
        value = float((a @ b).sum().item())
        torch.cuda.synchronize(device)
    except Exception as exc:  # pragma: no cover - hardware dependent
        return False, f"{type(exc).__name__}: {exc}"
    if value != float(256 * 256 * 256):
        return False, f"unexpected smoke-test value {value}"
    try:
        name = torch.cuda.get_device_name(index)
    except Exception:  # pragma: no cover - some drivers refuse the lookup
        name = f"cuda:{index}"
    return True, name


@dataclass(frozen=True)
class DeviceReport:
    mode: str
    device: str
    cuda_usable: bool
    smoke_test: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "device": self.device,
            "cuda_usable": self.cuda_usable,
            "smoke_test": self.smoke_test,
            "reason": self.reason,
        }


def describe_device(mode: str | None = None, *, index: int = 0,
                    smoke: bool = True) -> DeviceReport:
    """Resolve the mode into a device without raising.

    The probe runs at most once per mode per process; later calls return the
    cached report.  In ``auto`` mode a device resolved by a parent process is
    reused as-is, so a run that spawns child processes still probes CUDA only
    at start-up.
    """
    import torch

    mode = normalise_mode(mode or current_mode())
    key = (mode, index, smoke)
    cached = None if _cache_disabled() else _REPORT_CACHE.get(key)
    if cached is not None:
        return cached

    if mode == "auto" and smoke and not _cache_disabled():
        inherited = resolved_device()
        if inherited is not None:
            report = DeviceReport(
                mode, inherited, inherited != "cpu",
                "skipped (inherited)",
                f"auto mode: reusing {inherited} resolved at start-up",
            )
            _remember(key, report)
            return report

    if mode == "cpu":
        report = DeviceReport(mode, "cpu", False, "skipped",
                              "cpu mode: CUDA is never probed")
    else:
        usable, detail = _cuda_smoke_test(index) if smoke else (
            bool(torch.cuda.is_available()), "not probed")
        if usable:
            report = DeviceReport(mode, f"cuda:{index}", True,
                                  f"passed ({detail})",
                                  f"{mode} mode: GPU available")
        elif mode == "gpu":
            report = DeviceReport(mode, "cpu", False, f"failed ({detail})",
                                  "gpu mode requested but CUDA cannot be used")
        else:
            report = DeviceReport(mode, "cpu", False, f"failed ({detail})",
                                  "auto mode: falling back to CPU")
    if mode != "gpu" or report.cuda_usable:
        _remember(key, report)
    else:
        _REPORT_CACHE[key] = report
    return report


def announce_device(report: DeviceReport, *, verbose: bool = False) -> bool:
    """Print the resolution line once per process; returns True if printed."""
    key = (report.mode, 0, True)
    if not verbose and not _cache_disabled() and key in _ANNOUNCED:
        return False
    _ANNOUNCED.add(key)
    print(f"[device] mode={report.mode} -> {report.device} "
          f"({report.smoke_test})", flush=True)
    return True


def resolve_device(mode: str | None = None, *, index: int = 0,
                   smoke: bool = True, verbose: bool = False):
    """Return the torch device for the current (or given) mode.

    ``gpu`` never falls back: it raises so that a failed GPU run cannot be
    mistaken for a completed one.  The decision is cached, so the ``auto``
    smoke test runs once per process.
    """
    import torch

    report = describe_device(mode, index=index, smoke=smoke)
    announce_device(report, verbose=verbose)
    if report.mode == "gpu" and not report.cuda_usable:
        raise RuntimeError(
            "SCULPTOR device mode 'gpu' was requested but CUDA is not usable "
            f"({report.smoke_test}). Re-run with --device-mode cpu or "
            "--device-mode auto, or fix the CUDA installation."
        )
    return torch.device(report.device)


def device_summary(mode: str | None = None, *, index: int = 0) -> str:
    report = describe_device(mode, index=index)
    return (f"device mode {report.mode}: running on {report.device} "
            f"[{report.reason}; smoke test {report.smoke_test}]")


def device_summary_line(mode: str | None = None, *, index: int = 0) -> str:
    """One-line description that also states where the decision was made."""
    report = describe_device(mode, index=index)
    origin = ("inherited from the parent process"
              if report.smoke_test.startswith("skipped (inherited)")
              else "resolved in this process")
    return (f"[device] mode={report.mode} -> {report.device} "
            f"({report.reason}; {origin})")
