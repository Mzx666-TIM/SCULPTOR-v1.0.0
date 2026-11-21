"""Shared worker-pool helpers for the SCULPTOR pipelines.

Every parallel path in SCULPTOR is molecule-independent, so results are identical
to the serial path; the only effect of these helpers is to use more cores.
Worker counts default to an automatic, memory-aware value and can always be
forced back to 1 for small machines.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_MAX_WORKERS = 8


def _available_memory_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        pass
    try:
        if os.name == "nt":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullAvailPhys / (1024 ** 3)
        else:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            size = os.sysconf("SC_PAGE_SIZE")
            return pages * size / (1024 ** 3)
    except Exception:
        return None
    return None


def resolve_worker_count(
    requested: int | None,
    *,
    memory_per_worker_gb: float = 0.6,
    max_workers: int = DEFAULT_MAX_WORKERS,
    reserve_cores: int = 2,
) -> int:
    """Return the number of worker processes to use.

    ``requested`` semantics:

    * ``None`` or ``0`` -> automatic: as many workers as the machine can take,
      bounded by available RAM and by ``cpu_count - reserve_cores``.
    * ``1`` -> serial execution (exactly the historical behaviour).
    * ``n > 1`` -> at most ``n`` workers, never more than the CPU count.
    """

    cpu = os.cpu_count() or 1
    if requested is not None and requested == 1:
        return 1

    core_budget = max(1, min(cpu - reserve_cores, max_workers))
    memory = _available_memory_gb()
    memory_budget = core_budget
    if memory is not None and memory_per_worker_gb > 0:
        memory_budget = max(1, int(memory * 0.75 // memory_per_worker_gb))

    auto = max(1, min(core_budget, memory_budget))
    if requested is None or requested == 0:
        return auto
    return max(1, min(int(requested), cpu, auto))


def parallel_map(
    function: Callable[[T], R],
    items: Sequence[T],
    workers: int,
) -> list[R]:
    """Map ``function`` over ``items``, preserving input order.

    Falls back to a plain serial loop when only one worker is available, so the
    serial code path stays byte-identical to the historical implementation.
    """

    if workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, items, chunksize=max(1, len(items) // (workers * 8))))
