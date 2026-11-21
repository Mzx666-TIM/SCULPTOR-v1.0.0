"""On-disk cache for per-molecule property predictions.

The prediction of one molecule depends only on its canonical SMILES, on the
selected models and on the prediction seed. Cache entries are therefore keyed
by a fingerprint of that configuration, so a retrained model or a different
seed automatically invalidates the file instead of silently reusing stale
numbers.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

PROPERTY_CACHE_SCHEMA = 3


def _file_signature(value: Any) -> str | None:
    try:
        path = Path(str(value))
    except (TypeError, ValueError, OSError):
        return None
    try:
        if not path.is_file():
            return None
        stat = path.stat()
    except OSError:
        return None
    return "%s|%d|%d" % (path.resolve(), stat.st_size, stat.st_mtime_ns)


def config_fingerprint(kwargs: Mapping[str, Any], *, seed: int) -> str:
    """Stable hash of every setting that can change a prediction."""
    payload: dict[str, Any] = {}
    for key in sorted(kwargs):
        value = kwargs[key]
        if value is None and key.startswith("phase_fusion_") and key.endswith("_model_path"):
            from phase_fusion import DEFAULT_PHASE_FUSION_MODEL_PATHS
            target = key[len("phase_fusion_"):-len("_model_path")] + "_k"
            value = DEFAULT_PHASE_FUSION_MODEL_PATHS.get(target)
        if isinstance(value, (list, tuple)):
            payload[key] = [_file_signature(item) or str(item) for item in value]
            continue
        if isinstance(value, Mapping):
            payload[key] = dict(value)
            continue
        payload[key] = _file_signature(value) or str(value)
    payload["__seed_base__"] = int(seed)
    payload["__schema__"] = PROPERTY_CACHE_SCHEMA
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class PropertyCache:
    """Loaded disk cache; ``entries`` maps ``SMILES|seed`` to a prediction delta.

    The per-molecule prediction seed is part of the key because the 3D geometry
    is derived from it: the same molecule scored at a different position in a
    different stage is a genuinely different calculation and must not reuse an
    entry computed with another seed.
    """

    def __init__(self, path: str | Path | None, fingerprint: str) -> None:
        self.path = Path(path) if path else None
        self.fingerprint = fingerprint
        self.entries: dict[str, dict[str, Any]] = {}
        self.dirty = 0
        self._loaded_from: str = "empty"
        if self.path is not None and self.path.exists():
            try:
                payload = pickle.loads(self.path.read_bytes())
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("fingerprint") == fingerprint:
                entries = payload.get("entries")
                if isinstance(entries, dict):
                    self.entries = dict(entries)
                    self._loaded_from = "hit"
            elif payload is not None:
                self._loaded_from = "stale"

    @staticmethod
    def key(smiles: str, seed: int) -> str:
        return "%s|%d" % (smiles, int(seed))

    def get(self, smiles: str, seed: int) -> dict[str, Any] | None:
        return self.entries.get(self.key(smiles, seed))

    def put(self, smiles: str, seed: int, delta: dict[str, Any]) -> None:
        if not smiles:
            return
        self.entries[self.key(smiles, seed)] = delta
        self.dirty += 1

    def save(self, *, force: bool = False) -> None:
        if self.path is None or (not force and not self.dirty):
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": PROPERTY_CACHE_SCHEMA,
                "fingerprint": self.fingerprint,
                "entries": self.entries,
            }
            self.path.write_bytes(pickle.dumps(payload, protocol=4))
            self.dirty = 0
        except OSError:
            pass

    def describe(self) -> str:
        return "%s entries from %s cache" % (len(self.entries), self._loaded_from)
