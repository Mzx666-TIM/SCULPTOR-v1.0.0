from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import pickle


# Fingerprint configuration (upgraded 2026-08-06):
# radius 2 -> 4 and 2048 -> 8192 bits so that ring topology beyond the local
# 2-bond environment is visible and same-set fingerprint collisions among
# saturated hydrocarbon skeletons are eliminated (verified: the 6 observed
# max-similarity == 1.0 collisions all drop below 1.0 at radius 4 / 8192).
MORGAN_RADIUS = 4
MORGAN_FP_SIZE = 8192
MORGAN_FP_LABEL = f"morgan_r{MORGAN_RADIUS}_{MORGAN_FP_SIZE}"


@dataclass(frozen=True)
class FingerprintNoveltyResult:
    max_similarity: float
    novelty_score: float
    nearest_smiles: str | None


@dataclass(frozen=True)
class KnownFingerprintLibrary:
    smiles: tuple[str, ...]
    fingerprints: tuple[object, ...]


@lru_cache(maxsize=8)
def load_known_fingerprint_library(path: str | Path | None) -> KnownFingerprintLibrary:
    """Load canonical SMILES and precompute fingerprints for novelty scoring."""

    if path is None:
        return KnownFingerprintLibrary(smiles=(), fingerprints=())

    known_path = Path(path)
    if not known_path.exists():
        raise FileNotFoundError(f"known SMILES file not found: {known_path}")
    cache_path = known_path.with_suffix(known_path.suffix + f".{MORGAN_FP_LABEL}.pkl")
    source_stat = known_path.stat()
    if cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if (
                cached.get("source_size") == source_stat.st_size
                and cached.get("source_mtime_ns") == source_stat.st_mtime_ns
            ):
                return KnownFingerprintLibrary(
                    smiles=tuple(cached["smiles"]),
                    fingerprints=tuple(cached["fingerprints"]),
                )
        except (OSError, EOFError, AttributeError, KeyError, pickle.PickleError):
            pass

    smiles: list[str] = []
    fingerprints: list[object] = []
    for line in known_path.read_text(encoding="utf-8").splitlines():
        canonical = line.strip()
        if not canonical:
            continue
        try:
            fingerprints.append(_fingerprint_from_smiles(canonical))
        except ValueError:
            continue
        smiles.append(canonical)

    library = KnownFingerprintLibrary(
        smiles=tuple(smiles),
        fingerprints=tuple(fingerprints),
    )
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(
                {
                    "source_size": source_stat.st_size,
                    "source_mtime_ns": source_stat.st_mtime_ns,
                    "smiles": library.smiles,
                    "fingerprints": library.fingerprints,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        temporary.replace(cache_path)
    except OSError:
        pass
    return library


def fingerprint_novelty(
    smiles: str,
    library: KnownFingerprintLibrary,
) -> FingerprintNoveltyResult | None:
    """Return `1 - max Tanimoto similarity` against the known library."""

    if not library.fingerprints:
        return None

    DataStructs = _require_data_structs()
    fingerprint = _fingerprint_from_smiles(smiles)
    similarities = DataStructs.BulkTanimotoSimilarity(
        fingerprint,
        library.fingerprints,
    )
    if not similarities:
        return None

    max_similarity = max(float(value) for value in similarities)
    nearest_index = similarities.index(max_similarity)
    return FingerprintNoveltyResult(
        max_similarity=round(max_similarity, 6),
        novelty_score=round(1.0 - max_similarity, 6),
        nearest_smiles=library.smiles[nearest_index],
    )


def _fingerprint_from_smiles(smiles: str):
    Chem = _require_chem()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return _morgan_generator().GetFingerprint(mol)


def _require_chem():
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for fingerprint novelty scoring."
        ) from exc
    return Chem


@lru_cache(maxsize=1)
def _morgan_generator():
    try:
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for fingerprint novelty scoring."
        ) from exc
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_FP_SIZE,
    )


def _require_data_structs():
    try:
        from rdkit import DataStructs
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for fingerprint novelty scoring."
        ) from exc
    return DataStructs
