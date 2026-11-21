"""Mirror-symmetry scores of 3D conformers (path A / path B).

Both paths answer the same question in the same units: *how far is this
conformer from being mirror symmetric?*  The answer is a per-atom RMSD
(Angstrom) between the conformer and its mirrored copy, mapped to a score with
``1 / (1 + RMSD)`` - the same monotone map used by the other geometry
components of this project.

Path A - :func:`mirror_preservation_score`
    For molecules whose skeleton carries a non-trivial reflection: the atom
    pairing is the involutive graph automorphism used by the generator, so the
    score measures whether the optimised conformer really realises the
    topological mirror the generator imposed.  The best-fitting plane has a
    closed form (see :func:`best_mirror_plane`).

Path B - :func:`mirror_defect_score`
    For molecules without such an automorphism (asymmetric perturbation
    products).  No pairing is known, so the conformer is compared with its own
    mirror image over planes through its centroid and the smallest mismatch over
    all planes *and* all atom matchings is kept.  Also closed-form for a fixed
    matching (see :func:`_plane_normal_for_permutation`), which makes the
    alternating search (matching -> plane -> matching ...) converge quickly.

Both metrics read heavy-atom coordinates of the same conformer that feeds the
e3nn features, so no extra conformer generation is required when the caller
passes ``coords`` (the pipeline should pass the same ``seed`` it used for the
geometry features).

Notes / scope
    * Stereochemistry is not considered: the pipeline operates on
      constitution-level SMILES without stereo descriptors.
    * Explicit hydrogens in the input SMILES are rejected (the pipeline uses
      heavy-atom SMILES and adds hydrogens internally).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "MirrorSymmetryError",
    "NoMirrorAutomorphismError",
    "automorphisms",
    "decomposition_signature",
    "select_mirror",
    "involutions",
    "mirror_pairs",
    "heavy_coordinates",
    "best_mirror_plane",
    "paired_rmsd",
    "mirror_preservation_score",
    "mirror_defect_score",
    "mirror_score",
]

#: guard for the exhaustive automorphism enumeration (see :func:`automorphisms`)
AUTOMORPHISM_LIMIT = 200_000


class MirrorSymmetryError(RuntimeError):
    """Raised when the mirror metric cannot be evaluated reliably."""


class NoMirrorAutomorphismError(ValueError):
    """The molecule has no non-trivial reflection automorphism (use path B)."""


# --------------------------------------------------------------------------
# graph side: find the mirror (order-two) automorphism
# --------------------------------------------------------------------------
def _mol(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid SMILES: %s" % smiles)
    if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        raise MirrorSymmetryError(
            "explicit hydrogens are not supported, pass a heavy-atom SMILES: %s" % smiles
        )
    if mol.GetNumAtoms() == 0:
        raise ValueError("SMILES without heavy atoms: %s" % smiles)
    return mol


def automorphisms(smiles: str, max_matches: int = AUTOMORPHISM_LIMIT) -> list[tuple[int, ...]]:
    """All heavy-atom graph automorphisms of ``smiles`` (identity included).

    Each element maps ``atom index -> atom index``.  A saturated enumeration is
    reported instead of silently returning a truncated list.
    """

    mol = _mol(smiles)
    matches = mol.GetSubstructMatches(mol, uniquify=False, maxMatches=max_matches)
    if len(matches) >= max_matches:
        raise MirrorSymmetryError(
            "automorphism enumeration saturated for %s (%d matches); "
            "increase max_matches instead of trusting a truncated list" % (smiles, len(matches))
        )
    return [tuple(int(x) for x in m) for m in matches]


def _is_automorphism(smiles: str, sigma: tuple[int, ...]) -> bool:
    mol = _mol(smiles)
    n = mol.GetNumAtoms()
    if len(sigma) != n or sorted(sigma) != list(range(n)):
        return False
    bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}
    mapped = {tuple(sorted((sigma[u], sigma[v]))) for u, v in bonds}
    return mapped == bonds


def _half_graph_mirror(smiles: str, mol) -> tuple[int, ...] | None:
    """The generator's own reflection automorphism, or ``None`` if unavailable.

    ``carbon_graph_from_smiles`` renumbers atoms, so the half-graph pairing is
    only meaningful when every heavy atom is a carbon (always true for this
    project: saturated hydrocarbons).  The half-graph rules also refuse some
    fused cages (``cross-face dynamic edge``); those molecules fall back to the
    RDKit side.

    The *generation* decomposition (``find_generation_reflection_decomposition``,
    the axis the generator builds from and the source of the ``n1``/``n2``/
    ``n3`` partition the pipeline records) is preferred.  Note that a molecule
    can be built from more than one valid axis, so the canonical re-derivation
    of a SMILES need not reproduce the partition of the round that produced it;
    the looser decomposition is only a fallback, because it can return a
    formally valid but geometrically trivial axis (one swapped pair plus many
    fixed atoms).
    """

    if any(atom.GetAtomicNum() != 6 for atom in mol.GetAtoms()):
        return None
    try:
        from chemistry import carbon_graph_from_smiles
        from symmetry import (
            UnsupportedSymmetryError,
            find_generation_reflection_decomposition,
            find_reflection_decomposition,
        )

        num_nodes, edges = carbon_graph_from_smiles(smiles)
        if num_nodes != mol.GetNumAtoms():
            return None
    except (UnsupportedSymmetryError, ValueError, ImportError, AttributeError):
        return None
    permutation = None
    for finder in (find_generation_reflection_decomposition, find_reflection_decomposition):
        try:
            permutation = tuple(int(x) for x in finder(num_nodes, edges).permutation)
            break
        except (UnsupportedSymmetryError, ValueError):
            continue
    if permutation is None:
        return None
    if len(permutation) != mol.GetNumAtoms():
        return None
    return permutation


def decomposition_signature(smiles: str, sigma) -> tuple[int, int, int]:
    """``(n1, n2, n3)`` of a pairing, using the generator's own conventions.

    ``n1`` are the fixed atoms, ``n2`` the atoms of mirror pairs that are bonded
    to each other and ``n3`` the atoms of mirror pairs that are not (both are
    even, as in the manuscript).  Useful to cross-check a pairing against the
    ``n1``/``n2``/``n3`` columns the pipeline records.
    """

    mol = _mol(smiles)
    bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}
    pairs, singles = mirror_pairs(tuple(sigma))
    n2 = 2 * sum(1 for i, j in pairs if (i, j) in bonds)
    n3 = 2 * sum(1 for i, j in pairs if (i, j) not in bonds)
    return len(singles), n2, n3


def _rank_key(sigma: tuple[int, ...], mol) -> tuple[int, int, int]:
    """Ordering used by :func:`select_mirror` (larger is better)."""

    bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}
    fixed = sum(1 for i, j in enumerate(sigma) if i == j)
    pairs = sum(1 for i, j in enumerate(sigma) if i < j)
    cross = sum(1 for i, j in enumerate(sigma) if i < j and (i, j) in bonds)
    return fixed, cross, pairs


def select_mirror(smiles: str, max_matches: int = AUTOMORPHISM_LIMIT) -> tuple[int, ...] | None:
    """Pick the involutive, non-identity automorphism used as the mirror.

    The half-graph decomposition of the generator is preferred, so the pairing
    is identical to the one the generator built; when the half-graph rules
    refuse the molecule the same ordering is applied to the automorphisms found
    by RDKit: most fixed atoms, then most bonded (``n2``) pairs, then most
    pairs, with a lexicographic tie-break so the result never depends on
    enumeration order.

    Returns ``None`` when the molecule has no non-trivial involution.
    """

    mol = _mol(smiles)
    from_half_graph = _half_graph_mirror(smiles, mol)
    if from_half_graph is not None and _is_automorphism(smiles, from_half_graph):
        return from_half_graph

    n = mol.GetNumAtoms()
    best_key: tuple[int, int, int] | None = None
    best_sigma: tuple[int, ...] | None = None
    for sigma in automorphisms(smiles, max_matches=max_matches):
        if all(sigma[i] == i for i in range(n)):
            continue
        if any(sigma[sigma[i]] != i for i in range(n)):  # not an involution
            continue
        key = _rank_key(sigma, mol)
        if best_key is None or key > best_key or (key == best_key and sigma < best_sigma):
            best_key, best_sigma = key, sigma
    if best_sigma is not None and not _is_automorphism(smiles, best_sigma):
        raise MirrorSymmetryError("selected mapping is not a graph automorphism: %s" % smiles)
    return best_sigma


def mirror_pairs(sigma: tuple[int, ...]) -> tuple[list[tuple[int, int]], list[int]]:
    """Split an involution into mirror pairs and fixed (n1) atoms."""

    pairs = [(i, sigma[i]) for i in range(len(sigma)) if i < sigma[i]]
    singles = [i for i in range(len(sigma)) if sigma[i] == i]
    return pairs, singles


def involutions(smiles: str, max_matches: int = AUTOMORPHISM_LIMIT) -> list[tuple[int, ...]]:
    """Every non-identity involutive automorphism, in a deterministic order.

    A molecule can admit several mirror axes; :func:`select_mirror` returns the
    canonical one (the generator's), this function exposes all of them so a
    caller can also ask "how well does this conformer realise *any* of its
    mirror axes".
    """

    mol = _mol(smiles)
    n = mol.GetNumAtoms()
    found = []
    for sigma in automorphisms(smiles, max_matches=max_matches):
        if all(sigma[i] == i for i in range(n)):
            continue
        if any(sigma[sigma[i]] != i for i in range(n)):
            continue
        found.append(sigma)
    return sorted(found)


# --------------------------------------------------------------------------
# geometry side
# --------------------------------------------------------------------------
def _check_coordinates(smiles: str, coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            "coordinates must have shape (n_heavy_atoms, 3), got %s" % (coordinates.shape,)
        )
    n_expected = _mol(smiles).GetNumAtoms()
    if coordinates.shape[0] != n_expected:
        raise ValueError(
            "coordinates do not match %s: %d rows for %d heavy atoms"
            % (smiles, coordinates.shape[0], n_expected)
        )
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates contain non-finite values: %s" % smiles)
    return coordinates


def heavy_coordinates(smiles: str, seed: int = 0) -> np.ndarray:
    """Heavy-atom coordinates of the conformer used for the e3nn features.

    ``geometry_from_smiles`` appends hydrogens, so the heavy atoms are the
    leading rows of ``positions``; the element sequence is verified against the
    SMILES so this assumption can never break silently.
    """

    from property_model.data import geometry_from_smiles

    mol = _mol(smiles)
    geometry = geometry_from_smiles(smiles, seed=seed)
    positions = np.asarray(geometry.positions, dtype=float)
    atomic_numbers = np.asarray(geometry.atomic_numbers, dtype=int)
    n_heavy = mol.GetNumAtoms()
    expected = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=int)
    if positions.shape[0] != atomic_numbers.shape[0]:
        raise MirrorSymmetryError("geometry record is corrupt for %s" % smiles)
    if atomic_numbers.shape[0] < n_heavy or not np.array_equal(atomic_numbers[:n_heavy], expected):
        raise MirrorSymmetryError(
            "heavy-atom order of the conformer does not match the SMILES graph for %s" % smiles
        )
    return _check_coordinates(smiles, positions[:n_heavy])


def _plane_normal_for_permutation(centered: np.ndarray, sigma) -> np.ndarray:
    """Closed-form best mirror plane normal for a fixed atom matching.

    With ``x_i`` the centred coordinates, ``sigma`` a permutation and ``r_i``
    the residual of atom ``i`` against the mirror of its partner,

        T(n, d) = sum_i |r_i|^2
                = const - 4 n'Bn + 4d D'n + 4(n'Gn - 2d C'n + N d^2)

    where ``B = sum_i (x_sigma(i) - x_i) x_sigma(i)'``, ``G = X'X``,
    ``C = sum_i x_i`` and ``D = sum_i (x_sigma(i) - x_i) = 0`` because sigma is
    a permutation.  Minimising over ``d`` gives ``d = (C/N)'n`` (the optimal
    plane passes through the centroid; with centred coordinates ``d = 0``), and
    substituting leaves ``T(n) = const + 4 n'Hn`` with

        H = G - B - N k k',      k = C / N

    so the best normal is the eigenvector of ``H`` with the smallest
    eigenvalue.  For an involution this reduces to the ``8M'M + 4S'S - 2U'U``
    form used by the paired construction, which is checked in the test suite.
    """

    permuted = centered[np.asarray(sigma, dtype=int)]
    delta = permuted - centered
    gram = centered.T @ centered
    cross = delta.T @ permuted
    cross = 0.5 * (cross + cross.T)  # only the symmetric part enters n'Hn
    normal = np.linalg.eigh(gram - cross)[1][:, 0]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:  # fully degenerate geometry: deterministic axis
        return np.array([1.0, 0.0, 0.0])
    normal = normal / norm
    if normal[int(np.argmax(np.abs(normal)))] < 0:
        normal = -normal
    return normal


def best_mirror_plane(
    coordinates: np.ndarray,
    sigma,
) -> tuple[np.ndarray, float]:
    """Closed-form mirror plane minimising the RMSD for the pairing ``sigma``.

    Returns ``(normal, offset)`` for the plane ``x @ normal = offset``.  The
    plane always passes through the centroid of the heavy atoms, so the offset
    follows from the normal in closed form.  The normal is normalised and its
    sign is fixed deterministically (largest component positive).
    """

    coordinates = np.asarray(coordinates, dtype=float)
    centroid = coordinates.mean(axis=0)
    normal = _plane_normal_for_permutation(coordinates - centroid, sigma)
    return normal, float(centroid @ normal)


def paired_rmsd(
    coordinates: np.ndarray,
    sigma,
    normal: np.ndarray,
    offset: float,
) -> float:
    """Per-atom RMSD (Angstrom) between the conformer and its paired mirror.

    Every heavy atom is reflected about the plane and compared with the atom it
    is paired with (itself for n1 atoms), so the value is the usual "RMSD per
    atom" and is directly comparable with path B.
    """

    coordinates = np.asarray(coordinates, dtype=float)
    normal = np.asarray(normal, dtype=float)
    projected = coordinates @ normal - float(offset)
    reflected = coordinates - 2.0 * np.outer(projected, normal)
    permutation = np.asarray(sigma, dtype=int)
    difference = reflected[permutation] - coordinates
    return float(np.sqrt(np.mean(np.einsum("ij,ij->i", difference, difference))))


def mirror_preservation_score(
    smiles: str,
    seed: int = 0,
    coords: np.ndarray | None = None,
    sigma: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Path A.

    Raises :class:`NoMirrorAutomorphismError` when the molecule has no mirror
    automorphism (the caller should then use path B).
    """

    mapping = sigma if sigma is not None else select_mirror(smiles)
    if mapping is None:
        raise NoMirrorAutomorphismError("no non-trivial reflection automorphism found")
    if not _is_automorphism(smiles, mapping):
        raise MirrorSymmetryError("supplied mapping is not a graph automorphism: %s" % smiles)
    coordinates = (
        heavy_coordinates(smiles, seed=seed)
        if coords is None
        else _check_coordinates(smiles, coords)
    )
    pairs, singles = mirror_pairs(mapping)
    normal, offset = best_mirror_plane(coordinates, mapping)
    rmsd = paired_rmsd(coordinates, mapping, normal, offset)
    return {
        "score": float(1.0 / (1.0 + rmsd)),
        "rmsd": rmsd,
        "mode": "A",
        "n_pairs": len(pairs),
        "n1": len(singles),
        "n_atoms": int(coordinates.shape[0]),
        "plane_normal": [round(float(v), 6) for v in normal],
        "plane_offset": round(float(offset), 6),
    }


def mirror_defect_score(
    smiles: str,
    seed: int = 0,
    coords: np.ndarray | None = None,
    n_directions: int = 4096,
    n_exact: int = 16,
) -> dict[str, object]:
    """Path B: distance to the closest mirror plane over all atom matchings.

    A mirror plane of a point set must pass through its centroid, so only the
    normal is searched.  The scan is a dense Fibonacci sphere ranked with a
    cheap nearest-neighbour bound; the best candidates are then polished by the
    exact alternating scheme matching -> closed-form plane -> matching, and the
    best polished plane is kept.  The inertia frame is deliberately **not**
    used: for the high-symmetry cages that dominate this project the inertia
    tensor is isotropic and its axes carry no symmetry information.
    """

    coordinates = (
        heavy_coordinates(smiles, seed=seed)
        if coords is None
        else _check_coordinates(smiles, coords)
    )
    n_atoms = int(coordinates.shape[0])
    if n_atoms <= 3:
        # Any point set with at most three points is planar, hence mirror
        # symmetric: the defect is exactly zero by construction.
        return {
            "score": 1.0,
            "rmsd": 0.0,
            "mode": "B",
            "n_pairs": 0,
            "n1": 0,
            "n_atoms": n_atoms,
            "normal": None,
        }
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    normal, rmsd = _search_mirror_normal(centered, n_directions=n_directions, n_exact=n_exact)
    normal = np.asarray(normal, dtype=float)
    if normal[int(np.argmax(np.abs(normal)))] < 0:
        normal = -normal
    return {
        "score": float(1.0 / (1.0 + rmsd)),
        "rmsd": float(rmsd),
        "mode": "B",
        "n_pairs": 0,
        "n1": 0,
        "n_atoms": n_atoms,
        "normal": [round(float(v), 6) for v in normal],
    }


def _fibonacci_normals(n_directions: int) -> np.ndarray:
    golden = (1.0 + 5.0 ** 0.5) / 2.0
    k = np.arange(n_directions, dtype=float)
    z = 1.0 - 2.0 * (k + 0.5) / n_directions
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = 2.0 * np.pi * k / golden
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])


def _squared_distance_matrix(centered: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Squared distances between the point cloud and its reflection in a plane."""

    reflected = centered - 2.0 * np.outer(centered @ normal, normal)
    difference = centered[:, None, :] - reflected[None, :, :]
    return np.einsum("ijk,ijk->ij", difference, difference)


def _optimal_permutation(centered: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Best atom matching for one plane, as an array ``i -> sigma[i]``."""

    from scipy.optimize import linear_sum_assignment

    squared = _squared_distance_matrix(centered, normal)
    rows, cols = linear_sum_assignment(squared)
    sigma = np.empty(centered.shape[0], dtype=int)
    sigma[rows] = cols
    return sigma


def _defect_for_normal(centered: np.ndarray, normal: np.ndarray) -> float:
    """Exact smallest mirror RMSD for one plane (optimal atom matching)."""

    from scipy.optimize import linear_sum_assignment

    squared = _squared_distance_matrix(centered, normal)
    rows, cols = linear_sum_assignment(squared)
    return float(np.sqrt(np.mean(squared[rows, cols])))


def _nearest_neighbour_rmsd(
    centered: np.ndarray,
    gram: np.ndarray,
    radii: np.ndarray,
    normals: np.ndarray,
):
    """Vectorised upper bound of the mirror RMSD for many planes at once.

    Uses ``|x_i - R x_j|^2 = |x_i|^2 + |x_j|^2 - 2 x_i . R x_j`` with
    ``x_i . R x_j = G_ij - 2 (x_i . n)(x_j . n)``, so a whole block of plane
    normals is scored without a Python loop or an assignment problem.  Each
    value is an upper bound of the exact optimum for that plane (it comes from a
    valid, if greedy, matching), which is all the candidate ranking needs.
    """

    projection = normals @ centered.T  # (K, N)
    base = radii[:, None] + radii[None, :] - 2.0 * gram  # (N, N)
    for start in range(0, normals.shape[0], 512):
        block = projection[start : start + 512]
        squared = base[None, :, :] + 4.0 * np.einsum("ki,kj->kij", block, block)
        yield np.sqrt(np.maximum(squared.min(axis=2), 0.0).mean(axis=1))


def _polish_plane(
    centered: np.ndarray,
    normal: np.ndarray,
    max_iterations: int = 24,
) -> tuple[np.ndarray, float]:
    """Alternate optimal matching and closed-form plane until it stops improving.

    Each step cannot increase the RMSD (the matching is optimal for the current
    plane, and the next plane is optimal for that matching), so the iteration is
    monotonically decreasing and terminates at a fixed point.
    """

    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) or 1.0)
    best_normal, best_rmsd = normal, _defect_for_normal(centered, normal)
    for _ in range(max(int(max_iterations), 1)):
        sigma = _optimal_permutation(centered, best_normal)
        candidate = _plane_normal_for_permutation(centered, sigma)
        rmsd = _defect_for_normal(centered, candidate)
        if rmsd < best_rmsd - 1e-12:
            best_normal, best_rmsd = candidate, rmsd
        else:
            break
    return best_normal, float(best_rmsd)


def _search_mirror_normal(
    centered: np.ndarray,
    n_directions: int = 4096,
    n_exact: int = 16,
) -> tuple[np.ndarray, float]:
    """Dense scan of plane normals, exact re-ranking, then exact polish."""

    normals = _fibonacci_normals(max(int(n_directions), 16))
    gram = centered @ centered.T
    radii = np.einsum("ij,ij->i", centered, centered)
    proxies = np.concatenate(list(_nearest_neighbour_rmsd(centered, gram, radii, normals)))
    order = np.argsort(proxies, kind="stable")[: max(int(n_exact), 1)]
    # always keep a few spread-out starts so a bad ranking cannot trap the search
    spread = range(0, len(normals), max(len(normals) // 8, 1))
    candidates = [normals[i] for i in order] + [normals[i] for i in spread]

    best_normal: np.ndarray | None = None
    best_rmsd = float("inf")
    for candidate in candidates:
        normal, rmsd = _polish_plane(centered, candidate)
        if rmsd < best_rmsd:
            best_normal, best_rmsd = normal, rmsd
    if best_normal is None:
        raise MirrorSymmetryError("mirror plane search produced no candidate")
    return best_normal, float(best_rmsd)


def mirror_score(
    smiles: str,
    seed: int = 0,
    coords: np.ndarray | None = None,
) -> dict[str, object]:
    """Path A when a mirror automorphism exists, path B otherwise.

    Only :class:`NoMirrorAutomorphismError` triggers the fallback; every other
    failure (unparsable SMILES, failed embedding, saturated enumeration) is
    reported to the caller instead of being silently rerouted.
    """

    try:
        return mirror_preservation_score(smiles, seed=seed, coords=coords)
    except NoMirrorAutomorphismError as exc:
        result = mirror_defect_score(smiles, seed=seed, coords=coords)
        result["note"] = str(exc)[:70]
        return result
