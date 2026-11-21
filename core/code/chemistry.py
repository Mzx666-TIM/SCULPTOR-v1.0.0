from __future__ import annotations

import re
from dataclasses import dataclass

from graph import Edge, HalfGraph
from symmetry import (
    find_generation_reflection_decomposition,
    find_reflection_decomposition,
)


def _require_rdkit():
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for SMILES preprocessing. Install the optional "
            "chemistry dependency before calling this function."
        ) from exc
    return Chem


_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
_HYDROCARBON_FORMULA = re.compile(r"C(\d*)H(\d*)")
RING_COUNT_DEFINITION = "formula_cycle_rank_full_molecule_v1"
THREE_MEMBER_RING_STRAIN_WEIGHT = 6
FOUR_MEMBER_RING_STRAIN_WEIGHT = 3
FIVE_MEMBER_RING_PREFERENCE_WEIGHT = 2
SIX_MEMBER_RING_PREFERENCE_WEIGHT = 2


def degree_of_unsaturation_from_hydrocarbon_formula(formula: str) -> int:
    """Return the degree of unsaturation of a neutral C/H formula.

    For a connected saturated hydrocarbon every degree of unsaturation is an
    independent ring closure, so this value is also the graph cycle rank.  It
    must not be interpreted as a ring count for a structure containing
    multiple bonds, charges, radicals, heteroatoms, or disconnected fragments.
    """

    normalized = str(formula).strip()
    match = _HYDROCARBON_FORMULA.fullmatch(normalized)
    if match is None:
        raise ValueError(f"not a hydrocarbon formula: {formula}")
    carbon_count = int(match.group(1) or "1")
    hydrogen_count = int(match.group(2) or "1")
    unsaturation_twice = 2 * carbon_count + 2 - hydrogen_count
    if unsaturation_twice < 0 or unsaturation_twice % 2:
        raise ValueError(
            f"formula is inconsistent with a neutral closed-shell hydrocarbon: {formula}"
        )
    return unsaturation_twice // 2


def _is_connected_saturated_hydrocarbon(mol) -> bool:
    Chem = _require_rdkit()
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    for atom in mol.GetAtoms():
        if (
            atom.GetAtomicNum() != 6
            or atom.GetIsAromatic()
            or atom.GetFormalCharge() != 0
            or atom.GetNumRadicalElectrons() != 0
        ):
            return False
    return all(
        bond.GetBondType() == Chem.BondType.SINGLE and not bond.GetIsAromatic()
        for bond in mol.GetBonds()
    )


def _ring_count_from_mol(mol, formula: str) -> int:
    """Return the full-molecule graph cycle rank.

    The formula is authoritative for connected saturated hydrocarbons.  The
    graph value is retained as an invariant check here and as the fallback for
    general RDKit molecules containing heteroatoms or multiple bonds.
    """

    Chem = _require_rdkit()
    component_count = len(Chem.GetMolFrags(mol))
    graph_cycle_rank = mol.GetNumBonds() - mol.GetNumAtoms() + component_count
    if graph_cycle_rank < 0:
        raise ValueError("molecule graph has an invalid negative cycle rank")
    if _is_connected_saturated_hydrocarbon(mol):
        formula_ring_count = degree_of_unsaturation_from_hydrocarbon_formula(formula)
        if formula_ring_count != graph_cycle_rank:
            raise ValueError(
                "formula-derived ring count disagrees with the saturated "
                f"hydrocarbon graph cycle rank: {formula}"
            )
        return formula_ring_count
    return graph_cycle_rank


def ring_count_from_smiles(smiles: str) -> int:
    """Return a full-molecule ring closure count without other descriptors."""

    from rdkit.Chem import rdMolDescriptors

    Chem = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    formula = rdMolDescriptors.CalcMolFormula(mol)
    return _ring_count_from_mol(mol, formula)


def molar_mass_from_formula(formula: str) -> float:
    """Calculate standard average molar mass directly from a molecular formula."""

    normalized = str(formula).strip()
    if not normalized:
        raise ValueError("molecular formula must not be empty")

    Chem = _require_rdkit()
    periodic_table = Chem.GetPeriodicTable()
    position = 0
    molar_mass = 0.0
    token_count = 0
    for match in _FORMULA_TOKEN.finditer(normalized):
        if match.start() != position:
            raise ValueError(f"unsupported molecular formula: {formula}")
        symbol, count_text = match.groups()
        atomic_number = int(periodic_table.GetAtomicNumber(symbol))
        if atomic_number <= 0:
            raise ValueError(f"unknown element in molecular formula: {symbol}")
        atom_count = int(count_text) if count_text else 1
        if atom_count <= 0:
            raise ValueError("atom counts in a molecular formula must be positive")
        molar_mass += float(periodic_table.GetAtomicWeight(atomic_number)) * atom_count
        position = match.end()
        token_count += 1

    if token_count == 0 or position != len(normalized):
        raise ValueError(f"unsupported molecular formula: {formula}")
    return round(molar_mass, 6)


def carbon_graph_from_smiles(smiles: str) -> tuple[int, tuple[Edge, ...]]:
    """Parse a SMILES string and return its carbon-only skeleton graph."""

    Chem = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")

    carbon_old_to_new: dict[int, int] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            carbon_old_to_new[atom.GetIdx()] = len(carbon_old_to_new)

    edges: set[Edge] = set()
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in carbon_old_to_new and end in carbon_old_to_new:
            u, v = carbon_old_to_new[begin], carbon_old_to_new[end]
            edges.add((u, v) if u < v else (v, u))

    return len(carbon_old_to_new), tuple(sorted(edges))


def smiles_to_half_graph(smiles: str) -> HalfGraph:
    """Convert SMILES using any supported topological reflection axis."""

    num_nodes, edges = carbon_graph_from_smiles(smiles)
    return find_reflection_decomposition(num_nodes, edges).to_half_graph()


def smiles_to_generation_half_graph(smiles: str) -> HalfGraph:
    """Convert SMILES using the best axis that satisfies generation constraints."""

    num_nodes, edges = carbon_graph_from_smiles(smiles)
    return find_generation_reflection_decomposition(num_nodes, edges).to_half_graph()


def canonicalize_smiles(smiles: str) -> str:
    Chem = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def mol_from_carbon_edges(edges: list[Edge] | tuple[Edge, ...], n_carbons: int):
    """Build an RDKit molecule from carbon edges and add hydrogens."""

    mol = carbon_mol_from_edges(edges, n_carbons)
    Chem = _require_rdkit()
    return Chem.AddHs(mol)


def carbon_mol_from_edges(edges: list[Edge] | tuple[Edge, ...], n_carbons: int):
    """Build and sanitize an RDKit molecule from a carbon skeleton."""

    Chem = _require_rdkit()
    rw_mol = Chem.RWMol()
    for _ in range(n_carbons):
        rw_mol.AddAtom(Chem.Atom(6))
    for u, v in edges:
        rw_mol.AddBond(int(u), int(v), Chem.BondType.SINGLE)
    mol = rw_mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def generate_canonical_smiles(mol) -> str:
    Chem = _require_rdkit()
    return Chem.MolToSmiles(mol, canonical=True)


def canonical_smiles_from_carbon_edges(
    edges: list[Edge] | tuple[Edge, ...], n_carbons: int
) -> str:
    """Return a canonical implicit-hydrogen SMILES for a carbon skeleton."""

    return generate_canonical_smiles(carbon_mol_from_edges(edges, n_carbons))


@dataclass(frozen=True)
class MoleculeDescriptors:
    """Descriptors used by candidate screening.

    ``ring_count`` is the full-molecule cycle rank. ``rdkit_ring_count`` and the
    size counters come from the complete molecule's RDKit ring information and
    are intentionally kept separate because symmetric cages can have more SSSR
    rings than independent ring closures.
    """

    canonical_smiles: str
    formula: str
    molecular_weight: float
    sa_score: float
    carbon_count: int
    hydrogen_count: int
    h_to_c_ratio: float
    # Independent ring closures. For this toolkit's connected saturated
    # hydrocarbons this is the formula-derived cycle rank, not RDKit's SSSR
    # count, which can be larger for symmetric cage topologies.
    ring_count: int
    rdkit_ring_count: int
    three_member_rings: int
    four_member_rings: int
    five_member_rings: int
    six_member_rings: int
    strain_ring_score: int
    preferred_ring_score: int
    rotatable_bonds: int
    fraction_sp3: float


def descriptors_from_smiles(smiles: str) -> MoleculeDescriptors:
    """Compute lightweight RDKit descriptors for generated hydrocarbon screening."""

    Chem = _require_rdkit()
    from rdkit.Chem import Lipinski, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")

    canonical = Chem.MolToSmiles(mol, canonical=True)
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    hydrogen_count = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
    h_to_c_ratio = hydrogen_count / carbon_count if carbon_count else 0.0
    formula = rdMolDescriptors.CalcMolFormula(mol)
    ring_sizes = tuple(len(ring) for ring in mol.GetRingInfo().AtomRings())
    ring_count = _ring_count_from_mol(mol, formula)
    three_member_rings = ring_sizes.count(3)
    four_member_rings = ring_sizes.count(4)
    five_member_rings = ring_sizes.count(5)
    six_member_rings = ring_sizes.count(6)

    return MoleculeDescriptors(
        canonical_smiles=canonical,
        formula=formula,
        molecular_weight=molar_mass_from_formula(formula),
        sa_score=synthetic_accessibility_score_from_mol(mol),
        carbon_count=carbon_count,
        hydrogen_count=hydrogen_count,
        h_to_c_ratio=round(float(h_to_c_ratio), 6),
        ring_count=ring_count,
        rdkit_ring_count=len(ring_sizes),
        three_member_rings=three_member_rings,
        four_member_rings=four_member_rings,
        five_member_rings=five_member_rings,
        six_member_rings=six_member_rings,
        strain_ring_score=(
            three_member_rings * THREE_MEMBER_RING_STRAIN_WEIGHT
            + four_member_rings * FOUR_MEMBER_RING_STRAIN_WEIGHT
        ),
        preferred_ring_score=(
            five_member_rings * FIVE_MEMBER_RING_PREFERENCE_WEIGHT
            + six_member_rings * SIX_MEMBER_RING_PREFERENCE_WEIGHT
        ),
        rotatable_bonds=int(Lipinski.NumRotatableBonds(mol)),
        fraction_sp3=round(float(rdMolDescriptors.CalcFractionCSP3(mol)), 6),
    )


def synthetic_accessibility_score(smiles: str) -> float:
    Chem = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return synthetic_accessibility_score_from_mol(mol)


def synthetic_accessibility_score_from_mol(mol) -> float:
    try:
        from rdkit.Contrib.SA_Score import sascorer
    except ImportError as exc:
        raise RuntimeError(
            "RDKit SA_Score is required for synthetic accessibility scoring."
        ) from exc
    return round(float(sascorer.calculateScore(mol)), 6)
