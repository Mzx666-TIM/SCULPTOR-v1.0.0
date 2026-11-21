from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache

from chemistry import _require_rdkit


@dataclass(frozen=True)
class MMFFGeometryAnalysis:
    canonical_smiles: str
    embed_success: bool
    optimize_success: bool
    mmff_converged: bool
    mmff_energy_kcal_mol: float | None
    mmff_energy_per_atom: float | None
    mmff_energy_per_heavy_atom: float | None
    mmff_stability_score: float
    mmff_failure_reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@lru_cache(maxsize=8192)
def analyze_mmff_geometry(
    smiles: str,
    seed: int = 0,
    max_iters: int = 500,
    attempts: int = 6,
) -> MMFFGeometryAnalysis:
    """Embed and MMFF-optimize a molecule for early geometry screening."""

    Chem = _require_rdkit()
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    canonical = Chem.MolToSmiles(mol, canonical=True)
    mol_h = Chem.AddHs(mol)

    embed_status = -1
    for attempt in range(max(1, attempts)):
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed) + attempt * 1009
        params.useRandomCoords = attempt > 0
        mol_h.RemoveAllConformers()
        embed_status = AllChem.EmbedMolecule(mol_h, params)
        if embed_status == 0:
            break
    if embed_status != 0:
        return _failed_mmff(canonical, "embed_failed")

    if AllChem.MMFFHasAllMoleculeParams(mol_h):
        properties = AllChem.MMFFGetMoleculeProperties(mol_h)
        forcefield = AllChem.MMFFGetMoleculeForceField(mol_h, properties)
        method = "mmff"
    else:
        forcefield = AllChem.UFFGetMoleculeForceField(mol_h)
        method = "uff_fallback"
    if forcefield is None:
        return _failed_mmff(canonical, "forcefield_unavailable")

    try:
        optimize_status = forcefield.Minimize(maxIts=int(max_iters))
        energy = float(forcefield.CalcEnergy())
    except Exception:
        return _failed_mmff(canonical, f"{method}_optimization_failed")

    atom_count = max(mol_h.GetNumAtoms(), 1)
    heavy_count = max(sum(1 for atom in mol_h.GetAtoms() if atom.GetAtomicNum() > 1), 1)
    energy_per_atom = energy / atom_count
    energy_per_heavy = energy / heavy_count
    converged = optimize_status == 0
    stability = mmff_stability_score(
        energy_per_heavy_atom=energy_per_heavy,
        converged=converged,
    )
    return MMFFGeometryAnalysis(
        canonical_smiles=canonical,
        embed_success=True,
        optimize_success=True,
        mmff_converged=converged,
        mmff_energy_kcal_mol=round(energy, 6),
        mmff_energy_per_atom=round(energy_per_atom, 6),
        mmff_energy_per_heavy_atom=round(energy_per_heavy, 6),
        mmff_stability_score=round(stability, 6),
        mmff_failure_reason="" if method == "mmff" else method,
    )


def mmff_stability_score(
    energy_per_heavy_atom: float | None,
    converged: bool,
    reference: float = 25.0,
) -> float:
    if energy_per_heavy_atom is None or not math.isfinite(energy_per_heavy_atom):
        return 0.0
    penalty = max(0.0, energy_per_heavy_atom) / max(reference, 1e-9)
    score = 1.0 / (1.0 + penalty)
    if not converged:
        score *= 0.7
    return max(0.0, min(1.0, float(score)))


def _failed_mmff(canonical_smiles: str, reason: str) -> MMFFGeometryAnalysis:
    return MMFFGeometryAnalysis(
        canonical_smiles=canonical_smiles,
        embed_success=False,
        optimize_success=False,
        mmff_converged=False,
        mmff_energy_kcal_mol=None,
        mmff_energy_per_atom=None,
        mmff_energy_per_heavy_atom=None,
        mmff_stability_score=0.0,
        mmff_failure_reason=reason,
    )
