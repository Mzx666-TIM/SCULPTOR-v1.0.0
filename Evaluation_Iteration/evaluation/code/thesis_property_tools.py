from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from chemistry import _require_rdkit, molar_mass_from_formula
from property_uncertainty import novelty_adjusted_uncertainty


AVOGADRO = 6.02214076e23
ANGSTROM3_TO_CM3 = 1e-24
DELTA_HF_CO2_G_KJ_MOL = -393.5224
DELTA_HF_H2O_G_KJ_MOL = -241.8264
DELTA_HF_H2O_L_KJ_MOL = -285.8304
HARTREE_TO_KJ_MOL = 2625.499638
QM9_ATOMIC_REFERENCE_H_HARTREE = -0.500273
QM9_ATOMIC_REFERENCE_C_HARTREE = -37.846772
DELTA_HF_C_ATOM_G_KJ_MOL = 716.68
DELTA_HF_H_ATOM_G_KJ_MOL = 218.0

BONDI_RADII_ANGSTROM = {
    1: 1.20,
    6: 1.70,
}
DENSITY_LOW_NOVELTY_UNCERTAINTY_G_CM3 = 0.03
DENSITY_HIGH_NOVELTY_UNCERTAINTY_G_CM3 = 0.08


@dataclass(frozen=True)
class DensityEstimate:
    canonical_smiles: str
    molecular_weight_g_mol: float
    vdw_volume_angstrom3: float
    density_g_cm3: float
    uncertainty_g_cm3: float
    uncertainty_source: str
    radius_scale: float
    surface_points_per_atom: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HeatingValueEstimate:
    canonical_smiles: str
    formula: str
    carbon_count: int
    hydrogen_count: int
    molecular_weight_g_mol: float
    formation_enthalpy_kj_mol: float
    combustion_enthalpy_kj_mol: float
    heating_value_mj_kg: float
    water_phase: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MolarVolumeEstimate:
    molecular_weight_g_mol: float
    density_g_cm3: float
    molar_volume_cm3_mol: float
    uncertainty_cm3_mol: float | None
    source: str = "derived_from_molecular_weight_and_density"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VolumetricHeatingValueEstimate:
    volumetric_heating_value_mj_l: float
    uncertainty_mj_l: float | None
    source: str = "derived_from_density_and_mass_heating_value"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def molar_volume_from_density(
    molecular_weight_g_mol: float,
    density_g_cm3: float,
    density_uncertainty_g_cm3: float | None = None,
) -> MolarVolumeEstimate:
    """Derive molar volume from molecular weight and density.

    With molecular weight in g/mol and density in g/cm3, M/rho is directly in
    cm3/mol. Molecular weight is treated as exact; optional uncertainty is
    propagated from the density prediction.
    """

    molecular_weight = float(molecular_weight_g_mol)
    density = float(density_g_cm3)
    if not math.isfinite(molecular_weight) or molecular_weight <= 0.0:
        raise ValueError("molecular weight must be a positive finite number")
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("density must be a positive finite number")

    molar_volume = molecular_weight / density
    uncertainty = None
    if density_uncertainty_g_cm3 is not None:
        density_uncertainty = float(density_uncertainty_g_cm3)
        if not math.isfinite(density_uncertainty) or density_uncertainty < 0.0:
            raise ValueError("density uncertainty must be a non-negative finite number")
        uncertainty = molar_volume * density_uncertainty / density

    return MolarVolumeEstimate(
        molecular_weight_g_mol=round(molecular_weight, 6),
        density_g_cm3=round(density, 6),
        molar_volume_cm3_mol=round(molar_volume, 6),
        uncertainty_cm3_mol=(
            None if uncertainty is None else round(float(uncertainty), 6)
        ),
    )


def volumetric_heating_value_from_density(
    density_g_cm3: float,
    heating_value_mj_kg: float,
    density_uncertainty_g_cm3: float | None = None,
    heating_value_uncertainty_mj_kg: float | None = None,
) -> VolumetricHeatingValueEstimate:
    """Derive MJ/L from density and mass heating value with uncertainty."""

    density = float(density_g_cm3)
    heating_value = float(heating_value_mj_kg)
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("density must be a positive finite number")
    if not math.isfinite(heating_value) or heating_value <= 0.0:
        raise ValueError("heating value must be a positive finite number")

    volumetric_value = density * heating_value
    uncertainty = None
    if (
        density_uncertainty_g_cm3 is not None
        or heating_value_uncertainty_mj_kg is not None
    ):
        density_uncertainty = float(density_uncertainty_g_cm3 or 0.0)
        heating_uncertainty = float(heating_value_uncertainty_mj_kg or 0.0)
        if density_uncertainty < 0.0 or heating_uncertainty < 0.0:
            raise ValueError("uncertainties must be non-negative")
        uncertainty = math.sqrt(
            (heating_value * density_uncertainty) ** 2
            + (density * heating_uncertainty) ** 2
        )

    return VolumetricHeatingValueEstimate(
        volumetric_heating_value_mj_l=round(volumetric_value, 6),
        uncertainty_mj_l=None if uncertainty is None else round(uncertainty, 6),
    )


def estimate_vdw_density_from_smiles(
    smiles: str,
    seed: int = 0,
    radius_scale: float = 1.4,
    surface_points_per_atom: int = 960,
    novelty_score: float | None = None,
    high_novelty: bool = False,
) -> DensityEstimate:
    """Estimate density using the thesis-style vdW surface integral workflow.

    This mirrors the method described in Section 5.2.3: build a 3D structure,
    scale Bondi radii, keep exposed vdW surface points, estimate vdW volume with
    the divergence theorem, and convert molecular mass per vdW volume to g/cm3.
    """

    if radius_scale <= 0:
        raise ValueError("radius_scale must be positive")
    if surface_points_per_atom < 16:
        raise ValueError("surface_points_per_atom must be at least 16")

    Chem = _require_rdkit()
    from rdkit.Chem import AllChem, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    canonical = Chem.MolToSmiles(mol, canonical=True)
    mol_h = Chem.AddHs(mol)

    status = -1
    for attempt in range(4):
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed) + attempt * 1009
        params.useRandomCoords = attempt > 0
        mol_h.RemoveAllConformers()
        status = AllChem.EmbedMolecule(mol_h, params)
        if status == 0:
            break
    if status != 0:
        raise ValueError(f"failed to embed SMILES: {smiles}")
    try:
        AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        pass

    conformer = mol_h.GetConformer()
    atoms = []
    for atom in mol_h.GetAtoms():
        atomic_number = int(atom.GetAtomicNum())
        radius = _bondi_radius(atomic_number) * radius_scale
        position = conformer.GetAtomPosition(atom.GetIdx())
        atoms.append(
            (
                atomic_number,
                (float(position.x), float(position.y), float(position.z)),
                radius,
            )
        )

    unit_points = _fibonacci_sphere(surface_points_per_atom)
    volume = 0.0
    for atom_index, (_, center, radius) in enumerate(atoms):
        area_weight = 4.0 * math.pi * radius * radius / surface_points_per_atom
        for normal in unit_points:
            point = (
                center[0] + radius * normal[0],
                center[1] + radius * normal[1],
                center[2] + radius * normal[2],
            )
            if _is_buried(point, atom_index, atoms):
                continue
            volume += _dot(point, normal) * area_weight / 3.0

    volume = abs(volume)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    molecular_weight = molar_mass_from_formula(formula)
    molar_volume_cm3_mol = volume * ANGSTROM3_TO_CM3 * AVOGADRO
    density = molecular_weight / molar_volume_cm3_mol
    uncertainty = novelty_adjusted_uncertainty(
        DENSITY_LOW_NOVELTY_UNCERTAINTY_G_CM3,
        DENSITY_HIGH_NOVELTY_UNCERTAINTY_G_CM3,
        novelty_score=novelty_score,
        high_novelty=high_novelty,
    )
    return DensityEstimate(
        canonical_smiles=canonical,
        molecular_weight_g_mol=round(molecular_weight, 6),
        vdw_volume_angstrom3=round(volume, 6),
        density_g_cm3=round(float(density), 6),
        uncertainty_g_cm3=round(float(uncertainty.uncertainty), 6),
        uncertainty_source=uncertainty.source,
        radius_scale=radius_scale,
        surface_points_per_atom=surface_points_per_atom,
    )


def mass_heating_value_from_formation_enthalpy(
    smiles: str,
    formation_enthalpy_kj_mol: float,
    water_phase: str = "gas",
) -> HeatingValueEstimate:
    """Compute mass heating value from a standard formation enthalpy.

    The caller supplies the fuel standard formation enthalpy (an external
    quantum-chemistry result). This helper applies the hydrocarbon combustion
    stoichiometry and converts |Delta Hc| / M to MJ/kg.
    """

    Chem = _require_rdkit()
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    canonical = Chem.MolToSmiles(mol, canonical=True)
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    hydrogen_count = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
    if any(atom.GetAtomicNum() not in (1, 6) for atom in Chem.AddHs(mol).GetAtoms()):
        raise ValueError("heating value helper currently supports hydrocarbons only")

    water_hf = _water_formation_enthalpy(water_phase)
    product_enthalpy = (
        carbon_count * DELTA_HF_CO2_G_KJ_MOL
        + (hydrogen_count / 2.0) * water_hf
    )
    combustion_enthalpy = product_enthalpy - float(formation_enthalpy_kj_mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    molecular_weight = molar_mass_from_formula(formula)
    heating_value = -combustion_enthalpy / molecular_weight
    return HeatingValueEstimate(
        canonical_smiles=canonical,
        formula=formula,
        carbon_count=carbon_count,
        hydrogen_count=hydrogen_count,
        molecular_weight_g_mol=round(molecular_weight, 6),
        formation_enthalpy_kj_mol=round(float(formation_enthalpy_kj_mol), 6),
        combustion_enthalpy_kj_mol=round(float(combustion_enthalpy), 6),
        heating_value_mj_kg=round(float(heating_value), 6),
        water_phase=water_phase,
    )


def mass_heating_value_from_qm9_h298_proxy(
    smiles: str,
    molecular_h298_hartree: float,
    water_phase: str = "gas",
) -> HeatingValueEstimate:
    """Estimate heating value from a QM9-style molecular H298 prediction.

    The e3nn property model predicts a QM9-like molecular total enthalpy in
    Hartree. This converts it through a simple atomization-energy cycle, then
    reuses the same combustion helper as the formation-enthalpy route.
    """

    Chem = _require_rdkit()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    hydrogen_count = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
    if any(atom.GetAtomicNum() not in (1, 6) for atom in Chem.AddHs(mol).GetAtoms()):
        raise ValueError("heating value helper currently supports hydrocarbons only")

    atomization_kj_mol = (
        (
            carbon_count * QM9_ATOMIC_REFERENCE_C_HARTREE
            + hydrogen_count * QM9_ATOMIC_REFERENCE_H_HARTREE
        )
        - float(molecular_h298_hartree)
    ) * HARTREE_TO_KJ_MOL
    formation_enthalpy = (
        carbon_count * DELTA_HF_C_ATOM_G_KJ_MOL
        + hydrogen_count * DELTA_HF_H_ATOM_G_KJ_MOL
        - atomization_kj_mol
    )
    return mass_heating_value_from_formation_enthalpy(
        smiles,
        formation_enthalpy,
        water_phase=water_phase,
    )


def _bondi_radius(atomic_number: int) -> float:
    try:
        return BONDI_RADII_ANGSTROM[atomic_number]
    except KeyError as exc:
        raise ValueError(f"missing Bondi radius for atomic number {atomic_number}") from exc


def _water_formation_enthalpy(water_phase: str) -> float:
    if water_phase == "gas":
        return DELTA_HF_H2O_G_KJ_MOL
    if water_phase == "liquid":
        return DELTA_HF_H2O_L_KJ_MOL
    raise ValueError("water_phase must be 'gas' or 'liquid'")


def _fibonacci_sphere(count: int) -> tuple[tuple[float, float, float], ...]:
    points = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(count):
        y = 1.0 - (2.0 * index + 1.0) / count
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * index
        points.append((math.cos(theta) * radius, y, math.sin(theta) * radius))
    return tuple(points)


def _is_buried(
    point: tuple[float, float, float],
    atom_index: int,
    atoms: list[tuple[int, tuple[float, float, float], float]],
) -> bool:
    for other_index, (_, other_center, other_radius) in enumerate(atoms):
        if other_index == atom_index:
            continue
        if _distance_squared(point, other_center) < (other_radius - 1e-8) ** 2:
            return True
    return False


def _distance_squared(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def _dot(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
