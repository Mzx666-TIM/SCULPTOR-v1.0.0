"""Stages and operators shared by the generation and perturbation pipelines.

The per-``n1/n2/n3`` campaign folders were retired on 2026-09-22: both stages
now write flat tables plus one compact group summary, and every row keeps its
mirror split (``n1``/``n2``/``n3``) and ``group_key`` in its own columns. What
remains here is the shared vocabulary.
"""
from __future__ import annotations

PERTURBED_STAGE = "perturbed_from_hits"
ALLOWED_PERTURBATION_OPERATIONS = {
    "bond_rewire",
    "ch2_bridge_move",
    "local_skeleton_rearrange",
    "methyl_shift",
    "methylation",
    "ring_adjust",
    # Ring expansion: insert one CH2 into a strained three/four/five-membered
    # ring, so a cyclopropane becomes a cyclobutane and so on towards the
    # five/six-membered motifs JP-10 is built from.
    "ring_expand",
}
