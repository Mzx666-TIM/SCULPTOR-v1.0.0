from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

from chemistry import canonicalize_smiles
from property_screening import FuelPropertyThresholds, screening_status
from tier_routing import (
    CHANNEL_LABELS,
    CHANNEL_ORDER,
    NO_CHANNEL_LABEL,
    POTENTIAL1,
    POTENTIAL2,
    POTENTIAL3,
    TIER1,
    TIER2,
    TIER3,
    TIER4,
)


def mark_perturbation_seed_rows(
    rows: list[dict[str, object]],
    *,
    selection_basis: str = "tier123_pool",
) -> tuple[dict[str, object], ...]:
    """Mark candidate rows as perturbation seeds (unique by SMILES).

    The tier split and the potential pools decide who is perturbed;
    no calculation route is involved any more.
    """
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    status = {
        "potential_pool": "potential_pool",
        "iteration_pool": "iteration_pool",
    }.get(selection_basis, "tier_pool")
    for row in rows:
        canonical = canonicalize_smiles(str(row.get("canonical_smiles", "")).strip())
        if canonical in seen:
            continue
        seen.add(canonical)
        selected.append(
            {
                **row,
                "canonical_smiles": canonical,
                "perturbation_seed_status": status,
                "perturbation_seed_eligible": True,
                "seed_selection_basis": selection_basis,
                "feedback_role": "not_eligible_until_manual_confirmation",
                # 0 marks the first, non-iterative campaign; the iteration
                # wrapper overwrites this with the real round number.
                "seed_perturbation_round": int(row.get("seed_perturbation_round") or 0),
            }
        )
    _assign_carbon_group_seed_indices(selected)
    return tuple(selected)


def write_predicted_perturbation_seeds(
    rows: list[dict[str, object]],
    output_path: str | Path,
) -> tuple[dict[str, object], ...]:
    """Write the Tier 1-3 rows as the perturbation seed table."""
    selected = mark_perturbation_seed_rows(rows)
    _write_rows(Path(output_path), list(selected))
    return selected


def require_perturbation_seed_rows(
    path: str | Path,
    thresholds: FuelPropertyThresholds = FuelPropertyThresholds(),
) -> tuple[dict[str, str], ...]:
    """Validate a perturbation seed table.

    Seeds come from the channel pools only: the Tier 1-3 pool, the
    potential pool, or the pool an iteration round selected from. Every row must
    be marked ``perturbation_seed_eligible``, carry a carbon group index, and
    name the pool it was selected from.
    """

    rows = tuple(_read_rows(Path(path)))
    if not rows:
        raise ValueError("perturbation seed file contains no rows")
    for index, row in enumerate(rows, start=1):
        if str(row.get("perturbation_seed_eligible", "")).strip().lower() not in {
            "true",
            "1",
            "yes",
        }:
            raise ValueError(f"row {index} is not marked perturbation_seed_eligible")
        if int(row.get("carbon_group_index") or 0) <= 0:
            raise ValueError(f"row {index} is missing carbon_group_index")
        basis = str(row.get("seed_selection_basis", "")).strip().lower()
        if basis not in {"tier123_pool", "potential_pool", "iteration_pool"}:
            raise ValueError(f"row {index} has no recognised seed_selection_basis")
    return rows


def _assign_carbon_group_seed_indices(rows: list[dict[str, object]]) -> None:
    counters: dict[int, int] = {}
    rows.sort(
        key=lambda row: (
            int(row.get("carbon_count") or row.get("total_carbons") or 0),
            str(row.get("candidate_id", "")),
            str(row.get("canonical_smiles", "")),
        )
    )
    for row in rows:
        carbon_count = int(row.get("carbon_count") or row.get("total_carbons") or 0)
        if carbon_count <= 0:
            raise ValueError("finalized row is missing a positive carbon_count")
        counters[carbon_count] = counters.get(carbon_count, 0) + 1
        index = counters[carbon_count]
        row["carbon_group_index"] = index
        row["perturbation_seed_id"] = f"C{carbon_count}_seed_{index:04d}"


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_rows(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows and fieldnames is None:
            return
        if fieldnames is None:
            fieldnames = []
            for row in rows:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(part: float, whole: float) -> float:
    return part / whole * 100.0 if whole else 0.0


def _generation_metrics(output_dir: Path) -> dict[str, float]:
    summaries = [
        _read_json(cdir / "summary.json")
        for cdir in sorted(output_dir.glob("C*"))
        if cdir.is_dir()
    ]
    if not summaries:
        # Single-carbon runs write straight into the run folder.
        root_summary = output_dir / "summary.json"
        if root_summary.exists():
            summaries = [_read_json(root_summary)]
    metrics = {
        "requests": 0.0,
        "valid": 0.0,
        "passed_screen": 0.0,
        "unique_novel": 0.0,
        "temp_duplicates": 0.0,
        "seeds": 0.0,
        "property_failed": 0.0,
    }
    for summary in summaries:
        for key, field in (
            ("requests", "total_requested_samples"),
            ("valid", "valid_candidates"),
            ("passed_screen", "passed_pre_screen_before_temporary_dedup"),
            # ``temporary_smiles_count`` is the size of the temporary library, which
            # carries entries over from earlier carbon groups; the accepted-candidate
            # count is the number of newly stored unique structures of that stage.
            ("unique_novel", "passed_pre_screen"),
            ("temp_duplicates", "temporary_duplicate_rejections"),
            ("seeds", "perturbation_seed_count"),
            ("property_failed", "property_failed_count"),
        ):
            metrics[key] += float(summary.get(field) or 0)
    # The generation stage owns the ``C*`` pools whenever the folders exist - also
    # while no perturbation campaign has run yet, which is when the run root is
    # still empty. Only a folder whose generation wrote its pools flat uses the
    # run root.
    metrics.update(_tier_pool_counts(output_dir, prefer_root=False))
    metrics["valid_rate"] = _pct(metrics["valid"], metrics["requests"])
    metrics["efficiency"] = _pct(metrics["unique_novel"], metrics["requests"])
    return metrics


def _tier_pool_counts(output_dir: Path, *, prefer_root: bool = True) -> dict[str, float]:
    """Count the channel pools of a stage (per carbon, or run root).

    The stage reports and the flow-totals table present this split; it is the
    only channel split a run folder carries.

    ``prefer_root`` says whether the run-root pools belong to this stage: they do
    for the perturbation stage (``prefer_root=True``), while the generation stage
    reads its ``C*`` folders and only falls back to the run root when generation
    wrote the folder flat.
    """
    counts = {channel: 0.0 for channel in CHANNEL_POOLS}

    def add_pools(source_dir: Path) -> None:
        for channel in CHANNEL_POOLS:
            counts[channel] += len(_read_rows(source_dir / POOL_FILE_NAMES[channel]))

    if not prefer_root:
        carbon_dirs = sorted(
            (
                path
                for path in output_dir.iterdir()
                if path.is_dir() and path.name[:1] == "C" and path.name[1:].isdigit()
            ),
            key=lambda path: int(path.name[1:]),
        ) if output_dir.is_dir() else []
        if carbon_dirs:
            for cdir in carbon_dirs:
                add_pools(cdir)
            return counts
        # A single-carbon generation run writes its pools at the run root; a
        # perturbation-only folder has no generation pools at all.
        if _has_perturbation_stage(output_dir):
            return counts
    # The perturbation stage always owns the run-root pools.
    add_pools(output_dir)
    return counts


def _has_perturbation_stage(output_dir: Path) -> bool:
    return bool(
        (output_dir / "perturbation_generation_summary.json").exists()
        or (output_dir / "perturbed_candidates.csv").exists()
    )


def _perturbation_metrics(output_dir: Path) -> dict[str, float]:
    if not _has_perturbation_stage(output_dir):
        return None
    generation = _read_json(output_dir / "perturbation_generation_summary.json")
    pipeline = _read_json(output_dir / "perturbation_pipeline_summary.json")
    screened_path = output_dir / "perturbation_screened.csv"
    passed_screen_rows = None
    if screened_path.exists():
        screened_rows = _read_rows(screened_path)
        if screened_rows and "passes_screen" in screened_rows[0]:
            passed_screen_rows = sum(
                str(row.get("passes_screen") or "").strip().lower()
                in {"1", "true", "yes"}
                for row in screened_rows
            )
    property_failed_path = output_dir / "property_failed_candidates.csv"
    property_failed_rows = None
    if property_failed_path.exists():
        property_failed_rows = len(_read_rows(property_failed_path))
    generated_rows = float(
        generation.get("folder_generated_rows")
        or generation.get("generated_rows")
        or 0
    )
    property_rows = float(pipeline.get("merged_product_rows") or 0)
    metrics = {
        # ``folder_*`` counters sum every campaign of the run folder; the plain
        # ones describe the campaign that ran last, which is the fallback for
        # folders written before campaigns were merged.
        "requests": float(
            generation.get("folder_attempted_samples")
            or generation.get("attempted_samples")
            or 0
        ),
        "valid": generated_rows,
        "duplicates": float(
            generation.get("folder_duplicate_rows")
            or generation.get("duplicate_rows")
            or 0
        ),
        "failed": float(
            generation.get("folder_failed_attempts")
            or generation.get("failed_attempts")
            or 0
        ),
        "passed_screen": float(
            passed_screen_rows
            if passed_screen_rows is not None
            else pipeline.get("passed_screen_rows") or 0
        ),
        "property_failed": float(
            property_failed_rows
            if property_failed_rows is not None
            else pipeline.get("property_failed_rows") or 0
        ),
        "manual_review": float(pipeline.get("manual_review_rows") or 0),
        "carried_over": float(pipeline.get("carried_over_product_rows") or 0),
        "folder_products": property_rows,
    }
    manual_review_path = output_dir / "manual_review" / MANUAL_REVIEW_FILE
    if manual_review_path.exists():
        metrics["manual_review"] = float(len(_read_rows(manual_review_path)))
    metrics.update(_tier_pool_counts(output_dir, prefer_root=True))
    metrics["unique_novel"] = metrics["folder_products"]
    if not metrics.get("folder_products"):
        # Folders written before campaigns were merged describe one campaign,
        # which is the whole folder in that case.
        metrics["folder_products"] = metrics["valid"]
    metrics["valid_rate"] = _pct(metrics["valid"], metrics["requests"])
    metrics["efficiency"] = _pct(metrics["unique_novel"], metrics["requests"])
    return metrics


def channel_report_lines(metrics: dict[str, object]) -> list[str]:
    """One report line per channel: Tier 1-4, then Potential 1-3."""

    return [
        "%s: %.0f" % (CHANNEL_LABELS[channel], float(metrics.get(channel) or 0))
        for channel in CHANNEL_POOLS
    ]


def iteration_report_lines(output_dir: str | Path) -> list[str]:
    """Report lines of the iterative perturbation mode (empty when it did not run)."""

    summary_path = Path(output_dir) / ITERATION_SUMMARY_FILE
    if not summary_path.exists():
        return []
    rows = _read_rows(summary_path)
    if not rows:
        return []
    lines = ["", "Iterative perturbation (shared pool, static iteration model)"]
    lines.append(
        "Rounds run: %d | Tier 1 at the start: %s | after the last round: %s"
        % (
            len(rows),
            rows[0].get("tier1_total", "") if False else _round_baseline(rows),
            rows[-1].get("tier1_total", ""),
        )
    )
    for row in rows:
        lines.append(
            "Round %s: seed list %s (never perturbed %s) | perturbed %s leads | "
            "screened %s | duplicate rejections %s | new products %s | new Tier 1 %s "
            "| Tier 1 total %s"
            % (
                row.get("round", ""),
                row.get("seed_list_size", ""),
                row.get("fresh_seeds_in_list", ""),
                row.get("seed_rows", ""),
                row.get("selected_rows", ""),
                row.get("duplicate_rejections", ""),
                row.get("new_products", ""),
                row.get("new_tier1", ""),
                row.get("tier1_total", ""),
            )
        )
    return lines


def _round_baseline(rows: list[dict[str, object]]) -> object:
    """Tier 1 total before the first round (total minus what the round added)."""

    try:
        return int(rows[0].get("tier1_total") or 0) - int(rows[0].get("new_tier1") or 0)
    except (TypeError, ValueError):
        return ""


def build_generation_stage_report(
    output_dir: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Compute generation-stage metrics and write one-metric-per-line txt."""
    metrics = _generation_metrics(Path(output_dir))
    lines = [
        "Generation Stage Data Report",
        "=" * 32,
        f"Total generation requests: {metrics['requests']:.0f}",
        f"Valid molecules: {metrics['valid']:.0f}",
        f"Valid rate: {metrics['valid_rate']:.2f}%",
        f"Passed initial screening: {metrics['passed_screen']:.0f}",
        f"Unique novel molecules: {metrics['unique_novel']:.0f}",
        f"Generation efficiency (unique / requests): {metrics['efficiency']:.2f}%",
        *channel_report_lines(metrics),
        f"Temp library duplicate rejections: {metrics['temp_duplicates']:.0f}",
        f"Property prediction failures: {metrics['property_failed']:.0f}",
    ]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output_path": str(out), "metrics": metrics}


def build_perturbation_stage_report(
    output_dir: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Compute perturbation-stage metrics and write one-metric-per-line txt."""
    metrics = _perturbation_metrics(Path(output_dir))
    if metrics is None:
        return {"output_path": str(output_path), "present": False, "metrics": None}
    lines = [
        "Perturbation Stage Data Report",
        "=" * 32,
        f"Total perturbation attempts: {metrics['requests']:.0f}",
        f"Valid (unique perturbed products): {metrics['valid']:.0f}",
        f"Valid rate: {metrics['valid_rate']:.2f}%",
        f"Passed initial screening: {metrics['passed_screen']:.0f}",
        f"Unique novel molecules: {metrics['unique_novel']:.0f}",
        f"Generation efficiency (unique / requests): {metrics['efficiency']:.2f}%",
        *channel_report_lines(metrics),
        f"Duplicate rejections: {metrics['duplicates']:.0f}",
        f"Failed attempts: {metrics['failed']:.0f}",
        f"Manual review candidates: {metrics['manual_review']:.0f}",
        f"Property prediction failures: {metrics['property_failed']:.0f}",
        f"Products kept from earlier campaigns in this folder: {metrics['carried_over']:.0f}",
        f"Products stored in this folder in total: {metrics['folder_products']:.0f}",
    ] + iteration_report_lines(output_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output_path": str(out), "present": True, "metrics": metrics}


def stage_channel_members(output_dir: Path, stage: str) -> dict[str, set[str]]:
    """SMILES of the molecules each channel table of one stage holds.

    ``generation`` reads the ``C*`` folders (or the run root for a folder that
    generation wrote flat and no perturbation campaign has taken over);
    ``perturbation`` always reads the run root.
    """

    members = {channel: set() for channel in CHANNEL_POOLS}

    def add(folder: Path) -> None:
        for channel in CHANNEL_POOLS:
            for row in _read_rows(folder / POOL_FILE_NAMES[channel]):
                smiles = str(row.get("canonical_smiles") or "").strip()
                if smiles:
                    members[channel].add(smiles)

    if stage == "perturbation":
        add(output_dir)
        return members
    carbon_dirs = sorted(
        (
            child
            for child in output_dir.iterdir()
            if child.is_dir() and child.name[:1] == "C" and child.name[1:].isdigit()
        ),
        key=lambda child: int(child.name[1:]),
    ) if output_dir.is_dir() else []
    if carbon_dirs:
        for carbon_dir in carbon_dirs:
            add(carbon_dir)
    elif not _has_perturbation_stage(output_dir):
        add(output_dir)
    return members


def build_flow_totals_table(
    output_dir: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Combine generation + perturbation metrics into one totals table (CSV)."""
    root = Path(output_dir)
    gen = _generation_metrics(root)
    pert = _perturbation_metrics(root)
    total = dict(gen)
    rows = [("generation", gen)]
    if pert is not None:
        rows.append(("perturbation", pert))
        for key in ("requests", "valid", "passed_screen", "unique_novel"):
            total[key] = gen[key] + pert[key]
        # A molecule that both stages carry (a perturbation product equal to a
        # generated molecule) must be counted once, so the channel columns of the
        # total row are the size of the union of the two stage pools - exactly
        # what the merged ``analysis/totals`` tables hold.
        generation_members = stage_channel_members(root, "generation")
        perturbation_members = stage_channel_members(root, "perturbation")
        for channel in CHANNEL_POOLS:
            total[channel] = len(
                generation_members[channel] | perturbation_members[channel]
            )
    total["valid_rate"] = _pct(total["valid"], total["requests"])
    total["efficiency"] = _pct(total["unique_novel"], total["requests"])
    rows.append(("total", total))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "stage",
                "requests",
                "valid",
                "valid_rate_pct",
                "passed_screening",
                "unique_novel",
                "gen_efficiency_pct",
                *CHANNEL_POOLS,
            ]
        )
        for label, metrics in rows:
            writer.writerow(
                [
                    label,
                    f"{metrics['requests']:.0f}",
                    f"{metrics['valid']:.0f}",
                    f"{metrics['valid_rate']:.2f}",
                    f"{metrics['passed_screen']:.0f}",
                    f"{metrics['unique_novel']:.0f}",
                    f"{metrics['efficiency']:.2f}",
                    *[f"{metrics[channel]:.0f}" for channel in CHANNEL_POOLS],
                ]
            )
    return {
        "output_path": str(out),
        "generation": gen,
        "perturbation": pert,
        "total": total,
    }


def build_final_data_report(
    generation_report: str | Path,
    perturbation_report: str | Path | None,
    output_path: str | Path,
) -> dict[str, object]:
    """Integrate the two stage reports into one final data report txt."""
    parts: list[str] = []
    for report in (generation_report, perturbation_report):
        if report is None:
            continue
        path = Path(report)
        if path.exists():
            parts.append(path.read_text(encoding="utf-8").rstrip() + "\n")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return {"output_path": str(out), "sections": len(parts)}


MANUAL_REVIEW_FILE = "candidates_for_manual_calculation.csv"


def _remove_tree(path: Path, attempts: int = 6, delay: float = 0.5) -> None:
    """Remove a directory tree, retrying while a file is still open elsewhere.

    Result folders are routinely open in the control panel (fresh CSV previews)
    when ``organize-run-outputs`` runs, so a plain ``rmtree`` can hit
    ``WinError 32``; retry first, then delete whatever is still deletable.
    """
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt < attempts - 1:
                time.sleep(delay)
                continue
        except OSError:
            if attempt < attempts - 1:
                time.sleep(delay)
                continue
    for item in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            if item.is_file():
                item.unlink()
            else:
                item.rmdir()
        except OSError:
            continue
    try:
        path.rmdir()
    except OSError:
        pass


def _copy_file(source: Path, target: Path) -> bool:
    """Copy a file, tolerating a target that another program keeps open."""
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            shutil.copy2(source, target)
            return True
        except (PermissionError, OSError):
            time.sleep(0.4)
    return False


def _write_rows_tolerant(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> bool:
    """Write a CSV, tolerating a reader (e.g. Excel) holding the file open."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            _write_rows(path, rows, fieldnames)
            return True
        except (PermissionError, OSError):
            time.sleep(0.4)
    return False


#: property table each stage writes its scored molecules to
PERTURBATION_PROPERTY_FILE = "perturbation_property_predictions.csv"
GENERATION_PROPERTY_FILE = "property_predictions.csv"

#: keys a stage summary no longer carries (the retired route columns and the
#: input exports of the calculation route)
RETIRED_SUMMARY_KEYS = (
    "high_novelty_dft_candidates",
    "e3nn_queue_candidates",
    "final_dft_count",
    "final_gfn_xtb_count",
    "final_discard_count",
    "dft_export",
    "gfn_xtb_export",
    "high_novelty_complete_count",
    "high_novelty_complete_output",
    # the retired heating-value queue of the external calculator
    "pending_tdoc_count",
    "pending_tdoc_export",
)

#: tables no stage writes any more; a folder from an earlier build still has them
RETIRED_STAGE_FILES = ("pending_tdoc_candidates.csv",)


def normalize_stage_summaries(run_root: Path) -> int:
    """Bring every stage ``summary.json`` to the current contract.

    A folder written by an earlier build still names the retired route columns
    and input exports; the counts are re-derived from the channel tables that
    sit next to the summary, so the summary always agrees with the tables.
    """

    # The run-root pools belong to the perturbation stage once a campaign has run
    # there, so the root summary (a generation artefact) is only refreshed while
    # the root still describes that generation stage.
    targets: list[Path] = []
    if run_root.is_dir() and not _has_perturbation_stage(run_root):
        targets.append(run_root / "summary.json")
    if run_root.is_dir():
        targets.extend(
            child / "summary.json"
            for child in sorted(
                (
                    child
                    for child in run_root.iterdir()
                    if child.is_dir()
                    and child.name[:1] == "C"
                    and child.name[1:].isdigit()
                ),
                key=lambda child: int(child.name[1:]),
            )
        )
    rewritten = 0
    for path in targets:
        if not path.exists():
            continue
        try:
            payload = _read_json(path)
        except json.JSONDecodeError:
            continue
        if not payload:
            continue
        changed = False
        for key in RETIRED_SUMMARY_KEYS:
            if key in payload:
                payload.pop(key)
                changed = True
        for channel in CHANNEL_POOLS:
            count = len(_read_rows(path.parent / POOL_FILE_NAMES[channel]))
            key = "%s_count" % channel
            if payload.get(key) != count:
                payload[key] = count
                changed = True
        if changed:
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            rewritten += 1
    return rewritten


#: count columns of the per-round feedback table, renamed with the verdicts
FEEDBACK_COLUMN_RENAMES = {
    "property_dft_count": "property_pass_count",
    "property_gfn_xtb_count": "property_review_count",
    "property_discard_count": "property_reject_count",
}

#: channel columns of the retired one-axis group summary
RETIRED_GROUP_COLUMNS = ("dft", "gfn_xtb", "discard", "potential")


def normalize_stage_tables(run_root: Path) -> int:
    """Rewrite the per-stage feedback and group tables to the current columns.

    ``generation_feedback.csv`` is renamed in place (the counts already mean
    pass / review / reject), and a ``group_summary.csv`` written under the
    retired one-axis split is rebuilt from the property table next to it, so the
    counts match the channel definition the tables around it use.
    """

    from pipeline import _write_group_summary

    folders = [run_root]
    if run_root.is_dir():
        folders.extend(
            sorted(
                (
                    child
                    for child in run_root.iterdir()
                    if child.is_dir()
                    and child.name[:1] == "C"
                    and child.name[1:].isdigit()
                ),
                key=lambda child: int(child.name[1:]),
            )
        )
    rewritten = 0
    for folder in folders:
        feedback = folder / "generation_feedback.csv"
        if feedback.exists():
            rows = _read_rows(feedback)
            if rows and any(key in rows[0] for key in FEEDBACK_COLUMN_RENAMES):
                renamed = [
                    {
                        FEEDBACK_COLUMN_RENAMES.get(key, key): value
                        for key, value in row.items()
                    }
                    for row in rows
                ]
                _write_rows(feedback, renamed)
                rewritten += 1
        summary = folder / "group_summary.csv"
        property_path = folder / GENERATION_PROPERTY_FILE
        if not summary.exists() or not property_path.exists():
            continue
        try:
            header = summary.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            continue
        if not any(name in header.split(",") for name in RETIRED_GROUP_COLUMNS):
            continue
        property_rows = [
            {
                **row,
                "hard_threshold_verdict": str(row.get("screening_status") or ""),
            }
            for row in _read_rows(property_path)
        ]
        if not property_rows:
            continue
        total_carbons = 0
        for row in property_rows:
            try:
                total_carbons = max(
                    total_carbons,
                    int(row["n1"]) + int(row["n2"]) + int(row["n3"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
        _write_group_summary(folder, property_rows, total_carbons)
        rewritten += 1
    return rewritten


def normalize_candidate_input_metadata(run_root: Path) -> int:
    """Rename the retired route column inside the exported candidate snapshots.

    ``export-candidate-inputs`` copies the scored row of every candidate into
    ``dft_inputs/<id>/metadata.json``; folders exported by an earlier build
    still carry the old route column there, so it is rewritten to the neutral
    ``screening_status`` the run tables use.
    """

    rewritten = 0
    if not run_root.is_dir():
        return 0
    for path in run_root.rglob("metadata.json"):
        parts = path.relative_to(run_root).parts
        if not any(
            part == "dft_inputs" for part in parts
        ):
            continue
        try:
            payload = _read_json(path)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "final_calculation_route" not in payload:
            continue
        payload["screening_status"] = screening_status(
            payload.pop("final_calculation_route")
        )
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rewritten += 1
    return rewritten


def _channel_sort_key(row: dict[str, object]) -> tuple[int, float]:
    """Order inside a channel: more JP-10 beats first, then the composite score."""

    try:
        beats = int(row.get("jp10_beat_count") or 0)
    except (TypeError, ValueError):
        beats = 0
    try:
        score = float(row.get("composite_ranking_score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return (-beats, -score)


def _stored_beat_count(row: dict[str, object]) -> int | None:
    """JP-10 beats recorded in a scored row (None when the row has none)."""

    raw = str(row.get("jp10_beat_count") or "").strip()
    if raw.isdigit():
        return int(raw)
    beats = str(row.get("jp10_beats") or "").strip()
    if beats:
        return len([name for name in beats.split(";") if name.strip()])
    return None


def _stored_hard_threshold_pass(row: dict[str, object]) -> bool:
    """Whether every enabled hard threshold was met when the row was scored."""

    status = str(row.get("screening_status") or "").strip().lower()
    if status:
        return status == "pass"
    # Folders written before the neutral status column carry the route instead.
    return str(row.get("hard_threshold_verdict") or "").strip().lower() == "pass"


def _stored_hard_fail_count(row: dict[str, object]) -> int | None:
    """How many of the three compared hard thresholds a stored row misses.

    The potential channels are named by *(thresholds missed, properties beating
    JP-10)*, so re-deriving them needs the per-property verdicts rather than only
    their combined result. Rows from a folder written before those columns
    existed return ``None`` and the caller falls back to the combined verdict.
    """

    from tier_routing import hard_fail_count

    return hard_fail_count(row)


def retier_run_outputs(root: str | Path) -> dict[str, object]:
    """Re-derive every channel table of a run folder from its stored predictions.

    A molecule's channel is a pure function of what is already stored in its row -
    which of the three compared properties beat JP-10, and how many hard
    thresholds it misses - so a folder written under an earlier channel definition
    can be brought to the current one without re-running a single model. The
    property tables are rewritten with the channel columns, then one table per
    channel is written next to them. A molecule that misses thresholds and beats
    JP-10 on nothing matches no pool and keeps an empty channel.
    """

    from tier_routing import channel_for

    run_root = Path(root)
    carbon_dirs = sorted(
        (
            path
            for path in run_root.iterdir()
            if path.is_dir() and path.name[:1] == "C" and path.name[1:].isdigit()
        ),
        key=lambda path: int(path.name[1:]),
    ) if run_root.is_dir() else []

    targets = [(run_root, PERTURBATION_PROPERTY_FILE)]
    targets.extend((carbon_dir, GENERATION_PROPERTY_FILE) for carbon_dir in carbon_dirs)
    retiered: list[str] = []
    locked: list[str] = []
    channel_counts: dict[str, int] = {channel: 0 for channel in CHANNEL_POOLS}
    rows_read = 0
    rows_without_beats = 0
    rows_unrouted = 0
    for folder, filename in targets:
        property_path = folder / filename
        if not property_path.exists():
            continue
        rows = _read_rows(property_path)
        if not rows:
            continue
        channels: dict[str, list[dict[str, object]]] = {
            channel: [] for channel in CHANNEL_POOLS
        }
        rewritten: list[dict[str, object]] = []
        for row in rows:
            record = dict(row)
            beats = _stored_beat_count(record)
            if beats is None:
                rows_without_beats += 1
                rewritten.append(record)
                continue
            missed = _stored_hard_fail_count(record)
            channel, reason = channel_for(
                beats,
                missed_count=missed,
                hard_criteria_pass=_stored_hard_threshold_pass(record),
            )
            record["tier"] = channel
            record["tier_label"] = CHANNEL_LABELS.get(channel, NO_CHANNEL_LABEL)
            record["tier_reason"] = reason
            record["jp10_beat_count"] = beats
            record["hard_fail_count"] = "" if missed is None else int(missed)
            record["potential_candidate"] = channel in POTENTIAL_POOLS
            rewritten.append(record)
            if channel in channels:
                channels[channel].append(record)
            else:
                rows_unrouted += 1
        rows_read += len(rewritten)
        if not _write_rows_tolerant(property_path, rewritten):
            continue
        fieldnames = list(rewritten[0]) if rewritten else None
        for channel in CHANNEL_POOLS:
            selected = sorted(channels[channel], key=_channel_sort_key)
            pool_path = folder / POOL_FILE_NAMES[channel]
            if not _write_rows_tolerant(pool_path, selected, fieldnames):
                # A table held open elsewhere would leave a stale pool behind;
                # the caller sees it instead of a silently inconsistent folder.
                locked.append(str(pool_path))
            channel_counts[channel] += len(selected)
        retiered.append(str(property_path))
    return {
        "output_dir": str(run_root),
        "retiered_tables": retiered,
        "locked_files": locked,
        "rows_read": rows_read,
        "rows_without_beats": rows_without_beats,
        "rows_unrouted": rows_unrouted,
        "channel_counts": channel_counts,
    }


def normalize_run_outputs(root: str | Path) -> dict[str, object]:
    """Bring a run folder to the published contract.

    Every channel table is re-derived from the stored predictions, the
    manual-calculation pool is rebuilt from the tier tables, and the internal
    ``hard_threshold_verdict`` column is written out as the neutral
    ``screening_status``. Called by :func:`organize_run_outputs` after every
    campaign, so the folder on disk always matches the current contract.
    """
    run_root = Path(root)
    rewritten = 0
    for name in RETIRED_STAGE_FILES:
        for stale in run_root.rglob(name):
            try:
                stale.unlink()
                rewritten += 1
            except OSError:
                continue
    retiered = retier_run_outputs(run_root)
    iteration = retier_iteration_summary(run_root)
    rebuild_manual_review(run_root)
    normalize_stage_summaries(run_root)
    normalize_stage_tables(run_root)
    normalize_candidate_input_metadata(run_root)

    for path in sorted(run_root.rglob("*.csv")):
        parts = path.relative_to(run_root).parts
        if "analysis" in parts or "dft_inputs" in parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "hard_threshold_verdict" not in text:
            continue
        rows = _read_rows(path)
        if rows:
            public: list[dict[str, object]] = []
            for row in rows:
                record = dict(row)
                route = record.pop("hard_threshold_verdict", None)
                record["screening_status"] = screening_status(route)
                public.append(record)
            _write_rows(path, public)
        else:
            # Header-only table: rename the column in place.
            header = text.splitlines()[0].split(",")
            header = [
                "screening_status" if name == "hard_threshold_verdict" else name
                for name in header
            ]
            path.write_text(",".join(header) + "\n", encoding="utf-8")
        rewritten += 1
    return {
        "rewritten_tables": rewritten,
        "retiered": retiered,
        "iteration_ledger": iteration,
    }


def iteration_product_counts(run_root: Path) -> dict[str, object]:
    """Channel split of the iterative perturbation's products.

    Every product the iterative mode wrote carries the round that produced it in
    ``iteration_round``, so the split is a pure function of the stored rows. The
    round tags restart with each campaign, which makes a *single* round ambiguous
    once a folder holds more than one campaign, but the union of the tagged rows
    is exactly the set of iteration products - the ledger's own ``new_products``
    column sums to it - so the stage total stays exact.
    """

    per_round: dict[str, dict[str, int]] = {}
    aggregate: dict[str, int] = {channel: 0 for channel in CHANNEL_ORDER}
    tagged = 0
    property_path = run_root / PERTURBATION_PROPERTY_FILE
    if not property_path.exists():
        return {"per_round": per_round, "aggregate": aggregate, "tagged_products": 0,
                "per_round_products": {}}
    per_round_products: dict[str, int] = {}
    for row in _read_rows(property_path):
        tag = str(row.get("iteration_round") or "").strip()
        if not tag:
            continue
        channel = str(row.get("tier") or "").strip().lower()
        bucket = per_round.setdefault(tag, {name: 0 for name in CHANNEL_ORDER})
        per_round_products[tag] = per_round_products.get(tag, 0) + 1
        if channel in bucket:
            bucket[channel] += 1
            aggregate[channel] += 1
        tagged += 1
    return {
        "per_round": per_round,
        "per_round_products": per_round_products,
        "aggregate": aggregate,
        "tagged_products": tagged,
    }


def retier_iteration_summary(run_root: Path) -> dict[str, object]:
    """Bring the round ledger of the iterative mode to the current channels.

    Each ledger row keeps the per-round channel split. That split is re-derived
    from the products the round left in the folder, but only for the rounds whose
    tagged rows reproduce the row's own ``new_products`` count. Rounds that fail
    that test (folders whose rounds were re-tagged by a later campaign) keep the
    Tier 1 column and drop the other six channel columns instead of carrying a
    split that the folder cannot reproduce.
    """

    path = run_root / ITERATION_SUMMARY_FILE
    if not path.exists():
        return {"updated_rounds": 0, "cleared_rounds": 0, "rows": 0}
    rows = _read_rows(path)
    if not rows:
        return {"updated_rounds": 0, "cleared_rounds": 0, "rows": 0}
    counts = iteration_product_counts(run_root)
    per_round = counts["per_round"]
    products = counts["per_round_products"]
    channel_columns = [f"new_{channel}" for channel in CHANNEL_ORDER]
    updated = cleared = 0
    for row in rows:
        tag = str(row.get("round") or "").strip()
        bucket = per_round.get(tag)
        expected = int(_number(row.get("new_products")) or 0)
        if bucket is None or products.get(tag) != expected:
            for column in channel_columns:
                if column == f"new_{TIER1}":
                    continue
                if column in row:
                    row[column] = ""
            cleared += 1
            continue
        for column, channel in zip(channel_columns, CHANNEL_ORDER):
            if column in row:
                row[column] = int(bucket.get(channel, 0))
        updated += 1
    _write_rows(path, rows)
    return {"updated_rounds": updated, "cleared_rounds": cleared, "rows": len(rows)}


def rebuild_manual_review(run_root: Path) -> int:
    """Rebuild the manual-calculation pool from the tier tables.

    Pool = every Tier 1-3 molecule of the run (the seeds the campaign perturbed
    plus every Tier 1-3 product) together with the complete Potential 1 pool,
    with the channel kept on each row.  Runs during normalisation, so the pool
    always matches the channel tables of the folder.
    """
    tier_rows: list[dict[str, object]] = []
    for tier in ("tier1", "tier2", "tier3"):
        tier_rows.extend(_read_rows(run_root / f"{tier}_candidates.csv"))
    if not tier_rows:
        return 0
    seed_manifest = run_root / "perturbation_seeds.csv"
    seed_rows = _read_rows(seed_manifest) if seed_manifest.exists() else []
    # The Potential 1 pool belongs to the calculation list in full, across every
    # campaign of the run, so it is taken from the merged totals table.
    potential_rows: list[dict[str, object]] = []
    for candidate_path in (
        run_root / "analysis" / "totals" / "potential1_candidates_total.csv",
        run_root / "potential1_candidates.csv",
    ):
        if candidate_path.exists():
            potential_rows = _read_rows(candidate_path)
            break

    combined: list[dict[str, object]] = []
    seen: set[str] = set()
    for rows, origin in (
        (seed_rows, "perfect_symmetric"),
        (tier_rows, "perturbed_from_hits"),
        (potential_rows, "potential_pool"),
    ):
        for row in rows:
            smiles = str(row.get("canonical_smiles") or "").strip()
            tier = str(row.get("tier") or "").strip().lower()
            if not smiles or smiles in seen:
                continue
            if origin == "perturbed_from_hits" and tier not in {"tier1", "tier2", "tier3"}:
                continue
            if origin == "potential_pool" and tier != "potential1":
                continue
            seen.add(smiles)
            combined.append(
                {
                    **row,
                    "manual_review_id": "manual_%06d" % (len(combined) + 1),
                    "manual_review_status": "pending_manual_calculation",
                    "manual_review_tier": tier or str(row.get("seed_source_pool") or ""),
                    "candidate_origin": origin,
                }
            )
    combined.sort(
        key=lambda row: (
            str(row.get("manual_review_tier") or ""),
            -float(row.get("composite_ranking_score") or 0.0),
            str(row.get("canonical_smiles") or ""),
        )
    )
    target = run_root / "manual_review" / MANUAL_REVIEW_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(target, combined)
    return len(combined)


def organize_run_outputs(
    output_dir: str | Path,
    report_path: str | Path | None = None,
) -> dict[str, object]:
    """Organize a full run's outputs into an ``analysis/`` folder.

    Structure:
    - ``analysis/generation/<C*>`` : per-carbon generation files (split tables)
      plus ``generation_stage_report.txt`` stage report
    - ``analysis/perturbation/``   : perturbation outputs + manual review
      plus ``perturbation_stage_report.txt`` stage report
    - ``analysis/totals/``         : the channel pools merged across carbons
      (``tier1..tier4_candidates_total.csv`` with the full molecule rows plus
      ``tier_summary.csv`` counts), the whole-flow totals table, and the
      integrated ``final_data_report.txt``

    Only summary-level data (CSV/JSON) is copied; raw DFT inputs and per-seed
    metadata directories stay in the original output folder.
    """
    root = Path(output_dir)
    normalized = normalize_run_outputs(root)
    # An empty iteration scratch folder is left behind when a round finished and
    # its seed table was already merged; it carries no data, so it goes.
    scratch = root / "_iteration"
    try:
        if scratch.is_dir() and not any(scratch.iterdir()):
            scratch.rmdir()
    except OSError:
        pass
    analysis = root / "analysis"
    if analysis.exists():
        _remove_tree(analysis)
    # A file that is still open elsewhere can survive the removal; carry on and
    # let the copy step overwrite it instead of failing the whole run.
    analysis.mkdir(parents=True, exist_ok=True)
    generation_dir = analysis / "generation"
    totals_dir = analysis / "totals"
    perturbation_dir = analysis / "perturbation"
    perturbation_present = _perturbation_metrics(root) is not None
    folders = [generation_dir, totals_dir]
    if perturbation_present:
        folders.append(perturbation_dir)
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    copied = {"generation": 0, "perturbation": 0, "totals": 0}
    locked: list[str] = []

    for cdir in sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith("C")),
        key=lambda p: p.name,
    ):
        target = generation_dir / cdir.name
        target.mkdir(parents=True, exist_ok=True)
        for item in cdir.iterdir():
            if item.is_file():
                if _copy_file(item, target / item.name):
                    copied["generation"] += 1
                else:
                    locked.append(str(target / item.name))

    if perturbation_present:
        for item in root.iterdir():
            if item.is_file():
                name = item.name.lower()
                if "total" in name or "combined" in name:
                    if _copy_file(item, totals_dir / item.name):
                        copied["totals"] += 1
                    else:
                        locked.append(str(totals_dir / item.name))
                elif item.suffix.lower() in (".csv", ".json"):
                    if _copy_file(item, perturbation_dir / item.name):
                        copied["perturbation"] += 1
                    else:
                        locked.append(str(perturbation_dir / item.name))
        manual = root / "manual_review"
        if manual.is_dir():
            target = perturbation_dir / "manual_review"
            target.mkdir(parents=True, exist_ok=True)
            for item in manual.iterdir():
                if item.is_file():
                    if _copy_file(item, target / item.name):
                        copied["perturbation"] += 1
                    else:
                        locked.append(str(target / item.name))
    else:
        for item in root.iterdir():
            if item.is_file() and (
                "total" in item.name.lower() or "combined" in item.name.lower()
            ):
                if _copy_file(item, totals_dir / item.name):
                    copied["totals"] += 1
                else:
                    locked.append(str(totals_dir / item.name))

    generation_report = build_generation_stage_report(
        root,
        generation_dir / "generation_stage_report.txt",
    )
    tiers = build_tier_analysis(root, totals_dir)
    perturbation_report = None
    if perturbation_present:
        perturbation_report = build_perturbation_stage_report(
            root,
            perturbation_dir / "perturbation_stage_report.txt",
        )
    totals = build_flow_totals_table(root, totals_dir / "flow_totals.csv")
    final_report = build_final_data_report(
        generation_report["output_path"],
        perturbation_report["output_path"] if perturbation_report else None,
        totals_dir / "final_data_report.txt",
    )

    return {
        "output_dir": str(root),
        "analysis_dir": str(analysis),
        "structure": {
            "generation": str(generation_dir),
            "totals": str(totals_dir),
            **({"perturbation": str(perturbation_dir)} if perturbation_present else {}),
        },
        "copied": copied,
        "locked_files": locked,
        "normalized": normalized,
        "reports": {
            "generation_stage": generation_report["output_path"],
            "tier_summary": tiers["summary_path"],
            "perturbation_stage": (
                perturbation_report["output_path"] if perturbation_report else None
            ),
            "totals_table": totals["output_path"],
            "final_report": final_report["output_path"],
        },
    }


#: one table per channel: Tier 1-4 (every hard threshold met, split by how many of
#: the three compared properties beat JP-10) and Potential 1-3 (named by how many
#: hard thresholds are missed and how many properties still beat JP-10).
TIER_POOLS = (TIER1, TIER2, TIER3, TIER4)
POTENTIAL_POOLS = (POTENTIAL1, POTENTIAL2, POTENTIAL3)
CHANNEL_POOLS = CHANNEL_ORDER
POOL_FILE_NAMES = {
    channel: f"{channel}_candidates.csv" for channel in CHANNEL_ORDER
}
#: per-round table of the iterative perturbation mode, kept next to the pools
ITERATION_SUMMARY_FILE = "iteration_summary.csv"


def _number(value: object, default: float = 0.0) -> float:
    """Lenient float parse for a stored cell (default when blank/unparsable)."""

    try:
        if value is None or str(value).strip() == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)

ITERATION_TOTALS_FILE = "iteration_totals.csv"


def build_tier_analysis(root: str | Path, target_dir: str | Path) -> dict[str, object]:
    """Merge the channel pools into ``analysis/totals``.

    The per-carbon folders stay authoritative; this step gives the analysis tree
    one place with the merged pools and a count summary per stage and carbon,
    which is what a run-level review actually needs.
    """
    run_root = Path(root)
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    merged: dict[str, list[dict[str, object]]] = {
        channel: [] for channel in CHANNEL_POOLS
    }
    merged_smiles: dict[str, set[str]] = {channel: set() for channel in CHANNEL_POOLS}
    counts: list[dict[str, object]] = []

    carbon_dirs = (
        sorted(
            (
                p
                for p in run_root.iterdir()
                if p.is_dir() and p.name[:1] == "C" and p.name[1:].isdigit()
            ),
            key=lambda p: int(p.name[1:]),
        )
        if run_root.is_dir()
        else []
    )

    def collect_pools(source_dir: Path, stage: str, carbon: str) -> None:
        """Merge one stage's channel tables into the analysis tree."""

        for channel in CHANNEL_POOLS:
            rows = _read_rows(source_dir / POOL_FILE_NAMES[channel])
            counts.append(
                {
                    "stage": stage,
                    "carbon": carbon,
                    "tier": channel,
                    "count": len(rows),
                }
            )
            # A molecule can be both a generated candidate and a perturbation
            # product of the same run; the merged table keeps it once (the
            # earlier stage wins) so the totals, the count summary and the
            # whole-flow table all speak about molecules, not rows.
            for row in rows:
                smiles = str(row.get("canonical_smiles") or "").strip()
                if smiles and smiles in merged_smiles[channel]:
                    continue
                if smiles:
                    merged_smiles[channel].add(smiles)
                merged[channel].append(row)

    for cdir in carbon_dirs:
        collect_pools(cdir, "generation", cdir.name)

    perturbation_present = _perturbation_metrics(run_root) is not None
    if not carbon_dirs and not perturbation_present:
        # Single-carbon generation run: its pools live at the run root.
        collect_pools(run_root, "generation", run_root.name)

    if perturbation_present:
        collect_pools(run_root, "perturbation", "all")

    paths: dict[str, str] = {}
    locked: list[str] = []
    for channel in CHANNEL_POOLS:
        path = target / f"{channel}_candidates_total.csv"
        if not _write_rows_tolerant(path, merged[channel]):
            locked.append(str(path))
        paths[channel] = str(path)
        counts.append(
            {
                "stage": "total",
                "carbon": "all",
                "tier": channel,
                "count": len(merged[channel]),
            }
        )

    # Iterative perturbation: the round ledger is copied into totals, and the
    # count summary carries one aggregate row group for the whole iterative stage
    # (the round tags restart with every campaign, so a single round cannot be
    # separated reliably once a folder holds several campaigns).
    iteration_summary = run_root / ITERATION_SUMMARY_FILE
    if iteration_summary.exists():
        iteration_rows = _read_rows(iteration_summary)
        if iteration_rows and _write_rows_tolerant(
            target / ITERATION_TOTALS_FILE, iteration_rows
        ):
            paths["iteration"] = str(target / ITERATION_TOTALS_FILE)
        else:
            locked.append(str(target / ITERATION_TOTALS_FILE))
        iteration_counts = iteration_product_counts(run_root)
        for channel in CHANNEL_POOLS:
            counts.append(
                {
                    "stage": "perturbation_iteration",
                    "carbon": "all",
                    "tier": channel,
                    "count": int(iteration_counts["aggregate"].get(channel, 0)),
                }
            )

    summary_path = target / "tier_summary.csv"
    if not _write_rows_tolerant(summary_path, counts):
        locked.append(str(summary_path))
    return {
        "output_dir": str(target),
        "summary_path": str(summary_path),
        "paths": paths,
        "counts": counts,
        "locked_files": locked,
    }


#: candidate pools whose members become perturbation seeds (Tier 4 is discarded)
TIER_SEED_POOLS = (TIER1, TIER2, TIER3)

#: seed table built from the potential pools instead of the tier pools
POTENTIAL_SEED_TABLE = "perturbation_seeds_potential.csv"
#: share of each one-beaten potential pool (Potential 2, Potential 3) seeded
POTENTIAL_SEED_TOP_FRACTION = 0.10


def _potential_pool_rows(
    root: Path,
    filename: str,
) -> tuple[list[dict[str, object]], str]:
    """Rows of one potential pool, from the whole-run totals when they exist."""

    candidates = [
        root / "analysis" / "totals" / f"{Path(filename).stem}_total.csv",
        root / filename,
    ]
    carbon_dirs = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and path.name[:1] == "C" and path.name[1:].isdigit()
        ),
        key=lambda path: int(path.name[1:]),
    ) if root.is_dir() else []
    candidates.extend(carbon_dir / filename for carbon_dir in carbon_dirs)
    for path in candidates:
        if path.exists():
            rows = [dict(row) for row in _read_rows(path)]
            if rows:
                return rows, str(path)
    return [], ""


def _composite_score(row: dict[str, object]) -> float:
    try:
        return float(row.get("composite_ranking_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def build_potential_perturbation_seeds(
    output_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    one_item_fraction: float = POTENTIAL_SEED_TOP_FRACTION,
) -> dict[str, object]:
    """Write the second perturbation seed table of a run.

    Every Potential 1 molecule feeds it in full (it misses one hard threshold but
    beats JP-10 on the other two), and each one-beaten pool contributes its best
    ``one_item_fraction`` by composite score - Potential 2 (one threshold missed)
    and Potential 3 (two thresholds missed). The table sits next to
    ``perturbation_seeds_T123.csv`` and is written by the run flow itself, so the
    control panel can offer it without manual bookkeeping.
    """

    root = Path(output_dir)
    target = (
        Path(output_path) if output_path is not None else root / POTENTIAL_SEED_TABLE
    )
    two_item_rows, two_item_source = _potential_pool_rows(
        root, POOL_FILE_NAMES[POTENTIAL1]
    )
    fraction = min(max(float(one_item_fraction), 0.0), 1.0)
    one_item_pools: list[tuple[str, list[dict[str, object]], list[dict[str, object]], str]] = []
    for channel in (POTENTIAL2, POTENTIAL3):
        rows, source = _potential_pool_rows(root, POOL_FILE_NAMES[channel])
        ranked = sorted(rows, key=_composite_score, reverse=True)
        top_rows = (
            ranked[: max(1, int(round(len(ranked) * fraction)))] if ranked else []
        )
        one_item_pools.append((channel, ranked, top_rows, source))

    rows: list[dict[str, object]] = []
    seeded: list[tuple[str, list[dict[str, object]]]] = [
        (POTENTIAL1, two_item_rows),
        *[
            (f"{channel}_top{int(round(fraction * 100))}", top_rows)
            for channel, _, top_rows, _ in one_item_pools
        ],
    ]
    for pool, pool_rows in seeded:
        for row in pool_rows:
            if not str(row.get("canonical_smiles") or "").strip():
                continue
            rows.append({**row, "seed_source_pool": pool})

    selected = (
        mark_perturbation_seed_rows(rows, selection_basis="potential_pool")
        if rows
        else ()
    )
    if selected:
        _write_rows(target, list(selected))
    return {
        "output_dir": str(root),
        "output_path": str(target) if selected else "",
        "potential1_rows": len(two_item_rows),
        "potential2_rows": len(one_item_pools[0][1]),
        "potential2_selected": len(one_item_pools[0][2]),
        "potential3_rows": len(one_item_pools[1][1]),
        "potential3_selected": len(one_item_pools[1][2]),
        "one_item_fraction": fraction,
        "seed_rows": len(selected),
        "pool_sources": [
            source
            for source in (two_item_source, *(s for *_, s in one_item_pools))
            if source
        ],
        "warning": "" if selected else "no potential pool rows found",
    }


def build_t123_perturbation_seeds(
    output_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Write the perturbation seed table from the Tier 1-3 pools of a run.

    The two-axis routing stores every candidate that matches or beats JP-10
    on the hard-screened properties in the ``tier1`` / ``tier2`` / ``tier3``
    pools.  Those three pools are the molecules the perturbation stage is meant
    to explore, so this step re-runs
    the production seed writer (:func:`write_predicted_perturbation_seeds`) over
    them and stores the result inside the run folder as
    ``perturbation_seeds_T123.csv``.
    """
    root = Path(output_dir)
    target = (
        Path(output_path)
        if output_path is not None
        else root / "perturbation_seeds_T123.csv"
    )

    carbon_dirs = sorted(
        (
            p
            for p in root.iterdir()
            if p.is_dir() and p.name[:1] == "C" and p.name[1:].isdigit()
        ),
        key=lambda p: int(p.name[1:]),
    ) if root.is_dir() else []

    sources: list[tuple[str, Path, str]] = []
    if carbon_dirs:
        for cdir in carbon_dirs:
            for tier in TIER_SEED_POOLS:
                pool = cdir / f"{tier}_candidates.csv"
                if pool.exists():
                    sources.append((cdir.name, pool, tier))
    else:
        for tier in TIER_SEED_POOLS:
            pool = root / f"{tier}_candidates.csv"
            if pool.exists():
                sources.append((root.name, pool, tier))

    rows: list[dict[str, object]] = []
    per_carbon: dict[str, int] = {}
    per_pool: dict[str, int] = {}
    skipped_missing_smiles = 0
    for carbon, path, tier in sources:
        for row in _read_rows(path):
            if not str(row.get("canonical_smiles") or "").strip():
                skipped_missing_smiles += 1
                continue
            record = dict(row)
            route = record.pop("hard_threshold_verdict", None)
            record["screening_status"] = screening_status(route)
            record["seed_source_pool"] = tier
            rows.append(record)
            per_carbon[carbon] = per_carbon.get(carbon, 0) + 1
            per_pool[tier] = per_pool.get(tier, 0) + 1

    selected = write_predicted_perturbation_seeds(rows, target) if rows else ()
    fuel_hits = _merge_known_fuel_hit_tables(root, root / "known_fuel_hits_total.csv")
    return {
        "output_dir": str(root),
        "output_path": str(target) if rows else "",
        "pool_files": [str(path) for _, path, _ in sources],
        "per_carbon_counts": per_carbon,
        "per_pool_counts": per_pool,
        "pool_rows": len(rows),
        "unique_seeds": len(selected),
        "skipped_missing_smiles": skipped_missing_smiles,
        "known_fuel_hits": fuel_hits,
        "warning": "" if rows else "no Tier 1-3 candidate pools found",
    }


def _merge_known_fuel_hit_tables(
    root: Path,
    output_path: Path,
) -> dict[str, object]:
    """Merge per-carbon classic-fuel hit tables into one total table."""
    paths: list[Path] = []
    root_level = root / "known_fuel_hits.csv"
    if root_level.exists():
        paths.append(root_level)
    if root.is_dir():
        for candidate in sorted(root.iterdir()):
            if candidate.is_dir() and candidate.name.startswith("C"):
                hit = candidate / "known_fuel_hits.csv"
                if hit.exists():
                    paths.append(hit)
    aggregate: dict[str, dict[str, object]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = str(row.get("fuel_smiles") or "").strip()
                if not key:
                    continue
                entry = aggregate.setdefault(
                    key,
                    {
                        "fuel_name": str(row.get("fuel_name") or ""),
                        "fuel_smiles": key,
                        "hit_count": 0,
                        "hit_molecules": [],
                    },
                )
                entry["hit_count"] = int(entry["hit_count"]) + int(
                    str(row.get("hit_count") or 0) or 0
                )
                for molecule in str(row.get("hit_molecules") or "").split(";"):
                    if molecule and molecule not in entry["hit_molecules"]:
                        entry["hit_molecules"].append(molecule)
    if aggregate:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["fuel_name", "fuel_smiles", "hit_count", "hit_molecules"])
            for key in sorted(aggregate):
                entry = aggregate[key]
                writer.writerow(
                    [
                        entry["fuel_name"],
                        entry["fuel_smiles"],
                        entry["hit_count"],
                        ";".join(entry["hit_molecules"]),
                    ]
                )
    return {
        "output_path": str(output_path),
        "source_tables": [str(p) for p in paths],
        "total_fuels_hit": len(aggregate),
    }
#: potential pools derived from the (thresholds missed, properties beaten) profile
#: of each scored molecule (see tier_routing)
