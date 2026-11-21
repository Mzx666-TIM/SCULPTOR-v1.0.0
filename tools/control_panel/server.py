from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
JOB_ROOT = REPO_ROOT / "result" / "control_panel_jobs"
PYTHON_EXE = (
    REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if (REPO_ROOT / ".venv" / "Scripts" / "python.exe").exists()
    else Path(sys.executable)
)


@dataclass
class Job:
    id: str
    mode: str
    commands: list[list[str]]
    display_command: str
    status: str = "queued"
    returncode: int | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    log_path: str = ""
    output_dir: str = ""
    result_files: list[str] = field(default_factory=list)
    last_lines: list[str] = field(default_factory=list)
    current_stage: str = "prepare"
    current_stage_message: str = ""
    completed_commands: int = 0
    current_command_index: int = 0
    process_id: int | None = None
    stop_requested: bool = False

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["elapsed_seconds"] = self.elapsed_seconds
        payload["stages"] = stage_statuses(
            self.mode,
            self.status,
            self.current_stage,
            self.current_stage_message,
        )
        return payload

    @property
    def elapsed_seconds(self) -> float:
        end = self.ended_at or time.time()
        start = self.started_at or self.created_at
        return round(max(0.0, end - start), 1)


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def main() -> int:
    host = "127.0.0.1"
    port = int(os.environ.get("FUEL_PANEL_PORT", "8765"))
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    load_persisted_jobs()
    # Pre-warm the recent-outputs scan in the background so the first
    # /api/defaults call does not block on a full directory walk.
    threading.Thread(
        target=lambda: recent_paths(REPO_ROOT / "result"),
        daemon=True,
    ).start()
    server = ThreadingHTTPServer((host, port), Handler)
    safe_print(f"Fuel design control panel: http://{host}:{port}")
    server.serve_forever()
    return 0


def safe_print(message: str) -> None:
    try:
        if sys.stdout is not None:
            print(message)
    except OSError:
        return


class Handler(BaseHTTPRequestHandler):
    server_version = "FuelDesignControlPanel/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_file(STATIC_ROOT / "index.html")
            return
        if parsed.path.startswith("/static/"):
            rel = parsed.path.removeprefix("/static/")
            self._serve_file(STATIC_ROOT / unquote(rel))
            return
        if parsed.path == "/api/defaults":
            self._json(defaults_payload())
            return
        if parsed.path == "/api/jobs":
            with JOBS_LOCK:
                self._json({"jobs": [job.as_dict() for job in JOBS.values()]})
            return
        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                self._error(HTTPStatus.NOT_FOUND, "job not found")
                return
            refresh_job_results(job)
            self._json(job.as_dict())
            return
        if parsed.path == "/api/preview":
            query = parse_qs(parsed.query)
            path = query.get("path", [""])[0]
            try:
                self._json(preview_file(path))
            except ValueError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if parsed.path == "/api/browse":
            query = parse_qs(parsed.query)
            path = query.get("path", [""])[0]
            try:
                self._json(browse_path(path))
            except ValueError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/pick-folder":
                initial = str(payload.get("initial") or "")
                self._json(pick_folder(initial))
                return
            if parsed.path == "/api/build-command":
                command, display, output_dir = build_command(payload)
                self._json(
                    {
                        "command": command,
                        "display_command": display,
                        "output_dir": output_dir,
                    }
                )
                return
            if parsed.path == "/api/perturbation-inputs":
                self._json(perturbation_inputs(payload.get("output_dir")))
                return
            if parsed.path == "/api/run":
                command, display, output_dir = build_command(payload)
                job = create_job(
                    mode=str(payload.get("mode", "")),
                    command=command,
                    display_command=display,
                    output_dir=output_dir,
                )
                self._json(job.as_dict())
                return
            if parsed.path == "/api/resume":
                job_id = str(payload.get("job_id") or "")
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                if job is None:
                    raise ValueError("job not found")
                ensure_no_active_job()
                if job.status not in ("failed", "interrupted"):
                    raise ValueError("only failed or interrupted jobs can be resumed")
                job.status = "queued"
                job.returncode = None
                job.ended_at = None
                persist_job(job)
                threading.Thread(target=run_job, args=(job,), daemon=True).start()
                self._json(job.as_dict())
                return
            if parsed.path == "/api/stop":
                self._json(stop_job(str(payload.get("job_id") or "")))
                return
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        if not raw:
            return {}
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status=status)

    def _serve_file(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            if STATIC_ROOT.resolve() not in resolved.parents and resolved != STATIC_ROOT.resolve():
                self._error(HTTPStatus.FORBIDDEN, "forbidden")
                return
            if not resolved.exists() or not resolved.is_file():
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            content = resolved.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type(resolved))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except OSError:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "failed to read file")


def run_has_channel_tables(output_dir: object) -> bool:
    """Whether a run folder already carries the channel pool tables."""

    root = Path(str(output_dir or ""))
    if not root.is_dir():
        return False
    return any(
        (root / name).is_file()
        for name in (
            "tier1_candidates.csv",
            "tier2_candidates.csv",
            "tier3_candidates.csv",
            "tier4_candidates.csv",
            "potential1_candidates.csv",
            "potential2_candidates.csv",
            "potential3_candidates.csv",
        )
    )


def latest_run_dir() -> str:
    """Newest result folder that looks like a pipeline run (has C* folders)."""
    root = REPO_ROOT / "result"
    candidates: list[Path] = []
    if root.is_dir():
        for path in root.iterdir():
            if not path.is_dir():
                continue
            try:
                has_carbon = any(
                    child.is_dir()
                    and child.name[:1] == "C"
                    and child.name[1:].isdigit()
                    for child in path.iterdir()
                )
            except OSError:
                continue
            if has_carbon:
                candidates.append(path)
    if not candidates:
        return ""
    newest = max(candidates, key=lambda item: item.stat().st_mtime)
    return str(newest)


def perturbation_inputs(output_dir: object) -> dict[str, object]:
    """Seed table and temporary libraries the perturbation stage should read.

    A run folder carries two seed tables: the Tier 1-3 pools
    (``perturbation_seeds_T123.csv``) and the potential pools
    (``perturbation_seeds_potential.csv``). The generation chain writes both, so
    the panel can offer either one as the perturbation seed source.
    """
    value = str(output_dir or "").strip().rstrip("\\/")
    if not value:
        return {
            "output_dir": "",
            "seed_table": "",
            "seed_source": "missing",
            "potential_seed_table": "",
            "potential_seed_source": "missing",
            "generation_temporary": "",
            "perturbation_temporary": "",
            "t123_table": "",
        }
    root = Path(value)
    t123 = root / "perturbation_seeds_T123.csv"
    potential = root / "perturbation_seeds_potential.csv"
    seed_source = "tier123" if t123.exists() else "missing"
    return {
        "output_dir": str(root),
        "seed_table": str(t123),
        "seed_source": seed_source,
        "potential_seed_table": str(potential),
        "potential_seed_source": "potential" if potential.exists() else "missing",
        "stored_phase_fusion_schema": stored_phase_fusion_schema(root),
        "generation_temporary": str(root / "generation_temporary_smiles.txt"),
        "perturbation_temporary": str(root / "perturbation_temporary_smiles.txt"),
        "t123_table": str(t123),
    }


def _fusion_schema_of_table(path: Path) -> str:
    """Fusion-head version of a stored table, from its first data row."""

    if not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                feature_set = str(
                    row.get("melting_point_k_phase_fusion_feature_set") or ""
                ).strip().lower()
                if feature_set.endswith("ens3d"):
                    return "schema3_mlp"
                if feature_set.endswith("ens2d"):
                    return "schema2_mlp"
                return ""
    except OSError:
        return ""
    return ""


def stored_phase_fusion_schema(root: Path) -> str:
    """Fusion-head version the products already in a run folder were made with.

    Re-running into a folder that holds earlier products has to use the same
    fusion head, otherwise the merged tables would compare melting points from
    two different models. The panel uses this to pre-select the matching
    version, and the perturbation pipeline refuses a mismatch outright.
    """

    tables = [
        root / "perturbation_property_predictions.csv",
        root / "analysis" / "totals" / "tier1_candidates_total.csv",
    ]
    tables.extend(
        carbon_dir / "property_predictions.csv"
        for carbon_dir in sorted(root.glob("C*"))
    )
    for path in tables:
        schema = _fusion_schema_of_table(path)
        if schema:
            return schema
    return ""


def rescore_input_path() -> str:
    """Default rescore input: the newest run's Tier-1 pool."""
    newest = latest_run_dir()
    if not newest:
        return ""
    root = Path(newest)
    for candidate in (
        root / "analysis" / "totals" / "tier1_candidates_total.csv",
        root / "analysis" / "tiers" / "tier1_candidates_total.csv",
        root / "tier1_candidates.csv",
    ):
        if candidate.exists():
            return str(candidate)
    return ""


def defaults_payload() -> dict[str, object]:
    return {
        "repo_root": str(REPO_ROOT),
        "python_exe": str(PYTHON_EXE),
        "defaults": {
            "checkpoint": existing_or_default("Generation/bond_completion_2/model/grouped_20260925/stage2_bond_completion_grouped_seed2026.pt"),
            "semi_base_checkpoint": existing_or_default("Generation/bond_completion_1/model/grouped_20260925/stage1_semi_base_grouped_seed2026.pt"),
            "known_smiles": existing_or_default("Generation/screening/data/known_smiles_total_reference.txt"),
            "e3nn_checkpoint": existing_or_default("Evaluation_Iteration/evaluation/model/e3nn/e3nn_geometry_scorer_g_new.pt"),
            "sta_model": existing_or_default("Evaluation_Iteration/evaluation/model/STA/sta_820k_v2/sta_gdb_density_paper.pt"),
            "sta_density_model": existing_or_default("Evaluation_Iteration/evaluation/model/STA/sta_820k_v2/sta_gdb_density_paper.pt"),
            "sta_heating_value_model": existing_or_default("Evaluation_Iteration/evaluation/model/STA/sta_820k_v2/sta_gdb_formation_enthalpy_paper.pt"),
            "sta_melting_point_model": existing_or_default("Evaluation_Iteration/evaluation/model/STA/sta_820k_v2/sta_gdb_melting_point_paper.pt"),
            "sta_boiling_point_model": existing_or_default("Evaluation_Iteration/evaluation/model/STA/sta_820k_v2/sta_gdb_boiling_point_paper.pt"),
            "sta_flash_point_model": existing_or_default("Evaluation_Iteration/evaluation/model/STA/sta_820k_v2/sta_gdb_flash_point_paper.pt"),
            "sta_formation_enthalpy_model": existing_or_default("Evaluation_Iteration/evaluation/model/STA/sta_820k_v2/sta_gdb_formation_enthalpy_paper.pt"),
            "sta_isp_model": optional_existing("Evaluation_Iteration/evaluation/model/STA/sta_820k_v2/sta_gdb_isp_paper.pt"),
            "phase_fusion_schema_version": "schema2_mlp",
            "phase_fusion_melting_point_model": existing_or_default("Evaluation_Iteration/evaluation/model/schema 2/phase_fusion_gdb820k_mlp_ens2d/melting_point_k_fusion.joblib"),
            "phase_fusion_boiling_point_model": existing_or_default("Evaluation_Iteration/evaluation/model/schema 2/phase_fusion_gdb820k_mlp_ens2d/boiling_point_k_fusion.joblib"),
            "phase_fusion_flash_point_model": existing_or_default("Evaluation_Iteration/evaluation/model/schema 2/phase_fusion_gdb820k_mlp_ens2d/flash_point_k_fusion.joblib"),
            "last_run_dir": latest_run_dir(),
            "rescore_input": rescore_input_path(),
        },
        "recent_outputs": recent_paths(REPO_ROOT / "result"),
    }


def build_command(payload: dict[str, object]) -> tuple[list[list[str]], str, str]:
    mode = str(payload.get("mode") or "")
    args = payload.get("params")
    if not isinstance(args, dict):
        raise ValueError("params must be an object")
    # The device switch is a top-level CLI option, so it is prepended to every
    # command the panel launches and therefore also reaches the child workers.
    device_mode = str(args.get("device_mode") or "auto").strip().lower()
    if device_mode not in ("auto", "cpu", "gpu"):
        device_mode = "auto"
    command_prefix = [str(PYTHON_EXE), "-m", "fuel_design.cli",
                      "--device-mode", device_mode]
    output_dir = ""
    if mode == "symmetric":
        commands = build_symmetric_commands(args, command_prefix)
        output_dir = str(args.get("output_dir") or "")
    elif mode == "perturbation":
        commands = [
            command_prefix
            + build_perturbation_args(
                args,
                allow_replace=not bool(args.get("iterative_perturbation", False)),
            )
        ]
        output_dir = str(args.get("output_dir") or "")
        if (
            bool(args.get("iterative_perturbation", False))
            and output_dir
            and not run_has_channel_tables(output_dir)
        ):
            # Iterative mode only re-samples the leads a folder already holds, so
            # a folder straight from generation (its pools live in C*/, and the
            # run root is still empty) needs the first full campaign in front of
            # it - the same two-step order the production runs use.
            commands.insert(
                0,
                command_prefix
                + build_perturbation_args({**args, "iterative_perturbation": False}),
            )
        if output_dir:
            # After the full flow, organize total/split tables into analysis/ for
            # review and refresh the potential-pool seed table from them.
            commands.append(
                command_prefix
                + ["organize-run-outputs", "--output-dir", output_dir]
            )
            commands.append(
                command_prefix
                + ["build-potential-seeds", "--output-dir", output_dir]
            )
    elif mode == "rescore":
        commands = [command_prefix + build_rescore_args(args)]
        output_dir = str(Path(str(args.get("output") or "")).parent)
    else:
        raise ValueError(f"unknown mode: {mode}")
    env_prefix = f"$env:PYTHONPATH='{REPO_ROOT}'; "
    display = "\n".join(env_prefix + ps_join(command) for command in commands)
    return commands, display, output_dir


def build_symmetric_commands(args: dict[str, object], command_prefix: list[str]) -> list[list[str]]:
    required(args, "checkpoint", "known_smiles", "output_dir")
    carbon_min = int_value(args, "carbon_min", int_value(args, "total_carbons", 13))
    carbon_max = int_value(args, "carbon_max", carbon_min)
    if carbon_min <= 0 or carbon_max <= 0:
        raise ValueError("carbon range must be positive")
    if carbon_min > carbon_max:
        raise ValueError("carbon_min cannot be greater than carbon_max")
    if carbon_max > 17:
        raise ValueError("carbon_max cannot be greater than 17")
    output_dir = Path(str(args["output_dir"]))
    commands: list[list[str]] = []
    for total_carbons in range(carbon_min, carbon_max + 1):
        # Every carbon gets its own ``C<n>`` folder, also when the run covers a
        # single carbon: the perturbation stage owns the run-root pools, so a
        # flat single-carbon run would lose its generation channels as soon as a
        # perturbation campaign runs in the same folder.
        per_carbon_output = output_dir / f"C{total_carbons}"
        commands.append(
            command_prefix
            + build_symmetric_args(
                args,
                total_carbons=total_carbons,
                output_dir=str(per_carbon_output),
            )
        )
    # Newest architecture: the perturbation stage explores the Tier 1-3 pools, so
    # the run folder must carry that seed table (this step also merges the
    # per-carbon classic-fuel hit tables).
    commands.append(
        command_prefix
        + [
            "build-t123-seeds",
            "--output-dir",
            str(output_dir),
        ]
    )
    # Generate the analysis/ folder (generation + totals) as soon as
    # generation finishes; perturbation data is added when it exists.
    commands.append(
        command_prefix
        + ["organize-run-outputs", "--output-dir", str(output_dir)]
    )
    # The potential-pool seed table is derived from those totals, so it is
    # written after them and stays in step with every generation run.
    commands.append(
        command_prefix
        + ["build-potential-seeds", "--output-dir", str(output_dir)]
    )
    return commands


def build_symmetric_args(args: dict[str, object], *, total_carbons: int, output_dir: str) -> list[str]:
    required(args, "checkpoint", "known_smiles", "output_dir")
    result = [
        "run-pipeline",
        "--checkpoint",
        str(args["checkpoint"]),
        "--known-smiles",
        str(args["known_smiles"]),
        *(
            ["--temporary-smiles", str(args["temporary_smiles"])]
            if args.get("temporary_smiles")
            else []
        ),
        *(["--known-fuel-check"] if args.get("known_fuel_check") else []),
        "--output-dir",
        output_dir,
        "--total-carbons",
        str(total_carbons),
        "--samples",
        str(int_value(args, "samples", 200)),
        "--iterations",
        str(int_value(args, "iterations", 1)),
        "--threshold",
        str(float_value(args, "threshold", 0.94)),
        "--temperature",
        str(float_value(args, "temperature", 1.0)),
        "--top-k",
        str(int_value(args, "top_k", 2)),
        "--novelty-threshold",
        str(float_value(args, "novelty_threshold", 0.75)),
        "--density-min",
        str(float_value(args, "density_min", 0.85)),
        "--heating-value-min",
        str(float_value(args, "heating_value_min", 38.0)),
        "--melting-point-max",
        str(float_value(args, "melting_point_max", 273.15)),
        "--boiling-point-min",
        str(float_value(args, "boiling_point_min", 450.0)),
        "--flash-point-min",
        str(float_value(args, "flash_point_min", 310.0)),
        "--volumetric-heating-value-min",
        str(float_value(args, "volumetric_heating_value_min", 32.3)),
        "--missing-property-penalty",
        str(float_value(args, "missing_property_penalty", 0.15)),
        "--max-sa-score",
        str(float_value(args, "max_sa_score", 4.5)),
        "--screen-workers",
        str(int(float_value(args, "screen_workers", 1))),
        "--property-workers",
        str(int(float_value(args, "property_workers", 1))),
        "--reproducibility-scope",
        str(args.get("reproducibility_scope") or "tier12"),
        *(["--no-property-cache"] if args.get("disable_property_cache") else []),
        "--feedback-strength",
        str(float_value(args, "feedback_strength", 1.25)),
        "--property-feedback-warmup-iterations",
        str(int_value(args, "property_feedback_warmup_iterations", 20)),
        "--property-feedback-warmup-weight",
        str(float_value(args, "property_feedback_warmup_weight", 0.35)),
        "--property-feedback-weight",
        str(float_value(args, "property_feedback_weight", 0.70)),
        "--property-reward-slope",
        str(float_value(args, "property_reward_slope", 2.5)),
        "--partition-feedback-gain",
        str(float_value(args, "partition_feedback_gain", 6.0)),
        "--edge-feedback-gain",
        str(float_value(args, "edge_feedback_gain", 2.0)),
        "--feedback-exploration-fraction",
        str(float_value(args, "feedback_exploration_fraction", 0.12)),
        "--uncertainty-scale",
        str(float_value(args, "uncertainty_scale", 1.0)),
    ]
    add_optional(result, "--semi-base-checkpoint", args.get("semi_base_checkpoint"))
    add_optional(result, "--semi-base-threshold", args.get("semi_base_threshold", 0.73))
    add_objective_args(result, args)
    if bool(args.get("feedback_reward", True)):
        result.append("--feedback-reward")
    if not bool(args.get("online_property_feedback", True)):
        result.append("--disable-online-property-feedback")
    add_optional(result, "--seed", args.get("seed"))
    add_geometry_model_args(result, args)
    add_optional(result, "--e3nn-property-checkpoint", args.get("e3nn_property_checkpoint"))
    add_optional(result, "--e3nn-property-name", args.get("e3nn_property_name"))
    add_property_model_selection_args(result, args)
    add_heating_model_args(result, args)
    add_melting_model_args(result, args)
    if (
        not bool(args.get("use_density_screen", True))
        or not property_model_is_enabled(args, "density_model", "sta_density")
    ):
        result.append("--ignore-density")
    if (
        bool(args.get("ignore_heating_value", False))
        or not bool(args.get("use_heating_value_screen", True))
        or not property_model_is_enabled(args, "heating_model", "sta_heating")
    ):
        result.append("--ignore-heating-value")
    if bool(args.get("use_melting_point_screen", True)) and property_model_is_enabled(
        args, "melting_model", "phase_fusion_melting"
    ):
        result.append("--use-melting-point-screen")
    if bool(args.get("use_boiling_point_screen", False)) and property_model_is_enabled(
        args, "boiling_point_model", "phase_fusion_boiling"
    ):
        result.append("--use-boiling-point-screen")
    if bool(args.get("use_flash_point_screen", False)) and property_model_is_enabled(
        args, "flash_point_model", "phase_fusion_flash"
    ):
        result.append("--use-flash-point-screen")
    if bool(args.get("use_volumetric_heating_value_screen", False)):
        result.append("--use-volumetric-heating-value-screen")
    if bool(args.get("skip_dft_input_export", False)):
        result.append("--skip-dft-input-export")
    return result


def add_objective_args(result: list[str], args: dict[str, object]) -> None:
    mode = str(args.get("objective_mode") or "property_maximize")
    target_mode = mode == "target_property"
    volumetric_mode = bool(
        args.get("use_volumetric_heating_value_screen", False)
    )
    density_available = property_model_is_enabled(
        args, "density_model", "sta_density"
    )
    heating_available = property_model_is_enabled(
        args, "heating_model", "sta_heating"
    )
    melting_available = property_model_is_enabled(
        args, "melting_model", "phase_fusion_melting"
    )
    boiling_available = property_model_is_enabled(
        args, "boiling_point_model", "phase_fusion_boiling"
    )
    flash_available = property_model_is_enabled(
        args, "flash_point_model", "phase_fusion_flash"
    )
    if volumetric_mode and not (density_available and heating_available):
        raise ValueError("volumetric heating value mode requires both density and mass heating value prediction")
    density_selected = density_available and (
        bool(args.get("use_density_screen", True))
        or (target_mode and bool(args.get("select_target_density", True)))
    )
    heating_selected = heating_available and (
        bool(args.get("use_heating_value_screen", True))
        or (target_mode and bool(args.get("select_target_heating_value", True)))
    )
    melting_selected = melting_available and (
        bool(args.get("use_melting_point_screen", True))
        or (target_mode and bool(args.get("select_target_melting_point", False)))
    )
    boiling_selected = boiling_available and (
        bool(args.get("use_boiling_point_screen", False))
        or (target_mode and bool(args.get("select_target_boiling_point", False)))
    )
    flash_selected = flash_available and (
        bool(args.get("use_flash_point_screen", False))
        or (target_mode and bool(args.get("select_target_flash_point", False)))
    )
    density_weight = (
        0.0
        if volumetric_mode or not density_selected
        else float_value(args, "density_score_weight", 1.0)
    )
    heating_weight = (
        0.0
        if volumetric_mode or not heating_selected
        else float_value(args, "heating_value_score_weight", 1.0)
    )
    requested_volumetric_weight = float_value(
        args,
        "volumetric_heating_value_score_weight",
        1.0,
    )
    volumetric_weight = (
        requested_volumetric_weight
        if volumetric_mode and requested_volumetric_weight > 0.0
        else 1.0 if volumetric_mode else 0.0
    )
    result += [
        "--objective-mode",
        mode,
        "--objective-relative-window",
        str(float_value(args, "objective_relative_window", 0.15)),
        "--property-score-weight",
        str(float_value(args, "property_score_weight", 0.65)),
        "--e3nn-geometry-score-weight",
        str(float_value(args, "e3nn_geometry_score_weight", 0.10)),
        "--e3nn-symmetry-score-weight",
        str(float_value(args, "e3nn_symmetry_score_weight", 0.10)),
        "--sa-score-weight",
        "0.0",
        "--novelty-score-weight",
        str(float_value(args, "novelty_score_weight", 0.15)),
        "--density-score-weight",
        str(density_weight),
        "--heating-value-score-weight",
        str(heating_weight),
        "--volumetric-heating-value-score-weight",
        str(volumetric_weight),
        "--melting-point-score-weight",
        str(
            float_value(args, "melting_point_score_weight", 1.0)
            if melting_selected
            else 0.0
        ),
        "--boiling-point-score-weight",
        str(
            float_value(args, "boiling_point_score_weight", 1.0)
            if boiling_selected
            else 0.0
        ),
        "--flash-point-score-weight",
        str(
            float_value(args, "flash_point_score_weight", 1.0)
            if flash_selected
            else 0.0
        ),
        "--property-sa-score-weight",
        str(float_value(args, "property_sa_score_weight", 1.0)),
        "--structure-validity-score-weight",
        str(float_value(args, "structure_validity_score_weight", 1.0)),
        "--formation-enthalpy-score-weight",
        "0.0",
        "--molar-volume-score-weight",
        "0.0",
        "--isp-score-weight",
        "0.0",
    ]
    if (
        target_mode
        and density_available
        and bool(args.get("select_target_density", True))
        and not volumetric_mode
    ):
        add_optional(result, "--target-density", args.get("target_density"))
    if (
        target_mode
        and heating_available
        and bool(args.get("select_target_heating_value", True))
        and not volumetric_mode
    ):
        add_optional(result, "--target-heating-value", args.get("target_heating_value"))
    if (
        target_mode
        and bool(args.get("select_target_volumetric_heating_value", False))
        and volumetric_mode
    ):
        add_optional(
            result,
            "--target-volumetric-heating-value",
            args.get("target_volumetric_heating_value"),
        )
    if target_mode and melting_available and bool(
        args.get("select_target_melting_point", False)
    ):
        add_optional(result, "--target-melting-point", args.get("target_melting_point"))
    if target_mode and boiling_available and bool(
        args.get("select_target_boiling_point", False)
    ):
        add_optional(result, "--target-boiling-point", args.get("target_boiling_point"))
    if target_mode and flash_available and bool(
        args.get("select_target_flash_point", False)
    ):
        add_optional(result, "--target-flash-point", args.get("target_flash_point"))


def build_perturbation_args(
    args: dict[str, object],
    *,
    allow_replace: bool = True,
) -> list[str]:
    required(args, "finalized_hits", "output_dir")
    volumetric_mode = bool(
        args.get("use_volumetric_heating_value_screen", False)
    )
    density_enabled = (
        bool(args.get("use_density_screen", True))
        and property_model_is_enabled(args, "density_model", "sta_density")
    )
    heating_enabled = (
        bool(args.get("use_heating_value_screen", True))
        and property_model_is_enabled(args, "heating_model", "sta_heating")
    )
    melting_enabled = (
        bool(args.get("use_melting_point_screen", True))
        and property_model_is_enabled(args, "melting_model", "phase_fusion_melting")
    )
    boiling_enabled = (
        bool(args.get("use_boiling_point_screen", False))
        and property_model_is_enabled(args, "boiling_point_model", "phase_fusion_boiling")
    )
    flash_enabled = (
        bool(args.get("use_flash_point_screen", False))
        and property_model_is_enabled(args, "flash_point_model", "phase_fusion_flash")
    )
    if volumetric_mode and not (
        property_model_is_enabled(args, "density_model", "sta_density")
        and property_model_is_enabled(args, "heating_model", "sta_heating")
    ):
        raise ValueError("volumetric heating value mode requires both density and mass heating value prediction")
    density_weight = (
        0.0
        if volumetric_mode or not density_enabled
        else float_value(args, "density_score_weight", 1.0)
    )
    heating_weight = (
        0.0
        if volumetric_mode or not heating_enabled
        else float_value(args, "heating_value_score_weight", 1.0)
    )
    requested_volumetric_weight = float_value(
        args,
        "volumetric_heating_value_score_weight",
        1.0,
    )
    volumetric_weight = (
        requested_volumetric_weight
        if volumetric_mode and requested_volumetric_weight > 0.0
        else 1.0 if volumetric_mode else 0.0
    )
    result = [
        "run-perturbation-pipeline",
        "--finalized-hits",
        str(args["finalized_hits"]),
        "--output-dir",
        str(args["output_dir"]),
        "--samples",
        str(int_value(args, "samples", 100)),
        "--seed",
        str(int_value(args, "seed", 16)),
        "--novelty-threshold",
        str(float_value(args, "novelty_threshold", 0.75)),
        "--density-min",
        str(float_value(args, "density_min", 0.85)),
        "--heating-value-min",
        str(float_value(args, "heating_value_min", 38.0)),
        "--melting-point-max",
        str(float_value(args, "melting_point_max", 273.15)),
        "--boiling-point-min",
        str(float_value(args, "boiling_point_min", 450.0)),
        "--flash-point-min",
        str(float_value(args, "flash_point_min", 310.0)),
        "--volumetric-heating-value-min",
        str(float_value(args, "volumetric_heating_value_min", 32.3)),
        "--missing-property-penalty",
        str(float_value(args, "missing_property_penalty", 0.15)),
        "--max-sa-score",
        str(float_value(args, "max_sa_score", 4.5)),
        "--screen-workers",
        str(int(float_value(args, "screen_workers", 1))),
        "--property-workers",
        str(int(float_value(args, "property_workers", 1))),
        "--reproducibility-scope",
        str(args.get("reproducibility_scope") or "tier12"),
        *(["--no-property-cache"] if args.get("disable_property_cache") else []),
        "--property-score-weight",
        str(float_value(args, "property_score_weight", 0.70)),
        "--e3nn-geometry-score-weight",
        str(float_value(args, "e3nn_geometry_score_weight", 0.15)),
        "--e3nn-symmetry-score-weight",
        str(float_value(args, "e3nn_symmetry_score_weight", 0.15)),
        "--sa-score-weight",
        "0.0",
        "--novelty-score-weight",
        str(float_value(args, "novelty_score_weight", 0.0)),
        "--density-score-weight",
        str(density_weight),
        "--heating-value-score-weight",
        str(heating_weight),
        "--volumetric-heating-value-score-weight",
        str(volumetric_weight),
        "--melting-point-score-weight",
        str(
            float_value(args, "melting_point_score_weight", 1.0)
            if melting_enabled
            else 0.0
        ),
        "--boiling-point-score-weight",
        str(
            float_value(args, "boiling_point_score_weight", 1.0)
            if boiling_enabled
            else 0.0
        ),
        "--flash-point-score-weight",
        str(
            float_value(args, "flash_point_score_weight", 1.0)
            if flash_enabled
            else 0.0
        ),
        "--property-sa-score-weight",
        str(float_value(args, "property_sa_score_weight", 1.0)),
        "--structure-validity-score-weight",
        str(float_value(args, "structure_validity_score_weight", 1.0)),
        "--formation-enthalpy-score-weight",
        "0.0",
        "--molar-volume-score-weight",
        "0.0",
        "--isp-score-weight",
        "0.0",
        "--min-ring-count",
        str(int_value(args, "min_ring_count", 1)),
        "--max-ring-count",
        str(int_value(args, "max_ring_count", 6)),
        "--max-h-to-c",
        str(float_value(args, "max_h_to_c", 2.2)),
    ]
    add_optional(result, "--known-smiles", args.get("known_smiles"))
    add_optional(
        result,
        "--temporary-smiles",
        args.get("perturbation_generation_temporary")
        or args.get("temporary_smiles"),
    )
    add_optional(result, "--perturbation-temporary-smiles", args.get("perturbation_temporary_smiles"))
    if args.get("known_fuel_check"):
        result.append("--known-fuel-check")
    if bool(args.get("samples_per_seed_operation", False)):
        result.append("--samples-per-seed-operation")
    add_geometry_model_args(result, args)
    add_optional(result, "--e3nn-property-checkpoint", args.get("e3nn_property_checkpoint"))
    add_optional(result, "--e3nn-property-name", args.get("e3nn_property_name"))
    add_property_model_selection_args(result, args)
    add_heating_model_args(result, args)
    add_melting_model_args(result, args)
    add_optional(result, "--max-seeds", args.get("max_seeds"))
    operations = args.get("operations")
    if isinstance(operations, list):
        for operation in operations:
            if operation:
                result += ["--operation", str(operation)]
    if not density_enabled:
        result.append("--ignore-density")
    if heating_enabled:
        result.append("--use-heating-value")
        add_optional(result, "--heating-values-csv", args.get("heating_values_csv"))
    if melting_enabled:
        result.append("--use-melting-point-screen")
    if boiling_enabled:
        result.append("--use-boiling-point-screen")
    if flash_enabled:
        result.append("--use-flash-point-screen")
    if volumetric_mode:
        result.append("--use-volumetric-heating-value-screen")
    if bool(args.get("skip_mmff_geometry", False)):
        result.append("--skip-mmff-geometry")
    # Iterative mode accumulates every round, so replacing the stored products
    # is left to the first campaign the chain runs in front of it.
    if allow_replace and not bool(args.get("merge_existing_products", True)):
        result.append("--replace-existing-products")
    if bool(args.get("iterative_perturbation", False)):
        # Iterative mode: the same campaign, but every round re-samples the leads
        # that are not Tier 1 yet with the (static) iteration model.
        result += [
            "--iterations",
            str(max(int_value(args, "iterations", 3), 1)),
            "--per-round-budget",
            str(max(int_value(args, "per_round_budget", 2000), 1)),
            "--max-seeds-per-round",
            str(max(int_value(args, "max_seeds_per_round", 300), 1)),
            "--attempts-per-seed-operation",
            str(max(int_value(args, "attempts_per_seed_operation", 10), 1)),
            "--attempts-per-molecule",
            str(max(int_value(args, "attempts_per_molecule", 3), 0)),
            "--rounds-without-improvement",
            str(max(int_value(args, "rounds_without_improvement", 0), 0)),
        ]
        model_path = str(args.get("iteration_model") or "").strip()
        if model_path:
            result += ["--iteration-model", model_path]
        # Optional operator split of the iterative rounds: "name=count" pairs, so
        # the actual numbers stay tunable from the panel.
        raw_split = str(args.get("operator_attempts") or "").strip()
        for entry in [item.strip() for item in raw_split.replace(";", ",").split(",") if item.strip()]:
            name, _, count = entry.partition("=")
            if name.strip() and count.strip():
                result += ["--operator-attempt", "%s=%s" % (name.strip(), count.strip())]
        if raw_split:
            result += [
                "--operator-adaptation-rate",
                str(float_value(args, "operator_adaptation_rate", 0.5)),
            ]
        arm = str(args.get("ablation_arm") or "").strip()
        if arm:
            # Controlled ablation arm (A-E): same leads, rounds, attempts and
            # budget; only the named factor differs.
            result += ["--ablation-arm", arm.upper()]
    return result


def build_rescore_args(args: dict[str, object]) -> list[str]:
    required(args, "input", "output")
    result = [
        "rescore-candidates",
        "--input",
        str(args["input"]),
        "--output",
        str(args["output"]),
        "--density-min",
        str(float_value(args, "density_min", 0.85)),
        "--heating-value-min",
        str(float_value(args, "heating_value_min", 38.0)),
        "--melting-point-max",
        str(float_value(args, "melting_point_max", 273.15)),
        "--property-score-weight",
        str(float_value(args, "property_score_weight", 0.70)),
        "--e3nn-geometry-score-weight",
        str(float_value(args, "e3nn_geometry_score_weight", 0.10)),
        "--e3nn-symmetry-score-weight",
        str(float_value(args, "e3nn_symmetry_score_weight", 0.10)),
        "--sa-score-weight",
        "0.0",
        "--novelty-score-weight",
        str(float_value(args, "novelty_score_weight", 0.10)),
        "--density-score-weight",
        str(float_value(args, "density_score_weight", 1.0)),
        "--heating-value-score-weight",
        str(float_value(args, "heating_value_score_weight", 1.0)),
        "--melting-point-score-weight",
        str(float_value(args, "melting_point_score_weight", 1.0)),
        "--boiling-point-score-weight",
        str(float_value(args, "boiling_point_score_weight", 1.0)),
        "--flash-point-score-weight",
        str(float_value(args, "flash_point_score_weight", 1.0)),
        "--property-sa-score-weight",
        str(float_value(args, "property_sa_score_weight", 1.0)),
        "--formation-enthalpy-score-weight",
        "0.0",
        "--molar-volume-score-weight",
        "0.0",
        "--isp-score-weight",
        "0.0",
    ]
    add_optional(result, "--summary-output", args.get("summary_output"))
    return result


def create_job(mode: str, command: list[list[str]], display_command: str, output_dir: str) -> Job:
    ensure_no_active_job()
    job_id = uuid.uuid4().hex[:12]
    log_path = JOB_ROOT / f"{job_id}.log"
    job = Job(
        id=job_id,
        mode=mode,
        commands=command,
        display_command=display_command,
        log_path=str(log_path),
        output_dir=output_dir,
    )
    with JOBS_LOCK:
        JOBS[job_id] = job
    persist_job(job)
    thread = threading.Thread(target=run_job, args=(job,), daemon=True)
    thread.start()
    return job


def active_jobs() -> list[Job]:
    with JOBS_LOCK:
        return [job for job in JOBS.values() if job.status in ("queued", "running")]


def ensure_no_active_job() -> None:
    """Refuse to start a second run while one is still going.

    Two concurrent pipelines double the worker pools (each worker holds the
    torch/CUDA stack) and, when they share an output folder, they overwrite each
    other's tables. One job at a time is the supported mode.
    """
    running = active_jobs()
    if not running:
        return
    labels = ", ".join("%s(%s)" % (job.id, job.mode or "job") for job in running)
    raise ValueError(
        "another job is still running: %s - 面板一次只跑一个任务（并发会成倍占用内存，"
        "并且同目录写入会互相覆盖）" % labels
    )


def _terminate_process_tree(pid: int) -> int:
    """Kill a process and its children; returns how many were signalled."""
    if not pid:
        return 0
    try:
        import psutil  # noqa: WPS433 (only needed when stopping a job)
    except ImportError:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        return 1
    try:
        root = psutil.Process(pid)
    except psutil.Error:
        return 0
    victims = [root, *root.children(recursive=True)]
    killed = 0
    for victim in victims:
        try:
            victim.kill()
            killed += 1
        except psutil.Error:
            continue
    psutil.wait_procs(victims, timeout=20)
    return killed


def stop_job(job_id: str) -> dict[str, object]:
    """Stop a running job and mark it resumable."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise ValueError("job not found")
    if job.status not in ("queued", "running"):
        raise ValueError("only queued or running jobs can be stopped")
    job.stop_requested = True
    killed = _terminate_process_tree(int(job.process_id or 0))
    with JOBS_LOCK:
        job.status = "interrupted"
        job.current_stage_message = "Stopped from the control panel"
        job.ended_at = time.time()
        job.process_id = None
    persist_job(job)
    refresh_job_results(job)
    payload = job.as_dict()
    payload["processes_killed"] = killed
    return payload


def job_device_mode(job: Job) -> str:
    """The device mode the job was submitted with (defaults to ``auto``)."""
    for command in job.commands:
        for index, token in enumerate(command):
            if token == "--device-mode" and index + 1 < len(command):
                return command[index + 1].strip().lower()
            if token.startswith("--device-mode="):
                return token.split("=", 1)[1].strip().lower()
    return "auto"


def resolve_job_device(job: Job) -> str | None:
    """Resolve ``auto`` once for a job so its commands do not probe again.

    ``cpu`` needs no probe and ``gpu`` must be allowed to fail inside the
    command itself (so the error lands in that command's log), therefore only
    ``auto`` is resolved here.  A failure of the probe never blocks the job.
    """
    if job_device_mode(job) != "auto":
        return None
    code_dir = str(REPO_ROOT / "core" / "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    try:
        from device_policy import describe_device

        report = describe_device("auto")
    except Exception as exc:  # pragma: no cover - environment dependent
        safe_print(f"[device] panel probe skipped: {exc}")
        return None
    safe_print(f"[device] mode=auto -> {report.device} ({report.smoke_test}); "
               "every command of this job reuses this decision")
    return report.device


def run_job(job: Job) -> None:
    env = os.environ.copy()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # Resolve "auto" once for the whole job and hand the decision to every
    # command through the environment, so the CUDA smoke test does not run
    # again for each sub-command (or for each property-prediction call).
    inherited = resolve_job_device(job)
    if inherited:
        env["SCULPTOR_RESOLVED_DEVICE"] = inherited
    job.status = "running"
    job.started_at = time.time()
    persist_job(job)
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    log_file = Path(job.log_path)
    log_mode = "a" if log_file.exists() and job.completed_commands else "w"
    with log_file.open(log_mode, encoding="utf-8", errors="replace") as handle:
        handle.write(job.display_command + "\n\n")
        handle.flush()
        try:
            for index, command in enumerate(
                job.commands[job.completed_commands :],
                start=job.completed_commands + 1,
            ):
                job.current_command_index = index
                header = f"[{index}/{len(job.commands)}] start: {ps_join(command)}"
                handle.write(header + "\n")
                append_log_line(job, header)
                update_job_stage(job, "prepare", f"Starting command {index}/{len(job.commands)}")
                handle.flush()
                process = subprocess.Popen(
                    command,
                    cwd=str(REPO_ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                job.process_id = process.pid
                persist_job(job)
                if job.stop_requested:
                    # Stop was pressed while the command was still starting up.
                    _terminate_process_tree(process.pid)
                    job.process_id = None
                    job.status = "interrupted"
                    update_job_stage(job, "stopped", "Stopped from the control panel")
                    persist_job(job)
                    break
                assert process.stdout is not None
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    append_log_line(job, line.rstrip())
                job.returncode = process.wait()
                job.process_id = None
                if job.stop_requested:
                    job.status = "interrupted"
                    update_job_stage(job, "stopped", "Stopped from the control panel")
                    persist_job(job)
                    break
                if job.returncode != 0:
                    job.status = "failed"
                    update_job_stage(job, "failed", "Job failed")
                    persist_job(job)
                    break
                job.completed_commands = index
                persist_job(job)
            else:
                job.status = "succeeded"
                update_job_stage(job, "done", "Job completed")
        except Exception as exc:
            job.status = "failed"
            job.returncode = -1
            line = f"control panel failed to run command: {exc}"
            handle.write(line + "\n")
            append_log_line(job, line)
        finally:
            job.ended_at = time.time()
            refresh_job_results(job)
            persist_job(job)


def append_log_line(job: Job, line: str) -> None:
    with JOBS_LOCK:
        job.last_lines.append(line)
        job.last_lines = job.last_lines[-160:]
    marker = parse_stage_marker(line)
    if marker is not None:
        update_job_stage(job, marker[0], marker[1])


def parse_stage_marker(line: str) -> tuple[str, str] | None:
    prefix = "__FUEL_PANEL_STAGE__:"
    if not line.startswith(prefix):
        return None
    rest = line[len(prefix) :]
    stage, _, message = rest.partition(":")
    if not stage:
        return None
    return stage, message or stage


def update_job_stage(job: Job, stage: str, message: str = "") -> None:
    with JOBS_LOCK:
        job.current_stage = stage
        job.current_stage_message = message
    persist_job(job)


def refresh_job_results(job: Job) -> None:
    """Collect the summary-level result files of a job.

    Only the run root and the top level of each ``C*`` folder are listed: the
    per-seed perturbation folders and the candidate input folders hold thousands of
    files, and walking them on every poll is what made the control panel burn
    CPU and grow while a job was running.
    """
    files: list[str] = []
    candidates: list[Path] = []
    if job.output_dir:
        out = Path(job.output_dir)
        if out.exists():
            candidates.extend(_summary_level_files(out))
    candidates.append(Path(job.log_path))
    seen: set[str] = set()
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True):
        if not path.exists() or not path.is_file():
            continue
        text = str(path)
        if text not in seen:
            files.append(text)
            seen.add(text)
        if len(files) >= 80:
            break
    with JOBS_LOCK:
        job.result_files = files


def _summary_level_files(root: Path) -> list[Path]:
    found: list[Path] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return found
    for item in entries:
        if item.is_file() and item.suffix.lower() in (".csv", ".json"):
            found.append(item)
        elif item.is_dir() and item.name[:1] == "C" and item.name[1:].isdigit():
            try:
                children = list(item.iterdir())
            except OSError:
                continue
            for child in children:
                if child.is_file() and child.suffix.lower() in (".csv", ".json"):
                    found.append(child)
    return found


def job_state_path(job_id: str) -> Path:
    return JOB_ROOT / f"{job_id}.job.json"


def persist_job(job: Job) -> None:
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    path = job_state_path(job.id)
    temporary = path.with_suffix(".job.json.tmp")
    try:
        temporary.write_text(
            json.dumps(asdict(job), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        return


def load_persisted_jobs() -> None:
    loaded: dict[str, Job] = {}
    for path in JOB_ROOT.glob("*.job.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            job = Job(**payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if job.status in ("queued", "running"):
            job.status = "interrupted"
            job.ended_at = time.time()
            job.current_stage_message = "Control panel restarted; can resume from the incomplete command"
            persist_job(job)
        loaded[job.id] = job
    with JOBS_LOCK:
        JOBS.update(loaded)


def browse_path(path_value: str) -> dict[str, object]:
    path = safe_workspace_path(path_value)
    if not path.exists():
        raise ValueError("path does not exist")
    if path.is_file():
        return {
            "type": "file",
            "path": str(path),
            "entry": describe_entry(path),
        }
    entries = []
    children = sorted(
        path.iterdir(),
        key=lambda item: (not item.is_dir(), item.name.lower()),
    )
    for child in children[:200]:
        entries.append(describe_entry(child))
    parent = path.parent if path.resolve() != REPO_ROOT.resolve() else None
    return {
        "type": "directory",
        "path": str(path),
        "name": path.name or str(path),
        "parent": "" if parent is None else str(parent),
        "entries": entries,
        "truncated": len(children) > 200,
    }


def describe_entry(path: Path) -> dict[str, object]:
    if path.is_dir():
        dirs = 0
        files = 0
        try:
            for child in path.iterdir():
                if child.is_dir():
                    dirs += 1
                elif child.is_file():
                    files += 1
        except OSError:
            pass
        return {
            "type": "directory",
            "name": path.name,
            "path": str(path),
            "title": "Folder",
            "desc": f"Contains {dirs} subfolders and {files} files. Click to enter.",
            "detail": "",
        }
    return {
        "type": "file",
        "name": path.name,
        "path": str(path),
        **describe_file(path),
    }


def describe_file(path: Path) -> dict[str, str]:
    name = path.name.lower()
    title = "Output file"
    desc = "A file produced by this run."
    if name == "screened_candidates.csv":
        title = "Screened candidates"
        desc = "Candidates passing dedup, structure rules, and basic screening."
    elif name == "property_predictions.csv":
        title = "Property predictions"
        desc = "Complete results: density, heating value, melting reference, e3nn scores, threshold decisions, and composite scores."
    elif name == "tier1_candidates.csv" or name.startswith("tier1_candidates_total"):
        title = "Tier 1 pool"
        desc = "Matches or beats JP-10 on all three hard properties (density, mass heating value, melting point)."
    elif name == "tier2_candidates.csv" or name.startswith("tier2_candidates_total"):
        title = "Tier 2 pool"
        desc = "Beats JP-10 on two of the three hard properties."
    elif name == "tier3_candidates.csv" or name.startswith("tier3_candidates_total"):
        title = "Tier 3 pool"
        desc = "Meets every hard threshold and beats JP-10 on one of the three compared properties."
    elif name == "tier4_candidates.csv" or name.startswith("tier4_candidates_total"):
        title = "Tier 4 pool"
        desc = "Meets every hard threshold but beats JP-10 on none of the three compared properties."
    elif name in ("potential1_candidates.csv", "potential1_candidates_total.csv"):
        title = "Potential 1 pool"
        desc = "Beats JP-10 on two compared properties but misses one hard threshold."
    elif name in ("potential2_candidates.csv", "potential2_candidates_total.csv"):
        title = "Potential 2 pool"
        desc = "Misses one hard threshold and beats JP-10 on one of the three compared properties."
    elif name in ("potential3_candidates.csv", "potential3_candidates_total.csv"):
        title = "Potential 3 pool"
        desc = "Misses two hard thresholds and beats JP-10 on the remaining compared property."
    elif name == "tier_summary.csv":
        title = "Tier summary"
        desc = "Counts per stage and carbon for Tier 1-4 and Potential 1-3."
    elif name == "group_summary.csv":
        title = "Mirror-split summary"
        desc = "Per n1/n2/n3 counts of generated candidates per channel (Tier 1-4, Potential 1-3) for this carbon."
    elif name == "perturbation_group_summary.csv":
        title = "Perturbation group summary"
        desc = "Per seed x operation counts of generated perturbation products."
    elif name.lower() == "perturbation_seeds_t123.csv":
        title = "Perturbation seeds (Tier 1-3)"
        desc = "The molecules the perturbation stage explores: every Tier 1, Tier 2 and Tier 3 candidate of the run."
    elif name.lower() == "perturbation_seeds_potential.csv":
        title = "Perturbation seeds (potential pools)"
        desc = "Every Potential 1 molecule plus the best 10% of Potential 2 and of Potential 3, for the optional potential-pool campaign."
    elif name == "candidates_for_manual_calculation.csv":
        title = "Manual calculation pool"
        desc = "This run's seeds plus its Tier 1-3 perturbation products, with the full property record; pick from here for high-level calculation."
    elif name == "rejected_pre_property.csv":
        title = "Rejected before property prediction"
        desc = "Molecules failing dedup, MMFF, ring count, or basic structure screening."
    elif name == "property_failed_candidates.csv":
        title = "Property prediction failures"
        desc = "Molecules where property models or routing errored and need individual review."
    elif name == "missing_property_candidates.csv":
        title = "Missing property molecules"
        desc = ("Molecules whose selected properties are incomplete; they stay out of every channel until the "
                "data is filled in, and all available fields plus the penalties are kept.")
    elif name.endswith("summary.json") or name == "summary.json":
        title = "Run summary"
        desc = "Counts, paths, and key parameters for this run."
    elif name.endswith(".log"):
        title = "Run log"
        desc = "Actual commands and program output; check first when troubleshooting."
    elif name.endswith(".csv"):
        title = "Result table"
        desc = "Candidate or intermediate statistics table produced by the program."
    elif name.endswith(".json"):
        title = "Summary file"
        desc = "Structured statistics and configuration records produced by the program."
    return {
        "title": title,
        "desc": desc,
        "detail": file_detail(path),
    }


def file_detail(path: Path) -> str:
    try:
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="", errors="replace") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                rows = sum(1 for _ in reader)
            columns = ", ".join(header[:5])
            if len(header) > 5:
                columns += "..."
            return f"{rows} rows; columns: {columns}" if header else f"{rows} rows"
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                keys = ", ".join(list(data.keys())[:6])
                if len(data) > 6:
                    keys += "..."
                return f"JSON keys: {keys}"
            if isinstance(data, list):
                return f"JSON list: {len(data)} items"
        size = path.stat().st_size
        return f"{size} bytes"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "Summary unavailable"


def preview_file(path_value: str) -> dict[str, object]:
    path = safe_workspace_path(path_value)
    if not path.exists() or not path.is_file():
        raise ValueError("file does not exist")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="", errors="replace") as handle:
            reader = csv.DictReader(handle)
            rows = [row for _, row in zip(range(30), reader)]
            return {
                "type": "csv",
                "path": str(path),
                "columns": reader.fieldnames or [],
                "rows": rows,
            }
    if path.suffix.lower() == ".json":
        return {
            "type": "json",
            "path": str(path),
            "content": json.loads(path.read_text(encoding="utf-8")),
        }
    return {
        "type": "text",
        "path": str(path),
        "content": path.read_text(encoding="utf-8", errors="replace")[:20000],
    }


def stage_statuses(
    mode: str,
    status: str,
    current_stage: str,
    current_stage_message: str,
) -> list[dict[str, str]]:
    if mode == "symmetric":
        names = [
            ("prepare", "Preparing command"),
            ("generate", "Generating candidates"),
            ("screen", "Screening"),
            ("property", "Property prediction"),
            ("route", "Tier / Potential routing"),
            ("done", "Completed"),
        ]
    elif mode == "perturbation":
        names = [
            ("prepare", "Reading seeds"),
            ("generate", "Perturbation generation"),
            ("screen", "Dedup and screening"),
            ("property", "Property prediction"),
            ("route", "Result routing"),
            ("done", "Completed"),
        ]
    else:
        names = [
            ("prepare", "Reading tables"),
            ("score", "Rescoring"),
            ("sort", "Sorting output"),
            ("done", "Completed"),
        ]
    done = status == "succeeded"
    failed = status == "failed"
    stage_keys = [key for key, _ in names]
    current_index = stage_keys.index(current_stage) if current_stage in stage_keys else 0
    stages: list[dict[str, str]] = []
    for index, (key, label) in enumerate(names):
        state = "pending"
        if done:
            state = "done"
        elif failed:
            state = "done" if index < current_index else ("failed" if index == current_index else "pending")
        elif status == "running":
            state = "done" if index < current_index else ("running" if index == current_index else "pending")
        stages.append(
            {
                "key": key,
                "label": label,
                "state": state,
                "detail": current_stage_message if index == current_index else "",
            }
        )
    if failed:
        stages[-1]["state"] = "failed"
    return stages


def safe_workspace_path(path_value: str) -> Path:
    if not path_value:
        raise ValueError("missing path")
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    root = REPO_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("path must be inside workspace")
    return resolved


def required(args: dict[str, object], *names: str) -> None:
    missing = [name for name in names if not str(args.get(name) or "").strip()]
    if missing:
        raise ValueError("missing required parameter(s): " + ", ".join(missing))


def add_optional(args: list[str], flag: str, value: object) -> None:
    if value not in (None, ""):
        args.extend([flag, str(value)])


def add_geometry_model_args(command: list[str], args: dict[str, object]) -> None:
    model = str(args.get("geometry_model") or "e3nn_geometry")
    if model == "e3nn_geometry":
        add_optional(
            command,
            "--e3nn-geometry-checkpoint",
            args.get("e3nn_checkpoint")
            or existing_or_default("Evaluation_Iteration/evaluation/model/e3nn/e3nn_geometry_scorer_g_new.pt"),
        )
    elif model == "rule_geometry":
        return
    else:
        raise ValueError(f"unknown geometry model option: {model}")


def add_heating_model_args(command: list[str], args: dict[str, object]) -> None:
    model = str(args.get("heating_model") or "sta_heating")
    if model in ("sta", "sta_heating"):
        return
    elif model == "none":
        command.append("--no-cbh-bdc-heating")
    else:
        raise ValueError(f"unknown heating model option: {model}")


def add_property_model_selection_args(command: list[str], args: dict[str, object]) -> None:
    density_model = str(args.get("density_model") or "sta_density")
    heating_model = str(args.get("heating_model") or "sta_heating")
    melting_model = str(args.get("melting_model") or "phase_fusion_melting")
    boiling_model = str(args.get("boiling_point_model") or "phase_fusion_boiling")
    flash_model = str(args.get("flash_point_model") or "phase_fusion_flash")
    formation_model = str(args.get("formation_enthalpy_model") or "sta_formation")
    molar_volume_model = str(args.get("molar_volume_model") or "derived")
    isp_model = str(args.get("isp_model") or "none")
    mapped_density = {
        "sta_density": "sta",
        "sta": "sta",
        "none": "none",
    }.get(density_model)
    mapped_heating = {
        "sta_heating": "sta",
        "sta": "sta",
        "none": "none",
    }.get(heating_model)
    mapped_melting = {
        "sta_melting": "sta",
        "sta": "sta",
        "phase_fusion_melting": "phase_fusion",
        "phase_fusion": "phase_fusion",
        "none": "none",
    }.get(melting_model)
    mapped_formation = {
        "sta_formation": "sta",
        "sta": "sta",
        "none": "none",
    }.get(formation_model)
    mapped_boiling = {
        "none": "none",
        "sta_boiling": "sta",
        "sta": "sta",
        "phase_fusion_boiling": "phase_fusion",
        "phase_fusion": "phase_fusion",
    }.get(boiling_model)
    mapped_flash = {
        "none": "none",
        "sta_flash": "sta",
        "sta": "sta",
        "phase_fusion_flash": "phase_fusion",
        "phase_fusion": "phase_fusion",
    }.get(flash_model)
    mapped_molar_volume = {
        "derived": "derived",
        "none": "none",
        # Old saved panel settings now use the deterministic M/rho route.
        "sta_molar_volume": "derived",
        "sta": "derived",
    }.get(molar_volume_model)
    mapped_isp = {"none": "none", "sta_isp": "sta", "sta": "sta"}.get(isp_model)

    # Production heads: MLP ensembles without property-as-descriptor. The legacy
    # ExtraTrees variants were retired on 2026-09-22.
    phase_fusion_schema = str(args.get("phase_fusion_schema_version") or "schema2_mlp")
    _phase_fusion_mlp_dirs = {
        "schema2_mlp": "Evaluation_Iteration/evaluation/model/schema 2/"
                       "phase_fusion_gdb820k_mlp_ens2d",
        "schema3_mlp": "Evaluation_Iteration/evaluation/model/schema 3/"
                       "phase_fusion_gdb820k_mlp_ens3d",
    }
    if phase_fusion_schema not in _phase_fusion_mlp_dirs:
        raise ValueError(
            f"unknown phase fusion schema version: {phase_fusion_schema} "
            "(expected 'schema2_mlp' or 'schema3_mlp')"
        )
    phase_fusion_dir = _phase_fusion_mlp_dirs[phase_fusion_schema]

    if mapped_density is None:
        raise ValueError(f"unknown density model option: {density_model}")
    if mapped_heating is None:
        raise ValueError(f"unknown heating model option: {heating_model}")
    if mapped_melting is None:
        raise ValueError(f"unknown melting model option: {melting_model}")
    if mapped_boiling is None:
        raise ValueError(f"unknown boiling point model option: {boiling_model}")
    if mapped_formation is None:
        raise ValueError(f"unknown formation enthalpy model option: {formation_model}")
    if mapped_flash is None:
        raise ValueError(f"unknown flash point model option: {flash_model}")
    if mapped_molar_volume is None:
        raise ValueError(f"unknown molar volume model option: {molar_volume_model}")
    if mapped_isp is None:
        raise ValueError(f"unknown ISP model option: {isp_model}")
    command += ["--density-model", mapped_density]
    command += ["--heating-value-model", mapped_heating]
    command += ["--melting-point-model", mapped_melting]
    command += ["--boiling-point-model", mapped_boiling]
    command += ["--flash-point-model", mapped_flash]
    command += ["--formation-enthalpy-model", mapped_formation]
    command += ["--molar-volume-model", mapped_molar_volume]
    command += ["--isp-model", mapped_isp]
    if mapped_density == "sta":
        add_optional(
            command,
            "--sta-density-model",
            args.get("sta_density_model")
            or args.get("sta_density_model_path")
            or existing_or_default("Evaluation_Iteration/evaluation/model/STA/sta_820k_v2/sta_gdb_density_paper.pt"),
        )
    if mapped_heating == "sta":
        add_optional(
            command,
            "--sta-heating-value-model",
            args.get("sta_heating_value_model")
            or args.get("sta_heating_value_model_path")
            or args.get("sta_formation_enthalpy_model")
            or existing_or_default("Evaluation_Iteration/evaluation/model/STA/sta_820k_v2/sta_gdb_formation_enthalpy_paper.pt"),
        )
    if mapped_melting in ("sta", "phase_fusion"):
        add_optional(
            command,
            "--sta-melting-point-model",
            args.get("sta_melting_point_model")
            or args.get("sta_melting_point_model_path")
            or existing_or_default("Evaluation_Iteration/evaluation/model/STA/sta_820k_v2/sta_gdb_melting_point_paper.pt"),
        )
    if mapped_boiling in ("sta", "phase_fusion"):
        add_optional(
            command,
            "--sta-boiling-point-model",
            args.get("sta_boiling_point_model")
            or args.get("sta_boiling_point_model_path")
            or existing_or_default("Evaluation_Iteration/evaluation/model/STA/sta_820k_v2/sta_gdb_boiling_point_paper.pt"),
        )
    if mapped_flash in ("sta", "phase_fusion"):
        add_optional(
            command,
            "--sta-flash-point-model",
            args.get("sta_flash_point_model")
            or args.get("sta_flash_point_model_path")
            or existing_or_default("Evaluation_Iteration/evaluation/model/STA/sta_820k_v2/sta_gdb_flash_point_paper.pt"),
        )
    if mapped_melting == "phase_fusion":
        add_optional(
            command,
            "--phase-fusion-melting-point-model",
            args.get("phase_fusion_melting_point_model")
            or args.get("phase_fusion_melting_point_model_path")
            or existing_or_default(f"{phase_fusion_dir}/melting_point_k_fusion.joblib"),
        )
    if mapped_boiling == "phase_fusion":
        add_optional(
            command,
            "--phase-fusion-boiling-point-model",
            args.get("phase_fusion_boiling_point_model")
            or args.get("phase_fusion_boiling_point_model_path")
            or existing_or_default(f"{phase_fusion_dir}/boiling_point_k_fusion.joblib"),
        )
    if mapped_flash == "phase_fusion":
        add_optional(
            command,
            "--phase-fusion-flash-point-model",
            args.get("phase_fusion_flash_point_model")
            or args.get("phase_fusion_flash_point_model_path")
            or existing_or_default(f"{phase_fusion_dir}/flash_point_k_fusion.joblib"),
        )
    if mapped_formation == "sta":
        add_optional(
            command,
            "--sta-formation-enthalpy-model",
            args.get("sta_formation_enthalpy_model")
            or args.get("sta_formation_enthalpy_model_path")
            or existing_or_default("Evaluation_Iteration/evaluation/model/STA/sta_820k_v2/sta_gdb_formation_enthalpy_paper.pt"),
        )
    if mapped_isp == "sta":
        add_optional(
            command,
            "--sta-isp-model",
            args.get("sta_isp_model")
            or args.get("sta_isp_model_path")
            or optional_existing("Evaluation_Iteration/evaluation/model/STA/sta_820k_v2/sta_gdb_isp_paper.pt"),
        )
    add_optional(command, "--sta-property-model", args.get("sta_model_path"))


def add_melting_model_args(command: list[str], args: dict[str, object]) -> None:
    model = str(args.get("melting_model") or "phase_fusion_melting")
    if model in ("sta", "sta_melting", "phase_fusion", "phase_fusion_melting", "none"):
        return
    raise ValueError(f"unknown melting model option: {model}")


def int_value(args: dict[str, object], name: str, default: int) -> int:
    value = args.get(name, default)
    if value in (None, ""):
        return default
    return int(value)


def float_value(args: dict[str, object], name: str, default: float) -> float:
    value = args.get(name, default)
    if value in (None, ""):
        return default
    return float(value)


def property_model_is_enabled(
    args: dict[str, object],
    name: str,
    default: str,
) -> bool:
    return str(args.get(name) or default) != "none"


def existing_or_default(relative: str) -> str:
    path = REPO_ROOT / relative
    return str(path)


def optional_existing(relative: str) -> str:
    """Return the path only when the checkpoint ships with the package.

    Some optional heads (for example the ISP head) are not part of the released
    weights, so their panel default stays empty instead of pointing at a file
    that does not exist on the reproducer's machine.
    """
    path = REPO_ROOT / relative
    return str(path) if path.exists() else ""


_recent_paths_cache: tuple[float, list[str]] | None = None
RECENT_PATH_CACHE_SECONDS = 300.0
RECENT_PATH_MAX_FILES = 3000


def recent_paths(root: Path, limit: int = 80, max_files: int = RECENT_PATH_MAX_FILES) -> list[str]:
    """Return recent .csv/.json under outputs, bounded and cached.

    The perturbation runs create tens of thousands of deep per-seed files, so a
    full recursive scan is far too slow for every /api/defaults call. We prune
    the heavy subtrees, cap the number of stat() calls, and cache the result.
    """
    global _recent_paths_cache
    now = time.time()
    if _recent_paths_cache is not None and now - _recent_paths_cache[0] < RECENT_PATH_CACHE_SECONDS:
        return _recent_paths_cache[1]
    if not root.exists():
        _recent_paths_cache = (now, [])
        return []
    candidates: list[tuple[float, str]] = []
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in ("perturbed_from_hits", "dft_inputs", "manual_review")
        ]
        for name in filenames:
            if not (name.endswith(".csv") or name.endswith(".json")):
                continue
            scanned += 1
            full = os.path.join(dirpath, name)
            try:
                mtime = os.stat(full).st_mtime
            except OSError:
                continue
            candidates.append((mtime, full))
            if scanned >= max_files:
                break
        if scanned >= max_files:
            break
    candidates.sort(key=lambda item: item[0], reverse=True)
    result = [path for _, path in candidates[:limit]]
    _recent_paths_cache = (time.time(), result)
    return result


# There is exactly one folder picker: tools/control_panel/pick_folder.ps1, which
# drives the Windows Common Item Dialog (IFileOpenDialog with FOS_PICKFOLDERS).
# That is the dialog Explorer itself shows.  It replaced two earlier
# implementations - the Windows Forms FolderBrowserDialog and a Tk
# askdirectory - because both fall back to the legacy tree browser on machines
# whose shell is configured that way.  The dialog runs in its own process, so
# the panel thread never enters a GUI message loop.
PICKER_SCRIPT = Path(__file__).resolve().parent / "pick_folder.ps1"


def pick_folder(initial_dir: str = "") -> dict[str, object]:
    """Open the native folder picker and return the chosen path.

    Returns ``{"path": ...}``, ``{"canceled": True}`` or ``{"error": ...}``; the
    panel falls back to its built-in browser when the dialog cannot be used.
    """
    initial = initial_dir or str(REPO_ROOT)
    if os.name != "nt":
        return {"error": "The native folder picker is only available on Windows"}
    if not PICKER_SCRIPT.exists():
        return {"error": f"Folder picker script not found: {PICKER_SCRIPT}"}
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass",
             "-File", str(PICKER_SCRIPT), "-Initial", initial],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
    except FileNotFoundError:
        return {"error": "powershell.exe is not available on this machine"}
    except subprocess.TimeoutExpired:
        return {"error": "The folder picker did not respond"}
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-400:]
        return {"error": detail or "Could not open the folder picker"}
    lines = (result.stdout or "").strip().splitlines()
    chosen = lines[-1].strip() if lines else ""
    if not chosen:
        return {"canceled": True}
    return {"path": chosen}


def ps_join(command: list[str]) -> str:
    return " ".join(ps_quote(part) for part in command)


def ps_quote(value: str) -> str:
    if value == "":
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./:=\\")
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "''") + "'"


def content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".html":
        return "text/html; charset=utf-8"
    if suffix == ".css":
        return "text/css; charset=utf-8"
    if suffix == ".js":
        return "text/javascript; charset=utf-8"
    if suffix == ".json":
        return "application/json; charset=utf-8"
    return "application/octet-stream"


if __name__ == "__main__":
    raise SystemExit(main())
