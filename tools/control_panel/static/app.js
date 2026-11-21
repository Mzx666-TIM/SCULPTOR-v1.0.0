const state = {
  mode: "symmetric",
  defaults: {},
  activeJobId: null,
  pollTimer: null,
  currentOutputRoot: "",
  currentBrowsePath: "",
  lastJob: null,
  perturbationSeedTables: {},
};

const form = document.querySelector("#configForm");
const commandPreview = document.querySelector("#commandPreview");
const serverStatus = document.querySelector("#serverStatus");
const repoPath = document.querySelector("#repoPath");
const stageList = document.querySelector("#stageList");
const jobMeta = document.querySelector("#jobMeta");
const logBox = document.querySelector("#logBox");
const fileList = document.querySelector("#fileList");
const preview = document.querySelector("#preview");
const weightNotice = document.querySelector("#weightNotice");

const MODE_LABELS = {
  symmetric: "Symmetric Generation",
  perturbation: "Perturbation",
  rescore: "Rescore",
};

const STATUS_LABELS = {
  queued: "Queued",
  running: "Running",
  succeeded: "Succeeded",
  failed: "Failed",
  interrupted: "Interrupted (resumable)",
};

const COLUMN_LABELS = {
  candidate_id: "Candidate ID",
  operation: "Operation",
  canonical_smiles: "Canonical SMILES",
  parent_canonical_smiles: "Parent SMILES",
  carbon_count: "Carbons",
  hydrogen_count: "Hydrogens",
  formula: "Formula",
  molecular_weight: "Molecular weight (g/mol)",
  molecular_weight_g_mol: "Molecular weight (g/mol)",
  molecular_weight_source: "MW source",
  density_g_cm3: "Density",
  heating_value_mj_kg: "Mass heating value",
  melting_point_k: "Melting point",
  external_density_g_cm3: "Predicted density",
  external_density_model: "Density model",
  external_heating_value_mj_kg: "Predicted mass heating value",
  external_heating_value_model: "Heating value model",
  volumetric_heating_value_mj_l: "Volumetric heating value (MJ/L)",
  volumetric_heating_value_uncertainty: "Volumetric heating value uncertainty (MJ/L)",
  volumetric_heating_value_source: "Volumetric HV source",
  external_melting_point_k: "Predicted melting point",
  external_melting_point_model: "Melting point model",
  external_melting_point_usage: "Melting point usage",
  external_boiling_point_k: "Predicted boiling point",
  external_boiling_point_uncertainty: "Boiling point uncertainty",
  external_boiling_point_model: "Boiling point model",
  external_flash_point_k: "Predicted flash point",
  external_flash_point_uncertainty: "Flash point uncertainty",
  external_flash_point_model: "Flash point model",
  external_formation_enthalpy_kj_mol: "Predicted formation enthalpy",
  external_formation_enthalpy_uncertainty_kj_mol: "Formation enthalpy uncertainty",
  external_formation_enthalpy_model: "Formation enthalpy model",
  molar_volume_cm3_mol: "Converted molar volume (cm3/mol)",
  external_molar_volume_cm3_mol: "Converted molar volume (cm3/mol)",
  external_molar_volume_uncertainty: "Molar volume propagated uncertainty (cm3/mol)",
  external_molar_volume_model: "Molar volume source",
  external_isp: "Predicted ISP",
  external_isp_uncertainty: "ISP uncertainty",
  external_isp_model: "ISP model",
  cbh_bdc_formation_enthalpy_kj_mol: "CBH-BDC formation enthalpy",
  cbh_bdc_heating_value_mj_kg: "CBH-BDC mass heating value",
  cbh_bdc_heating_value_uncertainty: "CBH-BDC heating value uncertainty",
  cbh_bdc_status: "CBH-BDC status",
  cbh_bdc_failure_reason: "CBH-BDC error",
  e3nn_property_name: "e3nn heating proxy name",
  e3nn_property_prediction: "e3nn heating proxy value",
  e3nn_property_model_checkpoint: "e3nn heating proxy model",
  e3nn_property_prediction_status: "e3nn heating proxy status",
  e3nn_property_failure_reason: "e3nn heating proxy error",
  sta_property_model_checkpoint: "STA multi-property model",
  sta_property_prediction_status: "STA status",
  sta_property_failure_reason: "STA error",
  sta_flash_point_k: "STA flash point",
  sta_flash_point_k_uncertainty: "STA flash point uncertainty",
  sta_density_g_cm3: "STA density",
  sta_density_g_cm3_uncertainty: "STA density uncertainty",
  sta_formation_enthalpy_kj_mol: "STA formation enthalpy",
  sta_formation_enthalpy_kj_mol_uncertainty: "STA formation enthalpy uncertainty",
  sta_molar_volume_cm3_mol: "STA molar volume (cm3/mol)",
  sta_molar_volume_cm3_mol_uncertainty: "STA molar volume uncertainty (cm3/mol)",
  sta_isp: "STA ISP",
  sta_isp_uncertainty: "STA ISP uncertainty",
  sta_melting_point_k: "STA melting point",
  sta_boiling_point_k: "STA boiling point",
  sta_boiling_point_k_uncertainty: "STA boiling point uncertainty",
  sta_combustion_enthalpy_kj_mol: "STA combustion enthalpy",
  sta_mass_calorific_value_mj_kg: "STA mass heating value",
  fingerprint_novelty: "Novelty",
  property_quality_score: "Property quality score",
  composite_ranking_score: "Composite score",
  objective_mode: "Objective",
  objective_relative_window: "Objective scoring window",
  property_target_density_g_cm3: "Target density",
  property_target_heating_value_mj_kg: "Target heating value",
  property_target_volumetric_heating_value_mj_l: "Target volumetric heating value",
  property_target_melting_point_k: "Target melting point",
  property_target_boiling_point_k: "Target boiling point",
  property_target_flash_point_k: "Target flash point",
  structure_validity_score: "Structure validity score",
  missing_property_names: "Missing properties",
  missing_property_penalty_factor: "Missing property penalty factor",
  e3nn_geometry_score: "e3nn geometry score",
  symmetry_consistency: "e3nn symmetry score",
  sa_score: "SA score",
  screen_score: "Screen score",
  generation_reward_score: "Generation reward score",
  mean_reward_score: "Mean reward score",
  best_reward_score: "Best reward score",
  decision: "Feedback decision",
  next_threshold: "Next bond threshold",
  next_temperature: "Next sampling temperature",
  next_top_k: "Next top-k",
  tier: "Tier",
  tier_label: "Tier",
  tier_reason: "Tier reason",
  jp10_beats: "JP-10 properties beaten",
  jp10_beat_count: "JP-10 beats",
  potential_candidate: "Potential candidate",
  screening_status: "Screening status",
  property_decision: "Property decision",
  density_decision: "Density decision",
  heating_value_decision: "Heating value decision",
  melting_point_decision: "Melting point decision",
  failure_reasons: "Failure reasons",
  mmff_stability_score: "MMFF stability score",
  output_dir: "Output directory",
};

const VALUE_LABELS = {
  bond_rewire: "Bond rewire",
  ring_adjust: "Ring adjustment",
  methyl_shift: "Methyl shift",
  local_skeleton_rearrange: "Local skeleton rearrangement",
  ch2_bridge_move: "CH2 bridge move",
  methylation: "Methylation",
  ring_expand: "Ring expansion",
  pass: "Pass",
  reject: "Reject",
  review: "Review",
  below_threshold: "Below threshold",
  pre_screen_reject: "Pre-screen reject",
  exact_new_fingerprint_high: "Exact-new, fingerprint high",
  exact_new_fingerprint_low: "Exact-new, fingerprint low",
  tier_pool: "Tier pool",
  potential_pool: "Potential pool",
  iteration_pool: "Iteration pool",
  ignored: "Ignored",
  reference_only: "Reference only",
  missing_properties: "Missing properties",
  derived_from_density_and_mass_heating_value: "Derived from density and mass heating value",
  property_maximize: "Property-guided",
  target_property: "Target-property-guided",
  novelty_exploration: "Novelty exploration",
  gdb_multi_property: "GDB multi-property STA",
  sta_gdb_multi_property: "GDB multi-property STA",
  sta_gdb_multi_property_failed: "GDB multi-property STA failed",
  phase_fusion: "Phase-fusion model",
  calculated_from_molecular_formula: "Computed from molecular formula and standard atomic weights",
  derived_from_molecular_weight_and_density: "Derived from molecular weight and density",
  sta_gdb_boiling_point_paper: "STA boiling point",
  yang_cbh_bdc_delta_ml: "CBH-BDC / delta-ML",
  yang_cbh_bdc_stacking_regressor: "CBH-BDC Stacking",
  yang_cbh_bdc_failed: "CBH-BDC failed",
  heating_value_pending: "Heating value pending",
  predicted: "Predicted",
  not_requested: "Not requested",
  not_predicted: "Not predicted",
  failed: "Failed",
  True: "Yes",
  False: "No",
  true: "Yes",
  false: "No",
};

const COMPOSITE_WEIGHT_FIELDS = [
  "property_score_weight",
  "e3nn_geometry_score_weight",
  "e3nn_symmetry_score_weight",
  "novelty_score_weight",
];

const FILE_HINTS = [
  {
    pattern: /generation_feedback\.csv$/i,
    title: "Feedback reward records",
    desc: "Per-round SA, H/C ratio, MMFF, novelty, reward score, and auto-adjusted bond parameters.",
  },
  {
    pattern: /tier1_candidates(_total)?\.csv$/i,
    title: "Tier 1 pool",
    desc: "Matches or beats JP-10 on all three hard properties (density, mass heating value, melting point).",
  },
  {
    pattern: /tier2_candidates(_total)?\.csv$/i,
    title: "Tier 2 pool",
    desc: "Beats JP-10 on two of the three hard properties.",
  },
  {
    pattern: /tier3_candidates(_total)?\.csv$/i,
    title: "Tier 3 pool",
    desc: "Meets every hard threshold and beats JP-10 on one of the three compared properties.",
  },
  {
    pattern: /tier4_candidates(_total)?\.csv$/i,
    title: "Tier 4 pool",
    desc: "Meets every hard threshold but beats JP-10 on none of the three compared properties.",
  },
  {
    pattern: /potential1_candidates(_total)?\.csv$/i,
    title: "Potential 1 pool",
    desc: "Beats JP-10 on two compared properties but misses one hard threshold.",
  },
  {
    pattern: /potential2_candidates(_total)?\.csv$/i,
    title: "Potential 2 pool",
    desc: "Misses one hard threshold and beats JP-10 on one of the three compared properties.",
  },
  {
    pattern: /potential3_candidates(_total)?\.csv$/i,
    title: "Potential 3 pool",
    desc: "Misses two hard thresholds and beats JP-10 on the remaining compared property.",
  },
  {
    pattern: /tier_summary\.csv$/i,
    title: "Tier summary",
    desc: "Counts per stage and carbon for Tier 1-4 and Potential 1-3.",
  },
  {
    pattern: /group_summary\.csv$/i,
    title: "Mirror-split summary",
    desc: "Per n1/n2/n3 counts of generated candidates and their routes for this carbon.",
  },
  {
    pattern: /perturbation_group_summary\.csv$/i,
    title: "Perturbation group summary",
    desc: "Per seed x operation counts of generated perturbation products.",
  },
  {
    pattern: /perturbation_seeds_T123\.csv$/i,
    title: "Perturbation seeds (Tier 1-3)",
    desc: "The molecules the perturbation stage explores: every Tier 1, 2 and 3 candidate of the run.",
  },
  {
    pattern: /perturbation_seeds_potential\.csv$/i,
    title: "Perturbation seeds (potential pools)",
    desc: "Every Potential 1 molecule plus the best 10% of Potential 2 and of Potential 3, by composite score.",
  },
  {
    pattern: /candidates_for_manual_calculation\.csv$/i,
    title: "Manual calculation pool",
    desc: "This run's seeds plus its Tier 1-3 perturbation products, with the full property record.",
  },
  {
    pattern: /missing_property_candidates\.csv$/i,
    title: "Missing property molecules",
    desc: "Molecules with missing predictions in selected properties; all available fields and penalties are recorded.",
  },
  {
    pattern: /summary\.json$/i,
    title: "Run summary",
    desc: "Generation counts, classification counts, paths, and key statistics for this run.",
  },
  {
    pattern: /\.summary\.json$/i,
    title: "Rescore summary",
    desc: "Row counts, thresholds, and output locations after rescoring.",
  },
  {
    pattern: /perturbation_.*summary\.json$/i,
    title: "Perturbation summary",
    desc: "Statistics for perturbation generation, dedup, screening, and routing.",
  },
  {
    pattern: /generation_.*summary\.json$/i,
    title: "Generation summary",
    desc: "Attempts, valid counts, and acceptance for the generation stage.",
  },
  {
    pattern: /\.log$/i,
    title: "Run log",
    desc: "Actual commands and program output; check this first when troubleshooting.",
  },
  {
    pattern: /\.csv$/i,
    title: "Result table",
    desc: "Candidate or intermediate statistics table; click to preview the first 30 rows.",
  },
  {
    pattern: /\.json$/i,
    title: "Summary file",
    desc: "Structured statistics and configuration records.",
  },
];

const DIRECTORY_HINT = {
  title: "Output folder",
  desc: "All results of this task live here. Enter and browse subfolders and files.",
};

init();

async function init() {
  bindEvents();
  updateObjectivePanels();
  updateWeightNotice();
  await loadDefaults();
  switchMode("symmetric");
  syncPerturbationMode();
  await rebuildCommand();
  pollJobs();
}

// One place decides what a field change means, so a value that arrives from the
// folder picker behaves exactly like one that was typed.  Setting
// input.value from script does not fire an "input" event, which is why the
// dependent paths (seed table, temp dedup libraries, fusion schema) used to
// stay empty when the folder was chosen with Browse.
function handleFieldUpdate(target) {
  updateObjectivePanels();
  updateVolumetricMode();
  updateWeightNotice();
  const field = target && target.dataset ? target.dataset.field : null;
  if (field === "output_dir") {
    const section = target.closest(".mode-section");
    const mode = section ? section.dataset.section : null;
    if (mode === "perturbation") {
      derivePerturbationInputsFromOutput(target.value).then(rebuildCommand);
      return;
    }
    deriveTemporarySmilesFromOutput(target.value);
    syncFusionSchemaFromOutput(target.value).then(rebuildCommand);
    return;
  }
  rebuildCommand();
}


function bindEvents() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => switchMode(button.dataset.mode));
  });
  attachBrowseButtons();
  document.querySelectorAll("[data-perturbation-mode]").forEach((radio) => {
    radio.addEventListener("change", () => {
      syncPerturbationMode();
      rebuildCommand();
    });
  });
  form.addEventListener("input", debounce((event) => {
    handleFieldUpdate(event.target);
  }, 180));
  const objectiveMode = document.querySelector("#objectiveMode");
  if (objectiveMode) {
    objectiveMode.addEventListener("change", () => {
      updateObjectivePanels();
      applyObjectivePreset(objectiveMode.value);
      updateWeightNotice();
      rebuildCommand();
    });
  }
  document.querySelector("#copyCommand").addEventListener("click", async () => {
    await navigator.clipboard.writeText(commandPreview.value);
    setStatus("Command copied", "ok");
  });
  document.querySelector("#runButton").addEventListener("click", runCurrent);
  document.querySelector("#resumeButton").addEventListener("click", resumeCurrent);
  document.querySelector("#stopButton").addEventListener("click", stopCurrent);
  document.querySelectorAll('[data-field="known_fuel_check"]').forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      document.querySelectorAll('[data-field="known_fuel_check"]').forEach((other) => {
        if (other !== checkbox) other.checked = checkbox.checked;
      });
    });
  });
  // Switching the seed pool re-points the seed table field of the perturbation
  // mode at the table that belongs to the selected pool.
  document.querySelectorAll('[data-field="seed_pool"]').forEach((select) => {
    select.addEventListener("change", () => {
      const output = document.querySelector(
        '.mode-section[data-section="perturbation"] [data-field="output_dir"]'
      );
      derivePerturbationInputsFromOutput(output ? output.value : "").then(
        rebuildCommand
      );
    });
  });
}

function attachBrowseButtons() {
  document.querySelectorAll("[data-browse-for]").forEach((button) => {
    button.addEventListener("click", () => openDirPicker(button.dataset.browseFor));
  });
}

let pickerBusy = false;

async function openDirPicker(fieldName) {
  if (pickerBusy) {
    setStatus("Opening folder picker, please wait…", "warn");
    return;
  }
  pickerBusy = true;
  setStatus("Opening folder picker…", "warn");
  const input =
    document.querySelector(`.mode-section.active [data-field="${fieldName}"]`) ||
    document.querySelector(`[data-field="${fieldName}"]`);
  if (!input) {
    pickerBusy = false;
    return;
  }
  const startPath = input.value.trim() || state.defaults.repo_root || "F:\\fuel_design";
  const apply = (path) => {
    input.value = path;
    setStatus("Output folder selected", "ok");
    // Same follow-up as typing the path: the fields below depend on it.
    handleFieldUpdate(input);
  };
  try {
    const data = await postJson("/api/pick-folder", { initial: startPath });
    if (data.canceled) {
      setStatus("Standby", "ok");
      return;
    }
    if (data.error) {
      setStatus("System picker unavailable, using built-in browser", "warn");
      showDirPicker(startPath, apply);
      return;
    }
    apply(data.path);
  } catch (error) {
    setStatus("System picker unavailable, using built-in browser", "warn");
    showDirPicker(startPath, apply);
  } finally {
    pickerBusy = false;
  }
}

window.addEventListener("error", (event) => {
  const message = event.message || String(event.error || "Unknown JS error");
  setStatus("Page script error: " + message.slice(0, 120), "bad");
  console.error("page error:", event.error || event.message);
});

function showDirPicker(startPath, onPick) {
  const overlay = document.createElement("div");
  overlay.className = "picker-overlay";
  const box = document.createElement("div");
  box.className = "picker";
  overlay.appendChild(box);
  document.body.appendChild(overlay);

  let current = startPath;

  async function render() {
    box.innerHTML = "";
    const head = document.createElement("div");
    head.className = "picker-head";
    const title = document.createElement("span");
    title.className = "picker-title";
    title.textContent = `Select output folder: ${current}`;
    head.appendChild(title);
    const actions = document.createElement("div");
    actions.className = "picker-actions";

    const up = document.createElement("button");
    up.type = "button";
    up.textContent = "Up";
    up.addEventListener("click", async () => {
      try {
        const data = await getJson(`/api/browse?path=${encodeURIComponent(current)}`);
        if (data.parent) {
          current = data.parent;
          render();
        }
      } catch (error) {
        setStatus(String(error), "bad");
      }
    });
    actions.appendChild(up);

    const confirm = document.createElement("button");
    confirm.type = "button";
    confirm.className = "primary";
    confirm.textContent = "Select current folder";
    confirm.addEventListener("click", () => {
      onPick(current);
      overlay.remove();
    });
    actions.appendChild(confirm);

    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.textContent = "Cancel";
    cancel.addEventListener("click", () => overlay.remove());
    actions.appendChild(cancel);
    head.appendChild(actions);
    box.appendChild(head);

    const list = document.createElement("div");
    list.className = "picker-list";
    try {
      const data = await getJson(`/api/browse?path=${encodeURIComponent(current)}`);
      if (data.type === "file") {
        current = data.path;
        render();
        return;
      }
      (data.entries || []).forEach((entry) => {
        if (entry.type !== "directory") return;
        const row = document.createElement("button");
        row.type = "button";
        row.className = "picker-item";
        const name = document.createElement("span");
        name.className = "picker-item-name";
        name.textContent = entry.name || entry.title || entry.path;
        const choose = document.createElement("span");
        choose.className = "picker-item-choose";
        choose.textContent = "Select";
        row.append(name, choose);
        row.addEventListener("click", (event) => {
          if (event.target === choose) {
            onPick(entry.path);
            overlay.remove();
            return;
          }
          current = entry.path;
          render();
        });
        list.appendChild(row);
      });
    } catch (error) {
      const note = document.createElement("div");
      note.className = "file-note";
      note.textContent = String(error);
      list.appendChild(note);
    }
    box.appendChild(list);
  }

  render();
}

async function loadDefaults() {
  try {
    const data = await getJson("/api/defaults");
    state.defaults = data.defaults || {};
    repoPath.textContent = data.repo_root || "";
    fillDefaults();
    setStatus("Ready", "ok");
  } catch (error) {
    setStatus("Initialization failed", "bad");
    repoPath.textContent = String(error);
  }
}

function fillDefaults() {
  setAll("checkpoint", state.defaults.checkpoint);
  setAll("semi_base_checkpoint", state.defaults.semi_base_checkpoint);
  setAll("known_smiles", state.defaults.known_smiles);
  setAll("phase_fusion_schema_version", state.defaults.phase_fusion_schema_version || "schema2_mlp");
  const fallbackRoot = `${state.defaults.repo_root || "F:\\fuel_design"}\\SCULPTOR\\result`;
  const lastRun = state.defaults.last_run_dir || "";
  setModeField("symmetric", "output_dir", lastRun || fallbackRoot);
  setModeField("perturbation", "output_dir", lastRun || fallbackRoot);
  setModeField("perturbation", "known_smiles", state.defaults.known_smiles);
  const rescoreInput = state.defaults.rescore_input || "";
  if (rescoreInput) {
    setModeField("rescore", "input", rescoreInput);
    const rescoreDir = rescoreInput.replace(/[\\/][^\\/]*$/, "");
    setModeField("rescore", "output", `${rescoreDir}\\rescored_from_panel\\candidates_rescored.csv`);
    setModeField("rescore", "summary_output", `${rescoreDir}\\rescored_from_panel\\summary.json`);
  }
  const pertOutput = document.querySelector(
    '.mode-section[data-section="perturbation"] [data-field="output_dir"]'
  );
  derivePerturbationInputsFromOutput(pertOutput ? pertOutput.value : "");
  const symOutput = document.querySelector(
    '.mode-section[data-section="symmetric"] [data-field="output_dir"]'
  );
  deriveTemporarySmilesFromOutput(symOutput ? symOutput.value : "");
  syncFusionSchemaFromOutput(symOutput ? symOutput.value : "");
}

function setAll(field, value) {
  if (!value) return;
  document.querySelectorAll(`[data-field="${field}"]`).forEach((input) => {
    if (!input.value) input.value = value;
  });
}

function deriveTemporarySmilesFromOutput(outputDir) {
  const tempInput = document.querySelector('[data-field="temporary_smiles"]');
  if (!tempInput) return;
  const value = String(outputDir || "").trim();
  tempInput.value = value
    ? value.replace(/[\\/]+$/, "") + "\\generation_temporary_smiles.txt"
    : "";
}

function derivePerturbationTemporaryFromOutput(outputDir) {
  const tempInput = document.querySelector('[data-field="perturbation_temporary_smiles"]');
  if (!tempInput) return;
  const value = String(outputDir || "").trim();
  tempInput.value = value
    ? value.replace(/[\\/]+$/, "") + "\\perturbation_temporary_smiles.txt"
    : "";
}

function derivePerturbationGenerationTemporaryFromOutput(outputDir) {
  const tempInput = document.querySelector('[data-field="perturbation_generation_temporary"]');
  if (!tempInput) return;
  const value = String(outputDir || "").trim();
  tempInput.value = value
    ? value.replace(/[\\/]+$/, "") + "\\generation_temporary_smiles.txt"
    : "";
}

async function derivePerturbationInputsFromOutput(outputDir) {
  const seedInput = document.querySelector(
    '.mode-section[data-section="perturbation"] [data-field="finalized_hits"]'
  );
  const poolSelect = document.querySelector(
    '.mode-section[data-section="perturbation"] [data-field="seed_pool"]'
  );
  const value = String(outputDir || "").trim();
  if (!value) {
    if (seedInput) seedInput.value = "";
    state.perturbationSeedTables = {};
    deriveTemporarySmilesFromOutput("");
    derivePerturbationGenerationTemporaryFromOutput("");
    derivePerturbationTemporaryFromOutput("");
    return "";
  }
  const trimmed = value.replace(/[\\/]+$/, "");
  let data = null;
  try {
    data = await postJson("/api/perturbation-inputs", { output_dir: trimmed });
  } catch (error) {
    data = null;
  }
  // Server-side helper reports the seed tables of the run folder; the
  // folder-relative names are only a fallback.
  const tables = {
    tier123: (data && data.seed_table) || `${trimmed}\\perturbation_seeds_T123.csv`,
    potential:
      (data && data.potential_seed_table) ||
      `${trimmed}\\perturbation_seeds_potential.csv`,
  };
  const available = {
    tier123: !data || data.seed_source === "tier123",
    potential: !data || Boolean(data.potential_seed_source === "potential"),
  };
  state.perturbationSeedTables = { tables, available };
  applyStoredFusionSchema(data);
  const selected = (poolSelect && poolSelect.value) || "tier123";
  const seedTable = tables[selected] || tables.tier123;
  if (seedInput) {
    seedInput.value = seedTable;
    seedInput.title =
      selected === "potential"
        ? available.potential
          ? "Two-item potential + best 10% of the one-item potential pool"
          : "This table does not exist yet: run generation (or a perturbation campaign) first"
        : "Tier 1-3 seeds (current architecture)";
  }
  deriveTemporarySmilesFromOutput(trimmed);
  derivePerturbationGenerationTemporaryFromOutput(trimmed);
  derivePerturbationTemporaryFromOutput(trimmed);
  return seedTable;
}

// Re-running into a folder that already holds products must use the same fusion
// head as those products, otherwise the merged tables would compare melting
// points predicted by two different models. The pipeline refuses such a merge;
// here the matching version is preselected instead.
function applyStoredFusionSchema(data) {
  const storedSchema = (data && data.stored_phase_fusion_schema) || "";
  const schemaSelect = document.querySelector(
    '[data-field="phase_fusion_schema_version"]'
  );
  if (storedSchema && schemaSelect && schemaSelect.value !== storedSchema) {
    schemaSelect.value = storedSchema;
    schemaSelect.title =
      "Preselected: this run folder already holds products from this fusion head version";
  }
}

async function syncFusionSchemaFromOutput(outputDir) {
  const value = String(outputDir || "").trim().replace(/[\\/]+$/, "");
  if (!value) return;
  try {
    applyStoredFusionSchema(
      await postJson("/api/perturbation-inputs", { output_dir: value })
    );
  } catch (error) {
    // A folder without products simply keeps the default version.
  }
}

function setModeField(mode, field, value) {
  const input = document.querySelector(`[data-section="${mode}"] [data-field="${field}"]`);
  if (input && value && !input.value) input.value = value;
}

function switchMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.mode === mode);
  });
  document.querySelectorAll(".mode-section").forEach((section) => {
    section.classList.toggle("active", section.dataset.section === mode);
  });
  // Each mode has its own default weights, applied on switch (each sums to 1).
  const MODE_WEIGHTS = {
    symmetric: { property: "0.65", e3nn_geometry: "0.10", e3nn_symmetry: "0.10", novelty: "0.15" },
    perturbation: { property: "0.70", e3nn_geometry: "0.15", e3nn_symmetry: "0.15", novelty: "0" },
    rescore: { property: "0.70", e3nn_geometry: "0.10", e3nn_symmetry: "0.10", novelty: "0.10" },
  };
  const weights = MODE_WEIGHTS[mode] || MODE_WEIGHTS.symmetric;
  setFieldValue("property_score_weight", weights.property);
  setFieldValue("e3nn_geometry_score_weight", weights.e3nn_geometry);
  setFieldValue("e3nn_symmetry_score_weight", weights.e3nn_symmetry);
  setFieldValue("novelty_score_weight", weights.novelty);
  // Temp dedup library defaults are tied to the active mode's output folder.
  const activeOutput = document.querySelector(
    `.mode-section[data-section="${mode}"] [data-field="output_dir"]`
  );
  if (mode === "perturbation") {
    derivePerturbationInputsFromOutput(activeOutput ? activeOutput.value : "").then(
      rebuildCommand
    );
    return;
  }
  deriveTemporarySmilesFromOutput(activeOutput ? activeOutput.value : "");
  rebuildCommand();
}

function collectPayload() {
  const params = {};
  const active = document.querySelector(`.mode-section[data-section="${state.mode}"]`);
  const commonInputs = document.querySelectorAll(".subsection [data-field]");
  const modeInputs = active ? active.querySelectorAll("[data-field]") : [];
  [...modeInputs, ...commonInputs].forEach((input) => {
    const key = input.dataset.field;
    if (!key) return;
    if (input.type === "checkbox") {
      params[key] = input.checked;
    } else {
      params[key] = input.value.trim();
    }
  });
  if (state.mode === "perturbation") {
    params.operations = [...document.querySelectorAll("[data-operation]:checked")].map(
      (item) => item.dataset.operation
    );
    const chosen = document.querySelector("[data-perturbation-mode]:checked");
    params.iterative_perturbation = chosen
      ? chosen.dataset.perturbationMode === "iterative"
      : false;
  }
  return { mode: state.mode, params };
}

// The mode switch is a pair of radio buttons; the iterative-only settings are
// hidden while the full campaign is selected, so the two modes cannot be
// confused with a stray checkbox.
function syncPerturbationMode() {
  const chosen = document.querySelector("[data-perturbation-mode]:checked");
  const iterative = chosen ? chosen.dataset.perturbationMode === "iterative" : false;
  const block = document.getElementById("iterative-only");
  if (block) {
    block.hidden = !iterative;
    block.style.display = iterative ? "" : "none";
  }
}

async function rebuildCommand() {
  try {
    const payload = collectPayload();
    const data = await postJson("/api/build-command", payload);
    commandPreview.value = data.display_command || "";
  } catch (error) {
    commandPreview.value = String(error);
  }
}

async function runCurrent() {
  try {
    const payload = collectPayload();
    const job = await postJson("/api/run", payload);
    state.activeJobId = job.id;
    renderJob(job);
    setStatus("Running", "warn");
    startPolling();
  } catch (error) {
    setStatus("Start failed", "bad");
    logBox.textContent = String(error);
  }
}

async function resumeCurrent() {
  if (!state.activeJobId) return;
  try {
    const job = await postJson("/api/resume", { job_id: state.activeJobId });
    renderJob(job);
    setStatus("Resuming", "warn");
    startPolling();
  } catch (error) {
    setStatus("Resume failed", "bad");
    logBox.textContent = String(error);
  }
}

async function stopCurrent() {
  if (!state.activeJobId) return;
  try {
    setStatus("Stopping…", "warn");
    const job = await postJson("/api/stop", { job_id: state.activeJobId });
    renderJob(job);
    setStatus("Stopped, resumable", "ok");
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  } catch (error) {
    setStatus("Stop failed", "bad");
    logBox.textContent = String(error);
  }
}

function startPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  pollActiveJob();
  state.pollTimer = setInterval(pollActiveJob, 1500);
}

async function pollActiveJob() {
  if (!state.activeJobId) return;
  try {
    const job = await getJson(`/api/jobs/${state.activeJobId}`);
    renderJob(job);
    if (["succeeded", "failed", "interrupted"].includes(job.status)) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      const message = job.status === "succeeded"
        ? "Run completed"
        : job.status === "interrupted"
          ? "Job interrupted, resumable"
          : "Run failed";
      setStatus(message, job.status === "succeeded" ? "ok" : "bad");
    }
  } catch (error) {
    logBox.textContent = String(error);
  }
}

async function pollJobs() {
  try {
    const data = await getJson("/api/jobs");
    const jobs = data.jobs || [];
    if (!state.activeJobId && jobs.length) {
      state.activeJobId = jobs[jobs.length - 1].id;
      renderJob(jobs[jobs.length - 1]);
    }
  } catch {
    return;
  }
}

function renderJob(job) {
  state.lastJob = job;
  jobMeta.textContent = `Job ${job.id} · ${labelMode(job.mode)} · ${labelStatus(job.status)} · ${job.elapsed_seconds}s`;
  logBox.textContent = (job.last_lines || []).join("\n");
  logBox.scrollTop = logBox.scrollHeight;
  renderStages(job.stages || []);
  renderOutputPanel(job);
  const resumeButton = document.querySelector("#resumeButton");
  resumeButton.hidden = !["failed", "interrupted"].includes(job.status);
  const stopButton = document.querySelector("#stopButton");
  if (stopButton) {
    stopButton.hidden = !["queued", "running"].includes(job.status);
  }
}

function renderStages(stages) {
  stageList.innerHTML = "";
  stages.forEach((stage) => {
    const item = document.createElement("div");
    item.className = `stage ${stage.state}`;
    const label = document.createElement("span");
    label.className = "stage-label";
    label.textContent = stage.label;
    item.appendChild(label);
    if (stage.detail) {
      const detail = document.createElement("span");
      detail.className = "stage-detail";
      detail.textContent = stage.detail;
      item.appendChild(detail);
    }
    stageList.appendChild(item);
  });
}

function renderOutputPanel(job) {
  if (job.output_dir !== state.currentOutputRoot) {
    state.currentOutputRoot = job.output_dir || "";
    state.currentBrowsePath = "";
  }
  if (state.currentBrowsePath) return;
  fileList.innerHTML = "";
  if (job.output_dir) {
    fileList.appendChild(
      entryButton({
        type: "directory",
        title: DIRECTORY_HINT.title,
        desc: DIRECTORY_HINT.desc,
        detail: job.output_dir,
        path: job.output_dir,
      })
    );
  }
  if (job.log_path) {
    const hint = describeFile(job.log_path);
    fileList.appendChild(
      entryButton({
        type: "file",
        title: hint.title,
        desc: hint.desc,
        detail: job.log_path,
        path: job.log_path,
      })
    );
  }
}

async function browsePath(path) {
  preview.textContent = "Reading folder...";
  try {
    const data = await getJson(`/api/browse?path=${encodeURIComponent(path)}`);
    if (data.type === "file") {
      await previewPath(path);
      return;
    }
    state.currentBrowsePath = data.path;
    renderDirectory(data);
    preview.textContent = "Select a result file to view.";
  } catch (error) {
    preview.textContent = String(error);
  }
}

function renderDirectory(data) {
  fileList.innerHTML = "";
  const head = document.createElement("div");
  head.className = "folder-head";
  const name = document.createElement("span");
  name.className = "folder-name";
  name.textContent = `Current folder: ${data.path}`;
  head.appendChild(name);
  const actions = document.createElement("div");
  actions.className = "folder-actions";
  if (data.parent) {
    const parent = document.createElement("button");
    parent.type = "button";
    parent.textContent = "Up";
    parent.addEventListener("click", () => browsePath(data.parent));
    actions.appendChild(parent);
  }
  const root = document.createElement("button");
  root.type = "button";
  root.textContent = "Back to output folder";
  root.addEventListener("click", () => {
    if (state.currentOutputRoot) browsePath(state.currentOutputRoot);
  });
  actions.appendChild(root);
  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "Collapse";
  close.addEventListener("click", () => {
    state.currentBrowsePath = "";
    if (state.lastJob) renderOutputPanel(state.lastJob);
  });
  actions.appendChild(close);
  head.appendChild(actions);
  fileList.appendChild(head);
  (data.entries || []).forEach((entry) => fileList.appendChild(entryButton(entry)));
  if (data.truncated) {
    const note = document.createElement("div");
    note.className = "file-note";
    note.textContent = "This folder has many items; only the first 200 are shown.";
    fileList.appendChild(note);
  }
}

function entryButton(entry) {
  const button = document.createElement("button");
  button.className = `file-item ${entry.type === "directory" ? "folder-item" : ""}`;
  button.type = "button";
  button.title = entry.path;
  const title = document.createElement("span");
  title.className = "file-title";
  title.textContent = entry.type === "directory" ? `Folder: ${entry.name || entry.title}` : entry.title;
  const desc = document.createElement("span");
  desc.className = "file-desc";
  desc.textContent = entry.desc || "";
  const detail = document.createElement("span");
  detail.className = "file-path";
  detail.textContent = entry.detail || entry.path || "";
  button.append(title, desc, detail);
  button.addEventListener("click", () => {
    if (entry.type === "directory") {
      browsePath(entry.path);
    } else {
      previewPath(entry.path);
    }
  });
  return button;
}

async function previewPath(path) {
  preview.textContent = "Reading...";
  try {
    const data = await getJson(`/api/preview?path=${encodeURIComponent(path)}`);
    renderPreview(data);
  } catch (error) {
    preview.textContent = String(error);
  }
}

function renderPreview(data) {
  if (data.type === "csv") {
    const columns = data.columns || [];
    const rows = data.rows || [];
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    columns.forEach((col) => {
      const th = document.createElement("th");
      th.textContent = labelColumn(col);
      th.title = col;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);
    const tbody = document.createElement("tbody");
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      columns.forEach((col) => {
        const td = document.createElement("td");
        td.textContent = labelValue(row[col] || "");
        td.title = row[col] || "";
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    preview.innerHTML = "";
    preview.appendChild(table);
    return;
  }
  const pre = document.createElement("pre");
  pre.textContent = data.type === "json" ? JSON.stringify(data.content, null, 2) : data.content;
  preview.innerHTML = "";
  preview.appendChild(pre);
}

function updateWeightNotice() {
  if (!weightNotice) return;
  const total = COMPOSITE_WEIGHT_FIELDS.reduce((sum, field) => {
    const input = document.querySelector(`[data-field="${field}"]`);
    const value = Number(input?.value || 0);
    return sum + (Number.isFinite(value) ? value : 0);
  }, 0);
  const rounded = total.toFixed(2);
  const ok = Math.abs(total - 1) < 0.0001;
  weightNotice.className = `notice ${ok ? "ok" : "warn"}`;
  weightNotice.textContent = ok
    ? `Current weight total: ${rounded}`
    : `Warning: current weight total is ${rounded}, not 1.00. Adjust to 1, otherwise the program normalizes and effective shares change.`;
}

function updateObjectivePanels() {
  const mode = getFieldValue("objective_mode") || "property_maximize";
  document.querySelectorAll("[data-objective-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.objectivePanel !== mode;
  });
}

function applyObjectivePreset(mode) {
  if (mode === "novelty_exploration") {
    setFieldValue("property_score_weight", "0.20");
    setFieldValue("e3nn_geometry_score_weight", "0.05");
    setFieldValue("e3nn_symmetry_score_weight", "0.05");
    setFieldValue("novelty_score_weight", state.mode === "perturbation" ? "0" : "0.70");
    setFieldValue("novelty_threshold", "0.75");
    return;
  }
  const base =
    state.mode === "perturbation"
      ? { property: "0.70", e3nn_geometry: "0.15", e3nn_symmetry: "0.15", novelty: "0" }
      : { property: "0.65", e3nn_geometry: "0.10", e3nn_symmetry: "0.10", novelty: "0.15" };
  setFieldValue("property_score_weight", base.property);
  setFieldValue("e3nn_geometry_score_weight", base.e3nn_geometry);
  setFieldValue("e3nn_symmetry_score_weight", base.e3nn_symmetry);
  setFieldValue("novelty_score_weight", base.novelty);
}

function updateVolumetricMode() {
  const enabled = Boolean(getFieldValue("use_volumetric_heating_value_screen"));
  if (enabled) {
    setFieldValue("volumetric_heating_value_score_weight", "1.0");
    setFieldValue("select_target_volumetric_heating_value", true);
    setFieldValue("select_target_density", false);
    setFieldValue("select_target_heating_value", false);
  } else {
    setFieldValue("volumetric_heating_value_score_weight", "0.0");
  }
}

function getFieldValue(field) {
  const input = document.querySelector(`[data-field="${field}"]`);
  if (!input) return "";
  return input.type === "checkbox" ? input.checked : input.value;
}

function setFieldValue(field, value) {
  document.querySelectorAll(`[data-field="${field}"]`).forEach((input) => {
    if (input.type === "checkbox") {
      input.checked = Boolean(value);
    } else {
      input.value = String(value);
    }
  });
}

function shortPath(path) {
  const parts = path.split(/[\\/]/);
  if (parts.length <= 4) return path;
  return `${parts.slice(-4).join("\\")}`;
}

function labelMode(mode) {
  return MODE_LABELS[mode] || mode;
}

function labelStatus(status) {
  return STATUS_LABELS[status] || status;
}

function labelColumn(column) {
  return COLUMN_LABELS[column] || column;
}

function labelValue(value) {
  return VALUE_LABELS[value] || value;
}

function describeFile(path) {
  const normalized = path.replaceAll("\\", "/");
  return FILE_HINTS.find((item) => item.pattern.test(normalized)) || {
    title: "Output file",
    desc: "A file produced by this run; click to view its content.",
  };
}

function setStatus(text, mode) {
  serverStatus.textContent = text;
  serverStatus.style.background = mode === "bad" ? "var(--red)" : mode === "warn" ? "var(--amber)" : "var(--green)";
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function debounce(fn, delay) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}
