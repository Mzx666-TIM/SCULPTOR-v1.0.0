# Control Panel User Guide

This control panel visualizes project parameters, builds run commands, launches
programs, and shows logs and result tables.

## Desktop Version (recommended)

No server and no browser are needed; double-click to open a native window:

```text
tools\control_panel\open_control_panel.bat
```

You can also double-click `tools\control_panel\desktop_panel.pyw` directly, or
`tools\control_panel\open_control_panel.vbs` for a silent start. The window
interface is identical to the web version; configuration, launching, logs and
result browsing all happen inside the window.

## Supported Modes

1. Symmetric generation: runs `run-pipeline`. If a carbon range is set, the
   panel runs each carbon in sequence and writes results into `C7`, `C8`, etc.
   subfolders under the output folder.
2. Perturbation: runs `run-perturbation-pipeline`.
3. Rescore: runs `rescore-candidates`.

## Panel Features

- Choose a research objective before running:
  - Property-guided: no target values; rank by the composite of property score,
    e3nn geometry/symmetry and novelty (SA is included in the property score).
  - Target-property-guided: set density, heating value, melting point, etc.;
    property scores reward closeness to the targets.
  - Novelty exploration: uses novelty-weighted presets and prioritizes
    structurally different molecules.
- Configure attempts, random seed, 6-17 carbon range, thresholds, composite
  weights, and internal property score weights visually.
- Configure property prediction methods: density, mass heating value (derived
  from STA formation enthalpy), melting/boiling/flash (STA + schema2/schema3
  fusion heads), formation enthalpy, ISP, etc., all STA-related models; the
  panel uses the default model files and data paths automatically.
- Auto-generates PowerShell commands for review and copying.
- Perturbation picks its seed table from a "seed pool" selector: the Tier 1-3
  pools (`perturbation_seeds_T123.csv`) or the potential pools
  (`perturbation_seeds_potential.csv`, written by the generation and perturbation
  chains). Both tables live in the run folder and are refreshed automatically.
- "Keep the products already in the output folder" (on by default) merges a new
  perturbation campaign into the products the folder already holds, so every
  campaign of a run appears in one set of tables; untick it to replace them. The
  fusion-head version is preselected from the folder, and the pipeline refuses a
  merge that would mix two model versions.
- "Start Run" shows task status, stages, logs, and output files in the panel.
- Job state is persisted to disk; after an unexpected exit, the task can resume
  from the completed carbon instead of rerunning the whole range.
- The result area defaults to the output folder and run log; clicking the output
  folder enters subfolders to browse files.
- File entries show purpose, row count, main columns or JSON keys; clicking a
  CSV or JSON result file opens a preview.

## Option reference

The panel itself now carries only short labels.  This section holds the meaning
of every control, grouped the way the form is laid out.

### Research objective

| Control | Meaning |
|---|---|
| Compute device | `auto` (default) probes CUDA once with a short smoke test and takes the GPU path only if the probe succeeds, otherwise the CPU path. `cpu` never touches CUDA, so the whole workflow runs on the CPU - use it to reproduce a run on a machine without a GPU. `gpu` uses the GPU for the property heads and stops with an explanatory error when CUDA cannot actually be used, instead of falling back silently. Only the property heads (STA, phase fusion, e3nn) can use a GPU; generation, screening, perturbation and analysis always run on the CPU. The choice is forwarded to every command the panel launches. |
| Objective for this run | `Property-guided (no targets)` needs no target values and ranks candidates by the composite of property score, e3nn geometry/symmetry and novelty using the weights below. `Target-property-guided` switches property scoring from "higher/lower is better" to "closer to the target is better" and uses the target fields. `Novelty exploration` raises the novelty weight to explore chemical space; exact-match deduplication still runs first. |
| Target values | Only used by `Target-property-guided`: tick a property and give the target (density, mass heating value, melting point, boiling point, flash point, volumetric heating value). Unticked properties keep their default direction. |
| Relative window | Width of the "target reached" band around each target; 0.15 means the full score is reached 15% away from the target. |

### Symmetric generation

| Control | Meaning |
|---|---|
| Classic fuel hit check | Compare every batch of generated molecules with the classic-fuel library and write a hit table next to the run. |
| Semi-base / bond-completion checkpoint, thresholds | The two-stage GNN weights and their inference thresholds (defaults 0.73 and 0.94); the panel preselects the production checkpoints. |
| Total deduplication library | Reference library used for exact-match deduplication and novelty scoring; the panel points at `Generation/screening/data/known_smiles_total_reference.txt`. |
| Temporary smiles | File that records the structures already produced in this run, so a structure is evaluated at most once per run. |
| Carbon range, samples, iterations | Each carbon number in the range runs `iterations` rounds of `samples` requests; total requests = samples x iterations per carbon. |
| Temperature, top-k | Sampling temperature and the beam width of the skeleton enumeration. |
| Feedback controls | Online property feedback: overall strength, the warm-up weight and its number of rounds, the weight of property feedback afterwards, the reward slope, the split/edge feedback gains, the exploration fraction and the reward on/off switch. See the main README for the feedback design. |

### Perturbation

| Control | Meaning |
|---|---|
| Perturbation mode | The first switch in the section. **Full campaign** perturbs every Tier 1-3 seed once with every selected operator (10 attempts per seed and operator for the 784 seeds of the C9-C15 run, i.e. 54,880 requests). **Iterative** keeps a rolling queue and runs the rounds configured below, re-sampling the leads that are not Tier 1 yet. The round settings only apply to the iterative mode and are hidden while the full campaign is selected. |
| Perturbation seed pool | `Tier 1-3 pool` uses `perturbation_seeds_T123.csv` (every Tier 1, 2 and 3 candidate of the run). `Potential pools` uses `perturbation_seeds_potential.csv`: every Potential 1 molecule plus the best 10% of Potential 2 and of Potential 3 by composite score. Both tables live in the run folder and are refreshed automatically. |
| Add to the products already in the output folder | On by default: a new campaign is merged into the products the folder already holds so every campaign appears in one set of tables. Untick to replace them. The pipeline refuses a merge that would mix two fusion-head versions. |
| Attempts per group, samples per seed-operation, max seeds per round | Budget controls for a campaign. The checkbox switches between "per seed x operation" and a total budget. |
| Skip MMFF geometry | Skips the force-field geometry check to gain speed. |
| Operator attempts | Per-operator request split, for example `ch2_bridge_move=4,ring_expand=4,methyl_shift=2`; leaving it blank uses the weights of the iteration model. |
| Rounds without improvement, per-round feedback | Stop after N rounds without a new Tier 1 (0 = never); predict properties each round and feed them back; how strongly each round may shift the operator split. |

### Rescore

Input table, output table and summary path for `rescore-candidates`.

### Property thresholds

The hard thresholds (density minimum, mass heating value minimum, melting point
maximum, boiling point minimum, flash point minimum, volumetric heating value
minimum), the checkbox that enables each one, the novelty threshold, the
uncertainty scale, and the screening limits (maximum ring count, maximum SA
score, worker counts, reproducibility scope, missing-property penalty). A
candidate that fails an enabled threshold is routed to the Potential pools
instead of being discarded.

### Composite score weights

Weights of the four top-level terms - property composite, e3nn geometry, e3nn
symmetry and novelty. They must sum to 1; the panel warns and normalises
otherwise.

### Property score internal weights

Weights of density, mass heating value, volumetric heating value, melting point,
boiling point, flash point, SA score and structure validity inside the property
composite.

### Model paths

| Selector | Options |
|---|---|
| Density prediction method | `STA density model (default)` predicts with the GDB multi-property STA head; `Do not predict density` leaves the column empty and skips density thresholds. |
| Mass heating value prediction method | `STA formation-enthalpy derived (default)` computes the mass heating value from the STA formation enthalpy through Hess's law; `Do not predict mass heating value` excludes it from scoring, screening and missing-property checks. |
| Phase-fusion version | `2D MLP ensemble (default)` is the reported run; `3D MLP ensemble (optional)` is the descriptor variant. |
| Melting / boiling / flash point prediction method | `MLP fusion head (default)` uses the fusion ensemble; the `STA` option uses the STA head; `Do not predict ...` skips the property. |
| Formation enthalpy / ISP method | STA head by default; ISP can be disabled (the released weights do not include an ISP head). |
| 3D symmetry scoring method | `e3nn 3D geometry/symmetry scorer` scores spatial plausibility and symmetry from 3D conformers; `Rule-based geometry scoring` uses only the built-in geometry rules and needs no e3nn file. |

### Tooltips

Hovering the tier badges and the seed-pool fields in the result area shows the
longer description of that tier or pool; those texts are deliberately kept out of
the form itself.

## Notes

- In symmetric generation, `samples` is requests per round and `iterations` is
  the number of rounds; total requests = samples x iterations. This is not a
  guarantee of usable molecules produced.
- With feedback reward and per-round property feedback enabled, the program
  screens each round first, then predicts properties for passing candidates.
  Continuous property scores, composite scores, and the screening verdict
  join SA, H/C ratio, ring count, MMFF and novelty in the online reward.
- Online feedback adjusts the next round's bond threshold, sampling temperature,
  top-k, and `n1/n2/n3` sampling tendencies. Rewards for duplicate SMILES are
  halved so the model does not repeatedly exploit one high-scoring structure.
- "Property reward slope" (default 2.5) amplifies score differences near the
  pass boundary; "partition feedback gain" (default 6.0) controls how fast good
  `n1/n2/n3` partitions gain sampling share; "min exploration fraction" (default
  0.12) keeps 12% of attempts for other partitions.
- "Edge feedback gain" (default 2.0) learns the local bond types used by
  high-scoring molecules (endpoint classes, pre-bond degree, ring-closure
  distance) and directly corrects candidate edge probabilities next round.
- Default feedback strength is raised to 1.25. For long formal iterations, use
  defaults first; if properties do not move, raise the reward slope to 3.0-3.5
  instead of dropping the exploration fraction to 0.
- Structural rewards are hierarchical: valid mirror/valence and SMILES,
  connectivity, RDKit descriptors, MMFF embeddability, and SA/H/C/ring
  screening each add reward levels. Invalid bond actions or partitions receive
  zero/negative feedback.
- When at least 25% of a round's candidates duplicate the previous round, the
  program raises temperature, widens candidate edge ranges, and raises uniform
  exploration up to 35%.
- "Warmup property reward weight" (default 0.35) applies to early rounds;
  "property feedback weight" (default 0.70) applies later. The first 20 rounds
  prioritize validity and synthesizability, then property orientation.
- `generation_feedback.csv` records per-round property means, amplified target
  scores, pass/review/reject counts, and preferred `n1/n2/n3` partitions;
  `online_property_feedback.csv` stores molecules used in rewards with full
  predictions.
- Sampling temperature controls randomness: lower is more conservative, higher
  explores more diverse structures.
- Melting point screening defaults off; when off it only outputs predicted
  value, uncertainty, and possible interval without a screening verdict. Check
  "Melting point participates in screening" to include it in the hard thresholds.
- The model area shows method options, not underlying file paths; as long as
  defaults stay consistent, panel commands work after server migration.
- Composite weights should sum to 1; the panel shows the current total live.
- Default composite weights: property 0.65, e3nn geometry 0.10, e3nn symmetry
  0.10, novelty 0.15.
- Property internal weights set relative shares among enabled density, mass or
  volumetric heating value, melting/boiling/flash point, SA, and structure
  validity. SA is scored linearly (<=1.5 -> 1, 3.0 -> 0.5, >=4.5 -> 0);
  structure validity comes from canonical SMILES, connectivity, and MMFF
  stability.
- When volumetric mode is off, volumetric heating value is a reference output
  (density x mass heating value); when on, it replaces density and mass heating
  value in targets, scoring, and screening.
- Candidates missing a selected property are penalized 0.15 per missing item
  and go to `missing_property_candidates.csv` with all available fields kept.
- In target-property mode, "objective scoring window" is the relative deviation
  allowed; e.g., 0.15 means the property score drops to 0 at 15% deviation and
  reaches 1 exactly at the target.
- STA is not a master switch; each property has its own optional model:
  density, heating value, melting/flash point, formation enthalpy, and ISP can
  each choose an original method, STA, or none. STA choices write the
  corresponding `external_...` columns while keeping full `sta_...` raw
  prediction columns.
- Molar mass is computed from formula and built-in C/H atomic weights; molar
  volume is converted from molar mass and density with propagated uncertainty.
  Neither directly participates in scoring or rewards, but molar volume feeds
  the melting/boiling/flash fusion models.
- Dedup uses the SMILES-only GDB-17 500k library by default; fingerprints and
  exact sets are cached per carbon and reused.
- Run stages come from internal stage markers; at startup dependency loading
  may pause briefly at "Preparing command".
- The property score uses a 15% threshold window by default: 0.5 at threshold,
  1 at 15% better, 0 at 15% worse.

## Long-Run Resume

- Each panel task saves a `.job.json` state under `SCULPTOR/result/control_panel_jobs`.
- A carbon range runs one command per carbon; after an unexpected interruption,
  click "Resume Incomplete Job" to continue from the first unfinished command.
- Completed carbon output folders are not regenerated.

## STA Multi-Property Models

Prepare the GDB-13/GDB-17 multi-property training table:

```powershell
$env:PYTHONPATH='<SCULPTOR_ROOT>'
.\.venv\Scripts\python.exe -m fuel_design.cli prepare-sta-gdb-dataset --input <SCULPTOR_ROOT>\dataset\GDB\gdb13_g.csv --output <SCULPTOR_ROOT>\Evaluation_Iteration\evaluation\data\gdb13_sta_multi_property.csv
```

Train STA multi-property models:

```powershell
$env:PYTHONPATH='<SCULPTOR_ROOT>'
.\.venv\Scripts\python.exe -m fuel_design.cli train-sta-paper-property-models --data <SCULPTOR_ROOT>\Evaluation_Iteration\evaluation\data\gdb13_sta_multi_property.csv --output-dir <SCULPTOR_ROOT>\Evaluation_Iteration\evaluation\model\STA
```

Append STA predictions to an existing results table:

```powershell
$env:PYTHONPATH='<SCULPTOR_ROOT>'
.\.venv\Scripts\python.exe -m fuel_design.cli predict-sta-properties --input <SCULPTOR_ROOT>\result\<run>\property_predictions.csv --output <SCULPTOR_ROOT>\result\<run>\property_predictions_with_sta.csv --model <SCULPTOR_ROOT>\Evaluation_Iteration\evaluation\model\STA\sta_820k_v2\sta_gdb_density_paper.pt
```
