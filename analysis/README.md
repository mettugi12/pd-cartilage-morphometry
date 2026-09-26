# v9 analysis — canonical scripts

Copied from `v8/analysis/` at study-repo commit `c5d28c6` (2026-08-18); paths resolve
against `../results/` so they run from this folder unchanged.

Pinned application pipeline (see `../DATASET.md`): `cartilage-morphometry@1947e04`,
`knee-mr-seg@d62881e`, `knee-mr-pd2dess@6616c9a`.

| Script | Role |
|---|---|
| `run_baseline_grid_v8b.py` | progressor-arm longitudinal morphometry (166) |
| `run_nonprog_v8b.py` | stable-arm longitudinal morphometry (172) |
| `aggregate_v8.py` / `aggregate_v8b.py` | stats helpers + responsiveness table |
| `aggregate_discrimination_v8b.py` | **Table 3 (new): AUC / SDC95 / detection rates** |
| `diagnose_lateral_v8b.py` | lateral-drift diagnosis (supporting) |
| `qc_icp_nonprog_v8b.py` | stable-arm registration QC (supporting) |
| `download_nonprog_iw.py` | NAS pull for the new stable knees (data acquisition) |
| `convert_stage_nonprog.py` | DICOM→NIfTI + SAG affine QC (data acquisition) |
| `run_prog_icp_v9.py` | progressor-arm rerun capturing ICP metrics (deltas bit-identical to v8b) |
| `qc_labelfree_v9.py` | **predefined label-free QC** (ICP-only primary; orientation exploratory) → QC-pass CSVs + Tables 2/3/S1 sources |
| `make_figures_v9.py` | Figure 5 (ROC) + Figure S2 (distributions w/ 95%-spec threshold) — 150/151 |
| `make_figures_long_v9.py` | Figures 3–4 (longitudinal BA + scatter) — n=150 |
| `make_cohort_flow_v9.py` | Figure S1 (cohort flow, counts computed from data) |
| `diagnose_lateral_v9.py` | lateral drift diagnosis on the 150/151 frame |
| `build_tables_v9.py` | assembles `manuscript/tables/Table1–3, S1–S2` from results |

## v9.1 additions (Scientific Reports submission, 2026-09)

| Script | Role |
|---|---|
| `drift_adjusted_contrast_v9.py` | progressor-minus-stable excess per region/method with bootstrap CIs, PD/reference ratio, drift-corrected SRM, Cohen's d, tibial thirds, drift by KL grade (Supplementary Table S7) |
| `subject_overlap_sensitivity_v9_1.py` | five participants contribute opposite knees to both arms: subject-disjoint re-analysis + participant-clustered bootstrap (Supplementary Table S8) |
| `eval_seg_test_v9_1.py` | reproducible held-out segmentation evaluation (Dice / HD95 / ASSD, pooled and by OAI vs hospital source) → Supplementary Table S1 |
| `make_figure_seg_qual_v9.py` | Supplementary Fig. S4 (held-out contours, manual vs automated, paired DESS) |

### Site-specific paths

Data-acquisition and orchestration scripts (`download_nonprog_iw.py`, `convert_stage_nonprog.py`,
`run_*_v8b.py`, `run_prog_icp_v9.py`, `make_cohort_flow_v9.py`, `eval_seg_test_v9_1.py`,
`make_figure_seg_qual_v9.py`) carry the authors' local data roots as module-level constants
(`ROOT`, `OUT_ROOT`, `MERGED_V33`, `COHORT`, `EVAL`, `TRIPLE`, `REPO`) at the top of each file.
Edit those constants for your environment. The statistical scripts (`aggregate_*`, `qc_labelfree_v9.py`,
`drift_adjusted_contrast_v9.py`, `subject_overlap_sensitivity_v9_1.py`, `sample_size_v9.py`,
`make_figures*_v9.py`) read only the per-knee CSVs in `../results/` and run anywhere.
OAI imaging and the expert manual DESS morphometry (kMRI_QCart_Eckstein) must be obtained from
https://nda.nih.gov/oai under the OAI data-use agreement.

## v9.2 (symmetric follow-up handling)

`morphometry/cartilage_morphometry/validation/shared_mesh.py` gains `_followup_symmetric` and
`baseline_grid.regional_deltas(th48_status=...)`; enabled with
`PipelineConfig.long_followup_handling = "symmetric"` (`long_idw_radius_mm` 2.0, `long_denuded_dist_mm` 1.5).
The legacy zero-imputation rule remains the library default. Scripts: `run_long_abs_v9_1.py --mode symmetric`
(absolute regional thickness at both visits + expert absolute values), `make_v9_2_frames.py`,
`analysis_v9_1_frames.py --source v9_2`, `cross_sectional_abs_v9_1.py --source v9_2`,
`diagnose_drift_v9_1.py` (forward/reverse grid-owner test), `compare_followup_modes_v9_2.py`.
