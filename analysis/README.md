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
