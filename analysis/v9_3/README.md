# v9.3 analysis — reference-matched regions, total subchondral bone area, 75 %-definition reference

Scripts of the Scientific Reports submission (study repo `v9.3/analysis/`). They expect the study layout
`<study>/v9.3/analysis/` (this folder), `<study>/v9.3/results/` and the shared `<study>/v9/{analysis,results}/`
(helpers `aggregate_v8.py`, `aggregate_v8b.py`, `aggregate_discrimination_v8b.py` and the v9 QC-pass lists, all in
`../`). Four scripts carry the authors' local roots as module constants (`REPO`, `CAP`, `CAP_TIB`, `OAI`, `POOL`,
`COHORT`); set `REPO` to `../../morphometry` and the data roots to your environment.

| Script | Role |
|---|---|
| `build_long_abs_v9_3.py` | per-knee absolute thickness and 48-month change: first-crossing 75 % femoral regions (`femur_region="eckstein75"`), total-subchondral-bone-area means (`total_bone_area=True`), expert from the sagittal-DESS 75 %-definition projects paired within reading project; `--femur_region grid`, `--covered_area`, `--expert all` give the one-factor-removed variants (Supplementary Table S7) |
| `make_frames_v9_3.py` | restrict to the registration-QC-pass, one-knee-per-participant lists |
| `endpoints_v9_3.py` | every number of Tables 2–3 and Supplementary Tables S3–S5, S7 (`--frame leakfree` = primary 126/134; `full` = 150/151) |
| `cross_sectional_abs_v9_3.py` | regional absolute thickness vs the expert at both visits (Fig. 3, Supplementary Table S6) |
| `qc_decomposition_v9_3.py` | registration / orientation QC decomposition (Supplementary Table S2) |
| `run_factorial_v9_3.sh` | runs the three one-factor-removed variants end to end |
| `make_figures_v9_3.py`, `make_figures_long_v9_3.py`, `make_cohort_flow_v9_3.py` | Fig. 4–5, Supplementary Fig. S1–S2, S4 |
| `make_figure_regions_v9_3.py`, `template_panels_v9_3.py`, `viz_femur_regions_webstyle_v9_3.py`, `make_figure_measurement_v9_3.py`, `relabel_figure1_v9_3.py` | Fig. 1–2 and region diagnostics |

`results/` holds the aggregate endpoint files the manuscript builder reads (participant identifiers removed).
Per-knee values with anonymised participant/knee codes are in `../../data/`.
