# pd-cartilage-morphometry

Source code for the analysis pipeline of the manuscript:

> Cartilage Thickness and Its 48-Month Change From Routine Proton-Density
> Fat-Suppressed Knee MRI: Agreement With Expert Dual-Echo Steady-State
> Morphometry (submitted to Scientific Reports).

Fully automated cartilage morphometry from routine sagittal proton-density
fat-suppressed knee MRI: nnU-Net 11-structure segmentation, 4x depth
super-resolution (RECON), surface-based cartilage thickness with a
patient-specific baseline-grid longitudinal projection, label-free
registration quality control, and the statistical analyses reported in the
paper.

## Layout

| Folder | Contents |
|---|---|
| `segmentation/` | `knee_mr_seg` package: nnU-Net inference wrappers, RECON depth super-resolution, atlas post-processing, DICOM/NIfTI I/O; 11/13-class label maps in `configs/` |
| `morphometry/` | `cartilage_morphometry` package: surface extraction, ray-cast thickness, atlas-template and baseline-grid projections, regional statistics; validation harness in `scripts/` |
| `analysis/` | Study analysis scripts: cohort runners, label-free registration QC, discrimination and responsiveness tables, figures; `analysis/v9_3/` = the submitted version |
| `data/` | Per-knee 48-month change values, absolute regional thickness, quality-control flags and held-out segmentation metrics with anonymised participant / knee / case codes (no OAI participant IDs, no hospital identifiers) |

## What is NOT included

- **Trained model weights** (segmentation and super-resolution) and the
  hospital training cohort derive from clinical data and are excluded. They
  are available from the corresponding author on reasonable request, subject
  to institutional approval and a data-use agreement.
- **Imaging data.** Osteoarthritis Initiative (OAI) images and the expert
  QCart readings are publicly available from the NIMH Data Archive
  (https://nda.nih.gov/oai).

## Requirements

Python >= 3.10; `nnunetv2`, `torch`, `numpy`, `scipy`, `SimpleITK`,
`nibabel`, `scikit-image`, `trimesh`, `matplotlib`. See each subfolder's
`pyproject.toml`.

Analysis scripts reference local data paths (e.g., `E:/...`); adapt them to
your environment. They are provided for transparency and reproducibility of
the reported statistics.

## v9.3 (submitted version)

Measurement definitions matched to the expert reference (`morphometry/cartilage_morphometry/baseline_grid.py`):

- `regional_deltas(..., femur_region="eckstein75")` / `PipelineConfig.long_femur_region`: central femoral regions
  delimited as in the reference, from the trochlear-notch plane up to the first crossing of a plane at 75 % of the
  anterior–posterior notch-to-posterior-end distance (landmarks from `subregions.find_femur_eckstein_landmarks`).
- `regional_deltas(..., total_bone_area=True)` / `PipelineConfig.long_total_bone_area`: thickness averaged over the
  total subchondral bone area of the baseline plate (denuded bone = 0 mm at either visit; failed ray casts excluded
  from both visits), femur and tibia.
- Expert reference restricted to the sagittal-DESS 75 %-definition reading projects, paired within project
  (`analysis/v9_3/build_long_abs_v9_3.py`).

Library defaults are unchanged (legacy grid band, cartilage-covered means). Scripts and aggregate results in
`analysis/v9_3/`; per-knee data in `data/`. Pinned private development commits: `cartilage-morphometry@383d4b2`,
`knee-mr-seg@d62881e`, `knee-mr-pd2dess@6616c9a`.

## License

MIT (see `LICENSE`).
