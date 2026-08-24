# pd-cartilage-morphometry

Source code for the analysis pipeline of the manuscript:

> Longitudinal Cartilage Thickness Change From Routine Proton-Density
> Fat-Suppressed Knee MRI: Responsiveness and Progression Discrimination
> Against Dual-Echo Steady-State Morphometry (submitted).

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
| `analysis/` | Study analysis scripts: cohort runners, label-free registration QC, discrimination and responsiveness tables, figures |

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

## License

MIT (see `LICENSE`).
