# Template build scripts

The DESS bone+subchondral templates that `knee_mr_cartilage` uses as the
patient-registration target.

Per bone, the build does:
  per case (DESS seg):
    extract bone + cart binary masks
    marching_cubes(closed+smoothed bone) → bone surface mesh in physical mm
    per-vertex `subch_dist_mm` = distance to nearest cart voxel (mm)
    per-vertex `subch` = (dist <= 1mm)
    anisotropic scale to fixed bounds (femur: ML=75/AP=65 from interface;
                                       tibia: ML=75/AP=55 from full bone)
  cross-case:
    pick first case as reference
    register subsequent bones to reference via translation
    correspondence-average vertex positions + per-vertex scalars

Outputs at `E:/KneeMR/Datasets/Bone-Atlas/`:
  femur_template_full_with_subch.vtk    (152K verts; subch_prob, subch_dist_mm)
  tibia_template_full_with_subch.vtk    ( 85K verts; subch_prob, subch_dist_mm,
                                          compartment after add_tibia_compartment.py)

By construction: subchondral surface is a literal subset of bone vertices.

## Run (defaults match the MOAKS-Regeneration study cohort)

```bash
# 327 healthy KLG=0 RIGHT knees from OAI-MOAKS — currently the canonical cohort
python scripts/templates/build_femur_template.py
python scripts/templates/build_tibia_template.py
python scripts/templates/add_tibia_compartment.py     # adds med/lat label

python scripts/templates/view_template.py \
    --vtk "E:/KneeMR/Datasets/Bone-Atlas/femur_template_full_with_subch.vtk"
```

Each build script supports `--dess_glob` for a different cohort, `--n_run N` for
a quick test on N cases, and `--out_dir` to write somewhere other than
`E:/KneeMR/Datasets/Bone-Atlas/`.

## Cohort that produced the shipped templates

`E:/OAI_all/OAI_MOAKS/48M_PRED_MINI/predicted_3d_good_moaks_score_RIGHT/*.nii.gz`
(327 healthy KLG=0 RIGHT knees, OAI baseline). DESS labels: tibia bone=3 cart=4,
femur bone=1 cart=2.

The cohort selection is documented by the
`mettugi12/knee-mr-moaks-regeneration` study (the study that drove the build).
To rebuild with a different cohort, just point `--dess_glob` at it.

## When to rerun

Only when:
  - the cohort definition changes (different KLG filter, different N, etc.)
  - the build algorithm is updated meaningfully
  - we add a new bone (e.g. patella)
