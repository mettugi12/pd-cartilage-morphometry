PD vs DESS Slice Viewer
=======================

REQUIREMENTS
------------
Python 3.8+  (Anaconda / Miniconda recommended)
  numpy, matplotlib, scipy   (all standard in Anaconda)

tkinter must be available — it comes with standard Python and Anaconda.
If using a minimal Python install, run:
  pip install numpy matplotlib scipy


QUICK START (Windows)
---------------------
1. Install Python / Anaconda if not already installed.
   https://www.anaconda.com/download

2. Open a terminal in this folder and install dependencies (one-time):
     pip install numpy matplotlib scipy

3. Double-click  launch_viewer.bat
   — or from a terminal:
     python viewer_app.py


QUICK START (Mac / Linux)
--------------------------
1. Install dependencies:
     pip install numpy matplotlib scipy

2. Run:
     python viewer_app.py


FOLDER STRUCTURE
----------------
PD_DESS_Viewer/
  viewer_app.py
  launch_viewer.bat
  requirements.txt
  README.txt
  nifti/
    label_key.txt
    <patient_id>/
      DESS_MR.nii.gz              DESS MR (0.7 mm isotropic)
      DESS_labels.nii.gz          DESS segmentation label map
      PD_MR_registered.nii.gz    PD MR registered to DESS space
      PD_labels_registered.nii.gz Registered PD seg label map
      PD_MR_native.nii.gz        PD MR at native z-spacing (~40 slices)


CONTROLS
--------
  Patient dropdown  — select patient
  Slice slider      — scroll through DESS slices
  Arrow keys:  ← →  ±1 slice,   ↑ ↓  ±5 slices
  Label checkboxes  — toggle segmentation overlay
  Alpha slider      — overlay opacity


LABEL MAP KEY
-------------
  0 = background
  See nifti/label_key.txt for structure labels (1–11)
