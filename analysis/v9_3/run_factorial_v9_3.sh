#!/bin/bash
# 2026-09-30 — one-factor-removed variants of the v9.3 specification for Supplementary Table S7 (definitions sensitivity)
export PYTHONIOENCODING=utf-8
PY=~/miniconda3/envs/colab/python.exe
cd "$(dirname "$0")"
run () { tag=$1; shift; echo "=== $tag $(date)"; $PY build_long_abs_v9_3.py --tag $tag "$@" && $PY make_frames_v9_3.py --source $tag && $PY endpoints_v9_3.py --source $tag --frame leakfree; echo "=== done $tag $(date)"; }
run v9_3_gridband --femur_region grid
run v9_3_covered --covered_area
run v9_3_allprj --expert all
