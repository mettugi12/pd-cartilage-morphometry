"""v9.3 — Per-knee absolute + change table with the Chondrometrics-matched femoral regions and the
sagittal-DESS 75 % expert reference.

What changes vs v9.2 (everything else identical):
  automated femur  cMF / cLF = notch-anchored first-crossing 75 % regions
                   (cartilage-morphometry `baseline_grid.regional_deltas(..., femur_region="eckstein75")`;
                   landmarks from `subregions.find_femur_eckstein_landmarks`), recomputed from the exact
                   per-knee baseline-grid inputs captured from the v9.2 production run
                   (v9/analysis/capture_femur_vertices_v9_2.py -> E:/KneeMR/Studies/PD-vs-DESS/v9.2/femur_region_diag/).
                   Thickness over the TOTAL subchondral bone area of the baseline plate (`total_bone_area=True`):
                   articular bone inside the outer contour of the baseline plate without cartilage = 0 mm at either
                   visit (Chondrometrics ThCtAB convention); failed measurements dropped from both visits per vertex.
                   MFTC = cMF + MT, LFTC = cLF + LT recomputed.
  automated tibia  MT / LT (and a/c/p thirds): same regions as v9.2, but thickness over the total subchondral
                   bone area (`total_bone_area=True`), recomputed from captured tibia inputs
                   (E:/KneeMR/Studies/PD-vs-DESS/v9.2/tibia_region_diag/). Knees without a tibia capture are dropped.
  expert           OAI kMRI_QCart_Eckstein rows restricted to the sagittal-DESS 75 %-definition projects
                   (09B, 22 / 22b, 66) and paired within READPRJ (22 = 22b) before averaging, as the OAI
                   documentation requires for change scores (Eckstein Descrip. p3). v9.2 averaged every row,
                   mixing coronal FLASH (04/07/18), coronal MPR (08) and sagittal 60 % (09A) readings.

Sanity check: the grid-mode femur recomputed from the captured inputs must reproduce the v9.2 production
cMF / cLF values.

Output: ../results/v9_3_long_abs.csv (same columns as v9/results/v9_2_long_abs.csv)
Usage: python build_long_abs_v9_3.py [--n_run N] [--dry_run]

2026-09-30 factorial variants (one definition removed at a time from the final specification; output ../results/<tag>_long_abs.csv):
  --femur_region grid --tag v9_3_gridband      grid femoral band instead of the first-crossing 75 % regions
  --covered_area --tag v9_3_covered            cartilage-covered means instead of total subchondral bone area (femur and tibia)
  --expert all --tag v9_3_allprj               all reading projects averaged instead of the 75 %-definition sagittal-DESS projects
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
V9RES = HERE.parent.parent / "v9" / "results"
CAP = Path(r"E:/KneeMR/Studies/PD-vs-DESS/v9.2/femur_region_diag")
CAP_TIB = Path(r"E:/KneeMR/Studies/PD-vs-DESS/v9.2/tibia_region_diag")
TIB_REGIONS = ["MT", "LT", "aMT", "cMT", "pMT", "aLT", "cLT", "pLT"]
OAI = Path(r"E:/KneeMR/Datasets/OAI/OAICompleteData_ASCII")
REPO = r"c:/Users/mettu/OneDrive/바탕 화면/Connecteve_Research/KneeMR/Repos/cartilage-morphometry"
QCOL = {"cMF": "BMFMTH", "cLF": "BLFMTH", "MT": "WMTMTH", "LT": "WLTMTH"}
P75 = {"09B", "22", "22B", "66"}


def load_expert_75(projects=P75):
    """(pid str, side) -> eck_<reg>_{00m,48m,d}: 75 %-definition sagittal-DESS projects, paired within READPRJ."""
    tabs = {}
    for v, f in (("V00", "kMRI_QCart_Eckstein00.txt"), ("V06", "kmri_qcart_eckstein06.txt")):
        d = pd.read_csv(OAI / f, sep="|", low_memory=False)
        d.columns = [c.upper() for c in d.columns]
        d = d[d.READPRJ.astype(str).str.upper().isin(projects)].copy()
        d["pid"] = d.ID.astype(int).astype(str)
        d["side"] = np.where(d.SIDE.astype(str).str.contains("1"), "RIGHT", "LEFT")
        d["prj"] = d.READPRJ.astype(str).str.upper().str.replace("22B", "22")
        for reg, col in QCOL.items():
            d[reg] = pd.to_numeric(d[f"{v}{col}"], errors="coerce")
        tabs[v] = d[["pid", "side", "prj"] + list(QCOL)]
    m = tabs["V00"].merge(tabs["V06"], on=["pid", "side", "prj"], suffixes=("_00m", "_48m"))
    g = m.groupby(["pid", "side"]).mean(numeric_only=True)
    out = pd.DataFrame(index=g.index)
    for reg in QCOL:
        out[f"eck_{reg}_00m"], out[f"eck_{reg}_48m"] = g[f"{reg}_00m"], g[f"{reg}_48m"]
    for comp, (a, b) in (("MFTC", ("cMF", "MT")), ("LFTC", ("cLF", "LT"))):
        for k in ("00m", "48m"):
            out[f"eck_{comp}_{k}"] = out[f"eck_{a}_{k}"] + out[f"eck_{b}_{k}"]
    for reg in list(QCOL) + ["MFTC", "LFTC"]:
        out[f"eck_{reg}_d"] = out[f"eck_{reg}_48m"] - out[f"eck_{reg}_00m"]
    out["eck_projects"] = m.groupby(["pid", "side"]).prj.apply(lambda s: "+".join(sorted(set(s))))
    return out


def femur_from_capture(f: Path, bg, femur_region="eckstein75", tba=True):
    z = np.load(f, allow_pickle=True)
    shape = tuple(z["mask_shape"]); n = int(np.prod(shape))
    bone = np.unpackbits(z["bone"])[:n].reshape(shape).astype(bool)
    cart = np.unpackbits(z["cart"])[:n].reshape(shape).astype(bool)
    args = (z["points"], z["th00"], z["th48"], bone, cart, tuple(z["spacing"]))
    kw = dict(laterality="right_oriented", femur_unwrap="per_slice", th48_status=z["status"])
    grid = bg.regional_deltas("femur", *args, **kw)
    fr = {} if femur_region == "grid" else {"femur_region": femur_region}
    eck = bg.regional_deltas("femur", *args, total_bone_area=tba, **fr, **kw)
    info = dict(eck.get("_femur_region_info", {}))
    info.update({k: v for k, v in (eck.get("_tab_info") or {}).items() if not k.startswith("vertex_")})
    return grid, eck, info


def tibia_from_capture(f: Path, bg, tba=True):
    z = np.load(f, allow_pickle=True)
    shape = tuple(z["mask_shape"]); n = int(np.prod(shape))
    bone = np.unpackbits(z["bone"])[:n].reshape(shape).astype(bool)
    cart = np.unpackbits(z["cart"])[:n].reshape(shape).astype(bool)
    args = (z["points"], z["th00"], z["th48"], bone, cart, tuple(z["spacing"]))
    kw = dict(laterality="right_oriented", femur_unwrap="per_slice", th48_status=z["status"])
    grid = bg.regional_deltas("tibia", *args, **kw)
    tab = bg.regional_deltas("tibia", *args, total_bone_area=tba, **kw)
    info = {k: v for k, v in (tab.get("_tab_info") or {}).items() if not k.startswith("vertex_")}
    return grid, tab, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_run", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--exclude_22", action="store_true",
                    help="sensitivity: expert from projects 66 and 09B only (the 22/22b fraction is documented inconsistently); "
                         "knees without such a reading are dropped; output source v9_3ex22")
    ap.add_argument("--femur_region", default="eckstein75", choices=["eckstein75", "grid"])
    ap.add_argument("--covered_area", action="store_true", help="average over cartilage-covered vertices only (total_bone_area=False)")
    ap.add_argument("--expert", default="p75", choices=["p75", "all"], help="all = keep the v9.2 all-project average expert values")
    ap.add_argument("--tag", default=None, help="output source tag (default v9_3 / v9_3ex22)")
    a = ap.parse_args()
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from cartilage_morphometry import baseline_grid as bg

    base = pd.read_csv(V9RES / "v9_2_long_abs.csv")
    base["pid"] = base.pid.astype(str)
    if a.n_run:
        base = base.head(a.n_run)
    ex = load_expert_75({"09B", "66"} if a.exclude_22 else P75)
    src = a.tag or ("v9_3ex22" if a.exclude_22 else "v9_3")
    tba = not a.covered_area
    rows, maxdiff, missing = [], 0.0, []
    maxdiff_tib, missing_tib = 0.0, []
    for r in base.itertuples(index=False):
        rec = r._asdict()
        f = CAP / f"{r.pid}_{r.side}.npz"
        if not f.exists():
            missing.append(f"{r.pid}_{r.side}"); continue
        grid, eck, info = femur_from_capture(f, bg, a.femur_region, tba)
        for reg in ("cMF", "cLF"):
            for k in ("00m", "48m"):
                v2 = rec[f"pd_{reg}_{k}"]
                if np.isfinite(v2):
                    maxdiff = max(maxdiff, abs(grid[reg][k] - v2))
                rec[f"pd_{reg}_{k}"] = eck[reg][k]
            rec[f"pd_{reg}_d"] = eck[reg]["d"]
        ft = CAP_TIB / f"{r.pid}_{r.side}.npz"
        if not ft.exists():
            missing_tib.append(f"{r.pid}_{r.side}"); continue
        tgrid, ttab, tinfo = tibia_from_capture(ft, bg, tba)
        for reg in TIB_REGIONS:
            for k in ("00m", "48m"):
                v2 = rec[f"pd_{reg}_{k}"]
                if np.isfinite(v2):
                    maxdiff_tib = max(maxdiff_tib, abs(tgrid[reg][k] - v2))
                rec[f"pd_{reg}_{k}"] = ttab[reg][k]
            rec[f"pd_{reg}_d"] = ttab[reg]["d"]
        rec.update({"tibia_tab_den00_frac": tinfo.get("n_den00", np.nan) / max(tinfo.get("n_kept", 1), 1),
                    "tibia_tab_den48_frac": tinfo.get("n_den48", np.nan) / max(tinfo.get("n_kept", 1), 1)})
        for comp, (x, y) in (("MFTC", ("cMF", "MT")), ("LFTC", ("cLF", "LT"))):
            for k in ("00m", "48m", "d"):
                rec[f"pd_{comp}_{k}"] = rec[f"pd_{x}_{k}"] + rec[f"pd_{y}_{k}"]
        rec.update({"notch_ap_mm": info.get("ap_notch"), "notch_to_post_med_mm": info.get("ap_end_med", np.nan) - info.get("ap_notch", np.nan),
                    "femur_tab_den00_frac": info.get("n_den00", np.nan) / max(info.get("n_kept", 1), 1),
                    "femur_tab_den48_frac": info.get("n_den48", np.nan) / max(info.get("n_kept", 1), 1),
                    "femur_tab_fail00": info.get("n_fail00", np.nan), "femur_tab_fail48": info.get("n_fail48", np.nan)})
        key = (r.pid, r.side)
        if a.expert == "p75":
            for c in [c for c in rec if c.startswith("eck_")]:
                rec[c] = ex.at[key, c] if key in ex.index else np.nan
            rec["eck_projects"] = ex.at[key, "eck_projects"] if key in ex.index else ""
        else:
            rec["eck_projects"] = "all"   # v9.2 all-project average kept from the base table
        rows.append(rec)
        if a.dry_run and len(rows) >= 3:
            break
    out = pd.DataFrame(rows)
    print(f"[v9.3] {len(out)} knees; femur capture missing for {len(missing)}: {missing[:10]}")
    print(f"[check] grid-mode femur from captured inputs vs v9.2 production: max |diff| {maxdiff:.2e} mm")
    print(f"[check] grid-mode tibia from captured inputs vs v9.2 production: max |diff| {maxdiff_tib:.2e} mm; tibia capture missing {len(missing_tib)}")
    if "tibia_tab_den00_frac" in out:
        print(out.groupby("cohort")[["femur_tab_den00_frac", "femur_tab_den48_frac", "tibia_tab_den00_frac", "tibia_tab_den48_frac"]].median().round(3).to_string())
    print(f"[expert] 75 %-paired value available for {out.eck_cMF_d.notna().sum()}/{len(out)} knees; projects: "
          f"{out.eck_projects.value_counts().to_dict()}")
    print(out[["pd_cMF_00m", "pd_cMF_d", "eck_cMF_00m", "eck_cMF_d"]].describe().loc[["count", "mean", "std"]].round(3).to_string())
    if not a.dry_run:
        RES.mkdir(parents=True, exist_ok=True)
        if a.exclude_22:
            out = out[out.eck_cMF_d.notna()]
        out.to_csv(RES / f"{src}_long_abs.csv", index=False)
        print("wrote", RES / f"{src}_long_abs.csv", len(out), "knees")


if __name__ == "__main__":
    main()
