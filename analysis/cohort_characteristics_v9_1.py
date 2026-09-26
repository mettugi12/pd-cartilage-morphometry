"""v9.1 — Table 1: cohort characteristics (progressors vs stable knees; cross-sectional pairs).

Joins the primary-frame knee lists (150 / 151) and the 61 paired cross-sectional knees to
OAI public clinical files for age, sex, BMI, baseline KL grade, side, baseline minimum
medial JSW and its 48-month change.

Inputs
  ../results/v9_prog_qcpass.csv, ../results/v9_nonprog_qcpass.csv
  ../../v8/results/v8_cross_sectional.json                       (61 case ids + KL)
  E:/KneeMR/Studies/PD-vs-DESS/v3.2/cohort/cohort_v33_combined.csv (BMI_00, JSW_00, dJSW_48)
  E:/KneeMR/Datasets/OAI/OAICompleteData_ASCII/{Enrollees.txt, AllClinical00.txt}
Outputs
  ../results/v9_1_cohort_characteristics.md (+ .csv)

Usage: python cohort_characteristics_v9_1.py [--dry_run] [--n_run N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OAI = Path(r"E:/KneeMR/Datasets/OAI/OAICompleteData_ASCII")
COHORT = Path(r"E:/KneeMR/Studies/PD-vs-DESS/v3.2/cohort/cohort_v33_combined.csv")
CS_JSON = HERE.parent.parent / "v8" / "results" / "v8_cross_sectional.json"


def code(x):
    """OAI coded values look like '2: 2' or '1: Male' -> take the part before ':'."""
    if pd.isna(x):
        return np.nan
    s = str(x)
    return s.split(":")[0].strip() if ":" in s else s


def load_clinical():
    enr = pd.read_csv(OAI / "Enrollees.txt", sep="|", usecols=["ID", "P02SEX"])
    ac = pd.read_csv(OAI / "AllClinical00.txt", sep="|", usecols=["ID", "V00AGE", "P01BMI"], low_memory=False)
    d = enr.merge(ac, on="ID", how="left")
    d["sex"] = d.P02SEX.map(code).map({"1": "Male", "2": "Female"})
    d["age"] = pd.to_numeric(d.V00AGE, errors="coerce")
    d["bmi"] = pd.to_numeric(d.P01BMI, errors="coerce")
    return d[["ID", "sex", "age", "bmi"]].rename(columns={"ID": "pid"})


def summarize(df: pd.DataFrame):
    out = {}
    out["n"] = len(df)
    out["age"] = (df.age.mean(), df.age.std())
    out["female"] = ((df.sex == "Female").sum(), (df.sex == "Female").mean() * 100)
    out["bmi"] = (df.bmi.mean(), df.bmi.std())
    out["right"] = ((df.side == "RIGHT").sum(), (df.side == "RIGHT").mean() * 100)
    out["kl"] = df.KL_00.value_counts().reindex([0, 1, 2, 3, 4]).fillna(0).astype(int).to_dict()
    if "JSW_00" in df:
        out["jsw0"] = (df.JSW_00.mean(), df.JSW_00.std())
        out["djsw48"] = (df.dJSW_48.mean(), df.dJSW_48.std())
    return out


def fmt_ms(t, nd=1):
    return f"{t[0]:.{nd}f} ± {t[1]:.{nd}f}"


def fmt_np(t):
    return f"{t[0]} ({t[1]:.0f}%)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_run", type=int, default=None)
    ap.add_argument("--frame", default="leakfree", choices=["leakfree", "full"],
                    help="leakfree = v9_1_*_leakfree.csv (primary); full = v9_*_qcpass.csv (150/151)")
    a = ap.parse_args()
    PROG = RES / ("v9_1_prog_leakfree.csv" if a.frame == "leakfree" else "v9_prog_qcpass.csv")
    STAB = RES / ("v9_1_nonprog_leakfree.csv" if a.frame == "leakfree" else "v9_nonprog_qcpass.csv")
    SUF = "" if a.frame == "leakfree" else "_full"

    clin = load_clinical()
    coh = pd.read_csv(COHORT)[["pid", "side", "BMI_00", "JSW_00", "dJSW_48"]]
    prog = pd.read_csv(PROG)[["pid", "side", "KL_00"]]
    stab = pd.read_csv(STAB)[["pid", "side", "KL_00"]]
    cs = json.load(open(CS_JSON))["cross_sectional"]["per_case"]
    xs = pd.DataFrame({"pid": [int(c["case_id"][:7]) for c in cs],
                       "side": [c["case_id"].split("_")[1].upper() for c in cs],
                       "KL_00": [c["kl_grade"] for c in cs]})
    if a.n_run:
        prog, stab, xs = prog.head(a.n_run), stab.head(a.n_run), xs.head(a.n_run)
    prog = prog.merge(coh, on=["pid", "side"], how="left").merge(clin, on="pid", how="left")
    stab = stab.merge(coh, on=["pid", "side"], how="left").merge(clin, on="pid", how="left")
    xs = xs.merge(clin, on="pid", how="left")
    for nm, d in (("prog", prog), ("stab", stab), ("xs", xs)):
        miss = d[["age", "sex", "bmi"]].isna().sum().to_dict()
        print(nm, len(d), "missing:", miss)
    # BMI: prefer OAI P01BMI; fall back to cohort BMI_00
    for d in (prog, stab):
        d["bmi"] = d.bmi.fillna(d.BMI_00)

    P, S, X = summarize(prog), summarize(stab), summarize(xs)
    # tests progressor vs stable
    p_age = stats.ttest_ind(prog.age.dropna(), stab.age.dropna(), equal_var=False).pvalue
    p_bmi = stats.ttest_ind(prog.bmi.dropna(), stab.bmi.dropna(), equal_var=False).pvalue
    p_sex = stats.chi2_contingency(pd.crosstab(pd.concat([prog.sex, stab.sex]), np.r_[np.zeros(len(prog)), np.ones(len(stab))]))[1]
    p_side = stats.chi2_contingency(pd.crosstab(pd.concat([prog.side, stab.side]), np.r_[np.zeros(len(prog)), np.ones(len(stab))]))[1]
    p_kl = stats.chi2_contingency(pd.crosstab(pd.concat([prog.KL_00, stab.KL_00]), np.r_[np.zeros(len(prog)), np.ones(len(stab))]))[1]
    p_jsw0 = stats.ttest_ind(prog.JSW_00.dropna(), stab.JSW_00.dropna(), equal_var=False).pvalue
    p_djsw = stats.mannwhitneyu(prog.dJSW_48.dropna(), stab.dJSW_48.dropna()).pvalue

    def pf(p):
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    rows = [
        ["Knees, n", str(P["n"]), str(S["n"]), "", str(X["n"])],
        ["Age, years", fmt_ms(P["age"]), fmt_ms(S["age"]), pf(p_age), fmt_ms(X["age"])],
        ["Female sex, n (%)", fmt_np(P["female"]), fmt_np(S["female"]), pf(p_sex), fmt_np(X["female"])],
        ["Body-mass index, kg/m²", fmt_ms(P["bmi"]), fmt_ms(S["bmi"]), pf(p_bmi), fmt_ms(X["bmi"])],
        ["Right knee, n (%)", fmt_np(P["right"]), fmt_np(S["right"]), pf(p_side), fmt_np(X["right"])],
        ["Baseline Kellgren–Lawrence grade 0 / 1 / 2 / 3 / 4, n",
         " / ".join(str(P["kl"][k]) for k in (0, 1, 2, 3, 4)), " / ".join(str(S["kl"][k]) for k in (0, 1, 2, 3, 4)), pf(p_kl),
         " / ".join(str(X["kl"].get(k, 0)) for k in (0, 1, 2, 3, 4))],
        ["Baseline minimum medial JSW, mm", fmt_ms(P["jsw0"], 2), fmt_ms(S["jsw0"], 2), pf(p_jsw0), "—"],
        ["48-month change in minimum medial JSW, mm", fmt_ms(P["djsw48"], 2), fmt_ms(S["djsw48"], 2), pf(p_djsw), "—"],
    ]
    hdr = ["Characteristic", f"Progressors (n={P['n']})", f"Stable knees (n={S['n']})", "P value", f"Cross-sectional pairs (n={X['n']})"]
    md = ["# Table 1 (v9.1) — Cohort characteristics", "",
          "Progressor / stable = primary longitudinal frame after registration QC, one knee per participant, and exclusion of depth-network training-pool participants. "
          "Cross-sectional pairs = 61 same-visit PD/IW–DESS pairs (48-month OAI visit); JSW not tabulated for this cohort. "
          "Age, sex and BMI from OAI baseline clinical files (Enrollees, AllClinical00); KL grade from the baseline fixed-flexion radiograph reading used for cohort selection. "
          "Values mean ± SD or n (%). P values: Welch t test, χ² test, or Mann–Whitney U test (JSW change), progressors versus stable knees.", "",
          "| " + " | ".join(hdr) + " |", "|---|--:|--:|--:|--:|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    n_missing = int(pd.concat([prog, stab])[["age", "sex", "bmi"]].isna().any(axis=1).sum())
    md += ["", f"Knees with any missing clinical variable: {n_missing}."]
    text = "\n".join(md) + "\n"
    print(text)
    if not a.dry_run:
        (RES / f"v9_1_cohort_characteristics{SUF}.md").write_text(text, encoding="utf-8")
        pd.DataFrame(rows, columns=hdr).to_csv(RES / f"v9_1_cohort_characteristics{SUF}.csv", index=False)
        print("wrote", RES / f"v9_1_cohort_characteristics{SUF}.md")


if __name__ == "__main__":
    main()
