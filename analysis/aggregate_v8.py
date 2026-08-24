"""Aggregate the v8 cartilage-morphometry validation reports into manuscript tables.

Reads the two ValidationReport JSONs produced by
`scripts.validation.run_validation` (pinned cartilage-morphometry @ 74d678c):
  ../results/v8_cross_sectional.json   (n=61 PD<->DESS pairs)
  ../results/v8_longitudinal.json      (n=166 progressors, 48m)

Emits into ../results/:
  v8_cross_sectional_table.{csv,md}    -> manuscript Table 3 (thickness agreement)
  v8_longitudinal_table.{csv,md}       -> manuscript Table 4 (longitudinal change + QCart agreement)
  v8_vs_v33_comparison.md              -> headline v3.3 vs v8 deltas

Stats reproduce the v3.3 manuscript definitions: SRM = mean(d)/sd(d) with a BCa
bootstrap 95% CI (1000 iters, seed 42); per-knee Pearson r, Lin's CCC, ICC(A,1)
(two-way mixed, absolute agreement, single measure; McGraw & Wong 1996); Bland-Altman
bias = mean(PD - manual) with 95% LOA = bias +/- 1.96*SD.

--dry_run / --n_run are accepted for convention but this is a pure post-processor.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

# v3.3 manuscript reference numbers (Table 4 PD-auto + Table 3 r) for the comparison.
V33 = {
    "MFTC": {"srm": -1.59, "delta_um": -504, "r": 0.64, "ccc": 0.58, "icc": 0.58},
    "cMF":  {"srm": -1.62, "delta_um": -283, "r": 0.67, "ccc": 0.49, "icc": 0.49},
    "MT":   {"srm": -1.09, "delta_um": -221, "r": 0.48, "ccc": 0.46, "icc": 0.46},
    "femur_r_cs": 0.741, "tibia_r_cs": 0.721,
}


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------
def srm(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1))


def bca_ci_srm(d: np.ndarray, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """BCa bootstrap 95% CI for the SRM (matches the v3.3 manuscript)."""
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return (float("nan"), float("nan"))
    theta_hat = srm(d)
    rng = np.random.default_rng(seed)
    boots = np.array([srm(d[rng.integers(0, n, n)]) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    if len(boots) < 10:
        return (float("nan"), float("nan"))
    from scipy.stats import norm
    z0 = norm.ppf((boots < theta_hat).mean()) if 0 < (boots < theta_hat).mean() < 1 else 0.0
    # acceleration via jackknife
    jack = np.array([srm(np.delete(d, i)) for i in range(n)])
    jbar = jack.mean()
    num = ((jbar - jack) ** 3).sum()
    den = 6.0 * (((jbar - jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0
    zl, zu = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    def adj(z):
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = np.quantile(boots, [adj(zl), adj(zu)])
    return (float(lo), float(hi))


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    x, y = x[m], y[m]
    sx2 = x.var(ddof=0); sy2 = y.var(ddof=0)
    cov = ((x - x.mean()) * (y - y.mean())).mean()
    denom = sx2 + sy2 + (x.mean() - y.mean()) ** 2
    return float(2 * cov / denom) if denom != 0 else float("nan")


def icc_a1(x: np.ndarray, y: np.ndarray) -> float:
    """ICC(A,1): two-way mixed, absolute agreement, single measure (McGraw & Wong 1996)."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    M = np.column_stack([x[m], y[m]])
    n, k = M.shape  # n subjects, k=2 methods
    grand = M.mean()
    row_means = M.mean(axis=1)
    col_means = M.mean(axis=0)
    SSR = k * ((row_means - grand) ** 2).sum()          # between-subjects
    SSC = n * ((col_means - grand) ** 2).sum()           # between-methods
    SSE = ((M - row_means[:, None] - col_means[None, :] + grand) ** 2).sum()
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    denom = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    return float((MSR - MSE) / denom) if denom != 0 else float("nan")


def ba(pd: np.ndarray, man: np.ndarray):
    m = np.isfinite(pd) & np.isfinite(man)
    d = pd[m] - man[m]
    if len(d) < 2:
        return (float("nan"), float("nan"), float("nan"))
    bias = d.mean(); sd = d.std(ddof=1)
    return (float(bias), float(bias - 1.96 * sd), float(bias + 1.96 * sd))


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def dedup_one_per_subject(per_case: list, seed: int = 42) -> list:
    """One knee per subject (manuscript convention; seed 42). The harness runs all
    172 progressor knees; 6 subjects contribute both knees. Sort deterministically
    by (pid, side), then for each multi-knee subject pick one via a seeded RNG.
    The specific knee kept for those 6 subjects may differ from v3.3's stratified
    selection — immaterial to population SRM/r."""
    rng = np.random.default_rng(seed)
    by_pid: dict = {}
    for c in sorted(per_case, key=lambda c: (str(c.get("pid")), str(c.get("side")))):
        by_pid.setdefault(str(c.get("pid")), []).append(c)
    out = []
    for pid, knees in by_pid.items():
        out.append(knees[0] if len(knees) == 1 else knees[int(rng.integers(0, len(knees)))])
    return out


def long_pairs(cases: list, region: str):
    """Return (pd_delta_mm[], qcart_delta_mm[]) across cases for a region/composite.

    region in {MFTC, LFTC, cMF, cLF, MT, LT, ...}. PD delta from per_case bones/composites
    (template frame); QCart delta from per_case eckstein_qcart.<region>_d.
    """
    pd_list, qc_list = [], []
    for c in cases:
        eck = c.get("eckstein_qcart", {})
        if not eck.get("present"):
            qd = float("nan")
        else:
            qd = eck.get(f"{region}_d", float("nan"))
        # PD delta
        if region in ("MFTC", "LFTC"):
            pdd = c.get("composites", {}).get(f"{region}_d", float("nan"))
        else:
            pdd = float("nan")
            for bone in ("femur", "tibia"):
                rd = c["bones"].get(bone, {}).get("regions_delta", {})
                if region in rd:
                    pdd = rd[region]; break
        pd_list.append(pdd); qc_list.append(qd)
    return np.array(pd_list, float), np.array(qc_list, float)


def fmt(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def build_long_table(report: dict) -> tuple[list[dict], int]:
    cases = dedup_one_per_subject(report["per_case"])
    rows = []
    regions = ["MFTC", "cMF", "MT", "LFTC", "cLF", "LT"]
    for reg in regions:
        pd, qc = long_pairs(cases, reg)
        pd_um, qc_um = pd * 1000, qc * 1000
        lo, hi = bca_ci_srm(pd)
        qlo, qhi = bca_ci_srm(qc)
        bias, loa_lo, loa_hi = ba(pd_um, qc_um)
        rows.append({
            "region": reg,
            "n": int(np.isfinite(pd).sum()),
            "pd_delta_um": np.nanmean(pd_um), "pd_sd_um": np.nanstd(pd_um, ddof=1),
            "pd_srm": srm(pd), "srm_lo": lo, "srm_hi": hi,
            "qc_delta_um": np.nanmean(qc_um), "qc_sd_um": np.nanstd(qc_um, ddof=1),
            "qc_srm": srm(qc), "qc_srm_lo": qlo, "qc_srm_hi": qhi,
            "r": pearson_r(pd, qc), "ccc": lin_ccc(pd, qc), "icc": icc_a1(pd, qc),
            "ba_bias_um": bias, "loa_lo_um": loa_lo, "loa_hi_um": loa_hi,
            "n_qcart": int((np.isfinite(pd) & np.isfinite(qc)).sum()),
        })
    return rows, len(cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_run", type=int, default=None)
    ap.add_argument("--cs_json", type=Path, default=RESULTS / "v8_cross_sectional.json")
    ap.add_argument("--long_json", type=Path, default=RESULTS / "v8_longitudinal.json")
    args = ap.parse_args()

    cs = json.loads(args.cs_json.read_text(encoding="utf-8"))["cross_sectional"]
    lg = json.loads(args.long_json.read_text(encoding="utf-8"))["longitudinal"]

    # --- Table 3: cross-sectional thickness agreement ---
    cs_rows = []
    for bone, v in cs["per_bone"].items():
        cs_rows.append({
            "bone": bone, "n": v["n_cases"],
            "r_pearson": v["r_pearson_mean"], "r_pearson_sd": v["r_pearson_std"],
            "r_2d_smooth": v["r_2d_smooth_mean"], "r_2d_smooth_sd": v["r_2d_smooth_std"],
            "bias_mm": v["bias_mm_mean"], "bias_sd": v["bias_mm_std"],
            "mae_mm": v["mae_mm_mean"],
        })

    # --- Table 4: longitudinal (deduped to one knee per subject) ---
    lg_rows, n_long = build_long_table(lg)

    if args.dry_run:
        print(f"[dry] CS n_total={cs['n_total']} bones={[r['bone'] for r in cs_rows]}")
        print(f"[dry] LONG n_total={lg['n_total']} -> deduped n={n_long}")
        for r in lg_rows:
            print(f"  {r['region']}: PD SRM {fmt(r['pd_srm'])} ({fmt(r['srm_lo'])},{fmt(r['srm_hi'])}) "
                  f"d={fmt(r['pd_delta_um'],0)}um r={fmt(r['r'])} ccc={fmt(r['ccc'])} icc={fmt(r['icc'])} n={r['n']}")
        return

    RESULTS.mkdir(parents=True, exist_ok=True)

    # CSV: cross-sectional
    with open(RESULTS / "v8_cross_sectional_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cs_rows[0])); w.writeheader(); w.writerows(cs_rows)
    # CSV: longitudinal
    with open(RESULTS / "v8_longitudinal_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lg_rows[0])); w.writeheader(); w.writerows(lg_rows)

    # MD: cross-sectional (Table 3)
    md3 = ["# v8 Table 3 — Cross-sectional PD↔DESS cartilage thickness agreement",
           f"(n={cs['n_total']} pairs; cartilage-morphometry @ 74d678c, raycast_2d + thr0.5, no gate)\n",
           "| Bone | n | Pearson r (vertex) | r (2D smoothed) | Bias PD−DESS (mm) | MAE (mm) |",
           "|------|--:|--:|--:|--:|--:|"]
    for r in cs_rows:
        md3.append(f"| {r['bone'].capitalize()} | {r['n']} | {fmt(r['r_pearson'],3)} ± {fmt(r['r_pearson_sd'],3)} "
                   f"| {fmt(r['r_2d_smooth'],3)} ± {fmt(r['r_2d_smooth_sd'],3)} "
                   f"| {fmt(r['bias_mm'],3)} ± {fmt(r['bias_sd'],3)} | {fmt(r['mae_mm'],3)} |")
    (RESULTS / "v8_cross_sectional_table.md").write_text("\n".join(md3) + "\n", encoding="utf-8")

    # MD: longitudinal (Table 4)
    md4 = [f"# v8 Table 4 — Longitudinal cartilage thickness change (n={n_long} progressors, 48m)",
           "(cartilage-morphometry @ 74d678c. Δ in μm; SRM with BCa 95% CI, 1000 iters, seed 42. "
           "r/CCC/ICC = PD-auto vs manual QCart; BA bias = PD−manual.)\n",
           "| Region | PD Δ (μm) | PD SRM (95% CI) | QCart Δ (μm) | QCart SRM (95% CI) | r | CCC | ICC(A,1) | BA bias (μm) | 95% LOA (μm) | n (QCart) |",
           "|--------|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in lg_rows:
        md4.append(
            f"| {r['region']} | {fmt(r['pd_delta_um'],0)} ± {fmt(r['pd_sd_um'],0)} "
            f"| {fmt(r['pd_srm'])} ({fmt(r['srm_lo'])}, {fmt(r['srm_hi'])}) "
            f"| {fmt(r['qc_delta_um'],0)} ± {fmt(r['qc_sd_um'],0)} "
            f"| {fmt(r['qc_srm'])} ({fmt(r['qc_srm_lo'])}, {fmt(r['qc_srm_hi'])}) "
            f"| {fmt(r['r'])} | {fmt(r['ccc'])} | {fmt(r['icc'])} "
            f"| {fmt(r['ba_bias_um'],0)} | [{fmt(r['loa_lo_um'],0)}, {fmt(r['loa_hi_um'],0)}] | {r['n_qcart']} |")
    (RESULTS / "v8_longitudinal_table.md").write_text("\n".join(md4) + "\n", encoding="utf-8")

    # MD: v3.3 vs v8 comparison
    by = {r["region"]: r for r in lg_rows}
    cmp = ["# v8 vs v3.3 — headline comparison\n",
           "| Metric | v3.3 | v8 | Δ |", "|--------|--:|--:|--:|"]
    def cmp_row(label, v33, v8):
        d = (v8 - v33) if (v33 is not None and np.isfinite(v8)) else float("nan")
        cmp.append(f"| {label} | {fmt(v33)} | {fmt(v8)} | {fmt(d)} |")
    cmp_row("MFTC SRM (primary)", V33["MFTC"]["srm"], by["MFTC"]["pd_srm"])
    cmp_row("cMF SRM", V33["cMF"]["srm"], by["cMF"]["pd_srm"])
    cmp_row("MT SRM", V33["MT"]["srm"], by["MT"]["pd_srm"])
    cmp_row("MFTC r vs QCart", V33["MFTC"]["r"], by["MFTC"]["r"])
    cmp_row("cMF r vs QCart", V33["cMF"]["r"], by["cMF"]["r"])
    cmp_row("MT r vs QCart", V33["MT"]["r"], by["MT"]["r"])
    cs_by = {r["bone"]: r for r in cs_rows}
    if "femur" in cs_by:
        cmp_row("Femur CS r (2D)", V33["femur_r_cs"], cs_by["femur"]["r_2d_smooth"])
    if "tibia" in cs_by:
        cmp_row("Tibia CS r (2D)", V33["tibia_r_cs"], cs_by["tibia"]["r_2d_smooth"])
    (RESULTS / "v8_vs_v33_comparison.md").write_text("\n".join(cmp) + "\n", encoding="utf-8")

    print(f"[ok] wrote v8 tables to {RESULTS}")
    print("\n".join(md4))
    print("\n".join(cmp))


if __name__ == "__main__":
    main()
