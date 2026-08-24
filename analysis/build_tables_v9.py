"""Assemble the manuscript table files (../manuscript/tables/) from ../results/, so every
table the JMRI draft cites sits in one folder with final numbering and the 150/151 frame.

  Table1_cross_sectional.md        n=61 pairs (template harness) — from v8_cross_sectional_table.md
  Table2_longitudinal.md           PRIMARY block of v9_longitudinal_table_qcpass.md
  Table3_discrimination.md         PRIMARY block of v9_discrimination_table_qcpass.md
  TableS1_qc_decomposition.md      v9_qc_decomposition.md
  TableS2_full_cohort_sensitivity.md   sensitivity blocks of both v9 tables (166/168)
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
TABLES = HERE.parent / "manuscript" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def section(text: str, start_pat: str, end_pat: str | None) -> str:
    s = re.search(start_pat, text)
    if not s:
        raise RuntimeError(f"start not found: {start_pat}")
    body = text[s.start():]
    if end_pat:
        e = re.search(end_pat, body[1:])
        if e:
            body = body[: e.start() + 1]
    return body.strip() + "\n"


def main():
    cs = (RESULTS / "v8_cross_sectional_table.md").read_text(encoding="utf-8")
    cs = cs.replace("# v8 Table 3 — Cross-sectional PD↔DESS cartilage thickness agreement",
                    "# Table 1 — Cross-sectional PD↔DESS cartilage thickness agreement (n=61 pairs)")
    cs += ("\nNote: computed with the atlas-template harness (cartilage-morphometry 74d678c); the "
           "patella is not evaluated by this harness and bias is larger than under the earlier v3.3 "
           "harness. Longitudinal analyses (Tables 2–3) use the per-patient baseline-grid projection.\n")
    (TABLES / "Table1_cross_sectional.md").write_text(cs, encoding="utf-8")

    lg = (RESULTS / "v9_longitudinal_table_qcpass.md").read_text(encoding="utf-8")
    hdr = lg.splitlines()[1]
    prim = section(lg, r"## ICP-QC-pass — PRIMARY", r"\n## ")
    sens_l = section(lg, r"## Full cohort — sensitivity", None)
    (TABLES / "Table2_longitudinal.md").write_text(
        "# Table 2 — 48-month cartilage thickness change: PD-auto vs manual QCart (PRIMARY, ICP-QC-pass)\n"
        f"{hdr}\n\n{prim}", encoding="utf-8")

    dg = (RESULTS / "v9_discrimination_table_qcpass.md").read_text(encoding="utf-8")
    hdr2 = dg.splitlines()[1]
    prim2 = section(dg, r"## ICP-QC-pass — PRIMARY", r"\n## Full cohort")
    sens_d = section(dg, r"## Full cohort — sensitivity", None)
    (TABLES / "Table3_discrimination.md").write_text(
        "# Table 3 — Progressor vs stable discrimination (PRIMARY, ICP-QC-pass)\n"
        f"{hdr2}\n\n{prim2}", encoding="utf-8")

    (TABLES / "TableS1_qc_decomposition.md").write_text(
        (RESULTS / "v9_qc_decomposition.md").read_text(encoding="utf-8").replace(
            "# v9 — QC decomposition (MFTC)", "# Table S1 — QC decomposition (MFTC)"), encoding="utf-8")

    (TABLES / "TableS2_full_cohort_sensitivity.md").write_text(
        "# Table S2 — Full-cohort sensitivity analysis (no technical QC; 166 progressors / 168 stable)\n\n"
        "## A. Longitudinal change vs manual QCart\n" + sens_l.split("\n", 1)[1] +
        "\n## B. Discrimination\n" + sens_d.split("\n", 1)[1], encoding="utf-8")

    ss = RESULTS / "v9_sample_size.md"
    if ss.exists():
        (TABLES / "TableS3_sample_size.md").write_text(
            ss.read_text(encoding="utf-8").replace(
                "# v9 — Sample-size translation", "# Table S3 — Sample-size translation"), encoding="utf-8")

    for p in sorted(TABLES.glob("*.md")):
        print(f"[ok] {p.name}  ({p.stat().st_size} B)")


if __name__ == "__main__":
    main()
