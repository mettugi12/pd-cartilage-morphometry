"""v8 non-progressor arm expansion — download SAG_IW_TSE DICOMs from the NAS for
the 124 non-progressor knees missing PD segmentations (JMRI plan 3A scale-up).

Logic is download_v32_iw.py (v3.3) verbatim: probe releases -> pick main scan date
-> list series -> pick SAG+IW+laterality by DICOM header -> tar-over-SSH. Updated
per Guides/nas_ssh.md long-running-pull rules: tar stdout goes to a temp FILE (not
Popen.PIPE), and each series download retries 3x with 30 s backoff.

Input:   E:/KneeMR/Studies/PD-vs-DESS/v8/nonprog_pull/missing_nonprog.csv
         (pid, side, cohort, has00, has48 — timepoints already segmented are skipped)
Output:  E:/KneeMR/Studies/PD-vs-DESS/v8/nonprog_pull/IW_{00m,48m}/{pid}_{side}/*.dcm
Log:     E:/KneeMR/Studies/PD-vs-DESS/v8/nonprog_pull/download_nonprog.csv

CLI:
    python download_nonprog_iw.py --dry_run --n_run 2
    python download_nonprog_iw.py --n_run 5
    python download_nonprog_iw.py
"""

import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

NAS_SSH = "ssh -i ~/.ssh/id_rsa orthoinsight@121.138.151.7 -p 2222"
NAS_00M_BASE = "/volume1/homes/dssong/OAI/image03/OAI00MonthImages"
NAS_48M_BASE = "/volume1/homes/dssong/OAI/image03/OAI48MonthImages(1)/results"
RELEASES_00M = ["0.C.2", "0.E.1"]
RELEASES_48M = ["6.C.1", "6.C.2", "6.E.1", "6.E.2"]

OUT_ROOT = Path(r"E:/KneeMR/Studies/PD-vs-DESS/v8/nonprog_pull")
COHORT_CSV = OUT_ROOT / "missing_nonprog.csv"


def ssh_cmd(cmd, timeout=30):
    full = f'{NAS_SSH} "{cmd}"'
    try:
        result = subprocess.run(full, shell=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ""
    return result.stdout.decode("utf-8", errors="replace").strip()


def probe_release(pid, base, releases):
    for rel in releases:
        out = ssh_cmd(f"ls -d '{base}/{rel}/{pid}' 2>/dev/null || true", timeout=15)
        if out and str(pid) in out:
            return rel
    return None


def find_scan_date(pid, base, release):
    patient_dir = f"{base}/{release}/{pid}"
    listing = ssh_cmd(f"ls '{patient_dir}' 2>/dev/null", timeout=20)
    if not listing:
        return None, 0
    dates = [d.strip() for d in listing.split("\n") if d.strip()]
    best_date, best_count = None, 0
    for d in dates:
        cnt_out = ssh_cmd(f"ls -d '{patient_dir}/{d}'/*/ 2>/dev/null | wc -l", timeout=20)
        try:
            cnt = int(cnt_out.strip())
        except ValueError:
            continue
        if cnt > best_count:
            best_count, best_date = cnt, d
    return best_date, best_count


def list_series(pid, base, release, date):
    d = f"{base}/{release}/{pid}/{date}"
    out = ssh_cmd(
        f"for s in '{d}'/*/; do sid=$(basename \\\"$s\\\"); "
        f"cnt=$(ls -1 \\\"$s\\\" 2>/dev/null | wc -l); echo $sid:$cnt; done",
        timeout=60,
    )
    series = []
    for line in out.split("\n"):
        if ":" not in line:
            continue
        sid, cnt_str = line.rsplit(":", 1)
        try:
            cnt = int(cnt_str.strip())
        except ValueError:
            continue
        series.append((sid.strip(), cnt))
    return series


def read_series_description(pid, base, release, date, series_id):
    import pydicom

    remote = f"{base}/{release}/{pid}/{date}/{series_id}"
    tar_cmd = f"{NAS_SSH} \"cd '{remote}' && tar cf - 001 001.dcm 2>/dev/null\""
    try:
        result = subprocess.run(tar_cmd, shell=True, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        return None
    if not result.stdout:
        return None
    tmp = tempfile.mkdtemp()
    try:
        tar = tarfile.open(fileobj=io.BytesIO(result.stdout))
        tar.extractall(path=tmp)
        tar.close()
        candidate = None
        for name in ("001.dcm", "001"):
            p = os.path.join(tmp, name)
            if os.path.exists(p) and os.path.isfile(p) and os.path.getsize(p) > 0:
                candidate = p
                break
        if candidate is None:
            return None
        ds = pydicom.dcmread(candidate, force=True)
        return getattr(ds, "SeriesDescription", "") or ""
    except Exception:
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def count_dicoms(out_dir):
    if not out_dir.exists():
        return 0
    n = 0
    for p in out_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if name.endswith(".dcm") or name.isdigit() or (len(name) <= 5 and name.lstrip("0").isdigit()):
            n += 1
    return n


def download_series(pid, base, release, date, series_id, out_dir, expected_n, dry_run):
    """tar-over-SSH -> temp FILE (nas_ssh.md rule 1) -> extract. 3 retries."""
    existing = count_dicoms(out_dir)
    if expected_n and existing >= max(1, expected_n - 1):
        return f"SKIP_EXISTS({existing})"
    if dry_run:
        return "DRY_RUN"
    out_dir.mkdir(parents=True, exist_ok=True)

    remote = f"{base}/{release}/{pid}/{date}/{series_id}"
    tar_cmd = f"{NAS_SSH} \"cd '{remote}' && tar cf - .\""
    for attempt in range(1, 4):
        fd, tmpname = tempfile.mkstemp(suffix=".tar")
        os.close(fd)  # Windows: keep no open handle or unlink fails (WinError 32)
        tmptar = Path(tmpname)
        try:
            with tmptar.open("wb") as fout:
                rc = subprocess.call(tar_cmd, shell=True, stdin=subprocess.DEVNULL,
                                     stdout=fout, timeout=300)
            if rc == 0 and tmptar.stat().st_size > 0:
                with tarfile.open(str(tmptar), mode="r:") as tf:
                    tf.extractall(path=str(out_dir))
                n = count_dicoms(out_dir)
                return f"OK_{n}"
        except (subprocess.TimeoutExpired, tarfile.TarError):
            pass
        finally:
            tmptar.unlink(missing_ok=True)
        if attempt < 3:
            time.sleep(30)
    return "FAIL_TAR_3X"


def pick_iw_series(pid, base, release, date, target_lat, slice_list):
    iw_candidates = [(sid, n) for sid, n in slice_list if 30 <= n <= 45]
    sag_iw = []
    for sid, n in iw_candidates:
        desc = read_series_description(pid, base, release, date, sid)
        if desc is None:
            continue
        up = desc.upper()
        if "SAG" in up and "IW" in up:
            sag_iw.append((sid, n, desc, up))
            if target_lat in up:
                return sid, n, desc
    if len(sag_iw) == 1:
        sid, n, desc, _ = sag_iw[0]
        return sid, n, desc
    return None, 0, ""


def handle_timepoint(pid, side, tp_label, base, releases, out_sub, dry_run):
    target_lat = side.upper()
    release = probe_release(pid, base, releases)
    if release is None:
        return {"timepoint": tp_label, "release": "", "scan_date": "", "series": "",
                "desc": "", "n_slices": 0, "status": "FAIL_NO_RELEASE"}
    date, n_series = find_scan_date(pid, base, release)
    if not date or n_series < 5:
        return {"timepoint": tp_label, "release": release, "scan_date": date or "",
                "series": "", "desc": "", "n_slices": 0,
                "status": f"FAIL_NO_DATE({n_series})"}
    series_list = list_series(pid, base, release, date)
    sid, n, desc = pick_iw_series(pid, base, release, date, target_lat, series_list)
    if not sid:
        return {"timepoint": tp_label, "release": release, "scan_date": date,
                "series": "", "desc": "", "n_slices": 0, "status": "FAIL_NO_IW"}
    out_dir = OUT_ROOT / out_sub / f"{pid}_{side}"
    status = download_series(pid, base, release, date, sid, out_dir, n, dry_run)
    return {"timepoint": tp_label, "release": release, "scan_date": date,
            "series": sid, "desc": desc, "n_slices": n, "status": status}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_run", type=int, default=None)
    args = ap.parse_args()

    rows_in = []
    with open(COHORT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows_in.append({"pid": int(r["pid"]), "side": r["side"],
                            "need00": r["has00"] == "False", "need48": r["has48"] == "False"})
    if args.n_run is not None:
        rows_in = rows_in[: args.n_run]

    print(f"Knees: {len(rows_in)} | need00 {sum(r['need00'] for r in rows_in)} "
          f"| need48 {sum(r['need48'] for r in rows_in)} | {'DRY' if args.dry_run else 'DOWNLOAD'}")
    (OUT_ROOT / "IW_00m").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "IW_48m").mkdir(parents=True, exist_ok=True)

    log_rows, ok, fail = [], 0, 0
    t0 = time.time()
    for i, r in enumerate(rows_in, 1):
        pid, side = r["pid"], r["side"]
        el = (time.time() - t0) / 60
        print(f"[{i:3d}/{len(rows_in)}] {pid} {side}  ({el:.0f} min elapsed)", flush=True)
        for tp, base, rels, sub, need in (
                ("00m", NAS_00M_BASE, RELEASES_00M, "IW_00m", r["need00"]),
                ("48m", NAS_48M_BASE, RELEASES_48M, "IW_48m", r["need48"])):
            if not need:
                continue
            res = handle_timepoint(pid, side, tp, base, rels, sub, args.dry_run)
            print(f"    {tp} {res['release']:<6} date={res['scan_date']:<10} "
                  f"series={res['series']:<6} n={res['n_slices']:>3} {res['status']}", flush=True)
            log_rows.append({"pid": pid, "side": side, **res})
            if res["status"].startswith(("OK", "SKIP", "DRY")):
                ok += 1
            else:
                fail += 1
        if log_rows and (i % 5 == 0 or i == len(rows_in)):
            with open(OUT_ROOT / "download_nonprog.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
                w.writeheader(); w.writerows(log_rows)

    print(f"\nSummary: OK/SKIP {ok} | FAIL {fail} | {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
