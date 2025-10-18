#!/usr/bin/env python3
from pathlib import Path
import csv
import json
import statistics as stats

ROOT = Path("phase3_microchem_output")
MASTER = ROOT / "master_metrics.csv"


def read_master():
    rows = []
    with MASTER.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def to_float(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def group(rows, keys):
    out = {}
    for row in rows:
        k = tuple(row[k] for k in keys)
        out.setdefault(k, []).append(row)
    return out


def check_trends(rows):
    issues = []
    # Porosity should increase with Temperature_C and rubber_vol_frac
    by_mix = group(rows, ["Mix_ID"])
    for (mix_id,), rs in by_mix.items():
        by_temp = group(rs, ["Temperature_C"])
        last_mean = None
        for temp in ["200", "400", "600", "800"]:
            rtemp = [to_float(r.get("porosity")) for r in by_temp.get((temp,), []) if r.get("Analysis_Type") == "MicroCT"]
            rtemp = [x for x in rtemp if x is not None]
            if not rtemp:
                continue
            mean_p = stats.mean(rtemp)
            if last_mean is not None and mean_p + 1e-6 < last_mean:
                issues.append(f"Porosity non-monotonic for {mix_id} at {temp}C: {mean_p:.3f} < {last_mean:.3f}")
            last_mean = mean_p

    # Portlandite should decrease with temperature; C-S-H should decrease notably after 600C
    # Check by reading per-sample XRD files from index
    with (ROOT / "index.json").open() as f:
        idx = json.load(f)
    for entry in idx["entries"]:
        mix = entry["Mix_ID"]
        temp = entry["Temperature_C"]
        xrd_path = ROOT / entry["paths"]["xrd_phases_csv"]
        phases = {}
        with xrd_path.open() as f:
            r = csv.DictReader(f)
            for row in r:
                phases[row["phase"]] = float(row["wt_percent"])  # already normalized per file
        # Simple thresholds/trends checks by temp
        ch = phases.get("Portlandite (CH)", 0.0)
        csh = phases.get("C-S-H (amorphous)", 0.0)
        if temp in [400, 600, 800]:
            # CH should be <= 200C value for same mix
            baseline_path = None
            for e in idx["entries"]:
                if e["Mix_ID"] == mix and e["Temperature_C"] == 200 and e["paths"]["xrd_phases_csv"].split("_phases.csv")[0] == xrd_path.name.split("_phases.csv")[0].replace(f"-{temp}C", "-200C"):
                    baseline_path = ROOT / e["paths"]["xrd_phases_csv"]
                    break
            if baseline_path and baseline_path.exists():
                with baseline_path.open() as f:
                    r = csv.DictReader(f)
                    base_phases = {row["phase"]: float(row["wt_percent"]) for row in r}
                ch0 = base_phases.get("Portlandite (CH)", 0.0)
                if ch > ch0 + 1e-6:
                    issues.append(f"CH increased with T for {mix} at {temp}C: {ch:.2f} > {ch0:.2f}")
        # C-S-H after 600C should be less than at 400C
        if temp == 800:
            csh400 = None
            for e in idx["entries"]:
                if e["Mix_ID"] == mix and e["Temperature_C"] == 400 and e["paths"]["xrd_phases_csv"].split("_phases.csv")[0] == xrd_path.name.split("_phases.csv")[0].replace("-800C", "-400C"):
                    with (ROOT / e["paths"]["xrd_phases_csv"]).open() as f:
                        r = csv.DictReader(f)
                        csh400 = {row["phase"]: float(row["wt_percent"]) for row in r}.get("C-S-H (amorphous)", None)
                    break
            if csh400 is not None and csh > csh400 + 1e-6:
                issues.append(f"C-S-H increased at 800C vs 400C for {mix}: {csh:.2f} > {csh400:.2f}")

    # Rubber signatures: melt_phase_fraction should increase with rubber fraction and peak ~200-400C
    by_mix_temp = group(rows, ["Mix_ID", "Temperature_C"])    
    for (mix, temp), rs in by_mix_temp.items():
        melts = [to_float(r.get("melt_phase_fraction")) for r in rs if r.get("Analysis_Type") == "Rubber-Signature"]
        melts = [m for m in melts if m is not None]
        if not melts:
            continue
        m = stats.mean(melts)
        if mix == "HPRC-0" and m > 0.05:
            issues.append(f"Melt phase nonzero in HPRC-0 at {temp}C: {m:.2f}")

    return issues


if __name__ == "__main__":
    rows = read_master()
    issues = check_trends(rows)
    if issues:
        print("Validation issues detected:")
        for i in issues:
            print("-", i)
    else:
        print("All checks passed.")
