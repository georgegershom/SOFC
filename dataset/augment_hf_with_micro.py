from __future__ import annotations

import csv
import json
import os


ART_DIR = "/workspace/artifacts"
HF_CSV = os.path.join(ART_DIR, "inputs_HF.csv")
MICRO_DIR = os.path.join(ART_DIR, "micro")
META_JSON = os.path.join(MICRO_DIR, "microstructure_meta.json")
OUT_CSV = os.path.join(ART_DIR, "inputs_HF_micro.csv")


def main():
    with open(META_JSON, "r") as f:
        meta = json.load(f)
    files = [e["file"] if isinstance(e, dict) else e for e in meta["files"]]
    files = [os.path.join("micro", fn) for fn in files]
    if not files:
        raise RuntimeError("No micro files found in metadata")

    with open(HF_CSV, "r", newline="") as fin:
        r = csv.reader(fin)
        header = next(r)
        header_out = header + ["micro_file"]
        rows = list(r)

    out_rows = []
    for i, row in enumerate(rows):
        mf = files[i % len(files)]
        out_rows.append(row + [mf])

    with open(OUT_CSV, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(header_out)
        w.writerows(out_rows)


if __name__ == "__main__":
    main()
