#!/usr/bin/env python3
import argparse
import csv
import os
from typing import Dict, List


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, 'r') as f:
        return list(csv.DictReader(f))


def write_csv(rows: List[Dict[str, str]], out_path: str) -> None:
    if not rows:
        raise ValueError("No rows to write")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Deduce union header
    headers = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                headers.append(k)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def synthesize(sandy_path: str, clay_path: str, cases_path: str) -> Dict[str, List[Dict[str, str]]]:
    sandy = read_csv(sandy_path)
    for r in sandy:
        r['material'] = 'sand'
        r['dataset'] = 'sandy_soils'

    clay = read_csv(clay_path)
    for r in clay:
        r['material'] = 'clay'
        r['dataset'] = 'clay_soils'

    cases = read_csv(cases_path)
    for r in cases:
        r['dataset'] = 'case_study'
        # Light mapping for analysis convenience
        if r.get('case_type') == 'liquefaction_uplift':
            r['material'] = 'sand'
        elif r.get('case_type') == 'clay_slip_surface':
            r['material'] = 'clay'
        else:
            r['material'] = 'unknown'

    return {
        'sandy': sandy,
        'clay': clay,
        'cases': cases,
        'combined': sandy + clay + cases,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Synthesize geotechnical datasets and export combined CSVs')
    ap.add_argument('--sandy', default='data/synthetic/sandy_soils.csv')
    ap.add_argument('--clay', default='data/synthetic/clay_soils.csv')
    ap.add_argument('--cases', default='data/synthetic/case_studies.csv')
    ap.add_argument('--outdir', default='data/processed')
    args = ap.parse_args()

    result = synthesize(args.sandy, args.clay, args.cases)

    write_csv(result['sandy'], os.path.join(args.outdir, 'sandy_labeled.csv'))
    write_csv(result['clay'], os.path.join(args.outdir, 'clay_labeled.csv'))
    write_csv(result['cases'], os.path.join(args.outdir, 'case_studies_labeled.csv'))
    write_csv(result['combined'], os.path.join(args.outdir, 'combined_all.csv'))

    print('Wrote labeled and combined datasets to', args.outdir)


if __name__ == '__main__':
    main()
