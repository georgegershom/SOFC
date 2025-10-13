#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, Any

try:
    import requests
except Exception:
    print("The 'requests' package is required. Install with: pip install requests", file=sys.stderr)
    raise


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def download_with_retries(url: str, out_path: str, max_retries: int = 3, timeout: int = 60) -> None:
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp_out = out_path + ".part"
                with open(tmp_out, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp_out, out_path)
                return
        except Exception as e:
            if attempt >= max_retries:
                raise
            time.sleep(min(5 * attempt, 20))


def process_entry(entry: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    url = entry.get('url')
    rel = entry.get('relative_path')
    checksum = entry.get('sha256')
    if not url or not rel:
        return {"status": "skipped", "reason": "missing url or relative_path", "entry": entry}

    out_path = os.path.join(base_dir, rel)
    ensure_dir(os.path.dirname(out_path))

    try:
        download_with_retries(url, out_path)
        file_hash = sha256_of_file(out_path)
        verified = (checksum is None) or (checksum.lower() == file_hash.lower())
        return {"status": "downloaded", "path": out_path, "verified": verified, "sha256": file_hash}
    except Exception as e:
        return {"status": "error", "reason": str(e), "entry": entry}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets listed in a JSON config into data/raw/downloads.")
    parser.add_argument('--config', type=str, default='configs/sources.json')
    parser.add_argument('--outdir', type=str, default='data/raw/downloads')
    args = parser.parse_args()

    ensure_dir(args.outdir)

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    results = []
    for entry in cfg.get('sources', []):
        results.append(process_entry(entry, args.outdir))

    report_path = os.path.join(args.outdir, 'download_report.json')
    with open(report_path, 'w') as f:
        json.dump({"results": results}, f, indent=2)

    ok = sum(1 for r in results if r.get('status') == 'downloaded')
    print(f"Completed downloads: {ok}/{len(results)}. Report at {report_path}")


if __name__ == '__main__':
    main()
