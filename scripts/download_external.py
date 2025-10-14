#!/usr/bin/env python3
import argparse
import json
import os
import sys
import tarfile
import time
import urllib.request
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Dict, Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_file(url: str, dest_path: str, timeout: int = 600) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "dataset-fetcher/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest_path, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return dest_path


def sha256_file(path: str) -> str:
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def maybe_extract(archive_path: str, dest_dir: str, kind: str | None) -> None:
    if kind is None:
        # Try infer from extension
        lower = archive_path.lower()
        if lower.endswith(".zip"):
            kind = "zip"
        elif lower.endswith(".tar.gz") or lower.endswith(".tgz"):
            kind = "tar.gz"
        else:
            return
    if kind == "zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
    elif kind == "tar.gz":
        with tarfile.open(archive_path, mode="r:gz") as tf:
            tf.extractall(dest_dir)


def write_metadata(out_dir: str, meta: Dict[str, Any]) -> None:
    ensure_dir(out_dir)
    meta_path = os.path.join(out_dir, "source_info.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Download external datasets into data/external")
    parser.add_argument("--config", required=True, help="Path to JSON config with 'sources' list")
    parser.add_argument("--output-root", default=os.path.join("/workspace", "data", "external"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sources = cfg.get("sources", [])
    if not sources:
        print("No sources found in config", file=sys.stderr)
        sys.exit(1)

    ensure_dir(args.output_root)

    for src in sources:
        name = src.get("name")
        url = src.get("url")
        dest_dir = os.path.join(args.output_root, src.get("dest_dir", name.replace(" ", "_")))
        extract = src.get("extract", False)
        kind = src.get("kind")  # 'zip' | 'tar.gz' | None
        notes = src.get("notes")
        license_info = src.get("license")

        ensure_dir(dest_dir)
        ts = int(time.time())
        filename = src.get("filename") or os.path.basename(url) or f"download_{ts}"
        dest_path = os.path.join(dest_dir, filename)

        print(f"Downloading {name} from {url} ...")
        try:
            download_file(url, dest_path)
        except Exception as e:
            write_metadata(
                dest_dir,
                {
                    "name": name,
                    "url": url,
                    "status": "failed",
                    "error": str(e),
                    "notes": notes,
                    "license": license_info,
                    "attempted_at": ts,
                },
            )
            print(f"Failed: {name}: {e}", file=sys.stderr)
            continue

        file_hash = sha256_file(dest_path)
        meta = {
            "name": name,
            "url": url,
            "status": "downloaded",
            "path": dest_path,
            "sha256": file_hash,
            "notes": notes,
            "license": license_info,
            "downloaded_at": ts,
        }

        if extract:
            try:
                maybe_extract(dest_path, dest_dir, kind)
                meta["extracted"] = True
            except Exception as e:
                meta["extracted"] = False
                meta["extract_error"] = str(e)

        write_metadata(dest_dir, meta)
        print(f"Done: {name} -> {dest_dir}")


if __name__ == "__main__":
    main()
