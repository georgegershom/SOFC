#!/usr/bin/env python3
import os
import hashlib
import zipfile

from .dataset_builder import build_dataset


def _zipdir(dir_path: str, zip_path: str) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for f in files:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, start=os.path.dirname(dir_path))
                zf.write(abs_path, arcname=rel_path)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest_path = build_dataset()
    print(f"Dataset manifest written: {manifest_path}")

    dataset_root = os.path.dirname(manifest_path)
    dist_dir = "/workspace/dist"
    os.makedirs(dist_dir, exist_ok=True)
    zip_path = os.path.join(dist_dir, "phase3_dataset.zip")
    _zipdir(dataset_root, zip_path)
    digest = _sha256(zip_path)
    digest_path = zip_path + ".sha256"
    with open(digest_path, "w", encoding="utf-8") as f:
        f.write(digest + "  " + os.path.basename(zip_path) + "\n")
    print(f"Zipped dataset: {zip_path}\nSHA256: {digest}")


if __name__ == "__main__":
    main()