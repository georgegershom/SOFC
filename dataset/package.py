from __future__ import annotations

import os
import shutil


def main():
    src = "/workspace/artifacts"
    dst_dir = "/workspace/dist"
    os.makedirs(dst_dir, exist_ok=True)
    base = os.path.join(dst_dir, "sofc_multi_fidelity_dataset_v1")
    # Create zip archive of artifacts directory
    shutil.make_archive(base, "zip", src)
    print(base + ".zip")


if __name__ == "__main__":
    main()
