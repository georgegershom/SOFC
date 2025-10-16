import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SOFCWarpStressDataset(Dataset):
    def __init__(self, root: str, split: str = "train", normalize_warp: bool = True):
        self.root = Path(root)
        self.split = split
        self.normalize_warp = normalize_warp
        self.samples = []
        split_dir = self.root / split
        for child in sorted(split_dir.iterdir()):
            if child.is_dir() and (child / "warp.npy").exists() and (child / "stress.npz").exists():
                self.samples.append(child)
        # Load stats for normalization
        stats_path = self.root / "meta" / "stats.json"
        if stats_path.exists():
            with open(stats_path, "r") as f:
                self.stats = json.load(f)
        else:
            self.stats = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p = self.samples[idx]
        w = np.load(p / "warp.npy").astype(np.float32)
        s = np.load(p / "stress.npz")
        sigma_xx = s["sigma_xx"].astype(np.float32)
        sigma_yy = s["sigma_yy"].astype(np.float32)
        sigma_xy = s["sigma_xy"].astype(np.float32)

        if self.normalize_warp and "train" in self.stats:
            # Per-split scalar normalization by mean-of-stds
            w_std = float(self.stats["train"].get("warp_mean_of_stds", 1.0)) or 1.0
            w = w / w_std

        # Package as channels-first tensors
        w_t = torch.from_numpy(w)[None, ...]  # 1xHxW
        s_t = torch.from_numpy(np.stack([sigma_xx, sigma_yy, sigma_xy], axis=0))  # 3xHxW
        return {"warp": w_t, "stress": s_t, "path": str(p)}
