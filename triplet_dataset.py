#!/usr/bin/env python3
"""
triplet_dataset.py – Generates (prev, curr, next, HR) tuples.

Low-resolution stream is **854 × 480**; HR is still 1920 × 1080.

Splits:
    train : 70 %
    val   : 20 %
    test  : 10 %
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random

SPLIT_RATIO = (0.70, 0.20, 0.10)
_RANDOM_SEED = 42


class TripletDataset(Dataset):
    """
    Returns four tensors in [0,1], shape (C, H, W):
        prev480, curr480, next480, hr1080
    """
    def __init__(self,
                 lr_root: str | Path,
                 hr_root: str | Path,
                 split: str = "train"):
        assert split in {"train", "val", "test"}
        lr_root, hr_root = Path(lr_root), Path(hr_root)
        if not lr_root.exists() or not hr_root.exists():
            raise FileNotFoundError("LR or HR root folder not found.")

        seq_names = sorted([d.name for d in lr_root.iterdir() if d.is_dir()])
        random.Random(_RANDOM_SEED).shuffle(seq_names)

        n = len(seq_names)
        n_train = int(SPLIT_RATIO[0] * n)
        n_val   = int(SPLIT_RATIO[1] * n)

        if   split == "train": seq_names = seq_names[:n_train]
        elif split == "val"  : seq_names = seq_names[n_train:n_train+n_val]
        else                : seq_names = seq_names[n_train+n_val:]

        self.samples: List[Tuple[Path, Path, Path, Path]] = []
        self._to_tensor = T.ToTensor()

        for seq in seq_names:
            lr_frames = sorted((lr_root / seq).glob("*.jpg"))
            hr_frames = sorted((hr_root / seq).glob("*.jpg"))
            assert len(lr_frames) == len(hr_frames), f"Mismatch in {seq}"

            for i in range(1, len(lr_frames) - 1):
                self.samples.append(
                    (lr_frames[i-1], lr_frames[i], lr_frames[i+1],
                     hr_frames[i])
                )

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        p_prev, p_curr, p_next, p_hr = self.samples[idx]
        return tuple(self._to_tensor(Image.open(p).convert("RGB"))
                     for p in (p_prev, p_curr, p_next, p_hr))
