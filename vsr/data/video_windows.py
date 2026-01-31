import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from vsr.utils.io import read_image

def _sorted_frames(folder: str):
    exts = ("png", "jpg", "jpeg", "bmp", "webp")
    files = []
    for e in exts:
        files += glob(os.path.join(folder, f"*.{e}"))
    return sorted(files)

class VideoWindowDataset(Dataset):
    def __init__(self, root: str, scale: int = 2, window: int = 5):
        assert window % 2 == 1, "window must be odd"
        self.root = root
        self.scale = scale
        self.window = window
        self.half = window // 2
        self.samples = []

        seqs = sorted(glob(os.path.join(root, "*")))
        for seq in seqs:
            lr_dir = os.path.join(seq, "lr")
            hr_dir = os.path.join(seq, "hr")
            if not (os.path.isdir(lr_dir) and os.path.isdir(hr_dir)):
                continue
            lr_files = _sorted_frames(lr_dir)
            hr_files = _sorted_frames(hr_dir)
            if len(lr_files) == 0 or len(lr_files) != len(hr_files):
                continue
            for i in range(self.half, len(lr_files) - self.half):
                self.samples.append((lr_files, hr_files, i))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid sequences found under {root}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _to_tensor01(rgb: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return x

    def __getitem__(self, idx: int):
        lr_files, hr_files, c = self.samples[idx]
        window_lr = []
        for j in range(c - self.half, c + self.half + 1):
            rgb = read_image(lr_files[j])
            window_lr.append(self._to_tensor01(rgb))
        lr = torch.stack(window_lr, dim=0)  # T,C,H,W
        hr_rgb = read_image(hr_files[c])
        hr = self._to_tensor01(hr_rgb)  # C,Hs,Ws
        return lr, hr
