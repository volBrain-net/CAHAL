import json
import random

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from clusters import bins, anisotropy_bins, anisotropy_labels
from resolution_sampler import sample_resolutions


def get_ras_axes(aff, n_dims=3):
    aff_inv = np.linalg.inv(aff)
    return np.argmax(np.absolute(aff_inv[:n_dims, :n_dims]), axis=0)


def align_volume_to_ref(volume, aff, n_dims=3):
    """Reorient a volume to RAS using its affine, so all HR samples share one convention."""
    aff_flo = aff.copy()
    aff_ref = np.eye(4)

    ras_ref = get_ras_axes(aff_ref, n_dims)
    ras_flo = get_ras_axes(aff_flo, n_dims)

    aff_flo[:, ras_ref] = aff_flo[:, ras_flo]
    for i in range(n_dims):
        if ras_flo[i] != ras_ref[i]:
            volume = np.swapaxes(volume, ras_flo[i], ras_ref[i])
            swap_idx = np.where(ras_flo == ras_ref[i])
            ras_flo[swap_idx], ras_flo[i] = ras_flo[i], ras_flo[swap_idx]

    dot = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot[i] < 0:
            volume = np.flip(volume, [i])

    return volume


def load_hr_volume(path):
    img = nib.load(path)
    volume = np.array(img.dataobj)
    volume = align_volume_to_ref(volume, img.affine)
    volume[np.isnan(volume)] = 0
    volume[np.isinf(volume)] = 0
    volume[volume < 0] = 0
    return volume


def resample_volume(volume, resolutions):
    """Simulate LR acquisition: downsample to `resolutions` then upsample back to the HR grid."""
    original_shape = volume.shape[-3:]
    scale_factors = tuple(1.0 / r for r in resolutions)
    low = F.interpolate(volume.unsqueeze(0), scale_factor=scale_factors, mode="area")
    high = F.interpolate(low, size=original_shape, mode="trilinear").squeeze(0)
    return high


def load_csv_resolutions(csv_path):
    """Load the LR-resolution distribution CSV and assign volumetric/anisotropy clusters."""
    df = pd.read_csv(csv_path)
    for col in ["Resolution_X", "Resolution_Y", "Resolution_Z"]:
        df.loc[df[col] < 1, col] = 1

    df["res_product"] = df["Resolution_X"] * df["Resolution_Y"] * df["Resolution_Z"]

    df["cluster"] = pd.cut(df["res_product"], bins=bins, include_lowest=False).cat.codes + 1

    aniso = df[["Resolution_X", "Resolution_Y", "Resolution_Z"]].min(axis=1) / \
        df[["Resolution_X", "Resolution_Y", "Resolution_Z"]].max(axis=1)
    df["anisotropy_cluster"] = pd.cut(aniso, bins=anisotropy_bins, labels=anisotropy_labels,
                                       include_lowest=True).astype(int)
    return df


class ReconstructionDataset(Dataset):
    """
    HR/LR pair generator for training a super-resolution model.

    HR volumes are listed in a JSON file: {"hr_files": ["/path/a.nii.gz", ...]}.
    LR volumes are synthesized on the fly by resampling HR to a resolution drawn
    from `vol_bin`/`aniso_bin` (see clusters.py and resolution_sampler.py).
    """

    def __init__(self, hr_json, resolutions_df, vol_bin, aniso_bin, min_size=100, device="cpu"):
        with open(hr_json) as f:
            self.hr_files = json.load(f)["hr_files"]
        self.resolutions_df = resolutions_df
        self.vol_bin = vol_bin
        self.aniso_bin = aniso_bin
        self.min_size = min_size
        self.device = device

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, _):
        while True:
            path = random.choice(self.hr_files)
            try:
                hr = load_hr_volume(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue

            if any(d < self.min_size for d in hr.shape):
                continue

            hr = torch.from_numpy(hr).float().unsqueeze(0).to(self.device)  # (1, H, W, D)

            try:
                resolutions = sample_resolutions(self.vol_bin, self.aniso_bin, self.resolutions_df)
            except RuntimeError as e:
                print(f"Skipping {path}: {e}")
                continue
            
            lr = resample_volume(hr, resolutions)
            break

        lr = lr / lr.mean()
        hr = hr / hr.mean()
        return lr, hr, {"resolutions": resolutions}


def make_dataloader(hr_json, resolutions_df, vol_bin, aniso_bin, batch_size=1, num_workers=4, device="cpu"):
    dataset = ReconstructionDataset(hr_json, resolutions_df, vol_bin, aniso_bin, device=device)
    # CUDA tensors can't safely cross DataLoader worker process boundaries, so generation on GPU requires num_workers=0.
    on_gpu = torch.device(device).type != "cpu"
    num_workers = 0 if on_gpu else num_workers
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                       pin_memory=not on_gpu, persistent_workers=num_workers > 0, prefetch_factor=3 if num_workers > 0 else None)
