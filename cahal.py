"""
CAHAL Module Refactored (v2)
----------------------------
This module implements the CAHAL processing pipeline, including:
1. NIfTI Loading and Preprocessing (RAS alignment, Header fixing)
2. Optional Denoising (via independent executable)
3. Optional Isotropic Resampling (1mm)
4. Deep Learning Inference (NNUNet3D Ensemble)
5. Post-processing and Saving

"""

import time
import subprocess
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import nibabel as nii
from nibabel.orientations import io_orientation, ornt_transform, apply_orientation
from nibabel.processing import resample_to_output
from scipy.ndimage import zoom

import torch
import torch.nn as nn
import torch.nn.functional as F

import os 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:128"
torch.backends.cudnn.benchmark = False

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
DENOISE_EXECUTABLE = SCRIPT_DIR / "DenoiseImage"

# ANSI Colors for Terminal Output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    GREEN = "\033[32m"

# -----------------------------------------------------------------------------
# Logging Wrapper
# -----------------------------------------------------------------------------

class Logger:
    """Simple logger for formatted terminal output."""
    
    @staticmethod
    def _timestamp() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _log(prefix: str, msg: str, indent: int = 0, color: str = Colors.RESET) -> None:
        ind = "    " * indent
        print(f"{color}[{Logger._timestamp()}] {prefix}{Colors.RESET} {ind}{msg}")

    @classmethod
    def info(cls, msg: str, indent: int = 0) -> None:
        cls._log("INFO ", msg, indent, Colors.CYAN)

    @classmethod
    def warn(cls, msg: str, indent: int = 0) -> None:
        cls._log("WARN ", msg, indent, Colors.YELLOW)

    @classmethod
    def error(cls, msg: str, indent: int = 0) -> None:
        cls._log("ERROR", msg, indent, Colors.RED)

    @classmethod
    def success(cls, msg: str, indent: int = 0) -> None:
        cls._log("OK   ", msg, indent, Colors.GREEN)

# -----------------------------------------------------------------------------
# Neural Network Architecture (NNUNet3D)
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Basic 3D Convolutional Block with Instance Norm and LeakyReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downsampling Block: MaxPool3d + ConvBlock."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling Block: Generalized Trilinear Interpolation + Concat + ConvBlock."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Dynamically upsample to the skip connection size
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class NNUNet3D(nn.Module):
    """3D U-Net Architecture customized for the CAHAL task."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_features: int = 32):
        super().__init__()
        f = base_features
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = DownBlock(f, f * 2)
        self.enc3 = DownBlock(f * 2, f * 4)
        self.enc4 = DownBlock(f * 4, f * 8)
        self.bottom = DownBlock(f * 8, f * 16)

        # Decoder
        self.up4 = UpBlock(f * 16 + f * 8, f * 8)
        self.up3 = UpBlock(f * 8 + f * 4, f * 4)
        self.up2 = UpBlock(f * 4 + f * 2, f * 2)
        self.up1 = UpBlock(f * 2 + f, f)

        # Output
        self.final_conv = nn.Conv3d(f, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottom(e4)

        # Decoding
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        # Final output
        out = self.final_conv(d1)
        out = x + out         # Residual connection with input
        out = self.relu(out)  # Ensure non-negative output
        return out

# -----------------------------------------------------------------------------
# Image Processing Utilities
# -----------------------------------------------------------------------------

class NiftiUtils:
    """Static utilities for NIfTI file manipulation and orientation."""
    
    @staticmethod
    def fix_header(input_file: Union[str, Path], output_file: Union[str, Path]) -> Path:
        """Fix inconsistent NIfTI header information (sform/qform, voxel dimensions)."""
        img = nii.load(str(input_file))
        data = img.get_fdata()
        affine = img.affine
        
        fixed_img = nii.Nifti1Image(data, affine)
        # Harmonize qform/sform
        fixed_img.set_qform(affine, code='scanner')
        fixed_img.set_sform(affine, code='aligned')
        
        # Ensure pixel dims match affine scales
        header = fixed_img.header
        scales = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
        header['pixdim'][1:4] = scales
        
        nii.save(fixed_img, str(output_file))
        Logger.info(f"Fixed NIfTI header saved to: {output_file}", indent=1)
        return Path(output_file)

    @staticmethod
    def check_and_fix_singular_affine(filename: Union[str, Path]) -> None:
        """Detect singular affine matrix and overwrite with diagonal defaults if needed."""
        try:
            nii_obj = nii.load(str(filename))
            affine = nii_obj.affine
            det = np.linalg.det(affine[:3, :3])
            
            if abs(det) < 1e-10:
                Logger.warn(f"Singular affine (det={det:.2e}). Overwriting with diagonal voxel sizes.", indent=1)
                resolutions = nii_obj.header.get_zooms()[:3]
                
                new_affine = np.eye(4)
                np.fill_diagonal(new_affine[:3, :3], resolutions)
                
                nii_obj.header.set_sform(new_affine)
                nii_obj.header.set_qform(new_affine)
                
                # We need to construct a new image to save cleanly
                new_img = nii.Nifti1Image(nii_obj.get_fdata(), new_affine, nii_obj.header)
                nii.save(new_img, str(filename))
        except Exception as e:
            Logger.error(f"Failed to check/fix singular affine: {e}")

    @staticmethod
    def load_and_preprocess(filename: Union[str, Path], full_res: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray, nii.Nifti1Header, np.ndarray]]:
        """
        Load volume, ensure float32, check for empty content, and align to RAS.
        
        Returns:
            Tuple: (volume_data, original_affine, header, ras_aligned_affine)
        """
        try:
            img = nii.load(str(filename))
            affine = img.affine
            header = img.header
            
            Logger.info(f"Loaded volume shape: {img.shape}", indent=1)
            
            # Load data
            if full_res:
                volume = np.asanyarray(img.dataobj, dtype=np.float32)
            else:
                volume = np.asanyarray(img.dataobj[::2, ::2, ::2], dtype=np.float32)

            original_affine = affine.copy()

            # Handle dimensions (squeeze singleton)
            if volume.ndim > 3:
                volume = np.squeeze(volume)
            
            # Align to RAS
            volume, ras_affine = NiftiUtils.align_to_ras(volume, affine)

            # Check validity
            if np.sum(volume) == 0:
                Logger.warn("Empty volume detected.", indent=1)
                return None

            # Clean NaNs/Infs/Negative values
            volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
            volume[volume < 0] = 0
            
            return volume, original_affine, header, ras_affine

        except Exception as e:
            Logger.error(f"Error loading {filename}: {e}")
            raise

    @staticmethod
    def get_ras_axes(affine: np.ndarray, n_dims: int = 3) -> np.ndarray:
        """Identify RAS axes from affine matrix."""
        aff_inv = np.linalg.inv(affine)
        return np.argmax(np.absolute(aff_inv[0:n_dims, 0:n_dims]), axis=0)

    @staticmethod
    def align_to_ras(volume: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align volume to RAS orientation via affine manipulation.
        Extracted from Supersynth codebase.
        """
        n_dims = 3
        vol = volume.copy()
        aff = affine.copy().astype(np.float64)
        
        target_aff = np.eye(4) # Reference RAS
        
        try:
            current_ras = NiftiUtils.get_ras_axes(aff, n_dims)
            target_ras = NiftiUtils.get_ras_axes(target_aff, n_dims)
            
            if np.any(current_ras >= n_dims):
                Logger.warn("Invalid RAS axes, skipping axis alignment.")
                return vol, aff

            # 1. Align Axes
            aff[:, target_ras] = aff[:, current_ras]
            for i in range(n_dims):
                if current_ras[i] != target_ras[i]:
                    vol = np.swapaxes(vol, current_ras[i], target_ras[i])
                    # Update local tracking of axes
                    idx = np.where(current_ras == target_ras[i])[0][0]
                    current_ras[idx], current_ras[i] = current_ras[i], current_ras[idx]
            
            # 2. Align Directions
            dot_products = np.sum(aff[:3, :3] * target_aff[:3, :3], axis=0)
            for i in range(n_dims):
                if dot_products[i] < 0:
                    vol = np.flip(vol, axis=i)
                    aff[:, i] = -aff[:, i]
                    aff[:3, 3] = aff[:3, 3] - aff[:3, i] * (vol.shape[i] - 1)
            
            return vol, aff

        except Exception as e:
            Logger.error(f"Alignment failed: {e}. Returning original.")
            return volume, affine

    @staticmethod
    def save(volume: np.ndarray, output_path: Union[str, Path], affine: np.ndarray = np.eye(4), header: Optional[nii.Nifti1Header] = None):
        """Save numpy array to NIfTI."""
        s_vol = np.squeeze(volume).astype(np.float32)
        Logger.info(f"Saving NIfTI: {output_path}", indent=1)
        
        img = nii.Nifti1Image(s_vol, affine, header=header)
        nii.save(img, str(output_path))

    @staticmethod
    def resample_image_to_1mm(input_path: Path, output_path: Path) -> bool:
        """Resample NIfTI to 1mm isotropic via F.interpolate (PyTorch backend for speed if avail)."""
        try:
            img_obj = nii.load(str(input_path))
            # resample_to_output automatically calculates the new affine
            resampled = resample_to_output(img_obj, order=1, voxel_sizes=(1.0, 1.0, 1.0))
            # Save
            nii.save(resampled, str(output_path))
            Logger.success(f"Resampling complete: {output_path.name}")
            return True
        except Exception as e:
            Logger.error(f"Resampling error: {e}")
            return False

# -----------------------------------------------------------------------------
# Resolution Analysis
# -----------------------------------------------------------------------------

def analyze_volume_resolution(filename: Union[str, Path]) -> Tuple[int, int]:
    """
    Determine appropriate model clusters based on volume size and anisotropy.
    
    Returns:
        Tuple[int, int]: (volume_cluster_id, anisotropy_cluster_id)
    """
    bins_vol = [1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, np.inf]
    labels_vol = [1, 2, 3, 4, 5, 6, 7]

    img = nii.load(str(filename))
    res = list(img.header.get_zooms()[:3])
    Logger.info(f"Voxel resolutions: {res}", indent=1)
    
    # If all resolutions are below or equal to 1.0, assign negative clusters
    if all(r <= 1.0 for r in res):
        Logger.info("All resolutions <= 1.0mm. Not using CAHAL.", indent=1)
        return -1, -1
    # Floor resolution to 1.0
    res = [max(r, 1.0) for r in res]
    
    # Anisotropy
    min_r, max_r = min(res), max(res)
    ratio = min_r / max_r if max_r > 0 else 1.0
    
    if ratio <= 0.33:
        aniso_cluster = 1
    elif ratio <= 0.66:
        aniso_cluster = 2
    else:
        aniso_cluster = 3

    # Volume Cluster (product of resolutions)
    vol_prod = res[0] * res[1] * res[2]
    vol_cluster = labels_vol[-1] # Default to max
    for i in range(len(bins_vol) - 1):
        if bins_vol[i] <= vol_prod < bins_vol[i+1]:
            vol_cluster = labels_vol[i]
            break

    return vol_cluster, aniso_cluster

# -----------------------------------------------------------------------------
# Inference Engine
# -----------------------------------------------------------------------------

class InferenceEngine:
    """Handles model loading and inference execution."""

    @staticmethod
    def load_ensemble(vol_cluster: int, aniso_cluster: int, device: torch.device, modality_models: str) -> Optional[nn.Module]:
        """Load the PSNR-best model checkpoint for the given cluster."""
        nf = 64
        MODELS_DIR = SCRIPT_DIR / f"models_{modality_models.lower()}_half"
        metric = 'psnr'
        
        model = NNUNet3D(in_channels=1, out_channels=1, base_features=nf)
        name = (f"model_R{vol_cluster}_A{aniso_cluster}_NNUNET_nf{nf}_half_precission_"
                f"new_clustering_sequential_seg_loss_full_res_complex_anistropies_best_{metric}.pth")
        
        path = MODELS_DIR / name
        
        try:
            state = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            model.half() # Convert to half precision for inference
            Logger.info(f"Loaded {metric} model: {name}")
        except Exception as e:
            Logger.error(f"Failed to load {metric} model at {path}: {e}")
            return None
    
        return model

    @staticmethod
    def predict_ensemble(model: nn.Module, input_tensor: torch.Tensor, device: torch.device, ttda: bool = False) -> torch.Tensor:
        """
        Run inference on a pre-normalized tensor (values ≈ O(1)) with the single
        PSNR-best model.
        Normalization and denormalization are handled externally in CAHAL() using
        the volume global mean — no per-call normalization here, which would cause
        patch-boundary intensity offsets in tiled inference.
        """
        # autocast uses actual device.type — avoids hard failure on CPU-only runs.
        with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
            with torch.inference_mode():
                model.to(device)
                pred = model(input_tensor)
                if ttda:
                    # Depth-axis flip augmentation (B, C, D, H, W → flip dim 2)
                    pred_flipped = model(torch.flip(input_tensor, dims=[2]))
                    pred = (pred + torch.flip(pred_flipped, dims=[2])) / 2.0
                    del pred_flipped

        # NaN/Inf guard: InstanceNorm3d produces NaN when all spatial values are
        # identical (zero spatial variance), which can still occur on tiles that
        # slip through the background skip (e.g. near-constant border regions).
        # Replace affected voxels with input pass-through so the region retains
        # original signal rather than being silently zeroed.
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            n_bad = int((torch.isnan(pred) | torch.isinf(pred)).sum().item())
            Logger.warn(
                f"NaN/Inf in model output ({n_bad} voxels). "
                f"Input — mean: {input_tensor.float().mean():.4f}, "
                f"max: {input_tensor.float().max():.4f}. "
                "Replacing with input pass-through."
            )
            bad_mask = torch.isnan(pred) | torch.isinf(pred)
            pred = torch.where(bad_mask, input_tensor, pred)
            torch.nan_to_num_(pred, nan=0.0, posinf=0.0, neginf=0.0)

        return pred.squeeze()
    
    
    @staticmethod
    def predict_tiled(
        models: List[nn.Module], 
        input_tensor: torch.Tensor, 
        device: torch.device, 
        tile_size: Tuple[int, int, int] = (200, 200, 200), 
        stride_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Predicts a large volume by breaking it into overlapping tiles.
        Uses Hann importance weighting to blend edges.
        Applies mirror padding when tiles extend beyond volume boundaries.
        """
        B, C, D, H, W = input_tensor.shape

        # Create a Hann weight map for a single tile to smooth edges.
        # Kept on CPU — tiles are moved to device individually at inference time.
        def get_gaussian_map(shape):
            z, y, x = shape
            gauss_z = torch.hann_window(z, periodic=False)
            gauss_y = torch.hann_window(y, periodic=False)
            gauss_x = torch.hann_window(x, periodic=False)
            return gauss_z[:, None, None] * gauss_y[None, :, None] * gauss_x[None, None, :]

        weight_tile = get_gaussian_map(tile_size)  # CPU — matches CPU accumulators
        strides = [int(ts * stride_ratio) for ts in tile_size]

        # Calculate padding needed to ensure full coverage
        # We need to extend the volume if the last tile doesn't fit
        pad_d = max(0, ((D - tile_size[0] + strides[0] - 1) // strides[0]) * strides[0] + tile_size[0] - D)
        pad_h = max(0, ((H - tile_size[1] + strides[1] - 1) // strides[1]) * strides[1] + tile_size[1] - H)
        pad_w = max(0, ((W - tile_size[2] + strides[2] - 1) // strides[2]) * strides[2] + tile_size[2] - W)
        
        # Apply mirror padding if needed
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            Logger.info(f"Applying mirror padding: D={pad_d}, H={pad_h}, W={pad_w}", indent=2)
            # PyTorch padding format: (left, right, top, bottom, front, back)
            padding = (0, pad_w, 0, pad_h, 0, pad_d)
            input_padded = F.pad(input_tensor, padding, mode='reflect')
            D_padded, H_padded, W_padded = input_padded.shape[2:]
        else:
            input_padded = input_tensor
            D_padded, H_padded, W_padded = D, H, W

        # Log number of tiles
        num_tiles_d = max(1, (D_padded - tile_size[0]) // strides[0] + 1)
        num_tiles_h = max(1, (H_padded - tile_size[1]) // strides[1] + 1)
        num_tiles_w = max(1, (W_padded - tile_size[2]) // strides[2] + 1)
        num_tiles = num_tiles_d * num_tiles_h * num_tiles_w
        
        Logger.info(f"Predicting with tiled inference. Total tiles: {num_tiles} ({num_tiles_d}×{num_tiles_h}×{num_tiles_w})", indent=1)

        # CPU accumulators — keeps GPU footprint to one tile at a time (Fix 2.1).
        # fp32 precision for accurate Hann-weighted blending across many overlapping tiles.
        full_output_padded = torch.zeros((B, 1, D_padded, H_padded, W_padded), dtype=torch.float32)
        count_map_padded   = torch.zeros((B, 1, D_padded, H_padded, W_padded), dtype=torch.float32)

        # Iterate through the volume
        tile_idx = 0
        for z in range(0, D_padded - tile_size[0] + 1, strides[0]):
            for y in range(0, H_padded - tile_size[1] + 1, strides[1]):
                for x in range(0, W_padded - tile_size[2] + 1, strides[2]):
                    tile_idx += 1
                    
                    # Extract tile with proper bounds
                    z_end = min(z + tile_size[0], D_padded)
                    y_end = min(y + tile_size[1], H_padded)
                    x_end = min(x + tile_size[2], W_padded)
                    
                    z_slice = slice(z, z_end)
                    y_slice = slice(y, y_end)
                    x_slice = slice(x, x_end)
                    
                    tile = input_padded[:, :, z_slice, y_slice, x_slice]

                    # Actual tile shape (edge tiles may be smaller than tile_size)
                    actual_tile_shape = tile.shape[2:]

                    # Resolve Hann weight before any branching so the skip path can use it
                    if actual_tile_shape != tile_size:
                        current_weight = weight_tile[:actual_tile_shape[0], :actual_tile_shape[1], :actual_tile_shape[2]]
                    else:
                        current_weight = weight_tile

                    # Background tile skip: input is pre-normalized by global_mean so brain
                    # tissue has mean ≈ 1.0. Tiles with mean < 1e-4 are background or padding.
                    # Running inference on them causes InstanceNorm3d NaN (zero spatial
                    # variance). Fill with input pass-through so border voxels retain signal.
                    if tile.mean().item() < 1e-4:
                        full_output_padded[:, :, z_slice, y_slice, x_slice] += tile.float() * current_weight
                        count_map_padded[:, :, z_slice, y_slice, x_slice] += current_weight
                        # log skipped tile 
                        Logger.info(f"Skipping background tile with stats mean={tile.mean().item():.4f}, min={tile.min():.4f}, max={tile.max():.4f}", indent=2)
                        continue

                    # Per-tile normalization (Fix 2.3): compute mean from the original
                    # (unpadded) tile in fp32. The tile comes from the globally-normalized
                    # volume (values ≈ O(1)), so tile_mean is in the same scale.
                    # Clamped to 1e-4 to match the global_mean floor and prevent
                    # division by zero for any near-zero tile that passes the skip.
                    tile_mean = float(tile.float().mean().clamp(min=1e-4).item())

                    # Pad edge tiles to full tile_size for model input
                    if actual_tile_shape != tile_size:
                        pad_needed = [
                            (0, tile_size[2] - actual_tile_shape[2]),
                            (0, tile_size[1] - actual_tile_shape[1]),
                            (0, tile_size[0] - actual_tile_shape[0])
                        ]
                        tile = F.pad(tile, [p for pair in pad_needed for p in pair], mode='reflect')

                    # Normalize by local tile mean so the model receives a distribution
                    # matching its training inputs (mean ≈ 1.0). Cast to fp32 for the
                    # division, then back to fp16 for GPU memory efficiency.
                    tile_norm = (tile.float() / tile_mean).half()

                    # Move normalized tile to GPU for inference; input_padded stays on CPU (Fix 2.1 / Fix 2.2).
                    tile_gpu = tile_norm.to(device)
                    tile_pred = InferenceEngine.predict_ensemble(
                        models, tile_gpu, device, ttda=False
                    )
                    del tile_gpu, tile_norm  # free GPU and CPU normalized tile immediately
                    tile_pred = tile_pred.squeeze(0).unsqueeze(0)  # (1, D_tile, H_tile, W_tile)

                    # Crop prediction back to actual tile size if input was padded
                    if actual_tile_shape != tile_size:
                        tile_pred = tile_pred[:, :, :actual_tile_shape[0], :actual_tile_shape[1], :actual_tile_shape[2]]

                    # Progress indicator
                    if tile_idx % 10 == 0:
                        Logger.info(f"Processed {tile_idx}/{num_tiles} tiles", indent=2)
                        Logger.info(f"Tile stats - tile_mean: {tile_mean:.4f} | Input mean (norm): {tile.float().mean().item()/tile_mean:.4f}, min: {tile.min():.4f}, max: {tile.max():.4f} | Predicted mean: {tile_pred.float().mean():.4f}, max: {tile_pred.float().max():.4f}", indent=3)

                    # Denormalize: multiply by tile_mean to restore globally-normalized space.
                    # The chain is: original → /global_mean → /tile_mean → model → ×tile_mean → ×global_mean → original.
                    # Accumulate on CPU (Fix 2.1).
                    full_output_padded[:, :, z_slice, y_slice, x_slice] += (tile_pred.float().cpu() * tile_mean * current_weight)
                    count_map_padded[:, :, z_slice, y_slice, x_slice] += current_weight

        # Crop back to original size (accumulators are on CPU)
        full_output = full_output_padded[:, :, :D, :H, :W]
        count_map = count_map_padded[:, :, :D, :H, :W]

        # Normalize and return (already on CPU)
        return full_output / torch.clamp(count_map, min=1e-5)
    
    @staticmethod
    def estimate_full_volume_bytes(
        shape: Tuple[int, int, int],
        base_features: int = 64,
        n_stages: int = 4
    ) -> int:
        """
        Estimate peak GPU bytes for a full-volume no-grad forward pass.
        """
        V = shape[0] * shape[1] * shape[2]
        bytes_per_elem = 2  # fp16

        # Peak: skip_0 + upsample + cat(skip_0, upsample) at full resolution
        decoder_cat_peak = 4 * base_features * V * bytes_per_elem

        # Input tensor on GPU (fp16) + first TTDA prediction held alive during
        # the second forward pass
        io_tensors = 2 * V * bytes_per_elem

        return int((decoder_cat_peak + io_tensors) * 1.6)

    @staticmethod
    def predict_tiled_with_fallback(
        models: List[nn.Module], 
        input_tensor: torch.Tensor, 
        device: torch.device,
        tile_sizes: List[int] = [200, 128, 92, 64],
        stride_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Predicts using tiled inference with automatic fallback to smaller tiles on OOM.
        
        Args:
            models: List of neural network models
            input_tensor: Input volume tensor
            device: Torch device
            tile_sizes: List of tile sizes to try (in descending order)
            stride_ratio: Overlap ratio between tiles
        
        Returns:
            Predicted volume tensor
        """
        for tile_size in tile_sizes:
            try:
                Logger.info(f"Attempting tiled inference with tile size: {tile_size}³", indent=1)
                
                output = InferenceEngine.predict_tiled(
                    models=models,
                    input_tensor=input_tensor,
                    device=device,
                    tile_size=(tile_size, tile_size, tile_size),
                    stride_ratio=stride_ratio
                )
                
                Logger.success(f"Tiled inference succeeded with tile size: {tile_size}³", indent=1)
                return output
                
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                    raise  # Re-raise non-OOM errors
                
                Logger.warn(f"OOM with tile size {tile_size}³. Trying smaller tiles...", indent=1)
                torch.cuda.empty_cache()
                
                if tile_size == tile_sizes[-1]:  # Last attempt failed
                    raise RuntimeError(f"All tile sizes failed. Tried: {tile_sizes}")
        
        raise RuntimeError("Tiled inference failed for unknown reasons")

# -----------------------------------------------------------------------------
# External Tools
# -----------------------------------------------------------------------------

def run_ants_denoise(input_path: Path, output_path: Path) -> bool:
    """Execute external ANTs DenoiseImage tool."""
    cmd = [
        str(DENOISE_EXECUTABLE),
        "-d", "3",
        "-i", str(input_path),
        "-o", str(output_path)
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        if not output_path.exists():
            Logger.error(f"Denoise output missing: {output_path}")
            return False
        Logger.success(f"Denoising complete: {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        Logger.error(f"Denoise failed: {e.stderr}")
        return False
    except Exception as e:
        Logger.error(f"Denoise exec error: {e}")
        return False

# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------

def CAHAL(filename: str, denoise: bool = False, resample: bool = True, output_dir: Optional[str] = None,
             modality_models: str = "T1", gpu_memory_limit_gb: Optional[float] = None) -> str:
    """
    Main entry point for  CAHAL pipeline.

    Args:
        filename: Path to input NIfTI file.
        denoise: Whether to apply ANTs denoising step.
        resample: Whether to resample to 1mm isotropic before inference.
        output_dir: Optional custom output directory if not the same as input.
        modality_models: Whether to use models trained on T1 or FLAIR.
        gpu_memory_limit_gb: Effective GPU memory cap in GB.  Must be set when
            torch.cuda.set_per_process_memory_fraction() has been called, because
            mem_get_info() returns physical driver-level free memory and is unaware
            of the fraction cap.  When provided, the memory pre-check computes
            effective_free = cap - memory_reserved() instead of querying the driver.

    Returns:
        str: Path to the processed output file.
    """
    
    file_path = Path(filename).resolve()
    base_dir = file_path.parent
    dest_dir = Path(output_dir) if output_dir else base_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    Logger.info(f"Starting CAHAL pipeline on: {file_path.name}")
    start_time = time.time()

    final_output_path = dest_dir / f"CAHAL_processed_{file_path.name}"
    if final_output_path.exists():
        Logger.warn(f"Output already exists: {final_output_path}")
        return str(final_output_path)

    # 0. Header Analysis & Fixes
    vol_cluster, aniso_cluster = analyze_volume_resolution(file_path)

    current_file = file_path  # use original file for processing; header fixes are in-place
    if vol_cluster == -1:
        Logger.info("Volume does not meet CAHAL criteria. Returning original.", indent=1)
        return filename

    # 1. Denoise
    if denoise:
        denoised_path = dest_dir / f"denoised_{file_path.name}"
        t0 = time.time()
        if run_ants_denoise(current_file, denoised_path):
            current_file = denoised_path
            Logger.info(f"Denoising duration: {time.time() - t0:.2f}s", indent=1)
        else:
             Logger.warn("Denoising skipped due to error.", indent=1)

    # 2. Resample to 1mm
    if resample:
        resampled_path = dest_dir / f"resampled_{file_path.name}"
        t0 = time.time()
        if NiftiUtils.resample_image_to_1mm(current_file, resampled_path):
            current_file = resampled_path
            Logger.info(f"Resampling duration: {time.time() - t0:.2f}s", indent=1)
        else:
            Logger.warn("Resampling skipped due to error.", indent=1)

    # 3. Model Inference: first on cpu then owe will move to GPU to reduce memory OOM risk during loading
    device = torch.device("cpu")
    models = InferenceEngine.load_ensemble(vol_cluster, aniso_cluster, device, modality_models)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not models:
        raise RuntimeError("Failed to load ensemble models.")

    # Load data for inference
    loaded = NiftiUtils.load_and_preprocess(current_file, full_res=True)
    if not loaded:
        raise ValueError("Preprocessing returned empty data.")
        
    volume, orig_affine, orig_header, ras_affine = loaded
    
    # Prepare Tensor (Batch, Channel, D, H, W)
    original_shape = volume.shape  # D, H, W — stored before any padding for output crop

    # Compute global mean in fp32 from the original (unpadded) volume.
    # fp16 underflows MRI intensities below ~6e-5 to 0, making per-tile means
    # unreliable. A single global scalar is shared across all tiles — this is
    # the only correct way to ensure consistent intensity scale throughout tiled
    # inference and avoid patch-boundary artifacts in the blended output.
    input_tensor_fp32 = torch.from_numpy(volume[np.newaxis, np.newaxis, ...]).float()
    global_mean = float(torch.clamp(input_tensor_fp32.mean(), min=1e-4).item())
    Logger.info(f"Global mean for normalization: {global_mean:.4f}", indent=1)
    del input_tensor_fp32  # free before possible padding allocation

    # --- Guard A: bottleneck dimension safety ---
    # The U-Net has 4 MaxPool3d(2) stages → each spatial dim is divided by 16 at
    # the bottleneck.  InstanceNorm3d on a spatial dim of 1 has zero variance
    # (σ = 0), which produces NaN that propagates through fp16 cuDNN kernels and
    # can cause SIGSEGV rather than a recoverable Python exception.
    # Even with 1×1×1 isotropic resampling, very thin acquisitions (e.g. 5 slices
    # at 5 mm → 25 voxels after resampling) can hit this threshold.
    # Fix: symmetrically zero-pad any axis below 32 voxels (min safe bottleneck = 2).
    # The padding is recorded and the output is cropped back before saving.
    volume_pad_widths = None
    bottleneck_dims = [d // 16 for d in original_shape]
    if min(bottleneck_dims) < 2:
        pads = [max(0, 32 - d) for d in original_shape]
        pad_widths = [(p // 2, p - p // 2) for p in pads]
        volume = np.pad(volume, pad_widths, mode='constant', constant_values=0)
        volume_pad_widths = pad_widths
        Logger.warn(
            f"Volume {original_shape} → bottleneck {bottleneck_dims}: min dim < 32. "
            f"Symmetrically zero-padded to {volume.shape} for safe inference.",
            indent=1
        )

    # --- Guard B: fp16 activation overflow from hyperintense voxels ---
    # TRACE images in acute stroke carry lesion voxels 10–30× brighter than normal
    # parenchyma.  After global-mean normalization these can remain at 20–50 in
    # the normalised space.  He-init Conv3d + LeakyReLU across 4 encoder stages
    # can amplify these values beyond the fp16 maximum (~65504), producing inf in
    # cuDNN workspace memory and causing a kernel fault (SIGSEGV).  ADC images
    # have diffuse uniform intensities and do not trigger this path.
    # Threshold 50 is conservative: 50 × 2.5 overhead × 4-stage amplification
    # ≈ 500, far below 65504 at tile scale but unsafe for full-volume deep layers.
    force_tiled = False
    p999_normalized = float(np.percentile(volume, 99.9)) / global_mean
    if p999_normalized > 50.0:
        Logger.warn(
            f"Normalized 99.9th percentile = {p999_normalized:.1f} "
            "(fp16 max ~65504; activation overflow risk in deep encoder layers). "
            "Forcing tiled inference directly.",
            indent=1
        )
        force_tiled = True

    # Normalize in fp32 (values ≈ O(1)), then cast to fp16 for memory-efficient
    # inference. The model was trained on globally-normalized inputs — this
    # preserves that distribution contract at inference time.
    input_tensor_fp32 = torch.from_numpy(volume[np.newaxis, np.newaxis, ...]).float()
    input_tensor = (input_tensor_fp32 / global_mean).half().to(device)
    del input_tensor_fp32

    # --- Guard Q3: GPU memory pre-check ---
    # Estimate peak GPU bytes for a full-volume forward pass (skip connections +
    # bottleneck + input tensor + 2.5× overhead).  If the estimate exceeds 85% of
    # the currently free GPU memory, skip the full-volume attempt entirely and go
    # straight to tiled inference — avoiding the OOM-triggered fallback path and
    # its associated GPU state fragmentation (which previously caused SIGSEGV).
    if not force_tiled and device.type == 'cuda':
        idx = device.index if device.index is not None else 0
        if gpu_memory_limit_gb is not None:
            # set_per_process_memory_fraction() enforces the cap inside PyTorch's
            # allocator, but mem_get_info() queries the physical driver and returns
            # the full GPU free memory — it has no knowledge of the fraction.
            # Correct formula: effective_free = cap - already_reserved_by_pytorch
            # memory_reserved() is what PyTorch has already claimed from the driver
            # (includes cached-but-free blocks).  This accurately represents what
            # the allocator still has headroom for before hitting the fraction wall.
            cap_bytes   = int(gpu_memory_limit_gb * 1024 ** 3)
            free_bytes  = max(0, cap_bytes - torch.cuda.memory_reserved(idx))
            Logger.info(
                f"Memory pre-check (capped mode): cap={gpu_memory_limit_gb:.1f} GB, "
                f"reserved={torch.cuda.memory_reserved(idx)/1e9:.2f} GB, "
                f"effective free={free_bytes/1e9:.2f} GB.",
                indent=1
            )
        else:
            # No external cap — use driver-reported free memory directly.
            free_bytes, _ = torch.cuda.mem_get_info(idx)
            Logger.info(
                f"Memory pre-check: {free_bytes / 1e9:.2f} GB free (CUDA driver).",
                indent=1
            )
        required_bytes = InferenceEngine.estimate_full_volume_bytes(volume.shape)
        Logger.info(
            f"Memory pre-check: estimated {required_bytes / 1e9:.2f} GB required, "
            f"{free_bytes / 1e9:.2f} GB effective free.",
            indent=1
        )
        if required_bytes > free_bytes * 0.85:
            Logger.warn(
                f"Estimated footprint ({required_bytes / 1e9:.2f} GB) exceeds 85% of "
                f"effective free GPU memory ({free_bytes / 1e9:.2f} GB). "
                "Forcing tiled inference directly.",
                indent=1
            )
            force_tiled = True

    try:
        t0 = time.time()
        if force_tiled:
            # Pre-checks (Guard A, B, or Q3) determined full-volume inference is
            # unsafe or will OOM.  Move tensor to CPU immediately so predict_tiled
            # never holds a full-volume fp16 copy on the GPU (Fix 2.2).
            input_tensor = input_tensor.cpu()
            torch.cuda.empty_cache()
            Logger.info("Forced tiled inference (pre-check passed). Skipping full-volume attempt.", indent=1)
            output_volume = InferenceEngine.predict_tiled_with_fallback(
                models=models,
                input_tensor=input_tensor,
                device=device,
                tile_sizes=[200, 128, 92, 64],
                stride_ratio=0.5
            )
            final_output_path = dest_dir / f"CAHAL_tiled_inference_{file_path.name}"
        else:
            try:
                Logger.info("Attempting inference at original resolution...", indent=1)
                output_volume = InferenceEngine.predict_ensemble(models, input_tensor, device, ttda=False)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:

                if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                    raise e  # Re-raise non-OOM errors

                # Offload input_tensor to CPU before tiled inference (Fix 2.2).
                # After the failed full-volume attempt, input_tensor is still resident
                # on the GPU. predict_tiled would allocate input_padded on top of it,
                # holding two full-volume fp16 copies simultaneously.
                input_tensor = input_tensor.cpu()
                torch.cuda.empty_cache()
                Logger.warn(
                    f"OOM detected. Volume {original_shape} too large. "
                    "Using tiled inference with automatic fallback.",
                    indent=1
                )
                output_volume = InferenceEngine.predict_tiled_with_fallback(
                    models=models,
                    input_tensor=input_tensor,
                    device=device,
                    tile_sizes=[128, 92, 64],
                    stride_ratio=0.5
                )
                final_output_path = dest_dir / f"CAHAL_tiled_inference_{file_path.name}"

        Logger.info(f"Inference duration: {time.time() - t0:.2f}s", indent=1)

        # 4. Post-processing (Restore Orientation)
        # Denormalize: model output is in normalized space — restore original intensity scale.
        if torch.is_tensor(output_volume):
            output_volume = (output_volume.float() * global_mean).detach().cpu().numpy()
        else:
            output_volume = output_volume * global_mean

        # Crop out the safety padding added by Guard A, if any.
        # output_volume may be (D,H,W) from predict_ensemble or (1,1,D,H,W) from
        # predict_tiled — use [...] indexing to handle both shapes transparently.
        if volume_pad_widths is not None:
            slices = tuple(
                slice(pw[0], pw[0] + s)
                for pw, s in zip(volume_pad_widths, original_shape)
            )
            output_volume = output_volume[..., slices[0], slices[1], slices[2]]
            Logger.info(
                f"Cropped output back to original shape {original_shape} (Guard A padding removed).",
                indent=1
            )

        # Transform back from RAS to original orientation
        start_ornt = io_orientation(ras_affine)
        end_ornt = io_orientation(orig_affine)
        transform_ornt = ornt_transform(start_ornt, end_ornt)
        
        final_volume = apply_orientation(output_volume, transform_ornt)

        # Restore original header info
        if orig_header is not None:
            orig_header.set_sform(orig_affine)
            orig_header.set_qform(orig_affine)

        NiftiUtils.save(final_volume, final_output_path, affine=orig_affine, header=orig_header)
        Logger.success(f"Pipeline completed: {final_output_path}")

    except Exception as e:
        Logger.error(f"Inference pipeline failed: {e}")
        raise

    Logger.info(f"Total processing time: {time.time() - start_time:.2f}s")
    return str(final_output_path)