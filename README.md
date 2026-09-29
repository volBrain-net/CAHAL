# CAHAL Module

cahal.py implements the pipeline to use CAHAL [CLINICALLY APPLICABLE RESOLUTION ENHANCEMENT FOR LOW-RESOLUTION MRI SCANS] from native samples (raw):

1. Denoising using ANTs (linux library) `DenoiseImage`.
2. Resampling to 1×1×1 mm isotropic voxels using ANTs/ANTsPy.
3. CNN inference (single PSNR-best NN-UNet model per resolution/anisotropy cluster) for the CAHAL processing step.

This repository contains the pipeline script and expects model and data folders next to it.

## Requirements
- Python 3.8+
- Packages:
  - nibabel
  - numpy
  - torch
- `DenoiseImage` executable from ANTs must be placed next to `cahal.py` and be executable (chmod +x).
- CUDA is optional; the script will use CPU if no GPU is available.

Install (example):
```bash
pip install nibabel numpy antspyx torch torchio
```

## Files & Folders
- cahal.py — main pipeline script.
- models_*_half/ — trained model files. To download the models go to https://zenodo.org/records/23032648 and download T1w and FLAIR pth files. 


## Usage
Run the module directly:
```bash
python cahal_module_test.py
```
Behavior:
- Files in `data/` are processed in sequence.
- If a filename does not contain `denoised_`, script runs `DenoiseImage` and creates `processed_data/denoised_<orig>`.
- If filename does not contain `resampled_`, script resamples to 1mm and creates `processed_data/resampled_<name>`.
- After preprocessing the script loads the appropriate model and runs inference, saving results as `processed_data/CAHAL_processed_<name>`.

You can skip denoising/resampling by pre-naming files with `denoised_` and/or `resampled_`.

## Logging / Output
The script uses simple timestamped, leveled prints:
- INFO — general steps and timings
- WARN — recoverable issues (e.g., empty volume)
- ERROR — failures
- OK — successful major steps

Log lines include a timestamp and an indent parameter for nested messages.

## Notes & Tips
- Ensure voxel metadata (header/affine) is consistent; the script attempts to fix common header issues and sets output header zooms to 1.0 mm.
- The script expects input orientation alignment and will align to RAS internally.

## Troubleshooting
- "Denoised file not created": verify `DenoiseImage` presence and permissions; check subprocess STDOUT/STDERR (printed on error).
- Model loading errors: ensure model filenames and the `models/` folder exist and are readable.
- Memory / CUDA errors: try CPU mode (no CUDA) or reduce batch/volume size.

