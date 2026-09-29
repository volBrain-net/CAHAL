# REMIX reconstruction model — training code

Simplified, self-contained release of the training pipeline: a 3D U-Net that
reconstructs a high-resolution (HR) volume from a synthetically degraded
low-resolution (LR) volume, jointly supervised by a frozen segmentation model.

## Files

- `clusters.py` — volumetric-resolution and anisotropy bin definitions.
- `resolution_sampler.py` — samples an LR resolution triplet for a given
  (volume bin, anisotropy bin), analytically or by drawing from observed data.
- `data.py` — `ReconstructionDataset`: loads an HR volume, synthesizes its LR
  counterpart by resampling, applies augmentation, and returns an `(LR, HR)` pair.
- `model.py` — `NNUNet3D`, the 3D U-Net used for reconstruction.
- `losses.py` — reconstruction loss (`mixloss` = weighted MAE + frequency loss),
  Dice loss/metric, correlation, PSNR.
- `utils.py` — device setup, LR scheduler, early stopping, checkpointing, and
  `Segmentation_model`, a thin wrapper around a frozen segmentation network.
- `train.py` — training entry point.

## Data format

**HR samples** (`hr_json`): a JSON file listing paths to HR NIfTI volumes:

```json
{"hr_files": ["/data/subject001.nii.gz", "/data/subject002.nii.gz"]}
```

**LR resolution distribution** (`resolutions_csv`): a CSV with the resolution
(mm/voxel) of real LR acquisitions, used both to build the training curriculum
and as a fallback sampling source:

```csv
Resolution_X,Resolution_Y,Resolution_Z
2.2,1.09375,1.09375
1.0,1.0,1.0
```

**Segmentation model**: training uses a frozen, pretrained segmentation
network as an auxiliary loss. Plug your architecture into
`build_segmentation_model()` in `train.py`:

```python
from your_segmentation_model import SegmentationArchitecture

def build_segmentation_model(checkpoint_path, device):
    architecture = SegmentationArchitecture(...)
    seg_model = Segmentation_model(checkpoint_path, architecture, device)
    return nn.Sequential(seg_model.model, nn.Softmax(dim=1))
```

## Training

Models are trained progressively over a curriculum of volumetric-resolution
clusters (1 = near-isotropic 1mm, 7 = very low resolution) and, within each,
anisotropy clusters (3 = isotropic to 1 = highly anisotropic), transferring
weights from the previous cluster in the curriculum.

```bash
python train.py hr_files.json lr_resolutions.csv segmentation_checkpoint.pth \
    --output-dir models --nf 64 --lr 1e-3 --epochs 1000 --steps-per-epoch 20
```

Checkpoints (`model_R{vol_bin}_A{aniso_bin}_best_psnr.pth` /
`_best_dice.pth`) and TensorBoard logs are written to `--output-dir`.

## Requirements

pip install -r requirements.txt