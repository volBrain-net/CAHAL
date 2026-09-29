import argparse
import os

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from clusters import bins, bin_labels, anisotropy_labels
from data import make_dataloader, load_csv_resolutions
from losses import mixloss, correlation, calculate_psnr, mdice, mdice_loss
from model import NNUNet3D
from utils import (configure_device, EarlyStopping, ModelCheckpoint,
                    SequentialPlateauScheduler, Segmentation_model, DataGenerationLogger)

# Provide your own pretrained segmentation architecture here. It only needs
# to be a nn.Module producing per-class probability/logit volumes; it is kept
# frozen and used as an auxiliary supervision signal (see build_segmentation_model).
# from your_segmentation_model import SegmentationArchitecture

GRAD_CLIP_THRESHOLDS = {1e-3: 1.0, 1e-4: 2.0, 1e-5: 5.0, 1e-6: None}


def build_segmentation_model(checkpoint_path, device):
    architecture = None  # SegmentationArchitecture(...)
    if architecture is None:
        raise NotImplementedError(
            "Plug in your segmentation architecture in build_segmentation_model()."
        )
    seg_model = Segmentation_model(checkpoint_path, architecture, device)
    return nn.Sequential(seg_model.model, nn.Softmax(dim=1))


def adaptive_clip_grad(model, current_lr):
    max_norm = None
    for threshold in sorted(GRAD_CLIP_THRESHOLDS, reverse=True):
        if current_lr >= threshold:
            max_norm = GRAD_CLIP_THRESHOLDS[threshold]
            break
    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def train_cluster(config, train_loader, vol_bin, aniso_bin, segmentation_model, device, output_dir):
    model_name = os.path.join(output_dir, f"model_R{vol_bin}_A{aniso_bin}.pth")
    os.makedirs(output_dir, exist_ok=True)

    generator = NNUNet3D(in_channels=1, out_channels=1, base_features=config["nf"]).to(device)

    # Progressive transfer learning: start from the previous anisotropy cluster
    # (within the same resolution bin), or from the most isotropic cluster of
    # the previous resolution bin.
    best_path = model_name.replace(".pth", "_best_psnr.pth")
    if os.path.exists(best_path):
        generator.load_state_dict(torch.load(best_path))
        print(f"Model {best_path} already trained, skipping.")
        return
    elif aniso_bin < 3:
        prev = os.path.join(output_dir, f"model_R{vol_bin}_A{aniso_bin + 1}_best_psnr.pth")
        if os.path.exists(prev):
            generator.load_state_dict(torch.load(prev))
            print(f"Transferred weights from {prev}")
    elif vol_bin > 1:
        prev = os.path.join(output_dir, f"model_R{vol_bin - 1}_A3_best_psnr.pth")
        if os.path.exists(prev):
            generator.load_state_dict(torch.load(prev))
            print(f"Transferred weights from {prev}")

    optimizer = torch.optim.Adam(generator.parameters(), lr=config["lr"],
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    scheduler = SequentialPlateauScheduler(
        optimizer,
        schedulers_configs=[
            {"mode": "min", "factor": 0.5, "patience": config["patience_lr"], "min_lr": 1e-4},
            {"mode": "min", "factor": 0.5, "patience": config["patience_lr"], "min_lr": 1e-6},
        ],
        initial_lr=config["lr"],
    )
    early_stopping = EarlyStopping(patience=config["patience_early"])
    checkpoint_psnr = ModelCheckpoint(best_path, mode="max")
    checkpoint_dice = ModelCheckpoint(model_name.replace(".pth", "_best_dice.pth"), mode="max")
    writer = SummaryWriter(os.path.join(output_dir, "runs", f"R{vol_bin}_A{aniso_bin}"))
    scaler = GradScaler()

    accum = config["accum_grad_batches"]
    for epoch in range(config["epochs"]):
        generator.train()
        optimizer.zero_grad()
        losses_recon, losses_seg, losses_total = [], [], []
        metrics_cc, metrics_psnr, metrics_dice = [], [], []
        data_gen_logger = DataGenerationLogger()

        loader_iter = enumerate(tqdm(train_loader, desc=f"R{vol_bin} A{aniso_bin} epoch {epoch}",
                                      total=config["steps_per_epoch"]))
        for i, (lr, hr, info) in loader_iter:
            if i >= config["steps_per_epoch"]:
                break
            lr, hr = lr.to(device), hr.to(device)
            data_gen_logger.add_sample(info)

            with autocast():
                pred = generator(lr)
                with torch.no_grad():
                    seg_true = segmentation_model(hr)
                    seg_pred = segmentation_model(pred)

                loss_recon = mixloss(pred, hr) / accum
                loss_seg = mdice_loss(seg_true, seg_pred) / accum

                if loss_recon > 10.0:
                    print(f"Unstable batch (recon loss {loss_recon.item():.2f}), skipping.")
                    optimizer.zero_grad()
                    continue

                loss = config["w_segmentation"] * loss_seg + loss_recon

            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(optimizer)
                adaptive_clip_grad(generator, optimizer.param_groups[0]["lr"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            with torch.no_grad():
                metrics_cc.append(correlation(hr, pred).item())
                metrics_psnr.append(calculate_psnr(hr, pred).item())
                metrics_dice.append(mdice(seg_true, seg_pred).item())
            losses_recon.append(loss_recon.item() * accum)
            losses_seg.append(loss_seg.item() * accum)
            losses_total.append(loss.item() * accum)

        avg_recon = sum(losses_recon) / len(losses_recon)
        avg_dice = sum(metrics_dice) / len(metrics_dice)
        avg_psnr = sum(metrics_psnr) / len(metrics_psnr)

        writer.add_scalar("Train/Reconstruction", avg_recon, epoch)
        writer.add_scalar("Train/Dice", avg_dice, epoch)
        writer.add_scalar("Train/PSNR", avg_psnr, epoch)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)
        data_gen_logger.log_to_tensorboard(writer, epoch)

        scheduler.step(sum(losses_total) / len(losses_total))
        checkpoint_psnr.save_checkpoint(generator, avg_psnr)
        checkpoint_dice.save_checkpoint(generator, avg_dice)

        early_stopping(avg_recon)
        if early_stopping.early_stop:
            print(f"Early stopping R{vol_bin} A{aniso_bin} at epoch {epoch}")
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Train the reconstruction + segmentation model")
    parser.add_argument("hr_json", help="JSON file with {'hr_files': [paths...]}")
    parser.add_argument("resolutions_csv", help="CSV of observed LR resolutions (Resolution_X/Y/Z columns)")
    parser.add_argument("seg_checkpoint", help="Path to the pretrained segmentation model weights")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--min-vol-bin", type=int, default=1)
    parser.add_argument("--max-vol-bin", type=int, default=7)
    parser.add_argument("--min-aniso-bin", type=int, default=1)
    parser.add_argument("--max-aniso-bin", type=int, default=3)
    parser.add_argument("--nf", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w-segmentation", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--accum-grad-batches", type=int, default=1)
    parser.add_argument("--patience-early", type=int, default=400)
    parser.add_argument("--patience-lr", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = configure_device()

    resolutions_df = load_csv_resolutions(args.resolutions_csv)
    segmentation_model = build_segmentation_model(args.seg_checkpoint, device)

    config = {"lr": args.lr, "nf": args.nf, "w_segmentation": args.w_segmentation,
              "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
              "accum_grad_batches": args.accum_grad_batches,
              "patience_early": args.patience_early, "patience_lr": args.patience_lr}

    vol_range = [b for b in bin_labels if args.min_vol_bin <= b <= args.max_vol_bin]
    aniso_range = sorted([b for b in anisotropy_labels if args.min_aniso_bin <= b <= args.max_aniso_bin],
                          reverse=True)  # isotropic first, most anisotropic last

    for vol_bin in vol_range:
        for aniso_bin in aniso_range:
            # Very high resolution + strong anisotropy is not physically meaningful; skip.
            if vol_bin == 1 and aniso_bin in (1, 2):
                continue
            if vol_bin == 2 and aniso_bin == 1:
                continue

            print(f"Training resolution cluster {vol_bin} ({bins[vol_bin - 1]}-{bins[vol_bin]} mm^3), "
                  f"anisotropy cluster {aniso_bin}")

            train_loader = make_dataloader(args.hr_json, resolutions_df, vol_bin, aniso_bin,
                                            batch_size=args.batch_size, num_workers=args.workers,
                                            device=device)
            train_cluster(config, train_loader, vol_bin, aniso_bin, segmentation_model, device, args.output_dir)
