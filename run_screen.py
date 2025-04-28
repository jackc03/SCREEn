#!/usr/bin/env python3
"""
run_screen.py
=============

Train or evaluate the Three-Frame super-resolution model defined in `screen.py`
using the DAVIS-2017 720 p / 1080 p folders.

Additions (timestamped logging file):
  • --log_file is still supported, but whatever you pass (or the default
    "training_psnr.log") will have "_{DD-Mon_HH-MM}" appended automatically.
"""

from __future__ import annotations
import argparse, math, logging
from datetime import datetime
from pathlib import Path
import sys, os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --------------------------------------------------------------------------- #
#  Local imports
# --------------------------------------------------------------------------- #
this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)

from screen import SCREEn
from triplet_dataset import TripletDataset

# --------------------------- helper: timestamped name ----------------------- #
def add_timestamp_to_filename(fname: str | Path) -> str:
    """Return <stem>_DD-Mon_HH-MM<suffix> in the same directory."""
    p = Path(fname)
    ts = datetime.now().strftime("%d-%b_%H-%M")          # e.g. 28-Apr_14-37
    return str(p.with_name(f"{p.stem}_{ts}{p.suffix or '.log'}"))

# --------------------------- (rest is unchanged) ---------------------------- #
_Y_COEFFS = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)  # BT.601


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or x.size(1) != 3:
        raise ValueError("rgb_to_y expects (B,3,H,W) tensor in [0,1]")
    c = _Y_COEFFS.to(x.device, x.dtype)
    return (x * c).sum(dim=1, keepdim=True)


def psnr(pred: torch.Tensor, target: torch.Tensor, shave: int = 4) -> float:
    pred_y, target_y = rgb_to_y(pred), rgb_to_y(target)
    if shave:
        pred_y = pred_y[..., shave:-shave, shave:-shave]
        target_y = target_y[..., shave:-shave, shave:-shave]
    mse = F.mse_loss(pred_y, target_y, reduction='mean')
    return float('inf') if mse == 0 else 10.0 * math.log10(1.0 / mse.item())


@torch.no_grad()
def validate(model, loader, device) -> float:
    model.eval()
    total_psnr, n = 0.0, 0
    for prev720, cur720, next720, hr1080 in loader:
        prev720, cur720, next720, hr1080 = [
            t.to(device, non_blocking=True)
            for t in (prev720, cur720, next720, hr1080)
        ]
        sr = model(prev720, cur720, next720).clamp(0, 1)
        total_psnr += psnr(sr, hr1080) * prev720.size(0)
        n += prev720.size(0)
    return total_psnr / n


def main() -> None:
    ap = argparse.ArgumentParser("Train / test Three-Frame SR model.")
    ap.add_argument("--data_root", default="datasets", type=str)
    ap.add_argument("--mode", choices=["train", "test"], required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weights", type=str)
    ap.add_argument("--save_dir", default="ckpt", type=str)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_file", default="training_psnr.log", type=str,
                    help="Base name for the log file (timestamp is auto-added)")
    args = ap.parse_args()

    # ------------------------------- logging -------------------------------- #
    log_path = add_timestamp_to_filename(args.log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a"),
                  logging.StreamHandler(sys.stdout)],
    )
    logging.info("Logging to %s", log_path)

    # (… everything else exactly the same as the previous version …)
    # ----------------------------------------------------------------------- #
    root = Path(args.data_root)
    lr_dir = root / "davis_trainval_2017_720p" / "DAVIS" / "JPEGImages" / "Full-Resolution"
    hr_dir = root / "davis_trainval_2017_1080p" / "DAVIS" / "JPEGImages" / "Full-Resolution"

    if not (lr_dir.exists() and hr_dir.exists()):
        raise SystemExit(f"Could not find DAVIS folders under {root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = SCREEn().to(device)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device))
        logging.info("Loaded weights from %s", args.weights)

    if args.mode == "test":
        test_ld = DataLoader(
            TripletDataset(lr_dir, hr_dir, "test"),
            batch_size=1, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        avg_psnr = validate(model, test_ld, device)
        logging.info("Test-set PSNR: %.2f dB", avg_psnr)
        print(f"Test-set PSNR: {avg_psnr:.2f} dB")
        return

    train_ld = DataLoader(
        TripletDataset(lr_dir, hr_dir, "train"),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )
    val_ld = DataLoader(
        TripletDataset(lr_dir, hr_dir, "val"),
        batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for b, (prev720, cur720, next720, hr1080) in enumerate(train_ld, 1):
            prev720, cur720, next720, hr1080 = [
                t.to(device, non_blocking=True)
                for t in (prev720, cur720, next720, hr1080)
            ]
            opt.zero_grad(set_to_none=True)
            sr = model(prev720, cur720, next720)
            loss = F.l1_loss(sr, hr1080)
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * prev720.size(0)

            batch_psnr = psnr(sr.clamp(0, 1), hr1080)
            logging.info("Epoch %d  Batch %d/%d  PSNR %.2f dB",
                         epoch, b, len(train_ld), batch_psnr)

        epoch_loss /= len(train_ld.dataset)
        val_psnr = validate(model, val_ld, device)
        logging.info("Epoch %d  VALIDATE  PSNR %.2f dB", epoch, val_psnr)
        print(f"[{epoch:03d}/{args.epochs}]  L1={epoch_loss:.4f}  PSNR={val_psnr:.2f} dB")

        torch.save(model.state_dict(),
                   Path(args.save_dir) / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    main()
