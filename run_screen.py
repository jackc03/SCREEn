#!/usr/bin/env python3
"""
run_screen.py  —  train / test / demo for SCREEn VSR network
"""

from __future__ import annotations
import argparse, logging, math, os, sys, random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.utils as vutils         # demo helper

# ───────── distributed init (robust) ───────────────────────────────────────
if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    import torch.distributed as dist
    dist.init_process_group("nccl")

    LOCAL_RANK  = int(os.environ["LOCAL_RANK"])     # 0‥(nproc-1)
    NUM_VISIBLE = torch.cuda.device_count()         # how many this proc sees

    if LOCAL_RANK >= NUM_VISIBLE:                   # wrap if too few GPUs
        print(f"[rank{LOCAL_RANK}] only {NUM_VISIBLE} GPU(s) visible; "
              "wrapping index", file=sys.stderr, flush=True)
        LOCAL_RANK = LOCAL_RANK % max(1, NUM_VISIBLE)

    torch.cuda.set_device(LOCAL_RANK)
    WORLD_SIZE = dist.get_world_size()
else:   # single-GPU / CPU
    LOCAL_RANK = 0
    WORLD_SIZE = 1

# ───────── local imports ───────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from screen import SCREEn
from triplet_dataset import TripletDataset

# ───────── helpers ─────────────────────────────────────────────────────────
def add_timestamp(fname: str | Path) -> str:
    ts = datetime.now().strftime("%d-%b_%H-%M")
    p  = Path(fname)
    return str(p.with_name(f"{p.stem}_{ts}{p.suffix or '.log'}"))

_YCOEF = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


def rgb2y(t: torch.Tensor) -> torch.Tensor:
    return (t * _YCOEF.to(t.device, t.dtype)).sum(1, keepdim=True)


def psnr(pred: torch.Tensor, tgt: torch.Tensor, shave: int = 4) -> float:
    pred, tgt = rgb2y(pred), rgb2y(tgt)
    if shave:
        pred = pred[..., shave:-shave, shave:-shave]
        tgt  = tgt [..., shave:-shave, shave:-shave]
    mse = F.mse_loss(pred, tgt, reduction="mean")
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse.item())


@torch.no_grad()
def validate(model, loader, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for prev, cur, nxt, hr in loader:
        prev, cur, nxt, hr = [t.to(device, non_blocking=True)
                              for t in (prev, cur, nxt, hr)]
        sr = model(prev, cur, nxt).clamp(0, 1)
        tot += psnr(sr, hr) * prev.size(0)
        n   += prev.size(0)
    return tot / n


# ───────── main ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="datasets")
    ap.add_argument("--mode", choices=["train", "test", "demo"], required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)   # per GPU
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weights")
    ap.add_argument("--save_dir", default="ckpt")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_file", default="training_psnr.log")
    ap.add_argument("--demo_samples", type=int, default=3)
    args = ap.parse_args()

    # ---------- logging ----------------------------------------------------
    if LOCAL_RANK == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(add_timestamp(args.log_file), "a"),
                      logging.StreamHandler(sys.stdout)]
        )

    # ---------- paths ------------------------------------------------------
    root   = Path(args.data_root)
    lr_dir = root / "davis_trainval_2017_480p"  / "DAVIS" / "JPEGImages" / "Full-Resolution"
    hr_dir = root / "davis_trainval_2017_1080p" / "DAVIS" / "JPEGImages" / "Full-Resolution"
    if not (lr_dir.exists() and hr_dir.exists()):
        raise SystemExit("DAVIS folders not found")

    # ---------- model ------------------------------------------------------
    device = torch.device("cuda", LOCAL_RANK) if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.benchmark = True
    model = SCREEn().to(device)
    if args.weights:
        map_loc = {"cuda:0": f"cuda:{LOCAL_RANK}"}
        model.load_state_dict(torch.load(args.weights, map_location=map_loc))
        if LOCAL_RANK == 0:
            logging.info("Loaded weights from %s", args.weights)

    if WORLD_SIZE > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # ---------- datasets & loaders (unchanged) -----------------------------
    train_set = TripletDataset(lr_dir, hr_dir, "train")
    val_set   = TripletDataset(lr_dir, hr_dir, "val")
    test_set  = TripletDataset(lr_dir, hr_dir, "test")

    train_sam = DistributedSampler(train_set, shuffle=True)  if WORLD_SIZE > 1 else None
    val_sam   = DistributedSampler(val_set,   shuffle=False) if WORLD_SIZE > 1 else None
    test_sam  = DistributedSampler(test_set,  shuffle=False) if WORLD_SIZE > 1 else None

    train_ld = DataLoader(train_set, args.batch_size, sampler=train_sam,
                          shuffle=train_sam is None,
                          num_workers=args.num_workers, pin_memory=True)
    val_ld   = DataLoader(val_set, 1, sampler=val_sam, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    test_ld  = DataLoader(test_set, 1, sampler=test_sam, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # ---------- test mode --------------------------------------------------
    if args.mode == "test":
        psnr_val = validate(model, test_ld, device)
        if LOCAL_RANK == 0:
            print(f"Test PSNR: {psnr_val:.2f} dB")
        return

    # ---------- demo mode  (visual comparison)  --------------------------- #
    if args.mode == "demo":
        if LOCAL_RANK == 0:
            out_dir = Path("demo_out")
            out_dir.mkdir(exist_ok=True)
            model.eval()
            samples = random.sample(range(len(test_set)), args.demo_samples)
            logging.info("Demo samples: %s", samples)

            PATCH = 100
            H, W  = 1080, 1920          # HR frame size

            for idx in samples:
                prev, cur, nxt, hr = test_set[idx]

                with torch.no_grad():
                    sr = model(prev.unsqueeze(0).to(device),
                            cur .unsqueeze(0).to(device),
                            nxt .unsqueeze(0).to(device)).cpu()[0].clamp(0, 1)
                bilinear = F.interpolate(cur.unsqueeze(0), (H, W),
                                        mode="bilinear", align_corners=False)[0]

                rows = []               # will hold 4 concatenated rows
                for p in range(4):
                    y = random.randint(0, H - PATCH)
                    x = random.randint(0, W - PATCH)

                    crop_bi  = bilinear[:, y:y+PATCH, x:x+PATCH]
                    crop_sr  = sr      [:, y:y+PATCH, x:x+PATCH]
                    crop_gt  = hr      [:, y:y+PATCH, x:x+PATCH]

                    # row = [bilinear | network | GT]
                    row = torch.cat([crop_bi, crop_sr, crop_gt], dim=2)
                    rows.append(row)

                    # per-patch PSNR
                    b_psnr = psnr(crop_bi.unsqueeze(0), crop_gt.unsqueeze(0), shave=0)
                    s_psnr = psnr(crop_sr.unsqueeze(0), crop_gt.unsqueeze(0), shave=0)
                    logging.info("Sample %d patch %d  bil %.2f dB  net %.2f dB",
                                idx, p, b_psnr, s_psnr)

                grid = torch.cat(rows, dim=1)          # stack rows vertically
                vutils.save_image(grid, out_dir / f"demo_{idx:04d}.png")
                print(f"Saved demo_{idx:04d}.png")

        return

    # ---------- optimiser --------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ======================  TRAIN  ========================================
    for epoch in range(1, args.epochs + 1):
        if train_sam:
            train_sam.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0

        for b, (prev, cur, nxt, hr) in enumerate(train_ld, 1):
            prev, cur, nxt, hr = [t.to(device, non_blocking=True) for t in (prev, cur, nxt, hr)]
            opt.zero_grad(set_to_none=True)
            sr   = model(prev, cur, nxt)
            loss = F.l1_loss(sr, hr)
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * prev.size(0)

            # batch-level logging (rank-0)
            if LOCAL_RANK == 0:
                batch_psnr = psnr(sr.detach().clamp(0,1), hr)
                logging.info("Ep %d  Bt %d/%d  L1 %.4f  PSNR %.2f",
                             epoch, b, len(train_ld), loss.item(), batch_psnr)

        if WORLD_SIZE > 1:
            loss_tensor = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(loss_tensor)
            epoch_loss = loss_tensor.item()
        epoch_loss /= len(train_set)

        psnr_val = validate(model, val_ld, device)
        if LOCAL_RANK == 0:
            logging.info("Epoch %d | L1 %.4f | PSNR %.2f", epoch, epoch_loss, psnr_val)
            torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                       Path(args.save_dir) / f"epoch_{epoch:03d}.pt")

    if LOCAL_RANK == 0:
        logging.info("Finished training.")


if __name__ == "__main__":
    main()
