import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from vsr.data.video_windows import VideoWindowDataset
from vsr.models.lite_vsr import LiteVSR

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--train_root", required=True)
    p.add_argument("--val_root", required=True)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()

@torch.no_grad()
def eval_psnr(model, loader, device, amp):
    model.eval()
    psnrs = []
    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            pred = model(lr)
        mse = F.mse_loss(pred, hr, reduction="mean").item()
        psnr = 10.0 * torch.log10(torch.tensor(1.0) / torch.tensor(mse)).item() if mse > 0 else 99.0
        psnrs.append(psnr)
    return sum(psnrs) / len(psnrs) if psnrs else 0.0

def main():
    args = parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = VideoWindowDataset(args.train_root, scale=args.scale, window=args.window)
    val_ds = VideoWindowDataset(args.val_root, scale=args.scale, window=args.window)

    train_ld = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0
    )
    val_ld = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True
    )

    model = LiteVSR(scale=args.scale, window=args.window).to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best = -1.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_ld, desc=f"epoch {epoch}/{args.epochs}")
        for lr, hr in pbar:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                pred = model(lr)
                loss = F.l1_loss(pred, hr)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=float(loss.item()))

        psnr = eval_psnr(model, val_ld, device, args.amp)
        if psnr > best:
            best = psnr
            torch.save({"model": model.state_dict(), "scale": args.scale, "window": args.window}, args.out)
        print(f"[epoch {epoch}] val_psnr={psnr:.2f} best={best:.2f}")

if __name__ == "__main__":
    main()
