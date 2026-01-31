import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vsr.data.video_windows import VideoWindowDataset
from vsr.models.lite_vsr import LiteVSR
from vsr.utils.metrics import psnr_ssim

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--val_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--amp", action="store_true")
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = VideoWindowDataset(args.val_root, scale=args.scale, window=args.window)
    ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = LiteVSR(scale=ckpt.get("scale", args.scale), window=ckpt.get("window", args.window)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    psnrs, ssims = [], []
    for lr, hr in tqdm(ld, desc="eval"):
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            pred = model(lr)
        pred_np = pred[0].permute(1, 2, 0).float().cpu().numpy()
        hr_np = hr[0].permute(1, 2, 0).float().cpu().numpy()
        p, s = psnr_ssim(pred_np, hr_np)
        psnrs.append(p); ssims.append(s)

    print(f"PSNR: {np.mean(psnrs):.2f} dB  |  SSIM: {np.mean(ssims):.4f}")

if __name__ == "__main__":
    main()
