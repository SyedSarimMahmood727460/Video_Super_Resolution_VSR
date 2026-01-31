import argparse
import json
import os
import time
from typing import List
import numpy as np
import torch
import cv2

from vsr.models.lite_vsr import LiteVSR
from vsr.utils.io import read_video, write_video, read_image, write_image
from vsr.utils.timer import LatencyTimer
from vsr.utils.profiling import nvtx_range
def get_video_fps(video_path: str) -> float:
    """Detect the FPS of input video"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="video path or frames folder")
    p.add_argument("--output", required=True, help="output mp4 or output frames folder")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--precision", choices=["fp32", "amp"], default="amp")
    p.add_argument("--channels_last", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--measure", type=int, default=300)
    p.add_argument("--save_metrics", default="")
    p.add_argument("--fps", type=float, default=0.0, help="Output FPS (0 = auto-detect from input)")
    p.add_argument("--target_height", type=int, default=0, help="Resize output to this height after SR (preserve aspect). 0 = no resize")
    p.add_argument("--crf", type=int, default=16, help="x264 CRF (lower is higher quality, 0-51)")
    p.add_argument("--preset", type=str, default="slow", help="x264 preset (ultrafast..veryslow)")
    p.add_argument("--tune", type=str, default="film", help="x264 tune (film, animation, grain, etc.)")
    p.add_argument("--pix_fmt", type=str, default="yuv444p", help="Pixel format (e.g., yuv444p, yuv420p)")
    p.add_argument("--colorspace", type=str, default="bt709", help="Output colorspace metadata")
    p.add_argument("--color_primaries", type=str, default="bt709", help="Output color primaries metadata")
    p.add_argument("--color_trc", type=str, default="bt709", help="Output transfer metadata")
    p.add_argument("--color_range", type=str, default="tv", help="Output color range (tv or pc)")
    p.add_argument("--report_path", default="", help="Write a detailed performance report to this path")
    p.add_argument("--kernel_report_path", default="", help="Write a kernel performance report to this path")
    p.add_argument("--kernel_profile", action="store_true", help="Profile a single measured frame for kernel stats")
    p.add_argument("--kernel_profile_steps", type=int, default=1, help="Number of frames to profile for kernel stats")
    return p.parse_args()

def is_video(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}

def list_frames(folder: str) -> List[str]:
    exts = ("png", "jpg", "jpeg", "bmp", "webp")
    files = []
    for e in exts:
        files += [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(e)]
    return files

def to_tensor01(rgb_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb_u8).permute(2, 0, 1).float().div(255.0)

def to_uint8(rgb01: torch.Tensor) -> np.ndarray:
    x = rgb01.clamp(0, 1).mul(255.0).byte().permute(1, 2, 0).cpu().numpy()
    return x

@torch.no_grad()
def main():
    args = parse()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (args.precision == "amp") and (device == "cuda")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = LiteVSR(scale=ckpt.get("scale", args.scale), window=ckpt.get("window", args.window)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    if args.channels_last and device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if args.compile and device == "cuda":
        model = torch.compile(model, mode="max-autotune")

    torch.backends.cudnn.benchmark = True

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {mem_gb:.1f} GB")

    if is_video(args.input):
        frames = list(read_video(args.input))
        # Auto-detect FPS if not specified
        if args.fps <= 0:
            args.fps = get_video_fps(args.input)
            print(f"Auto-detected input FPS: {args.fps:.2f}")
    else:
        files = list_frames(args.input)
        frames = [read_image(p) for p in files]
        if args.fps <= 0:
            args.fps = 30.0  # Default for image sequences

    if len(frames) == 0:
        raise RuntimeError("No frames found")

    input_h = int(frames[0].shape[0])
    if input_h not in (360, 480, 720):
        raise RuntimeError("Input height must be exactly 360, 480, or 720.")
    if args.target_height and args.target_height > 0:
        if input_h == 360 and args.target_height not in (480, 720, 1080):
            raise RuntimeError("For 360p input, target must be 480, 720, or 1080.")
        if input_h == 480 and args.target_height not in (720,):
            raise RuntimeError("For 480p input, target must be 720.")
        if input_h == 720 and args.target_height not in (1080,):
            raise RuntimeError("For 720p input, target must be 1080.")

    duration_sec = len(frames) / args.fps
    print(f"Processing {len(frames)} frames @ {args.fps:.2f} fps ({duration_sec:.1f}s)...")

    assert args.window % 2 == 1
    half = args.window // 2
    padded = [frames[0]] * half + frames + [frames[-1]] * half

    out_frames = []
    timer = LatencyTimer()
    stage_times = {"prepare_ms": [], "model_ms": [], "post_ms": [], "total_ms": []}
    kernel_profiled = 0
    kernel_report_text = ""

    start_evt = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    end_evt = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

    total = len(frames)
    warmup_n = min(args.warmup, total) if args.benchmark else 0
    measure_n = min(args.measure, total - warmup_n) if args.benchmark else total
    end_i = warmup_n + measure_n

    for i in range(total):
        if i % 50 == 0:
            print(f"  Frame {i+1}/{total}")
        total_start = time.perf_counter()
        with nvtx_range("prepare_window"):
            window = padded[i : i + args.window]
            lr = torch.stack([to_tensor01(f) for f in window], dim=0).unsqueeze(0)

            if device == "cuda":
                lr = lr.pin_memory()
                lr = lr.to(device, non_blocking=True)
                if args.channels_last:
                    # lr is (B,T,C,H,W). channels_last works only for rank-4 (N,C,H,W).
                    lr = lr.contiguous()  # keep default contiguous for 5D
        if device == "cuda":
            torch.cuda.synchronize()
        prepare_ms = (time.perf_counter() - total_start) * 1000.0


        with nvtx_range("model_forward"):
            model_start = time.perf_counter()
            if device == "cuda":
                start_evt.record()
            do_profile = (
                args.kernel_profile
                and device == "cuda"
                and kernel_profiled < max(0, int(args.kernel_profile_steps))
                and i >= warmup_n
            )
            if do_profile:
                from torch.profiler import ProfilerActivity, profile

                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                ) as prof:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                        sr = model(lr)
                kernel_profiled += 1
                if args.kernel_report_path and not kernel_report_text:
                    kernel_report_text = prof.key_averages().table(
                        sort_by="cuda_time_total",
                        row_limit=40,
                    )
            else:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    sr = model(lr)
            if device == "cuda":
                end_evt.record()
                end_evt.synchronize()
            model_ms = (time.perf_counter() - model_start) * 1000.0

        with nvtx_range("postprocess"):
            post_start = time.perf_counter()
            if device == "cuda":
                t_ms = float(start_evt.elapsed_time(end_evt))
            else:
                t_ms = 0.0

            if args.benchmark and (warmup_n <= i < end_i):
                timer.add_ms(t_ms)

            sr_u8 = to_uint8(sr[0])
            
            # Optional target resize after SR
            if args.target_height and args.target_height > 0:
                h, w = sr_u8.shape[:2]
                if h != args.target_height:
                    scale = args.target_height / float(h)
                    tw = max(1, int(round(w * scale)))
                    # Use higher-quality resize for upscales; area for downscales.
                    interp = cv2.INTER_LANCZOS4 if args.target_height > h else cv2.INTER_AREA
                    sr_u8 = cv2.resize(sr_u8, (tw, args.target_height), interpolation=interp)
            
            out_frames.append(sr_u8)
            
            # Clear GPU cache periodically for memory-constrained GPUs
            if device == "cuda" and i % 10 == 0:
                torch.cuda.empty_cache()
        post_ms = (time.perf_counter() - post_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        if warmup_n <= i < end_i:
            stage_times["prepare_ms"].append(prepare_ms)
            stage_times["model_ms"].append(model_ms)
            stage_times["post_ms"].append(post_ms)
            stage_times["total_ms"].append(total_ms)
    print(f"Processed {len(out_frames)} frames")
    

    if os.path.splitext(args.output)[1].lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        print(f"Writing video @ {args.fps:.2f} fps...")
        write_video(
            args.output,
            out_frames,
            fps=args.fps,
            crf=args.crf,
            preset=args.preset,
            tune=args.tune,
            pixelformat=args.pix_fmt,
            colorspace=args.colorspace,
            color_primaries=args.color_primaries,
            color_trc=args.color_trc,
            color_range=args.color_range,
        )
        print(f" Video saved: {args.output}")
    else:
        os.makedirs(args.output, exist_ok=True)
        for i, fr in enumerate(out_frames):
            write_image(os.path.join(args.output, f"{i:06d}.png"), fr)

    report = {
        "device": device,
        "amp": amp,
        "channels_last": bool(args.channels_last),
        "compile": bool(args.compile),
        "frames_total": total,
        "fps": args.fps,
        "duration_sec": total / args.fps,
        "crf": args.crf,
        "preset": args.preset,
        "tune": args.tune,
        "bench_warmup": warmup_n,
        "bench_measure": measure_n,
        "stage_times_ms": {
            "prepare_mean": float(np.mean(stage_times["prepare_ms"])) if stage_times["prepare_ms"] else 0.0,
            "model_mean": float(np.mean(stage_times["model_ms"])) if stage_times["model_ms"] else 0.0,
            "post_mean": float(np.mean(stage_times["post_ms"])) if stage_times["post_ms"] else 0.0,
            "total_mean": float(np.mean(stage_times["total_ms"])) if stage_times["total_ms"] else 0.0,
        },
    }
    if args.benchmark and timer.times_ms:
        s = timer.summary()
        report.update({
            "lat_mean_ms": s.mean_ms,
            "lat_p50_ms": s.p50_ms,
            "lat_p90_ms": s.p90_ms,
            "lat_p99_ms": s.p99_ms,
            "fps_est": s.fps,
        })

    print(json.dumps(report, indent=2))
    if args.save_metrics:
        with open(args.save_metrics, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    if args.report_path:
        total_mean = report["stage_times_ms"]["total_mean"]
        stage_means = {
            "prepare": report["stage_times_ms"]["prepare_mean"],
            "model": report["stage_times_ms"]["model_mean"],
            "postprocess": report["stage_times_ms"]["post_mean"],
        }
        bottleneck = max(stage_means, key=stage_means.get) if total_mean > 0 else "unknown"
        with open(args.report_path, "w", encoding="utf-8") as f:
            f.write("LiteVSR Performance Report\n")
            f.write("==========================\n")
            f.write(f"Frames: {total}\n")
            f.write(f"FPS (input/output): {args.fps:.2f}\n")
            f.write(f"Duration (sec): {total / args.fps:.2f}\n")
            f.write(f"Scale: {args.scale}x, Window: {args.window}\n")
            f.write(f"Target height: {args.target_height}\n")
            f.write(f"Precision: {args.precision}\n")
            f.write(f"Encoder: crf={args.crf}, preset={args.preset}, tune={args.tune}\n\n")
            f.write("Stage timing (ms, mean over measured frames)\n")
            for k, v in stage_means.items():
                f.write(f"  {k}: {v:.2f}\n")
            f.write(f"  total: {total_mean:.2f}\n\n")
            if total_mean > 0:
                f.write("Bottleneck analysis\n")
                for k, v in stage_means.items():
                    pct = (v / total_mean) * 100.0 if total_mean > 0 else 0.0
                    f.write(f"  {k}: {pct:.1f}%\n")
                f.write(f"  bottleneck: {bottleneck}\n")
            if device == "cuda":
                try:
                    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    f.write(f"\nMax GPU memory allocated: {max_mem:.2f} GB\n")
                except Exception:
                    pass
    if args.kernel_report_path and kernel_report_text:
        with open(args.kernel_report_path, "w", encoding="utf-8") as f:
            f.write("Kernel Performance Report\n")
            f.write("=========================\n")
            f.write(kernel_report_text)

if __name__ == "__main__":
    main()


