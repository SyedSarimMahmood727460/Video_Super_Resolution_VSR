import os
import cv2
import numpy as np
import imageio.v3 as iio

def read_image(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def write_image(path: str, rgb: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

def read_video(path: str):
    for frame in iio.imiter(path):
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        yield frame.astype(np.uint8)

def write_video(
    path: str,
    frames_rgb_uint8,
    fps: float = 30.0,
    crf: int = 16,
    preset: str = "slow",
    tune: str = "film",
    pixelformat: str = "yuv444p",
    colorspace: str = "bt709",
    color_primaries: str = "bt709",
    color_trc: str = "bt709",
    color_range: str = "tv",
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Higher-quality defaults (yuv444p preserves chroma better than yuv420p)
    # Note: yuv444p can be less compatible with some players.
    iio.imwrite(
        path,
        list(frames_rgb_uint8),
        fps=fps,
        codec="libx264",
        pixelformat=pixelformat,
        ffmpeg_params=[
            "-crf", str(crf),
            "-preset", preset,
            "-tune", tune,
            "-colorspace", colorspace,
            "-color_primaries", color_primaries,
            "-color_trc", color_trc,
            "-color_range", color_range,
            "-x264-params",
            f"colorprim={color_primaries}:transfer={color_trc}:colormatrix={colorspace}:fullrange=off",
        ],
    )
