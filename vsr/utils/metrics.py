import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0, 0, 255).astype(np.uint8)

def psnr_ssim(pred01: np.ndarray, gt01: np.ndarray) -> tuple[float, float]:
    pred = to_uint8(pred01)
    gt = to_uint8(gt01)
    psnr = float(peak_signal_noise_ratio(gt, pred, data_range=255))
    ssim = float(structural_similarity(gt, pred, channel_axis=2, data_range=255))
    return psnr, ssim
