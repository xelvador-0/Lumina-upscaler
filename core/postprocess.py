# Pre & post processing
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EnhancementParams:
    pre_blur: float = 0.0
    noise: float = 0.0
    compression: float = 0.0
    details: float = 0.0
    halo: float = 0.20
    sharpen: float = 0.0
    luma_blend: float = 0.0
    chroma_preserve: float = 0.0
    film_grain: float = 0.0
    anime_darken: float = 0.0

    def clamp(self) -> "EnhancementParams":
        return EnhancementParams(
            pre_blur=float(np.clip(self.pre_blur, -1.0, 1.0)),
            noise=float(np.clip(self.noise, 0.0, 1.0)),
            compression=float(np.clip(self.compression, 0.0, 1.0)),
            details=float(np.clip(self.details, 0.0, 1.0)),
            halo=float(np.clip(self.halo, 0.0, 1.0)),
            sharpen=float(np.clip(self.sharpen, 0.0, 1.0)),
            luma_blend=float(np.clip(self.luma_blend, 0.0, 0.5)),
            chroma_preserve=float(np.clip(self.chroma_preserve, 0.0, 1.0)),
            film_grain=float(np.clip(self.film_grain, 0.0, 0.08)),
            anime_darken=float(np.clip(self.anime_darken, 0.0, 1.0)),
        )


def preprocess_image(img_rgb: np.ndarray, params: EnhancementParams) -> np.ndarray:
    p = params.clamp()
    denoise_strength = max(p.noise, p.compression * 0.8)
    if denoise_strength <= 0.02 and abs(p.pre_blur) <= 0.02:
        return img_rgb

    out = img_rgb.astype(np.float32, copy=True)

    if denoise_strength > 0.02:
        u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        d = 5 if denoise_strength < 0.5 else 7
        filtered = cv2.bilateralFilter(u8, d=d, sigmaColor=10.0 + denoise_strength * 35.0, sigmaSpace=8.0 + denoise_strength * 18.0)
        alpha = min(0.60, denoise_strength * 0.70)
        out = cv2.addWeighted(out, 1.0 - alpha, filtered.astype(np.float32) / 255.0, alpha, 0.0)

    if p.pre_blur < -0.02:
        blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=abs(p.pre_blur) * 0.85)
        out = cv2.addWeighted(out, 1.0 - abs(p.pre_blur) * 0.4, blurred, abs(p.pre_blur) * 0.4, 0.0)
    elif p.pre_blur > 0.02:
        blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=0.8)
        out = np.clip(out + (out - blurred) * (p.pre_blur * 0.35), 0.0, 1.0)

    return np.clip(out, 0.0, 1.0)


def postprocess_image(
    orig_rgb: np.ndarray,
    ai_rgb: np.ndarray,
    params: EnhancementParams,
    grain_seed: int | None = None,
) -> np.ndarray:
    p = params.clamp()
    needs_orig = (p.luma_blend > 0.001) or (p.chroma_preserve > 0.01)
    needs_luma_fx = (
        (p.anime_darken > 0.02)
        or (p.details > 0.02)
        or (p.sharpen > 0.02)
        or (p.film_grain > 0.001)
    )

    if not needs_orig and not needs_luma_fx:
        return np.clip(ai_rgb, 0.0, 1.0)

    target_h, target_w = ai_rgb.shape[:2]
    ai_f = np.clip(ai_rgb.astype(np.float32, copy=False), 0.0, 1.0)
    ai_ycrcb = cv2.cvtColor(ai_f, cv2.COLOR_RGB2YCrCb)
    y_work = ai_ycrcb[:, :, 0].copy()

    orig_ycrcb = None
    if needs_orig:
        orig_up = cv2.resize(orig_rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC) if orig_rgb.shape[:2] != (target_h, target_w) else orig_rgb
        orig_ycrcb = cv2.cvtColor(np.clip(orig_up.astype(np.float32, copy=False), 0.0, 1.0), cv2.COLOR_RGB2YCrCb)
        if p.luma_blend > 0.001:
            y_work = y_work * (1.0 - p.luma_blend) + orig_ycrcb[:, :, 0] * p.luma_blend

    if p.anime_darken > 0.02:
        blur_line = cv2.GaussianBlur(y_work, (0, 0), sigmaX=1.2)
        y_work = y_work + np.minimum(y_work - blur_line, 0.0) * (p.anime_darken * 0.9)

    if p.details > 0.02:
        blur_fine = cv2.GaussianBlur(y_work, (0, 0), sigmaX=0.7)
        blur_med = cv2.GaussianBlur(y_work, (0, 0), sigmaX=1.8)
        y_work = y_work + (blur_fine - blur_med) * (p.details * 0.70)

    if p.sharpen > 0.02:
        blur_usm = cv2.GaussianBlur(y_work, (0, 0), sigmaX=1.1)
        high_freq = y_work - blur_usm
        mask = np.clip((np.abs(high_freq) - 0.006) / 0.025, 0.0, 1.0)
        boost = high_freq * mask * (p.sharpen * 1.15)
        if p.halo > 0.01:
            kernel = np.ones((3, 3), dtype=np.uint8)
            sharpened = y_work + boost
            clamped = np.clip(sharpened, cv2.erode(y_work, kernel), cv2.dilate(y_work, kernel))
            y_work = sharpened * (1.0 - p.halo * 0.75) + clamped * (p.halo * 0.75)
        else:
            y_work = y_work + boost

    if p.film_grain > 0.001:
        rng = np.random.default_rng(grain_seed)
        noise = cv2.GaussianBlur(rng.normal(0.0, p.film_grain, size=(target_h, target_w)).astype(np.float32), (3, 3), sigmaX=0.55)
        y_work = y_work + noise * np.clip(4.0 * y_work * (1.0 - y_work), 0.15, 1.0)

    y_work = np.clip(y_work, 0.0, 1.0)

    if orig_ycrcb is not None and p.chroma_preserve > 0.01:
        cr = ai_ycrcb[:, :, 1] * (1.0 - p.chroma_preserve) + orig_ycrcb[:, :, 1] * p.chroma_preserve
        cb = ai_ycrcb[:, :, 2] * (1.0 - p.chroma_preserve) + orig_ycrcb[:, :, 2] * p.chroma_preserve
    else:
        cr, cb = ai_ycrcb[:, :, 1], ai_ycrcb[:, :, 2]

    return np.clip(cv2.cvtColor(np.stack([y_work, cr, cb], axis=-1).astype(np.float32), cv2.COLOR_YCrCb2RGB), 0.0, 1.0)
