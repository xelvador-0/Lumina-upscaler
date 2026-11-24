# Smart tiling
from __future__ import annotations

import gc
import threading
from typing import Callable, Optional

import cv2
import numpy as np


class CancelledError(Exception):
    pass


def _build_1d_feather(length: int, pad_start: int, pad_end: int) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32)

    if pad_start > 0:
        n = min(pad_start, length)
        t = np.linspace(0.0, np.pi, n + 2, dtype=np.float32)[1:-1]
        weights[:n] = np.minimum(weights[:n], 0.5 * (1.0 - np.cos(t)))

    if pad_end > 0:
        n = min(pad_end, length)
        t = np.linspace(0.0, np.pi, n + 2, dtype=np.float32)[1:-1]
        weights[-n:] = np.minimum(weights[-n:], 0.5 * (1.0 + np.cos(t)))

    return np.clip(weights, 1e-4, 1.0)


def _compute_tile_starts(total: int, tile_size: int, overlap: int) -> list[int]:
    if total <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, total - tile_size, stride))
    last = total - tile_size
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


class SmartTiler:
    def __init__(
        self,
        tile_size: int = 128,
        overlap: int = 16,
        scale: int = 2,
        low_vram_gc: bool = True,
    ) -> None:
        if tile_size < 32:
            raise ValueError("tile_size must be >= 32")
        if overlap < 0 or overlap >= tile_size // 2:
            raise ValueError("overlap must be >= 0 and < tile_size // 2")
        self.tile_size = int(tile_size)
        self.overlap = int(overlap)
        self.scale = int(scale)
        self.low_vram_gc = bool(low_vram_gc)

    def process(
        self,
        img_rgb: np.ndarray,
        tile_fn: Callable[[np.ndarray], np.ndarray],
        progress_cb: Optional[Callable[[int, int], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> np.ndarray:
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got {img_rgb.shape}")

        h, w, _ = img_rgb.shape
        s = self.scale

        if h <= self.tile_size and w <= self.tile_size:
            if stop_event is not None and stop_event.is_set():
                raise CancelledError("Stopped by user")
            out = tile_fn(np.ascontiguousarray(img_rgb, dtype=np.float32))
            if progress_cb:
                progress_cb(1, 1)
            return np.clip(out, 0.0, 1.0)

        pad_bottom = max(0, self.tile_size - h)
        pad_right = max(0, self.tile_size - w)
        if pad_bottom > 0 or pad_right > 0:
            work = cv2.copyMakeBorder(img_rgb, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101)
        else:
            work = img_rgb

        wh, ww, _ = work.shape
        ys = _compute_tile_starts(wh, self.tile_size, self.overlap)
        xs = _compute_tile_starts(ww, self.tile_size, self.overlap)

        accum = np.zeros((wh * s, ww * s, 3), dtype=np.float32)
        weight_map = np.zeros((wh * s, ww * s, 1), dtype=np.float32)

        total_tiles = len(ys) * len(xs)
        done = 0

        for yi, y0 in enumerate(ys):
            y1 = min(y0 + self.tile_size, wh)
            ov_top = (self.overlap * s) if yi > 0 else 0
            ov_bottom = (self.overlap * s) if yi < len(ys) - 1 else 0
            wy = _build_1d_feather((y1 - y0) * s, ov_top, ov_bottom)[:, None]

            for xi, x0 in enumerate(xs):
                if stop_event is not None and stop_event.is_set():
                    raise CancelledError("Stopped by user")

                x1 = min(x0 + self.tile_size, ww)
                ov_left = (self.overlap * s) if xi > 0 else 0
                ov_right = (self.overlap * s) if xi < len(xs) - 1 else 0
                wx = _build_1d_feather((x1 - x0) * s, ov_left, ov_right)[None, :]
                mask = (wy * wx)[:, :, None]

                tile_in = np.ascontiguousarray(work[y0:y1, x0:x1, :], dtype=np.float32)
                tile_out = tile_fn(tile_in)

                oy0, oy1 = y0 * s, y1 * s
                ox0, ox1 = x0 * s, x1 * s
                accum[oy0:oy1, ox0:ox1, :] += tile_out * mask
                weight_map[oy0:oy1, ox0:ox1, :] += mask

                done += 1
                if progress_cb:
                    progress_cb(done, total_tiles)
                if self.low_vram_gc and (done % 64 == 0):
                    gc.collect()

        merged = accum / np.maximum(weight_map, 1e-6)
        return np.clip(merged[: h * s, : w * s, :], 0.0, 1.0)
