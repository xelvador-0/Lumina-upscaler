# Temporal stabilizer
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class TemporalStabilizer:
    def __init__(
        self,
        strength: float = 0.25,
        scene_cut_threshold: float = 0.22,
        flow_proxy_width: int = 480,
    ) -> None:
        self.strength = float(np.clip(strength, 0.0, 0.65))
        self.scene_cut_threshold = float(scene_cut_threshold)
        self.flow_proxy_width = int(flow_proxy_width)

        self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self._prev_proxy_gray: Optional[np.ndarray] = None
        self._prev_enhanced: Optional[np.ndarray] = None
        self._grid_x: Optional[np.ndarray] = None
        self._grid_y: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_proxy_gray = None
        self._prev_enhanced = None

    def _to_proxy_gray(self, rgb_f32: np.ndarray) -> np.ndarray:
        h, w = rgb_f32.shape[:2]
        if w > self.flow_proxy_width:
            ph = max(16, int(round(h * (self.flow_proxy_width / float(w)))))
            small = cv2.resize(rgb_f32, (self.flow_proxy_width, ph), interpolation=cv2.INTER_AREA)
        else:
            small = rgb_f32
        return cv2.cvtColor(np.clip(small * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    def _ensure_grid(self, h: int, w: int) -> None:
        if self._grid_x is None or self._grid_x.shape != (h, w):
            self._grid_x, self._grid_y = np.meshgrid(
                np.arange(w, dtype=np.float32),
                np.arange(h, dtype=np.float32),
            )

    def stabilize(self, orig_rgb: np.ndarray, enhanced_rgb: np.ndarray) -> np.ndarray:
        if self.strength <= 0.005:
            return enhanced_rgb

        curr_gray = self._to_proxy_gray(orig_rgb)
        if (
            self._prev_proxy_gray is None
            or self._prev_enhanced is None
            or self._prev_proxy_gray.shape != curr_gray.shape
            or self._prev_enhanced.shape != enhanced_rgb.shape
        ):
            self._prev_proxy_gray = curr_gray
            self._prev_enhanced = enhanced_rgb.copy()
            return enhanced_rgb

        mad = float(np.mean(np.abs(curr_gray.astype(np.float32) - self._prev_proxy_gray.astype(np.float32)))) / 255.0
        if mad > self.scene_cut_threshold:
            self._prev_proxy_gray = curr_gray
            self._prev_enhanced = enhanced_rgb.copy()
            return enhanced_rgb

        flow_small = self._dis.calc(curr_gray, self._prev_proxy_gray, None)
        out_h, out_w = enhanced_rgb.shape[:2]
        ph, pw = curr_gray.shape[:2]

        flow_full = cv2.resize(flow_small, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        flow_full[:, :, 0] *= out_w / float(pw)
        flow_full[:, :, 1] *= out_h / float(ph)

        self._ensure_grid(out_h, out_w)
        assert self._grid_x is not None and self._grid_y is not None
        warped_prev = cv2.remap(
            self._prev_enhanced,
            self._grid_x + flow_full[:, :, 0],
            self._grid_y + flow_full[:, :, 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        diff = np.mean(np.abs(enhanced_rgb - warped_prev), axis=2, keepdims=True)
        alpha = (self.strength * np.clip(1.0 - (diff - 0.02) / 0.10, 0.0, 1.0)).astype(np.float32)
        blended = np.clip(enhanced_rgb * (1.0 - alpha) + warped_prev * alpha, 0.0, 1.0)

        self._prev_proxy_gray = curr_gray
        self._prev_enhanced = blended.copy()
        return blended
