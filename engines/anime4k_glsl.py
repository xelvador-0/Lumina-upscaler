# Anime4K GLSL engine
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_MAT4_OFFSET_RE = re.compile(r"mat4\(([^)]+)\)\s*\*\s*(go_[01])\(([-\d.]+)\s*,\s*([-\d.]+)\)")
_MAT4_DENSE_RE = re.compile(r"mat4\(([^)]+)\)\s*\*\s*(g_\d+)")
_BIAS_RE = re.compile(r"result\s*\+=\s*vec4\(([^)]+)\)\s*;")


def _parse_floats(csv_str: str) -> np.ndarray:
    return np.fromstring(csv_str, sep=",", dtype=np.float32)


@dataclass
class Conv3x3Pass:
    desc: str
    save_name: str
    bind_name: str
    uses_crelu: bool
    kernels: Dict[Tuple[int, int, int], np.ndarray] = field(default_factory=dict)
    bias: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))


@dataclass
class Dense1x1Pass:
    desc: str
    save_name: str
    binds: List[str]
    weights: Dict[int, np.ndarray] = field(default_factory=dict)
    bias: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    adds_main: bool = False


class Anime4KGLSLModel:
    def __init__(self, shader_path: str | Path) -> None:
        self.shader_path = Path(shader_path)
        if not self.shader_path.is_file():
            raise FileNotFoundError(f"Shader not found: {self.shader_path}")

        self.conv_passes: List[Conv3x3Pass] = []
        self.dense_pass: Optional[Dense1x1Pass] = None
        self.has_depth_to_space: bool = False
        self.scale: int = 1
        self._parse(self.shader_path.read_text(encoding="utf-8", errors="ignore"))

    def _parse(self, text: str) -> None:
        for blk in re.split(r"(?=//!DESC\s+)", text):
            if not blk.strip().startswith("//!DESC"):
                continue

            desc_m = re.search(r"//!DESC\s+(.+)", blk)
            desc = desc_m.group(1).strip() if desc_m else "Unknown"
            save_m = re.search(r"//!SAVE\s+(\S+)", blk)
            save_name = save_m.group(1).strip() if save_m else "MAIN"
            binds = re.findall(r"//!BIND\s+(\S+)", blk)

            if "Depth-to-Space" in desc or "i0.y * 2 + i0.x" in blk:
                self.has_depth_to_space = True
                self.scale = 2
                continue

            dense_matches = _MAT4_DENSE_RE.findall(blk)
            if dense_matches:
                weights = {int(g.split("_")[1]): _parse_floats(m).reshape(4, 4) for m, g in dense_matches}
                bias_m = _BIAS_RE.search(blk)
                bias = _parse_floats(bias_m.group(1)) if bias_m else np.zeros(4, dtype=np.float32)
                self.dense_pass = Dense1x1Pass(
                    desc=desc,
                    save_name=save_name,
                    binds=[b for b in binds if b != "MAIN"],
                    weights=weights,
                    bias=bias,
                    adds_main="MAIN_tex(" in blk,
                )
                continue

            conv_matches = _MAT4_OFFSET_RE.findall(blk)
            if conv_matches:
                kernels: Dict[Tuple[int, int, int], np.ndarray] = {}
                uses_crelu = False
                for mat_str, go_name, dx_s, dy_s in conv_matches:
                    branch = 0 if go_name == "go_0" else 1
                    if branch == 1:
                        uses_crelu = True
                    kernels[(branch, int(round(float(dx_s))), int(round(float(dy_s))))] = _parse_floats(mat_str).reshape(4, 4)
                bias_m = _BIAS_RE.search(blk)
                bias = _parse_floats(bias_m.group(1)) if bias_m else np.zeros(4, dtype=np.float32)
                self.conv_passes.append(
                    Conv3x3Pass(desc, save_name, binds[0] if binds else "MAIN", uses_crelu, kernels, bias)
                )

        if not self.conv_passes or self.dense_pass is None:
            raise ValueError(f"Unsupported shader: {self.shader_path.name}")

    def infer_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        h, w, _ = tile_rgb.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = tile_rgb
        tensors: Dict[str, np.ndarray] = {"MAIN": rgba}

        for layer in self.conv_passes:
            src = tensors[layer.bind_name]
            out = np.broadcast_to(layer.bias, (h, w, 4)).copy()
            if layer.uses_crelu:
                pos_pad = cv2.copyMakeBorder(np.maximum(src, 0.0), 1, 1, 1, 1, cv2.BORDER_REPLICATE)
                neg_pad = cv2.copyMakeBorder(np.maximum(-src, 0.0), 1, 1, 1, 1, cv2.BORDER_REPLICATE)
                for (branch, dx, dy), mat in layer.kernels.items():
                    buf = pos_pad if branch == 0 else neg_pad
                    out += buf[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w, :] @ mat
            else:
                src_pad = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
                for (_, dx, dy), mat in layer.kernels.items():
                    out += src_pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w, :] @ mat
            tensors[layer.save_name] = out

        assert self.dense_pass is not None
        dense = self.dense_pass
        last = np.broadcast_to(dense.bias, (h, w, 4)).copy()
        for feat_idx, bind_name in enumerate(dense.binds):
            feat = tensors[bind_name]
            if feat_idx * 2 in dense.weights:
                last += np.maximum(feat, 0.0) @ dense.weights[feat_idx * 2]
            if feat_idx * 2 + 1 in dense.weights:
                last += np.maximum(-feat, 0.0) @ dense.weights[feat_idx * 2 + 1]

        if self.has_depth_to_space:
            res = np.empty((h * 2, w * 2, 1), dtype=np.float32)
            res[0::2, 0::2, 0] = last[:, :, 0]
            res[0::2, 1::2, 0] = last[:, :, 1]
            res[1::2, 0::2, 0] = last[:, :, 2]
            res[1::2, 1::2, 0] = last[:, :, 3]
            return np.clip(cv2.resize(tile_rgb, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR) + res, 0.0, 1.0)

        return np.clip(tile_rgb + last[:, :, :3], 0.0, 1.0) if dense.adds_main else np.clip(last[:, :, :3], 0.0, 1.0)


class Anime4KChainEngine:
    def __init__(self, shader_paths: List[str | Path]) -> None:
        self.models = [Anime4KGLSLModel(p) for p in shader_paths]
        total = 1
        for m in self.models:
            total *= m.scale
        self.scale = total

    def infer_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        cur = tile_rgb
        for model in self.models:
            cur = model.infer_tile(cur)
        return cur
