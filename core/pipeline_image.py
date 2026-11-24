# Image & frame pipeline
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.hardware import inspect_directml, recommend_tile_size
from core.paths import get_app_dir
from core.postprocess import EnhancementParams, postprocess_image, preprocess_image
from core.tiling import SmartTiler
from engines.anime4k_glsl import Anime4KChainEngine
from engines.onnx_dml import OnnxDirectMLEngine

_EXIF_COLOR_EXTS = {".jpg", ".jpeg", ".webp", ".bmp"}


def load_image_rgb(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    data = np.fromfile(str(path), dtype=np.uint8)
    # Use IMREAD_COLOR for JPEG/WebP/BMP so OpenCV applies EXIF orientation automatically (H4)
    flag = cv2.IMREAD_COLOR if path.suffix.lower() in _EXIF_COLOR_EXTS else cv2.IMREAD_UNCHANGED
    img = cv2.imdecode(data, flag)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")

    # Proper 16-bit PNG/TIFF normalization (H5)
    maxv = 65535.0 if img.dtype == np.uint16 else 255.0
    alpha: Optional[np.ndarray] = None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3].astype(np.float32) / maxv
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)

    return np.clip(img.astype(np.float32) / maxv, 0.0, 1.0), alpha


def save_image_rgb(path: Path, img_rgb: np.ndarray, alpha: Optional[np.ndarray] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    u8_rgb = np.clip(np.round(img_rgb * 255.0), 0, 255).astype(np.uint8)

    if alpha is not None:
        if alpha.shape[:2] != u8_rgb.shape[:2]:
            alpha = cv2.resize(alpha, (u8_rgb.shape[1], u8_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        a_u8 = np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)
        out_img = np.dstack([cv2.cvtColor(u8_rgb, cv2.COLOR_RGB2BGR), a_u8])
    else:
        out_img = cv2.cvtColor(u8_rgb, cv2.COLOR_RGB2BGR)

    ext = path.suffix.lower() or ".png"
    encode_params: List[int] = []
    if ext in (".jpg", ".jpeg"):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 96]
    elif ext == ".webp":
        encode_params = [cv2.IMWRITE_WEBP_QUALITY, 96]

    ok, buf = cv2.imencode(ext, out_img, encode_params)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    buf.tofile(str(path))


class FrameEnhancer:
    def __init__(
        self,
        mode: str = "real",
        scale: int = 2,
        model_paths: Optional[List[Path]] = None,
        shaders_dir: Optional[Path] = None,
        device_id: Optional[int] = None,
        use_cpu: bool = False,
        tile_size: Optional[int] = None,
        overlap: Optional[int] = None,
        params: Optional[EnhancementParams] = None,
        restore_anime: bool = True,
        warn_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.mode = mode
        self.scale = int(scale)
        self.warnings: List[str] = []
        self._auto_tile = tile_size is None
        self._warn_cb = warn_cb

        hw = inspect_directml(scale=min(4, self.scale), device_id=device_id)
        self.device_id = hw.recommended_device_id if device_id is None else device_id
        self.selected_vram_mb = next((a.vram_mb for a in hw.adapters if a.index == self.device_id), 0)

        self.params = EnhancementParams() if params is None else params.clamp()
        if mode == "anime" and self.params.anime_darken == 0.0 and self.params.sharpen > 0.0:
            self.params.anime_darken = 0.25

        self.stages: List[Tuple[str, int, Optional[Tuple[int, int]], Callable[[np.ndarray], np.ndarray]]] = []

        if mode == "real":
            # Deduplicate identical consecutive model paths (H1)
            raw_models = [Path(p).resolve() for p in (model_paths or []) if p is not None]
            valid_models: List[Path] = []
            for mp in raw_models:
                if not valid_models or valid_models[-1] != mp:
                    valid_models.append(mp)
            if not valid_models:
                raise ValueError("Model required for real mode")

            loaded: Dict[Path, OnnxDirectMLEngine] = {}
            for mp in valid_models:
                if mp not in loaded:
                    loaded[mp] = OnnxDirectMLEngine(mp, device_id=self.device_id, use_cpu=use_cpu, scale=None, warn_cb=warn_cb)
                eng = loaded[mp]
                lbl = f"{mp.name} ({eng.scale}x, {eng.active_provider}, {'FP16' if eng.is_fp16 else 'FP32'})"
                self.stages.append((lbl, eng.scale, eng.fixed_hw, eng.infer_tile))

            if len(self.stages) >= 2 and self.stages[0][1] > 1:
                self.warnings.append(
                    f"Model 1 ({valid_models[0].name}) is {self.stages[0][1]}x; set Model 1 to None or a 1x model to avoid double-upscaling."
                )

            total_scale = 1
            for _, st_scale, _, _ in self.stages:
                total_scale *= st_scale
            last_upscale = next((st for st in reversed(self.stages) if st[1] >= 2), None)
            while last_upscale is not None and total_scale < self.scale:
                self.stages.append(last_upscale)
                total_scale *= last_upscale[1]

            self.backend_label = " -> ".join(st[0] for st in self.stages)

        elif mode == "anime":
            sdir = shaders_dir or (get_app_dir() / "shaders")
            chain: List[Path] = []
            restore_glsl = sdir / "Anime4K_Restore_CNN_M.glsl"
            upscale_glsl = sdir / "Anime4K_Upscale_CNN_x2_M.glsl"

            if restore_anime and restore_glsl.is_file():
                chain.append(restore_glsl)
            passes_2x = 0 if self.scale <= 1 else (1 if self.scale == 2 else (2 if self.scale == 4 else 3))
            if passes_2x > 0:
                if not upscale_glsl.is_file():
                    raise FileNotFoundError(f"Missing shader: {upscale_glsl}")
                for _ in range(passes_2x):
                    chain.append(upscale_glsl)

            if not chain:
                raise FileNotFoundError(f"No .glsl shaders found in {sdir}")

            loaded_sh: Dict[Path, Anime4KChainEngine] = {}
            for sh in chain:
                if sh not in loaded_sh:
                    loaded_sh[sh] = Anime4KChainEngine([sh])
                eng = loaded_sh[sh]
                self.stages.append((sh.name, eng.scale, None, eng.infer_tile))

            self.backend_label = "Anime4K CPU (" + " -> ".join(p.name for p in chain) + ")"
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # H7: Compute auto tile size from the REAL maximum stage model scale, not just target scale
        max_stage_scale = max((st[1] for st in self.stages), default=self.scale)
        if tile_size is None:
            self.tile_size = recommend_tile_size(self.selected_vram_mb, scale=max_stage_scale, fp16=True)
        else:
            self.tile_size = int(tile_size)
        self.overlap = (16 if self.tile_size >= 128 else 8) if overlap is None else int(overlap)

    def enhance_frame(
        self,
        orig_rgb: np.ndarray,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        grain_seed: Optional[int] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> np.ndarray:
        orig_h, orig_w = orig_rgb.shape[:2]
        cur = preprocess_image(orig_rgb, self.params)

        for idx, (_, st_scale, fixed_hw, tile_fn) in enumerate(self.stages):
            t_size = self.tile_size
            t_ov = self.overlap
            # M4: Use exact fixed_hw when tile is Auto (or clamp when larger) to avoid wasted padding
            if fixed_hw is not None:
                max_side = min(fixed_hw)
                if self._auto_tile or t_size > max_side:
                    t_size = max_side
                if t_ov >= t_size // 2:
                    t_ov = max(8, t_size // 8)

            tiler = SmartTiler(tile_size=t_size, overlap=t_ov, scale=st_scale, low_vram_gc=True)
            cb = progress_cb if idx == len(self.stages) - 1 else None
            cur = tiler.process(cur, tile_fn, progress_cb=cb, stop_event=stop_event)

        target_w, target_h = orig_w * self.scale, orig_h * self.scale
        if cur.shape[1] != target_w or cur.shape[0] != target_h:
            interp = cv2.INTER_AREA if target_w < cur.shape[1] else cv2.INTER_CUBIC
            cur = cv2.resize(cur, (target_w, target_h), interpolation=interp)

        return postprocess_image(orig_rgb, cur, self.params, grain_seed=grain_seed)


def render_preview_comparison(
    orig_rgb: np.ndarray,
    enhancer: FrameEnhancer,
    output_path: Path,
    crop_size: int = 256,
) -> Path:
    h, w = orig_rgb.shape[:2]
    ch, cw = min(h, crop_size), min(w, crop_size)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = orig_rgb[y0 : y0 + ch, x0 : x0 + cw, :]

    after = enhancer.enhance_frame(crop, grain_seed=42)
    ah, aw = after.shape[:2]
    before = cv2.resize(crop, (aw, ah), interpolation=cv2.INTER_CUBIC)

    combined = np.concatenate([before, np.ones((ah, 4, 3), dtype=np.float32), after], axis=1)
    u8 = np.clip(np.round(combined * 255.0), 0, 255).astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale_txt = max(0.55, min(1.2, aw / 420.0))
    cv2.putText(u8, "BEFORE", (12, 32), font, scale_txt, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(u8, "BEFORE", (12, 32), font, scale_txt, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(u8, "AFTER", (aw + 16, 32), font, scale_txt, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(u8, "AFTER", (aw + 16, 32), font, scale_txt, (255, 255, 255), 2, cv2.LINE_AA)

    save_image_rgb(output_path, u8.astype(np.float32) / 255.0)
    return output_path


def enhance_single_image(
    input_path: Path,
    output_path: Path,
    mode: str = "real",
    scale: int = 2,
    model_path: Optional[Path] = None,
    model_chain: Optional[List[Path]] = None,
    shaders_dir: Optional[Path] = None,
    device_id: Optional[int] = None,
    use_cpu: bool = False,
    tile_size: Optional[int] = None,
    overlap: Optional[int] = None,
    params: Optional[EnhancementParams] = None,
    restore_anime: bool = True,
) -> dict:
    t0 = time.perf_counter()
    orig_rgb, alpha = load_image_rgb(input_path)
    orig_h, orig_w = orig_rgb.shape[:2]

    models: List[Path] = list(model_chain) if model_chain else ([model_path] if model_path else [])
    enhancer = FrameEnhancer(
        mode=mode,
        scale=scale,
        model_paths=models,
        shaders_dir=shaders_dir,
        device_id=device_id,
        use_cpu=use_cpu,
        tile_size=tile_size,
        overlap=overlap,
        params=params,
        restore_anime=restore_anime,
    )

    try:
        from tqdm import tqdm  # type: ignore

        pbar = None

        def _progress(done: int, total: int) -> None:
            nonlocal pbar
            if pbar is None:
                pbar = tqdm(total=total, desc=f"Tiling ({enhancer.tile_size}px)", unit="tile")
            pbar.n = done
            pbar.refresh()
            if done >= total:
                pbar.close()

        final_rgb = enhancer.enhance_frame(orig_rgb, progress_cb=_progress, grain_seed=123)
    except ImportError:
        final_rgb = enhancer.enhance_frame(orig_rgb, grain_seed=123)

    save_image_rgb(output_path, final_rgb, alpha=alpha)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "input_resolution": f"{orig_w}x{orig_h}",
        "output_resolution": f"{final_rgb.shape[1]}x{final_rgb.shape[0]}",
        "backend": enhancer.backend_label,
        "tile_size": enhancer.tile_size,
        "overlap": enhancer.overlap,
        "warnings": enhancer.warnings,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
