# Universal ONNX engine (DirectML / CUDA / ROCm / CoreML / Multi-core CPU)
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np


class OnnxDirectMLEngine:
    def __init__(
        self,
        model_path: str | Path,
        device_id: int = 0,
        use_cpu: bool = False,
        scale: Optional[int] = None,
        warn_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise ImportError("Install onnxruntime-directml") from exc

        self.ort = ort
        self.device_id = int(device_id)
        self.warn_cb = warn_cb
        (
            self.session,
            self.active_provider,
            self.input_name,
            self.output_name,
            self.is_fp16,
            self.input_dtype,
            self.fixed_hw,
            detected_scale,
        ) = self._init_and_verify_session(ort, device_id=self.device_id, use_cpu=use_cpu)
        self.scale = scale if scale is not None else detected_scale

    def _make_opts(self, ort, opt_level, is_cpu: bool):
        opts = ort.SessionOptions()
        opts.enable_mem_pattern = is_cpu
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = opt_level
        if is_cpu:
            opts.intra_op_num_threads = max(1, os.cpu_count() or 4)
            opts.inter_op_num_threads = 1
        return opts

    def _init_and_verify_session(self, ort, device_id: int, use_cpu: bool):
        available = ort.get_available_providers()
        attempts = []

        if not use_cpu:
            if "CUDAExecutionProvider" in available:
                attempts.append((f"CUDA (device_id={device_id})", [("CUDAExecutionProvider", {"device_id": int(device_id)})], ort.GraphOptimizationLevel.ORT_ENABLE_ALL, False))
            if "DmlExecutionProvider" in available:
                attempts.append((f"DirectML (device_id={device_id}, BASIC)", [("DmlExecutionProvider", {"device_id": int(device_id)})], ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, False))
                attempts.append((f"DirectML (device_id={device_id}, NO_OPT)", [("DmlExecutionProvider", {"device_id": int(device_id)})], ort.GraphOptimizationLevel.ORT_DISABLE_ALL, False))
            if "ROCMExecutionProvider" in available:
                attempts.append((f"ROCm (device_id={device_id})", [("ROCMExecutionProvider", {"device_id": int(device_id)})], ort.GraphOptimizationLevel.ORT_ENABLE_ALL, False))
            if "CoreMLExecutionProvider" in available:
                attempts.append(("CoreML", ["CoreMLExecutionProvider"], ort.GraphOptimizationLevel.ORT_ENABLE_ALL, False))

        attempts.append((f"CPU ({os.cpu_count() or 4} threads)", ["CPUExecutionProvider"], ort.GraphOptimizationLevel.ORT_ENABLE_ALL, True))

        last_err: Optional[Exception] = None
        for label, providers, opt_level, is_cpu in attempts:
            try:
                opts = self._make_opts(ort, opt_level, is_cpu)
                sess = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=providers)

                inp = sess.get_inputs()[0]
                out = sess.get_outputs()[0]
                input_name, output_name = inp.name, out.name
                is_fp16 = "float16" in str(inp.type).lower()
                input_dtype = np.float16 if is_fp16 else np.float32

                fixed_hw: Optional[Tuple[int, int]] = None
                if len(inp.shape) == 4 and isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int):
                    fixed_hw = (int(inp.shape[2]), int(inp.shape[3]))

                probe_h, probe_w = fixed_hw if fixed_hw else (32, 32)
                dummy = np.zeros((1, 3, probe_h, probe_w), dtype=input_dtype)
                probe_out = sess.run([output_name], {input_name: dummy})[0]

                if (
                    len(inp.shape) == 4
                    and len(out.shape) == 4
                    and isinstance(inp.shape[2], int)
                    and isinstance(out.shape[2], int)
                    and inp.shape[2] > 0
                ):
                    detected_scale = max(1, int(round(out.shape[2] / inp.shape[2])))
                elif probe_out.ndim == 4 and probe_out.shape[2] >= probe_h:
                    detected_scale = max(1, int(round(probe_out.shape[2] / float(probe_h))))
                else:
                    detected_scale = 4

                return sess, label, input_name, output_name, is_fp16, input_dtype, fixed_hw, detected_scale
            except Exception as exc:
                last_err = exc
                continue

        raise RuntimeError(f"ONNX init failed: {last_err}")

    def _switch_to_cpu(self, reason: Exception) -> None:
        opts = self._make_opts(self.ort, self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL, is_cpu=True)
        self.session = self.ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.active_provider = f"CPU Fallback ({os.cpu_count() or 4} threads)"
        if self.warn_cb:
            self.warn_cb(f"{self.model_path.name} switched to CPU fallback: {reason}")

    def infer_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        orig_h, orig_w = tile_rgb.shape[:2]
        work = tile_rgb

        if self.fixed_hw is not None:
            fh, fw = self.fixed_hw
            if orig_h > fh or orig_w > fw:
                raise ValueError(f"Tile {orig_w}x{orig_h} exceeds model input {fw}x{fh}")
            if orig_h < fh or orig_w < fw:
                work = cv2.copyMakeBorder(work, 0, fh - orig_h, 0, fw - orig_w, cv2.BORDER_REFLECT_101)

        chw = np.transpose(work, (2, 0, 1))[None, ...].astype(self.input_dtype, copy=False)
        try:
            out = self.session.run([self.output_name], {self.input_name: chw})[0]
        except RuntimeError as exc:
            if "CPU" not in self.active_provider:
                self._switch_to_cpu(exc)
                out = self.session.run([self.output_name], {self.input_name: chw})[0]
            else:
                raise

        hwc = np.transpose(out[0], (1, 2, 0)).astype(np.float32, copy=False)
        if self.fixed_hw is not None:
            hwc = hwc[: orig_h * self.scale, : orig_w * self.scale, :]
        return np.clip(hwc, 0.0, 1.0)
