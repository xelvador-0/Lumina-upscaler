# Hardware & DirectML detection
from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0


@dataclass
class GPUAdapterInfo:
    index: int
    name: str
    vram_mb: int
    driver_version: str = "Unknown"
    is_integrated: bool = False


@dataclass
class DirectMLReport:
    os_name: str
    python_version: str
    cpu_threads: int
    onnxruntime_version: str
    available_providers: List[str]
    directml_available: bool
    adapters: List[GPUAdapterInfo]
    recommended_device_id: int
    recommended_tile_size: int
    recommended_overlap: int


VRAM_BUDGET = {
    1: {"base_gb": 0.5, "pixels_per_gb": 90_000},
    2: {"base_gb": 0.7, "pixels_per_gb": 60_000},
    4: {"base_gb": 0.9, "pixels_per_gb": 40_000},
}

SAFE_TILE_SIZES = (64, 96, 128, 160, 192, 224, 256, 288, 384, 512)


def _query_registry_vram_map() -> dict[str, int]:
    ps = (
        "Get-ItemProperty -Path 'HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' "
        "-ErrorAction SilentlyContinue | "
        "Select-Object DriverDesc, @{N='VRAM';E={$_.'HardwareInformation.qwMemorySize'}} | "
        "ConvertTo-Json -Compress"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=6, check=False, creationflags=_NO_WINDOW,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {}
        raw = json.loads(proc.stdout.strip())
        if isinstance(raw, dict):
            raw = [raw]
        out: dict[str, int] = {}
        for item in raw:
            desc = str(item.get("DriverDesc") or "").strip().lower()
            vram = item.get("VRAM")
            if desc and isinstance(vram, (int, float)) and vram > 0:
                out[desc] = int(vram) // (1024 * 1024)
        return out
    except Exception:
        return {}


def detect_windows_gpus() -> List[GPUAdapterInfo]:
    if platform.system().lower() != "windows":
        return []

    ps_script = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name, AdapterRAM, DriverVersion | "
        "ConvertTo-Json -Compress"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True, text=True, timeout=8, check=False, creationflags=_NO_WINDOW,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return []

        raw = json.loads(proc.stdout.strip())
        if isinstance(raw, dict):
            raw = [raw]

        reg_vram = _query_registry_vram_map()
        adapters: List[GPUAdapterInfo] = []
        for idx, item in enumerate(raw):
            name = str(item.get("Name") or f"GPU {idx}").strip()
            ram_bytes = item.get("AdapterRAM") or 0
            try:
                vram_mb = max(0, int(ram_bytes) // (1024 * 1024))
            except (ValueError, TypeError):
                vram_mb = 0
            # Use 64-bit registry VRAM when available (fixes 4GB uint32 cap on modern 6GB-24GB GPUs)
            reg_mb = reg_vram.get(name.lower(), 0)
            if reg_mb > vram_mb:
                vram_mb = reg_mb

            driver = str(item.get("DriverVersion") or "Unknown")
            low = name.lower()
            is_igpu = any(
                tag in low
                for tag in ("uhd", "hd graphics", "iris", "radeon(tm) graphics", "vega")
            ) and ("arc" not in low)
            adapters.append(GPUAdapterInfo(idx, name, vram_mb, driver, is_igpu))
        return adapters
    except Exception:
        return []


def recommend_tile_size(vram_mb: int, scale: int = 2, fp16: bool = True) -> int:
    scale_key = 4 if scale >= 4 else (2 if scale >= 2 else 1)
    budget = VRAM_BUDGET[scale_key]

    if vram_mb <= 0:
        return 128 if scale_key >= 4 else 192

    vram_gb = vram_mb / 1024.0
    usable_gb = max(0.15, vram_gb * 0.75 - (budget["base_gb"] * (0.5 if fp16 else 1.0)))
    max_pixels = usable_gb * budget["pixels_per_gb"] * (1.0 if fp16 else 0.5)
    raw_side = int(math.sqrt(max(4096.0, max_pixels)))

    if vram_mb <= 1200:
        cap = 128 if scale_key >= 4 else 160
    elif vram_mb <= 2200:
        cap = 192 if scale_key >= 4 else 256
    elif vram_mb <= 4500:
        cap = 256 if scale_key >= 4 else 288
    elif vram_mb <= 8500:
        cap = 384 if scale_key >= 4 else 512
    else:
        cap = 512

    target = min(raw_side, cap)
    chosen = SAFE_TILE_SIZES[0]
    for s in SAFE_TILE_SIZES:
        if s <= target:
            chosen = s
        else:
            break
    return chosen


def inspect_directml(scale: int = 2, device_id: Optional[int] = None) -> DirectMLReport:
    ort_version = "Not Installed"
    providers: List[str] = []
    dml_ok = False

    try:
        import onnxruntime as ort  # type: ignore

        ort_version = getattr(ort, "__version__", "Unknown")
        providers = list(ort.get_available_providers())
        dml_ok = any(p in providers for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider", "CoreMLExecutionProvider"))
    except ImportError:
        pass

    adapters = detect_windows_gpus()
    if device_id is not None:
        rec_device = device_id
    elif adapters:
        # Prefer discrete GPU with highest VRAM
        discrete = sorted([a for a in adapters if not a.is_integrated and a.vram_mb >= 900], key=lambda x: x.vram_mb, reverse=True)
        rec_device = discrete[0].index if discrete else adapters[0].index
    else:
        rec_device = 0

    selected_vram = next((a.vram_mb for a in adapters if a.index == rec_device), 0)
    rec_tile = recommend_tile_size(selected_vram, scale=scale, fp16=True)
    rec_overlap = 16 if rec_tile >= 128 else 8

    return DirectMLReport(
        os_name=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python_version=platform.python_version(),
        cpu_threads=os.cpu_count() or 4,
        onnxruntime_version=ort_version,
        available_providers=providers,
        directml_available=dml_ok,
        adapters=adapters,
        recommended_device_id=rec_device,
        recommended_tile_size=rec_tile,
        recommended_overlap=rec_overlap,
    )


def format_hardware_report(report: DirectMLReport, scale: int = 2) -> str:
    lines = [
        "=== Hardware Diagnostics ===",
        f"OS / Python  : {report.os_name} | Python {report.python_version} | CPU Threads: {report.cpu_threads}",
        f"ONNX Runtime : {report.onnxruntime_version}",
        f"Providers    : {', '.join(report.available_providers) if report.available_providers else 'None'}",
        f"GPU Accel    : {'YES' if report.directml_available else 'NO (CPU Only)'}",
    ]
    if report.adapters:
        lines.append("GPUs         :")
        for gpu in report.adapters:
            kind = "iGPU" if gpu.is_integrated else "Discrete"
            vram_txt = f"{gpu.vram_mb} MB" if gpu.vram_mb > 0 else "Shared"
            lines.append(f"  [{gpu.index}] {gpu.name} | {vram_txt} | {kind}")
    lines.extend([
        f"Device       : {report.recommended_device_id}",
        f"Safe Tile    : {report.recommended_tile_size} (overlap {report.recommended_overlap})",
        "============================",
    ])
    return "\n".join(lines)
