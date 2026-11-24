#!/usr/bin/env python3
# CLI
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from core.hardware import format_hardware_report, inspect_directml
from core.paths import get_app_dir
from core.pipeline_image import FrameEnhancer, enhance_single_image, load_image_rgb, save_image_rgb
from core.postprocess import EnhancementParams
from video.ffmpeg_pipe import enhance_video, find_ffmpeg_binaries

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".flv"}


def _resolve_model(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    if path.is_file():
        return path
    cand = get_app_dir() / "models" / path.name
    return cand if cand.is_file() else path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lumina", description="Lumina Upscaler CLI")
    p.add_argument("--check-gpu", action="store_true")
    p.add_argument("-i", "--input", type=Path)
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--mode", choices=["real", "anime"], default="real")
    p.add_argument("--scale", type=int, choices=[1, 2, 4, 8], default=2)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--model2", type=Path, default=None)
    p.add_argument("--shaders-dir", type=Path, default=None)
    p.add_argument("--no-anime-restore", action="store_true")
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--tile", type=int, default=None)
    p.add_argument("--overlap", type=int, default=None)
    p.add_argument("--pre-blur", type=float, default=0.0)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--compression", type=float, default=0.0)
    p.add_argument("--details", type=float, default=0.0)
    p.add_argument("--halo", type=float, default=0.20)
    p.add_argument("--sharpen", type=float, default=0.0)
    p.add_argument("--luma-blend", type=float, default=0.0)
    p.add_argument("--grain", type=float, default=0.0)
    p.add_argument("--ffmpeg", type=Path, default=None)
    p.add_argument("--anti-flicker", type=float, default=0.20)
    p.add_argument("--codec", choices=["h264", "h265", "prores"], default="h264")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.check_gpu:
            print(format_hardware_report(inspect_directml(scale=min(4, args.scale), device_id=args.device), scale=min(4, args.scale)))
            return 0

        if not args.input or not args.output:
            parser.error("-i/--input and -o/--output are required")

        if not args.input.exists():
            raise FileNotFoundError(f"Input not found: {args.input}")

        params = EnhancementParams(
            pre_blur=args.pre_blur, noise=args.noise, compression=args.compression,
            details=args.details, halo=args.halo, sharpen=args.sharpen,
            luma_blend=args.luma_blend, film_grain=args.grain,
        )

        m1 = _resolve_model(args.model)
        m2 = _resolve_model(args.model2)
        model_chain: List[Path] = [m for m in (m1, m2) if m is not None]

        # M8: Auto-pick single model in models/ if omitted in real mode
        if args.mode == "real" and not model_chain:
            avail = sorted((get_app_dir() / "models").glob("*.onnx"))
            if len(avail) == 1:
                model_chain = [avail[0]]
            else:
                raise ValueError("Pass --model <path> for real mode (or place a single .onnx in models/)")

        # Folder batch mode for images
        if args.input.is_dir():
            files = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            if not files:
                raise ValueError(f"No images found in {args.input}")
            args.output.mkdir(parents=True, exist_ok=True)
            enhancer = FrameEnhancer(
                mode=args.mode, scale=args.scale, model_paths=model_chain, shaders_dir=args.shaders_dir,
                device_id=args.device, use_cpu=args.cpu, tile_size=args.tile, overlap=args.overlap,
                params=params, restore_anime=not args.no_anime_restore,
            )
            ok, fail = 0, 0
            for idx, img_file in enumerate(files, start=1):
                try:
                    rgb, alpha = load_image_rgb(img_file)
                    out_rgb = enhancer.enhance_frame(rgb, grain_seed=idx)
                    out_path = args.output / f"{img_file.stem}_{args.mode}_{args.scale}x.png"
                    save_image_rgb(out_path, out_rgb, alpha=alpha)
                    ok += 1
                    print(f"[{idx}/{len(files)}] {out_path.name}")
                except Exception as exc:
                    fail += 1
                    print(f"[{idx}/{len(files)}] FAILED ({img_file.name}): {exc}", file=sys.stderr)
            print(f"Done: {ok} succeeded, {fail} failed.")
            return 0 if fail == 0 else 1

        if args.input.suffix.lower() in VIDEO_EXTS:
            ff_bin, fp_bin = find_ffmpeg_binaries(args.ffmpeg)
            enhancer = FrameEnhancer(
                mode=args.mode, scale=args.scale, model_paths=model_chain, shaders_dir=args.shaders_dir,
                device_id=args.device, use_cpu=args.cpu, tile_size=args.tile, overlap=args.overlap,
                params=params, restore_anime=not args.no_anime_restore,
            )
            res = enhance_video(
                input_path=args.input, output_path=args.output, enhancer=enhancer,
                ffmpeg_path=ff_bin, ffprobe_path=fp_bin, anti_flicker=args.anti_flicker,
                codec=args.codec, crf=args.crf, keep_audio=not args.no_audio, resume=not args.no_resume,
                progress_cb=lambda c, t, f: print(f"\r{c}/{t} ({f:.2f} fps)", end="", flush=True),
                log_cb=print,
            )
            print(f"\nDone ({res['elapsed_sec']}s) -> {res['output']}")
            return 0

        res = enhance_single_image(
            input_path=args.input, output_path=args.output, mode=args.mode, scale=args.scale,
            model_chain=model_chain if model_chain else None, shaders_dir=args.shaders_dir,
            device_id=args.device, use_cpu=args.cpu, tile_size=args.tile, overlap=args.overlap,
            params=params, restore_anime=not args.no_anime_restore,
        )
        print(f"Done ({res['elapsed_sec']}s) -> {res['output']} [{res['output_resolution']}]")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
