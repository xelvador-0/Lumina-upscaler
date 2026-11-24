# Video FFmpeg pipe
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from core.paths import get_app_dir
from core.pipeline_image import FrameEnhancer
from video.temporal import TemporalStabilizer

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0


@dataclass
class VideoInfo:
    width: int
    height: int
    fps_num: int
    fps_den: int
    fps: float
    total_frames: int
    duration_sec: float
    has_audio: bool
    color_matrix: str = "bt709"
    color_range: str = "tv"
    rotation: int = 0


def _escape_concat_path(p: Path) -> str:
    return p.resolve().as_posix().replace("'", r"'\''")


def _extract_rotation(v_stream: dict) -> int:
    for sd in v_stream.get("side_data_list", []):
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"])))
            except (ValueError, TypeError):
                pass
    rot_tag = v_stream.get("tags", {}).get("rotate")
    if rot_tag is not None:
        try:
            return int(round(float(rot_tag)))
        except (ValueError, TypeError):
            pass
    return 0


def _detect_color_matrix(v_stream: dict, height: int) -> Tuple[str, str]:
    cs = str(v_stream.get("color_space") or "").lower()
    if cs in ("bt709",):
        mat = "bt709"
    elif cs in ("smpte170m", "bt470bg", "bt601"):
        mat = "bt601"
    else:
        mat = "bt709" if height >= 720 else "bt601"
    cr = str(v_stream.get("color_range") or "").lower()
    rng = "pc" if cr in ("pc", "full") else "tv"
    return mat, rng


def find_ffmpeg_binaries(custom_ffmpeg: Optional[Path] = None) -> Tuple[Path, Optional[Path]]:
    base_dir = get_app_dir()
    candidates: List[Path] = [Path(custom_ffmpeg)] if custom_ffmpeg is not None else []
    for name in ("ffmpeg.exe", "ffmpeg"):
        candidates.append(base_dir / "ffmpeg" / name)
        candidates.append(base_dir / name)

    ffmpeg_path = next((c.resolve() for c in candidates if c.is_file()), None)
    if ffmpeg_path is None:
        which_ff = shutil.which("ffmpeg")
        if which_ff:
            ffmpeg_path = Path(which_ff).resolve()
    if ffmpeg_path is None:
        raise FileNotFoundError("ffmpeg not found")

    ffprobe_path = next(
        ((ffmpeg_path.parent / p).resolve() for p in ("ffprobe.exe", "ffprobe") if (ffmpeg_path.parent / p).is_file()),
        None,
    )
    if ffprobe_path is None:
        which_pr = shutil.which("ffprobe")
        if which_pr:
            ffprobe_path = Path(which_pr).resolve()
    return ffmpeg_path, ffprobe_path


def probe_video(video_path: Path, ffmpeg_path: Path, ffprobe_path: Optional[Path] = None) -> VideoInfo:
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise ValueError(f"Invalid video: {video_path}")

    if ffprobe_path is not None and ffprobe_path.is_file():
        cmd = [str(ffprobe_path), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video_path)]
        meta = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=_NO_WINDOW).stdout)
        v_stream = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), None)
        if v_stream is None:
            raise ValueError(f"No video stream in {video_path}")
        has_audio = any(s.get("codec_type") == "audio" for s in meta.get("streams", []))

        width, height = int(v_stream["width"]), int(v_stream["height"])
        # H2: Account for FFmpeg auto-rotation on portrait phone videos
        rot = _extract_rotation(v_stream)
        if abs(rot) % 180 == 90:
            width, height = height, width

        mat, rng = _detect_color_matrix(v_stream, height)
        rate_str = v_stream.get("avg_frame_rate") or v_stream.get("r_frame_rate") or "24/1"
        if "/" in rate_str:
            n, d = rate_str.split("/", 1)
            fps_num, fps_den = max(1, int(n)), max(1, int(d))
        else:
            fps_num, fps_den = max(1, int(round(float(rate_str) * 1000))), 1000
        fps = fps_num / float(fps_den)
        duration = float(v_stream.get("duration") or meta.get("format", {}).get("duration") or 0.0)
        nb = v_stream.get("nb_frames")
        total = int(nb) if (nb and str(nb).isdigit()) else (max(1, int(round(duration * fps))) if duration > 0 else 0)
        return VideoInfo(width, height, fps_num, fps_den, fps, total, duration, has_audio, mat, rng, rot)

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid dimensions: {width}x{height}")
    mat = "bt709" if height >= 720 else "bt601"
    return VideoInfo(width, height, max(1, int(round(fps * 1000))), 1000, fps, total, total / fps if fps > 0 else 0.0, True, mat, "tv", 0)


def extract_single_video_frame(video_path: Path, frame_index: int, ffmpeg_path: Path, info: VideoInfo) -> np.ndarray:
    seek_sec = max(0.0, frame_index / max(1.0, info.fps))
    vf = f"scale=in_color_matrix={info.color_matrix}:in_range={info.color_range},format=rgb24"
    cmd = [
        str(ffmpeg_path), "-v", "error", "-ss", f"{seek_sec:.4f}", "-i", str(video_path),
        "-vf", vf, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True, creationflags=_NO_WINDOW)
    expected = info.width * info.height * 3
    if len(proc.stdout) < expected:
        raise RuntimeError("Preview frame read failed")
    return np.frombuffer(proc.stdout[:expected], dtype=np.uint8).reshape((info.height, info.width, 3)).astype(np.float32) / 255.0


def _build_codec_args(codec: str, crf: int, mat: str, rng: str) -> List[str]:
    c = codec.lower()
    crf_safe = max(0, min(51, int(crf)))
    ff_cs = "bt709" if mat == "bt709" else "smpte170m"
    ff_pri = "bt709" if mat == "bt709" else "smpte170m"
    ff_trc = "bt709" if mat == "bt709" else "smpte170m"
    if c == "h265":
        return [
            "-vf", f"scale=out_color_matrix={mat}:out_range={rng},format=yuv420p",
            "-c:v", "libx265", "-crf", str(crf_safe), "-preset", "medium", "-tag:v", "hvc1",
            "-colorspace", ff_cs, "-color_primaries", ff_pri, "-color_trc", ff_trc, "-color_range", rng,
        ]
    if c == "prores":
        return [
            "-vf", f"scale=out_color_matrix={mat}:out_range={rng},format=yuv422p10le",
            "-c:v", "prores_ks", "-profile:v", "2",
            "-colorspace", ff_cs, "-color_primaries", ff_pri, "-color_trc", ff_trc, "-color_range", rng,
        ]
    return [
        "-vf", f"scale=out_color_matrix={mat}:out_range={rng},format=yuv420p",
        "-c:v", "libx264", "-crf", str(crf_safe), "-preset", "medium",
        "-colorspace", ff_cs, "-color_primaries", ff_pri, "-color_trc", ff_trc, "-color_range", rng,
    ]


def _state_file_for(output_path: Path) -> Path:
    return output_path.parent / f".{output_path.stem}_resume.json"


def _settings_hash(enhancer: FrameEnhancer, codec: str, crf: int, anti_flicker: float) -> str:
    payload = json.dumps({
        "backend": enhancer.backend_label,
        "scale": enhancer.scale,
        "codec": codec,
        "crf": crf,
        "anti_flicker": round(anti_flicker, 3),
        "params": asdict(enhancer.params),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def enhance_video(
    input_path: Path,
    output_path: Path,
    enhancer: FrameEnhancer,
    ffmpeg_path: Path,
    ffprobe_path: Optional[Path] = None,
    anti_flicker: float = 0.25,
    codec: str = "h264",
    crf: int = 18,
    keep_audio: bool = True,
    resume: bool = True,
    stop_event: Optional[threading.Event] = None,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    t0 = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = probe_video(input_path, ffmpeg_path, ffprobe_path)

    out_w = (info.width * enhancer.scale) & ~1
    out_h = (info.height * enhancer.scale) & ~1
    cfg_hash = _settings_hash(enhancer, codec, crf, anti_flicker)

    state_file = _state_file_for(output_path)
    start_frame = 0
    segments: List[str] = []

    if resume and state_file.is_file():
        try:
            st = json.loads(state_file.read_text(encoding="utf-8"))
            if (
                st.get("input") == str(input_path)
                and st.get("out_w") == out_w
                and st.get("out_h") == out_h
                and st.get("codec") == codec
                and st.get("settings_hash") == cfg_hash
            ):
                valid_segs = [s for s in st.get("segments", []) if Path(s).is_file()]
                if valid_segs and int(st.get("completed_frames", 0)) > 0:
                    start_frame = int(st["completed_frames"])
                    segments = valid_segs
                    if log_cb:
                        log_cb(f"Resuming from frame {start_frame}/{info.total_frames}")
        except Exception:
            pass

    seg_ext = ".mov" if codec.lower() == "prores" else ".mkv"
    seg_path = output_path.parent / f".{output_path.stem}_part_{len(segments):03d}{seg_ext}"

    # H6: Explicit color matrix on decode to eliminate BT.709/601 shift
    vf_parts: List[str] = []
    if start_frame > 0:
        vf_parts.append(f"select=gte(n\\,{start_frame}),setpts=PTS-STARTPTS")
    vf_parts.append(f"scale=in_color_matrix={info.color_matrix}:in_range={info.color_range},format=rgb24")
    dec_cmd = [str(ffmpeg_path), "-v", "error", "-i", str(input_path), "-vf", ",".join(vf_parts), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]

    enc_cmd = [
        str(ffmpeg_path), "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}", "-r", f"{info.fps_num}/{info.fps_den}",
        "-i", "pipe:0",
        *_build_codec_args(codec, crf, info.color_matrix, info.color_range),
        str(seg_path),
    ]

    enc_err_file = tempfile.TemporaryFile(mode="w+b")
    decoder = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW)
    encoder = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=enc_err_file, creationflags=_NO_WINDOW)

    stabilizer = TemporalStabilizer(strength=anti_flicker)
    frame_bytes = info.width * info.height * 3
    processed_this_run = 0
    current_frame = start_frame
    stopped_early = False

    try:
        assert decoder.stdout is not None and encoder.stdin is not None
        while True:
            if stop_event is not None and stop_event.is_set():
                stopped_early = True
                break
            raw = decoder.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break

            orig_f32 = np.frombuffer(raw, dtype=np.uint8).reshape((info.height, info.width, 3)).astype(np.float32) / 255.0
            enhanced_f32 = enhancer.enhance_frame(orig_f32, grain_seed=current_frame)
            if anti_flicker > 0.005:
                enhanced_f32 = stabilizer.stabilize(orig_f32, enhanced_f32)
            if enhanced_f32.shape[1] != out_w or enhanced_f32.shape[0] != out_h:
                enhanced_f32 = enhanced_f32[:out_h, :out_w, :]

            encoder.stdin.write(np.clip(np.round(enhanced_f32 * 255.0), 0, 255).astype(np.uint8).tobytes())
            processed_this_run += 1
            current_frame += 1

            if progress_cb:
                progress_cb(current_frame, max(info.total_frames, current_frame), processed_this_run / max(0.001, time.perf_counter() - t0))
    finally:
        for pipe in (encoder.stdin, decoder.stdout):
            if pipe:
                try:
                    pipe.close()
                except Exception:
                    pass
        if stopped_early:
            decoder.terminate()
        decoder.wait(timeout=10)
        encoder.wait(timeout=30)
        enc_err_file.seek(0)
        enc_err = enc_err_file.read().decode("utf-8", errors="ignore")
        enc_err_file.close()
        if encoder.returncode not in (0, None) and processed_this_run > 0:
            raise RuntimeError(f"FFmpeg encoder error: {enc_err.strip()}")

    if processed_this_run > 0 and seg_path.is_file():
        segments.append(str(seg_path))

    if stopped_early:
        state_file.write_text(json.dumps({
            "input": str(input_path), "output": str(output_path),
            "out_w": out_w, "out_h": out_h, "codec": codec, "settings_hash": cfg_hash,
            "completed_frames": current_frame, "total_frames": info.total_frames, "segments": segments,
        }, indent=2), encoding="utf-8")
        return {"status": "paused", "completed_frames": current_frame, "total_frames": info.total_frames, "output": str(output_path), "elapsed_sec": round(time.perf_counter() - t0, 2)}

    # M2: Warn if source video ended significantly earlier than expected
    warning_msg = None
    if info.total_frames > 0 and current_frame < int(info.total_frames * 0.90):
        warning_msg = f"Source video may be truncated/corrupt: decoded {current_frame}/{info.total_frames} frames."
        if log_cb:
            log_cb(f"WARNING: {warning_msg}")

    concat_list = output_path.parent / f".{output_path.stem}_concat.txt"
    try:
        if len(segments) == 1:
            video_source_args = ["-i", segments[0]]
        else:
            concat_list.write_text("\n".join(f"file '{_escape_concat_path(Path(s))}'" for s in segments), encoding="utf-8")
            video_source_args = ["-f", "concat", "-safe", "0", "-i", str(concat_list)]

        mux_cmd = [str(ffmpeg_path), "-y", "-v", "error", *video_source_args]
        if keep_audio and info.has_audio:
            mux_cmd.extend(["-i", str(input_path), "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "copy", "-shortest"])
        else:
            mux_cmd.extend(["-map", "0:v:0", "-c:v", "copy"])
        if output_path.suffix.lower() == ".mp4":
            mux_cmd.extend(["-movflags", "+faststart"])
        mux_cmd.append(str(output_path))

        mux_proc = subprocess.run(mux_cmd, capture_output=True, text=True, creationflags=_NO_WINDOW)
        if mux_proc.returncode != 0:
            if keep_audio and info.has_audio:
                aac_proc = subprocess.run([
                    str(ffmpeg_path), "-y", "-v", "error", *video_source_args,
                    "-i", str(input_path), "-map", "0:v:0", "-map", "1:a?",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output_path),
                ], capture_output=True, text=True, creationflags=_NO_WINDOW)
                if aac_proc.returncode != 0:
                    raise RuntimeError(f"FFmpeg mux error: {aac_proc.stderr.strip()}")
            else:
                raise RuntimeError(f"FFmpeg mux error: {mux_proc.stderr.strip()}")
    finally:
        for s in segments:
            try:
                Path(s).unlink(missing_ok=True)
            except Exception:
                pass
        concat_list.unlink(missing_ok=True)
        state_file.unlink(missing_ok=True)

    return {
        "status": "completed",
        "completed_frames": current_frame,
        "total_frames": info.total_frames,
        "input_resolution": f"{info.width}x{info.height}",
        "output_resolution": f"{out_w}x{out_h}",
        "backend": enhancer.backend_label,
        "warning": warning_msg,
        "output": str(output_path),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
