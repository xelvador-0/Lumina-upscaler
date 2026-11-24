#!/usr/bin/env python3
# GUI
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, List

from core.config import load_presets, load_user_settings, save_user_settings
from core.hardware import format_hardware_report, inspect_directml
from core.paths import get_app_dir, get_bundle_dir
from core.pipeline_image import (
    FrameEnhancer,
    load_image_rgb,
    render_preview_comparison,
    save_image_rgb,
)
from core.postprocess import EnhancementParams
from core.tiling import CancelledError
from video.ffmpeg_pipe import (
    enhance_video,
    extract_single_video_frame,
    find_ffmpeg_binaries,
    probe_video,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_PROCESSED_RE = re.compile(r"_(real|anime)_[1248]x$", re.IGNORECASE)

BASE_DIR = get_app_dir()
DEFAULT_IMG_IN = BASE_DIR / "input_photos"
DEFAULT_IMG_OUT = BASE_DIR / "output_photos"
DEFAULT_VID_OUT = BASE_DIR / "output_videos"
MODELS_DIR = BASE_DIR / "models"
SHADERS_DIR = BASE_DIR / "shaders"
FFMPEG_DIR = BASE_DIR / "ffmpeg"

PRESETS: Dict[str, Dict[str, float]] = load_presets()


class LuminaGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Lumina Upscaler")
        self.root.geometry("700x690")
        self.root.minsize(640, 620)
        self.root.resizable(True, True)
        self.root.configure(bg="#F0F0F0")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._apply_window_icon()

        style = ttk.Style()
        for theme in ("winnative", "vista", "clam", "default"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break

        for folder in (DEFAULT_IMG_IN, DEFAULT_IMG_OUT, DEFAULT_VID_OUT, MODELS_DIR, SHADERS_DIR, FFMPEG_DIR):
            try:
                folder.mkdir(exist_ok=True)
            except OSError:
                pass

        self.is_running = False
        self.stop_event = threading.Event()
        self._build_ui()
        self._refresh_models()
        self._auto_detect_ffmpeg()
        self._restore_saved_settings()

    def _apply_window_icon(self) -> None:
        for base in (get_app_dir(), get_bundle_dir()):
            ico = base / "assets" / "icon.ico"
            if ico.is_file():
                try:
                    self.root.iconbitmap(default=str(ico))
                    return
                except Exception:
                    pass

    def _on_close(self) -> None:
        self._save_current_settings()
        if self.is_running:
            if not messagebox.askyesno("Exit", "A job is running. Stop cleanly and exit?"):
                return
            self.stop_event.set()
            self._log("Stopping cleanly before exit...")
            self.root.after(400, self._wait_and_destroy)
            return
        self.root.destroy()

    def _wait_and_destroy(self) -> None:
        if self.is_running:
            self.root.after(250, self._wait_and_destroy)
        else:
            self.root.destroy()

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}

        # Tabs
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="x", padx=10, pady=(8, 4))

        self.tab_photos = tk.Frame(self.tabs, bg="#F0F0F0")
        self.tab_video = tk.Frame(self.tabs, bg="#F0F0F0")
        self.tabs.add(self.tab_photos, text="  Photos  ")
        self.tabs.add(self.tab_video, text="  Video  ")

        self._build_photos_tab()
        self._build_video_tab()

        # Models
        ai_frame = tk.LabelFrame(self.root, text=" Models ", bg="#F0F0F0", fg="#000000", font=("Segoe UI", 9, "bold"))
        ai_frame.pack(fill="x", **pad)

        r1 = tk.Frame(ai_frame, bg="#F0F0F0")
        r1.pack(fill="x", padx=8, pady=5)

        tk.Label(r1, text="Mode:", bg="#F0F0F0", fg="#000000").pack(side="left")
        self.mode_var = tk.StringVar(value="real")
        mode_cb = ttk.Combobox(r1, textvariable=self.mode_var, values=["real", "anime"], state="readonly", width=8)
        mode_cb.pack(side="left", padx=(4, 14))
        mode_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_mode_change())

        tk.Label(r1, text="Scale:", bg="#F0F0F0", fg="#000000").pack(side="left")
        self.scale_var = tk.StringVar(value="2")
        ttk.Combobox(r1, textvariable=self.scale_var, values=["1", "2", "4", "8"], state="readonly", width=5).pack(side="left", padx=(4, 14))

        tk.Label(r1, text="Tile:", bg="#F0F0F0", fg="#000000").pack(side="left")
        self.tile_var = tk.StringVar(value="Auto")
        ttk.Combobox(r1, textvariable=self.tile_var, values=["Auto", "128", "160", "192", "256", "384", "512"], state="readonly", width=7).pack(side="left", padx=(4, 14))

        tk.Label(r1, text="GPU:", bg="#F0F0F0", fg="#000000").pack(side="left")
        self.device_var = tk.StringVar(value="Auto")
        ttk.Combobox(r1, textvariable=self.device_var, values=["Auto", "0", "1", "2", "CPU"], state="readonly", width=6).pack(side="left", padx=(4, 10))

        tk.Button(r1, text="Refresh", command=self._refresh_models, bg="#E1E1E1", fg="#000000").pack(side="right")

        r2 = tk.Frame(ai_frame, bg="#F0F0F0")
        r2.pack(fill="x", padx=8, pady=3)
        tk.Label(r2, text="Model 1:", width=10, anchor="w", bg="#F0F0F0", fg="#000000").pack(side="left")
        self.model1_var = tk.StringVar(value="None")
        self.model1_cb = ttk.Combobox(r2, textvariable=self.model1_var, state="readonly")
        self.model1_cb.pack(side="left", fill="x", expand=True)

        r3 = tk.Frame(ai_frame, bg="#F0F0F0")
        r3.pack(fill="x", padx=8, pady=(3, 7))
        tk.Label(r3, text="Model 2:", width=10, anchor="w", bg="#F0F0F0", fg="#000000").pack(side="left")
        self.model2_var = tk.StringVar(value="None")
        self.model2_cb = ttk.Combobox(r3, textvariable=self.model2_var, state="readonly")
        self.model2_cb.pack(side="left", fill="x", expand=True)

        # Quality
        q_frame = tk.LabelFrame(self.root, text=" Quality ", bg="#F0F0F0", fg="#000000", font=("Segoe UI", 9, "bold"))
        q_frame.pack(fill="x", **pad)

        pr_row = tk.Frame(q_frame, bg="#F0F0F0")
        pr_row.pack(fill="x", padx=8, pady=(4, 3))
        tk.Label(pr_row, text="Preset:", bg="#F0F0F0", fg="#000000").pack(side="left", padx=(0, 6))
        for name in PRESETS:
            tk.Button(pr_row, text=name, command=lambda n=name: self._apply_preset(n), bg="#E1E1E1", fg="#000000", padx=8).pack(side="left", padx=3)

        qr1 = tk.Frame(q_frame, bg="#F0F0F0")
        qr1.pack(fill="x", padx=8, pady=4)
        self.noise_var = self._add_spin(qr1, "Denoise:", 0.0, 0.0, 1.0, 0.05)
        self.comp_var = self._add_spin(qr1, "Comp:", 0.0, 0.0, 1.0, 0.05)
        self.details_var = self._add_spin(qr1, "Details:", 0.0, 0.0, 1.0, 0.05)
        self.sharpen_var = self._add_spin(qr1, "Sharpen:", 0.0, 0.0, 1.0, 0.05)

        qr2 = tk.Frame(q_frame, bg="#F0F0F0")
        qr2.pack(fill="x", padx=8, pady=(2, 6))
        self.halo_var = self._add_spin(qr2, "Dehalo:", 0.20, 0.0, 1.0, 0.05)
        self.luma_var = self._add_spin(qr2, "Luma:", 0.0, 0.0, 0.5, 0.02)
        self.grain_var = self._add_spin(qr2, "Grain:", 0.0, 0.0, 0.08, 0.004)

        self._on_mode_change()

        # Controls
        btn_row = tk.Frame(self.root, bg="#F0F0F0")
        btn_row.pack(fill="x", padx=10, pady=6)

        self.start_btn = tk.Button(btn_row, text="START", width=13, height=2, command=self._start_processing, bg="#FFFFFF", fg="#000000", font=("Segoe UI", 10, "bold"), relief="raised", bd=2)
        self.start_btn.pack(side="left")

        self.stop_btn = tk.Button(btn_row, text="STOP", width=12, height=2, command=self._request_stop, bg="#E1E1E1", fg="#000000", state="disabled")
        self.stop_btn.pack(side="left", padx=6)

        self.preview_btn = tk.Button(btn_row, text="Preview", width=12, height=2, command=self._run_preview, bg="#E1E1E1", fg="#000000")
        self.preview_btn.pack(side="left", padx=4)

        tk.Button(btn_row, text="Check GPU", width=11, height=2, command=self._check_gpu, bg="#E1E1E1", fg="#000000").pack(side="left", padx=4)
        tk.Button(btn_row, text="Open Output", width=12, height=2, command=self._open_output_folder, bg="#E1E1E1", fg="#000000").pack(side="right")

        # Log
        self.progress_label = tk.Label(self.root, text="Idle", anchor="w", bg="#F0F0F0", fg="#000000")
        self.progress_label.pack(fill="x", padx=10)

        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=(2, 4))

        log_frame = tk.LabelFrame(self.root, text=" Log ", bg="#F0F0F0", fg="#000000")
        log_frame.pack(fill="both", expand=True, padx=10, pady=(2, 10))

        self.log_box = scrolledtext.ScrolledText(log_frame, height=8, bg="#FFFFFF", fg="#000000", font=("Consolas", 9), state="disabled")
        self.log_box.pack(fill="both", expand=True, padx=6, pady=6)
        self._log("Ready.")

    def _add_spin(self, parent: tk.Frame, label: str, default: float, min_v: float, max_v: float, step: float) -> tk.DoubleVar:
        tk.Label(parent, text=label, bg="#F0F0F0", fg="#000000").pack(side="left")
        var = tk.DoubleVar(value=default)
        tk.Spinbox(parent, from_=min_v, to=max_v, increment=step, textvariable=var, width=6, bg="#FFFFFF").pack(side="left", padx=(3, 12))
        return var

    def _apply_preset(self, name: str) -> None:
        cfg = PRESETS[name]
        self.noise_var.set(cfg["noise"])
        self.comp_var.set(cfg["comp"])
        self.details_var.set(cfg["details"])
        self.sharpen_var.set(cfg["sharpen"])
        self.halo_var.set(cfg["halo"])
        self.luma_var.set(cfg["luma"])
        self.grain_var.set(cfg["grain"])
        self.flicker_var.set(cfg["flicker"])
        self._log(f"Preset: {name}")

    def _build_photos_tab(self) -> None:
        self.source_type = tk.StringVar(value="folder")
        r_row = tk.Frame(self.tab_photos, bg="#F0F0F0")
        r_row.pack(fill="x", padx=8, pady=(8, 2))
        tk.Radiobutton(r_row, text="Folder", variable=self.source_type, value="folder", command=self._on_photo_source_change, bg="#F0F0F0", fg="#000000").pack(side="left", padx=(0, 15))
        tk.Radiobutton(r_row, text="Single File", variable=self.source_type, value="file", command=self._on_photo_source_change, bg="#F0F0F0", fg="#000000").pack(side="left")

        in_row = tk.Frame(self.tab_photos, bg="#F0F0F0")
        in_row.pack(fill="x", padx=8, pady=4)
        tk.Label(in_row, text="Input :", width=8, anchor="w", bg="#F0F0F0", fg="#000000").pack(side="left")
        self.img_in_var = tk.StringVar(value=str(DEFAULT_IMG_IN))
        tk.Entry(in_row, textvariable=self.img_in_var, bg="#FFFFFF", fg="#000000").pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(in_row, text="Browse...", width=10, command=self._browse_img_input, bg="#E1E1E1").pack(side="left")

        out_row = tk.Frame(self.tab_photos, bg="#F0F0F0")
        out_row.pack(fill="x", padx=8, pady=(2, 10))
        tk.Label(out_row, text="Output:", width=8, anchor="w", bg="#F0F0F0", fg="#000000").pack(side="left")
        self.img_out_var = tk.StringVar(value=str(DEFAULT_IMG_OUT))
        tk.Entry(out_row, textvariable=self.img_out_var, bg="#FFFFFF", fg="#000000").pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(out_row, text="Browse...", width=10, command=self._browse_img_output, bg="#E1E1E1").pack(side="left")

    def _build_video_tab(self) -> None:
        v_in = tk.Frame(self.tab_video, bg="#F0F0F0")
        v_in.pack(fill="x", padx=8, pady=(8, 3))
        tk.Label(v_in, text="Video :", width=8, anchor="w", bg="#F0F0F0", fg="#000000").pack(side="left")
        self.vid_in_var = tk.StringVar(value="")
        tk.Entry(v_in, textvariable=self.vid_in_var, bg="#FFFFFF", fg="#000000").pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(v_in, text="Browse...", width=10, command=self._browse_vid_input, bg="#E1E1E1").pack(side="left")

        v_out = tk.Frame(self.tab_video, bg="#F0F0F0")
        v_out.pack(fill="x", padx=8, pady=3)
        tk.Label(v_out, text="Output:", width=8, anchor="w", bg="#F0F0F0", fg="#000000").pack(side="left")
        self.vid_out_var = tk.StringVar(value=str(DEFAULT_VID_OUT))
        tk.Entry(v_out, textvariable=self.vid_out_var, bg="#FFFFFF", fg="#000000").pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(v_out, text="Browse...", width=10, command=self._browse_vid_output, bg="#E1E1E1").pack(side="left")

        ff_row = tk.Frame(self.tab_video, bg="#F0F0F0")
        ff_row.pack(fill="x", padx=8, pady=3)
        tk.Label(ff_row, text="FFmpeg:", width=8, anchor="w", bg="#F0F0F0", fg="#000000").pack(side="left")
        self.ffmpeg_var = tk.StringVar(value="")
        tk.Entry(ff_row, textvariable=self.ffmpeg_var, bg="#FFFFFF", fg="#000000").pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(ff_row, text="Browse...", width=10, command=self._browse_ffmpeg, bg="#E1E1E1").pack(side="left")

        opt_row = tk.Frame(self.tab_video, bg="#F0F0F0")
        opt_row.pack(fill="x", padx=8, pady=(4, 8))
        self.flicker_var = self._add_spin(opt_row, "Anti-Flicker:", 0.20, 0.0, 0.5, 0.05)

        tk.Label(opt_row, text="Codec:", bg="#F0F0F0", fg="#000000").pack(side="left")
        self.codec_var = tk.StringVar(value="h264")
        ttk.Combobox(opt_row, textvariable=self.codec_var, values=["h264", "h265", "prores"], state="readonly", width=7).pack(side="left", padx=(3, 12))

        tk.Label(opt_row, text="CRF:", bg="#F0F0F0", fg="#000000").pack(side="left")
        self.crf_var = tk.StringVar(value="18")
        ttk.Combobox(opt_row, textvariable=self.crf_var, values=["14", "16", "18", "20", "22"], state="readonly", width=4).pack(side="left", padx=(3, 12))

        self.keep_audio_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_row, text="Audio", variable=self.keep_audio_var, bg="#F0F0F0", fg="#000000").pack(side="left")
        self.resume_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_row, text="Resume", variable=self.resume_var, bg="#F0F0F0", fg="#000000").pack(side="left", padx=(8, 0))

    def _log(self, msg: str) -> None:
        def _append() -> None:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.root.after(0, _append)

    def _set_status(self, text: str) -> None:
        self.root.after(0, lambda: self.progress_label.configure(text=text))

    def _auto_detect_ffmpeg(self) -> None:
        try:
            ff, _ = find_ffmpeg_binaries()
            self.ffmpeg_var.set(str(ff))
        except FileNotFoundError:
            self.ffmpeg_var.set("")

    def _restore_saved_settings(self) -> None:
        st = load_user_settings()
        if not st:
            return
        for key, var in (
            ("mode", self.mode_var), ("scale", self.scale_var), ("tile", self.tile_var),
            ("device", self.device_var), ("codec", self.codec_var), ("crf", self.crf_var),
            ("noise", self.noise_var), ("comp", self.comp_var), ("details", self.details_var),
            ("sharpen", self.sharpen_var), ("halo", self.halo_var), ("luma", self.luma_var),
            ("grain", self.grain_var), ("flicker", self.flicker_var),
            ("keep_audio", self.keep_audio_var), ("resume", self.resume_var),
        ):
            if key in st:
                try:
                    var.set(st[key])
                except Exception:
                    pass
        if st.get("model1") in self.model1_cb["values"]:
            self.model1_var.set(st["model1"])
        if st.get("model2") in self.model2_cb["values"]:
            self.model2_var.set(st["model2"])
        self._on_mode_change()

    def _save_current_settings(self) -> None:
        save_user_settings({
            "mode": self.mode_var.get(), "scale": self.scale_var.get(), "tile": self.tile_var.get(),
            "device": self.device_var.get(), "model1": self.model1_var.get(), "model2": self.model2_var.get(),
            "codec": self.codec_var.get(), "crf": self.crf_var.get(),
            "noise": float(self.noise_var.get()), "comp": float(self.comp_var.get()),
            "details": float(self.details_var.get()), "sharpen": float(self.sharpen_var.get()),
            "halo": float(self.halo_var.get()), "luma": float(self.luma_var.get()),
            "grain": float(self.grain_var.get()), "flicker": float(self.flicker_var.get()),
            "keep_audio": bool(self.keep_audio_var.get()), "resume": bool(self.resume_var.get()),
        })

    def _on_photo_source_change(self) -> None:
        if self.source_type.get() == "folder":
            self.img_in_var.set(str(DEFAULT_IMG_IN))
        elif Path(self.img_in_var.get()).is_dir():
            self.img_in_var.set("")

    def _on_mode_change(self) -> None:
        state = "readonly" if self.mode_var.get() == "real" else "disabled"
        self.model1_cb.configure(state=state)
        self.model2_cb.configure(state=state)

    def _refresh_models(self) -> None:
        onnx_files = sorted(p.name for p in MODELS_DIR.glob("*.onnx"))
        # H1: Both Model 1 and Model 2 allow "None" so single-model jobs never force a second model
        opts = ["None"] + onnx_files
        self.model1_cb["values"] = opts
        self.model2_cb["values"] = opts
        if self.model1_var.get() not in opts:
            self.model1_var.set("None")
        if self.model2_var.get() not in opts:
            if onnx_files:
                self.model2_var.set(next((m for m in onnx_files if not m.lower().startswith("1x")), onnx_files[0]))
            else:
                self.model2_var.set("None")

    def _snapshot_settings(self) -> dict:
        """M7: Read all Tk variables safely on the main thread before launching worker threads."""
        return {
            "mode": self.mode_var.get(),
            "scale": int(self.scale_var.get()),
            "tile": None if self.tile_var.get() == "Auto" else int(self.tile_var.get()),
            "device": self.device_var.get(),
            "model1": self.model1_var.get(),
            "model2": self.model2_var.get(),
            "params": EnhancementParams(
                noise=float(self.noise_var.get()),
                compression=float(self.comp_var.get()),
                details=float(self.details_var.get()),
                halo=float(self.halo_var.get()),
                sharpen=float(self.sharpen_var.get()),
                luma_blend=float(self.luma_var.get()),
                film_grain=float(self.grain_var.get()),
            ),
            "source_type": self.source_type.get(),
            "img_in": self.img_in_var.get().strip(),
            "img_out": self.img_out_var.get().strip(),
            "vid_in": self.vid_in_var.get().strip(),
            "vid_out": self.vid_out_var.get().strip(),
            "ffmpeg": self.ffmpeg_var.get().strip(),
            "anti_flicker": float(self.flicker_var.get()),
            "codec": self.codec_var.get(),
            "crf": int(self.crf_var.get()),
            "keep_audio": bool(self.keep_audio_var.get()),
            "resume": bool(self.resume_var.get()),
            "is_video": self.tabs.index(self.tabs.select()) == 1,
        }

    def _build_enhancer(self, cfg: dict) -> FrameEnhancer:
        models: List[Path] = []
        if cfg["mode"] == "real":
            for m in (cfg["model1"], cfg["model2"]):
                if m and m != "None" and (MODELS_DIR / m).is_file():
                    models.append(MODELS_DIR / m)
            if not models:
                raise FileNotFoundError(f"Select at least one .onnx model in {MODELS_DIR}")

        dev_str = cfg["device"]
        enhancer = FrameEnhancer(
            mode=cfg["mode"],
            scale=cfg["scale"],
            model_paths=models,
            shaders_dir=SHADERS_DIR,
            device_id=int(dev_str) if dev_str.isdigit() else None,
            use_cpu=(dev_str == "CPU"),
            tile_size=cfg["tile"],
            params=cfg["params"],
            warn_cb=lambda msg: self._log(f"WARNING: {msg}"),
        )
        for w in enhancer.warnings:
            self._log(f"WARNING: {w}")
        return enhancer

    def _browse_img_input(self) -> None:
        if self.source_type.get() == "folder":
            folder = filedialog.askdirectory(initialdir=self.img_in_var.get() or str(BASE_DIR))
            if folder:
                self.img_in_var.set(folder)
        else:
            fp = filedialog.askopenfilename(initialdir=str(BASE_DIR), filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("All", "*.*")])
            if fp:
                self.img_in_var.set(fp)

    def _browse_img_output(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.img_out_var.get() or str(BASE_DIR))
        if folder:
            self.img_out_var.set(folder)

    def _browse_vid_input(self) -> None:
        fp = filedialog.askopenfilename(initialdir=str(BASE_DIR), filetypes=[("Videos", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.ts"), ("All", "*.*")])
        if fp:
            self.vid_in_var.set(fp)

    def _browse_vid_output(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.vid_out_var.get() or str(BASE_DIR))
        if folder:
            self.vid_out_var.set(folder)

    def _browse_ffmpeg(self) -> None:
        fp = filedialog.askopenfilename(initialdir=str(FFMPEG_DIR), filetypes=[("FFmpeg", "ffmpeg.exe;ffmpeg"), ("All", "*.*")])
        if fp:
            self.ffmpeg_var.set(fp)

    def _open_path(self, target: Path) -> None:
        resolved = target.resolve()
        if not resolved.exists():
            return
        if sys.platform.startswith("win"):
            os.startfile(str(resolved))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(resolved)])
        else:
            subprocess.Popen(["xdg-open", str(resolved)])

    def _open_output_folder(self) -> None:
        out_dir = Path(self.vid_out_var.get() if self.tabs.index(self.tabs.select()) == 1 else self.img_out_var.get())
        out_dir.mkdir(parents=True, exist_ok=True)
        self._open_path(out_dir)

    def _check_gpu(self) -> None:
        scale = min(4, int(self.scale_var.get()))
        dev_str = self.device_var.get()
        dev_id = int(dev_str) if dev_str.isdigit() else None
        self._set_status("Checking hardware...")

        def _worker() -> None:
            report = inspect_directml(scale=scale, device_id=dev_id)
            self._log("\n" + format_hardware_report(report, scale=scale))
            self._set_status("Idle")

        threading.Thread(target=_worker, daemon=True).start()

    def _request_stop(self) -> None:
        if self.is_running:
            self.stop_event.set()
            self._log("Stopping...")
            self.stop_btn.configure(state="disabled")

    def _run_preview(self) -> None:
        if self.is_running:
            return
        cfg = self._snapshot_settings()
        self.is_running = True
        self.start_btn.configure(state="disabled")
        self.preview_btn.configure(state="disabled")
        self._set_status("Generating preview...")

        def _worker() -> None:
            try:
                enhancer = self._build_enhancer(cfg)
                if cfg["is_video"]:
                    vp = Path(cfg["vid_in"])
                    if not vp.is_file():
                        raise FileNotFoundError("Select a video file first.")
                    ff_custom = Path(cfg["ffmpeg"]) if cfg["ffmpeg"] else None
                    ff_bin, fp_bin = find_ffmpeg_binaries(ff_custom)
                    info = probe_video(vp, ff_bin, fp_bin)
                    sample_rgb = extract_single_video_frame(vp, max(0, info.total_frames // 2), ff_bin, info)
                    out_dir = Path(cfg["vid_out"])
                else:
                    ip = Path(cfg["img_in"])
                    if cfg["source_type"] == "folder":
                        imgs = sorted(p for p in ip.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS) if ip.is_dir() else []
                        if not imgs:
                            raise FileNotFoundError(f"No images in {ip}")
                        ip = imgs[0]
                    elif not ip.is_file():
                        raise FileNotFoundError(f"Not found: {ip}")
                    sample_rgb, _ = load_image_rgb(ip)
                    out_dir = Path(cfg["img_out"])

                out_dir.mkdir(parents=True, exist_ok=True)
                preview_path = out_dir / "preview_before_after.png"
                render_preview_comparison(sample_rgb, enhancer, preview_path, crop_size=256)
                self._log(f"Preview [{enhancer.backend_label}] -> {preview_path}")
                self._set_status("Preview ready")
                self.root.after(0, lambda p=preview_path: self._open_path(p))
            except Exception as exc:
                self._log(f"ERROR: {exc}")
                self._set_status("Idle")
                self.root.after(0, lambda e=str(exc): messagebox.showerror("Error", e))
            finally:
                self._finish_job()

        threading.Thread(target=_worker, daemon=True).start()

    def _start_processing(self) -> None:
        if self.is_running:
            return
        cfg = self._snapshot_settings()
        if cfg["is_video"]:
            self._start_video_job(cfg)
        else:
            self._start_photos_job(cfg)

    def _start_photos_job(self, cfg: dict) -> None:
        in_path = Path(cfg["img_in"])
        out_dir = Path(cfg["img_out"])
        if cfg["source_type"] == "folder":
            if not in_path.is_dir():
                messagebox.showerror("Error", f"Folder not found: {in_path}")
                return
            same_dir = in_path.resolve() == out_dir.resolve()
            files = sorted(
                p for p in in_path.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                and not (same_dir and _PROCESSED_RE.search(p.stem))
            )
            if not files:
                messagebox.showinfo("Empty", f"No images in {in_path}")
                return
        else:
            if not in_path.is_file():
                messagebox.showerror("Error", f"File not found: {in_path}")
                return
            files = [in_path]

        self.is_running = True
        self.stop_event.clear()
        self.start_btn.configure(state="disabled", text="Working...")
        self.stop_btn.configure(state="normal")
        self.preview_btn.configure(state="disabled")
        self.progress["maximum"] = len(files)
        self.progress["value"] = 0

        def _worker() -> None:
            ok_count = 0
            fail_count = 0
            try:
                enhancer = self._build_enhancer(cfg)
                self._log(f"[Photos] {enhancer.backend_label} (tile={enhancer.tile_size})")
                out_dir.mkdir(parents=True, exist_ok=True)
                used_names: set[str] = set()

                for idx, img_file in enumerate(files, start=1):
                    if self.stop_event.is_set():
                        self._log("Stopped.")
                        break
                    self._set_status(f"{idx}/{len(files)}: {img_file.name}")
                    try:
                        orig_rgb, alpha = load_image_rgb(img_file)
                        out_rgb = enhancer.enhance_frame(orig_rgb, grain_seed=idx, stop_event=self.stop_event)
                        base_name = f"{img_file.stem}_{enhancer.mode}_{enhancer.scale}x"
                        out_name = f"{base_name}.png"
                        if out_name.lower() in used_names:
                            out_name = f"{base_name}_{img_file.suffix.lstrip('.')}.png"
                        used_names.add(out_name.lower())
                        out_file = out_dir / out_name
                        save_image_rgb(out_file, out_rgb, alpha=alpha)
                        ok_count += 1
                        self._log(f"  [{idx}/{len(files)}] Done -> {out_file.name} [{out_rgb.shape[1]}x{out_rgb.shape[0]}]")
                    except CancelledError:
                        self._log("Stopped mid-image.")
                        break
                    except Exception as exc:
                        # H3: Continue batch on corrupt/unreadable file
                        fail_count += 1
                        self._log(f"  [{idx}/{len(files)}] FAILED ({img_file.name}): {exc}")
                    self.root.after(0, lambda v=idx: self.progress.configure(value=v))

                if not self.stop_event.is_set():
                    summary = f"{ok_count} done, {fail_count} failed" if fail_count else f"Processed {ok_count} image(s)"
                    self._set_status(summary)
                    self.root.after(0, lambda s=summary: messagebox.showinfo("Done", s))
            except Exception as exc:
                self._log(f"ERROR: {exc}")
                self.root.after(0, lambda e=str(exc): messagebox.showerror("Error", e))
            finally:
                self._finish_job()

        threading.Thread(target=_worker, daemon=True).start()

    def _start_video_job(self, cfg: dict) -> None:
        vid_path = Path(cfg["vid_in"])
        out_dir = Path(cfg["vid_out"])
        if not vid_path.is_file():
            messagebox.showerror("Error", "Select a video file.")
            return

        ff_custom = Path(cfg["ffmpeg"]) if cfg["ffmpeg"] else None
        try:
            ff_bin, fp_bin = find_ffmpeg_binaries(ff_custom)
        except FileNotFoundError as exc:
            messagebox.showerror("FFmpeg", str(exc))
            return

        ext = ".mov" if cfg["codec"] == "prores" else ".mp4"
        out_file = out_dir / f"{vid_path.stem}_{cfg['mode']}_{cfg['scale']}x{ext}"

        self.is_running = True
        self.stop_event.clear()
        self.start_btn.configure(state="disabled", text="Working...")
        self.stop_btn.configure(state="normal")
        self.preview_btn.configure(state="disabled")
        self.progress["value"] = 0

        def _worker() -> None:
            try:
                enhancer = self._build_enhancer(cfg)
                self._log(f"[Video] {vid_path.name} | {enhancer.backend_label} (tile={enhancer.tile_size})")

                def _prog(cur: int, total: int, fps_speed: float) -> None:
                    eta = (total - cur) / max(0.01, fps_speed) if total > cur else 0.0
                    self.root.after(0, lambda: self.progress.configure(maximum=max(1, total), value=cur))
                    self._set_status(f"Frame {cur}/{total} ({(cur / max(1, total)) * 100:.1f}%) | {fps_speed:.2f} fps | ETA {eta:.0f}s")

                res = enhance_video(
                    input_path=vid_path,
                    output_path=out_file,
                    enhancer=enhancer,
                    ffmpeg_path=ff_bin,
                    ffprobe_path=fp_bin,
                    anti_flicker=cfg["anti_flicker"],
                    codec=cfg["codec"],
                    crf=cfg["crf"],
                    keep_audio=cfg["keep_audio"],
                    resume=cfg["resume"],
                    stop_event=self.stop_event,
                    progress_cb=_prog,
                    log_cb=self._log,
                )
                if res["status"] == "paused":
                    self._log(f"Paused at frame {res['completed_frames']}/{res['total_frames']}")
                    self._set_status(f"Paused ({res['completed_frames']}/{res['total_frames']})")
                else:
                    self._log(f"Done ({res['elapsed_sec']}s) -> {out_file.name} [{res['input_resolution']} -> {res['output_resolution']}]")
                    self._set_status("Done")
                    self.root.after(0, lambda: messagebox.showinfo("Done", f"Saved: {out_file}"))
            except Exception as exc:
                self._log(f"ERROR: {exc}")
                self.root.after(0, lambda e=str(exc): messagebox.showerror("Error", e))
            finally:
                self._finish_job()

        threading.Thread(target=_worker, daemon=True).start()

    def _finish_job(self) -> None:
        self.is_running = False
        self.root.after(0, lambda: (
            self.start_btn.configure(state="normal", text="START"),
            self.stop_btn.configure(state="disabled"),
            self.preview_btn.configure(state="normal"),
        ))


def main() -> None:
    root = tk.Tk()
    LuminaGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
