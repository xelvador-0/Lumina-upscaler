# Automated test suite
from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.config import load_presets
from core.hardware import recommend_tile_size
from core.pipeline_image import load_image_rgb
from core.postprocess import EnhancementParams, postprocess_image, preprocess_image
from core.tiling import CancelledError, SmartTiler
from video.ffmpeg_pipe import _extract_rotation


class TestLuminaCore(unittest.TestCase):
    def test_smart_tiler_seam_accuracy(self) -> None:
        img = np.zeros((170, 210, 3), dtype=np.float32)
        cv2.circle(img, (105, 85), 45, (0.8, 0.4, 0.2), -1)
        img = cv2.GaussianBlur(img, (5, 5), 1.0)

        tiler = SmartTiler(tile_size=96, overlap=16, scale=2)
        tiled = tiler.process(img, lambda t: cv2.resize(t, (t.shape[1] * 2, t.shape[0] * 2), interpolation=cv2.INTER_LINEAR))
        full = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
        self.assertLess(float(np.max(np.abs(tiled - full))), 0.01)

    def test_smart_tiler_cancellation(self) -> None:
        ev = threading.Event()
        ev.set()
        tiler = SmartTiler(tile_size=64, overlap=8, scale=2)
        with self.assertRaises(CancelledError):
            tiler.process(np.zeros((128, 128, 3), dtype=np.float32), lambda t: t, stop_event=ev)

    def test_fast_path_identity(self) -> None:
        arr = np.full((64, 64, 3), 0.4, dtype=np.float32)
        p = EnhancementParams()
        self.assertIs(preprocess_image(arr, p), arr)
        out = postprocess_image(arr, arr, p)
        self.assertEqual(float(np.max(np.abs(out - arr))), 0.0)

    def test_16bit_png_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test16.png"
            cv2.imwrite(str(p), np.full((32, 32, 3), 32768, dtype=np.uint16))
            rgb, _ = load_image_rgb(p)
            self.assertAlmostEqual(float(np.mean(rgb)), 0.5, places=2)

    def test_vram_tile_recommendations(self) -> None:
        self.assertLessEqual(recommend_tile_size(1024, 4), 128)
        self.assertGreaterEqual(recommend_tile_size(8192, 2), 384)
        self.assertEqual(recommend_tile_size(16384, 4), 512)

    def test_rotation_extraction(self) -> None:
        self.assertEqual(_extract_rotation({"side_data_list": [{"rotation": -90}]}), -90)
        self.assertEqual(_extract_rotation({"tags": {"rotate": "90"}}), 90)
        self.assertEqual(_extract_rotation({}), 0)

    def test_presets_loaded(self) -> None:
        presets = load_presets()
        self.assertIn("Pure AI", presets)
        self.assertIn("Anime", presets)

    def test_cli_clean_error(self) -> None:
        cli_path = Path(__file__).resolve().parent.parent / "cli.py"
        r = subprocess.run([sys.executable, str(cli_path), "-i", "nonexistent_file.png", "-o", "out.png"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()
