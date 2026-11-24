# Config & presets loader
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from core.paths import get_app_dir, get_bundle_dir

DEFAULT_PRESETS: Dict[str, Dict[str, float]] = {
    "Pure AI": {"noise": 0.0, "comp": 0.0, "details": 0.0, "sharpen": 0.0, "halo": 0.20, "luma": 0.0, "grain": 0.0, "flicker": 0.20},
    "Anime": {"noise": 0.0, "comp": 0.0, "details": 0.0, "sharpen": 0.10, "halo": 0.25, "luma": 0.0, "grain": 0.0, "flicker": 0.20},
    "Real": {"noise": 0.05, "comp": 0.0, "details": 0.25, "sharpen": 0.25, "halo": 0.20, "luma": 0.08, "grain": 0.012, "flicker": 0.25},
    "Decompress": {"noise": 0.15, "comp": 0.25, "details": 0.20, "sharpen": 0.20, "halo": 0.30, "luma": 0.05, "grain": 0.010, "flicker": 0.30},
}


def load_presets() -> Dict[str, Dict[str, float]]:
    for base in (get_app_dir(), get_bundle_dir()):
        p = base / "config" / "presets.json"
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data:
                    return data
            except Exception:
                pass
    return dict(DEFAULT_PRESETS)


def _settings_path() -> Path:
    return get_app_dir() / "config" / "settings.json"


def load_user_settings() -> Dict[str, Any]:
    p = _settings_path()
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_user_settings(data: Dict[str, Any]) -> None:
    p = _settings_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass
