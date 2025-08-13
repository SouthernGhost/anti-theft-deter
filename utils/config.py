import json
import os
from pathlib import Path

# Project root (utils/ is one level below root)
ROOT_DIR = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT_DIR / 'settings.json'

# Load template from existing root config.py as requested
try:
    # These provide the default/template configuration
    from config import CONFIG as CONFIG_TEMPLATE  # type: ignore
except Exception:
    # Minimal fallback template if root config.py is not available
    CONFIG_TEMPLATE = {
        "model_path": "yolo11n.pt",
        "stream_mode": False,
        "video_source": "videos/vid0h.mp4",
        "ip_camera_url": "http://127.0.0.1:8080/video",
        "bathroom_zone": {"x1": 0.01, "y1": 0.5, "x2": 0.99, "y2": 0.99},
        "window_size": [1280, 720],
        "annotations": {"bathroom_zone": True, "persons": True, "items": True, "show_fps": True},
        "merchandise_classes": [],
        "person_classes": [0],
        "max_reconnect_attempts": 10,
        "reconnect_delay": 5,
        "confidence_threshold": 0.25,
        "max_detections": 50,
        "imgsz": 640,
        "abandoned_timeout_seconds": 5,
        "association_overlap_threshold": 0.3,
        "association_min_duration_seconds": 0.0,
        "suppress_alarm_for_unassociated_items": True,
        "log_file": "logs/alerts.log",
        "images_folder": str((ROOT_DIR / 'imgs').resolve())
    }


def _ensure_settings_file() -> None:
    """Create settings.json from template if it does not exist."""
    if not SETTINGS_FILE.exists():
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(CONFIG_TEMPLATE, f, indent=2)
        except Exception:
            pass


def _load_settings() -> dict:
    """Load settings.json from project root."""
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception:
        # Fallback to template if file unreadable
        cfg = dict(CONFIG_TEMPLATE)

    return cfg


# Ensure and load configuration at import time
_ensure_settings_file()
CONFIG = _load_settings()


