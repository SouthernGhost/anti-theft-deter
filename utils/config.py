import json
import os
from pathlib import Path
import requests
from typing import final

# Project root (utils/ is one level below root)
ROOT_DIR = Path(__file__).resolve().parents[1]
SETTINGS_FILE = (Path.home() / 'BathroomMonitor/settings.json')
MODEL_PATH = str(Path.home() / "BathroomMonitor/yolo11n.pt")
AUDIO_FILE = str(Path.home()/ "BathroomMonitor/audio/speech1.wav")
IMAGES_FOLDER = str(Path.home() / "Pictures/Merchandise in Zone/")
LOGS_FOLDER = str(Path.home() / "BathroomMonitor/logs/alerts.log")

COCO_ITEMS = [24,25,26,27,28,39,40,41,42,43,44,45,63,64,65,66,67,73,74,76,77,78,79]
# Load template from existing root config.py as requested
try:
    # These provide the default/template configuration
    from config import CONFIG as CONFIG_TEMPLATE  # type: ignore
except Exception:
    # Minimal fallback template if root config.py is not available
    CONFIG_TEMPLATE = {
        "model_path": MODEL_PATH,
        "audio_file": AUDIO_FILE,
        "audio_device": 0,
        "stream_mode": False,
        "video_source": "",
        "ip_camera_url": "",
        "bathroom_zone": {"x1": 0.01, "y1": 0.5, "x2": 0.99, "y2": 0.99},
        "window_size": [1280, 720],
        "show_stats": False,
        "stats_scale_factor": False,
        "annotations": {"bathroom_zone": False, "persons": False, "items": False, "show_fps": False},
        "merchandise_classes": [1],
        "person_classes": [],
        "max_reconnect_attempts": 10,
        "reconnect_delay": 5,
        "detection_frequency": 0.1,
        "confidence_threshold": 0.5,
        "max_detections": 6,
        "imgsz": 416,
        "abandoned_timeout_seconds": 5,
        "association_overlap_threshold": 0.3,
        "association_min_duration_seconds": 0.0,
        "suppress_alarm_for_unassociated_items": False,
        "min_announcement_interval": 0.001,
        "person_annotation_threshold": 0.3,
        "person_association_threshold": 0.3,
        "log_file": LOGS_FOLDER,
        "images_folder": IMAGES_FOLDER
    }

def _create_settings_file():
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(CONFIG_TEMPLATE, f, indent=2)
    except Exception:
        pass


def _ensure_settings_file() -> None:
    """Create settings.json from template if it does not exist."""
    if not SETTINGS_FILE.exists():
        _create_settings_file()


def _load_settings() -> dict:
    """Load settings.json from project root."""
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception:
        # Fallback to template if file unreadable
        cfg = dict(CONFIG_TEMPLATE)

    return cfg


def _save_settings(config:dict):
    """Save settings."""
    if not SETTINGS_FILE.exists():
        _create_settings_file()
    else:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)


def _get_asset_files():
    base = Path.home() / "BathroomMonitor"
    audio = base / "audio/speech1.wav"
    model = base / "hands_detector.pt"  # Replace with your model filename
    urls = {
        audio: "https://raw.githubusercontent.com/SouthernGhost/anti-theft-deter/main/assets/speech1.wav",
        model: "https://raw.githubusercontent.com/SouthernGhost/anti-theft-deter/main/assets/hands_detector.pt"
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    for path, url in urls.items():
        if url and not path.exists():
            print('Downloading assets...')
            os.makedirs(path.parent, exist_ok=True)
            r = requests.get(url, headers=headers)
            if r.ok: open(path, "wb").write(r.content)
            else: print(f"Failed {url} ({r.status_code})")


