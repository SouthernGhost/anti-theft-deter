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
        "audio_device": None,
        "stream_mode": False,
        "video_source": "",
        "ip_camera_url": "",
        "bathroom_zone": {"x1": 0.01, "y1": 0.5, "x2": 0.99, "y2": 0.99},
        "window_size": [1280, 720],
        "show_stats": False,
        "stats_scale_factor": False,
        "annotations": {"bathroom_zone": False, "persons": False, "items": False, "show_fps": False},
        "merchandise_classes": COCO_ITEMS,
        "person_classes": [0],
        "max_reconnect_attempts": 10,
        "reconnect_delay": 5,
        "detection_frequency": 0.1,
        "confidence_threshold": 0.25,
        "max_detections": 50,
        "imgsz": 416,
        "abandoned_timeout_seconds": 5,
        "association_overlap_threshold": 0.3,
        "association_min_duration_seconds": 0.0,
        "suppress_alarm_for_unassociated_items": True,
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


import os
import requests
from pathlib import Path

def _get_audio_file():
    path = Path.home() / "BathroomMonitor/audio/"
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, 'speech1.wav')
    repo_url = 'https://raw.githubusercontent.com/SouthernGhost/anti-theft-deter/main/audio/speech1.wav'

    if not os.path.isfile(file_path):
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        print(f"Downloading {repo_url}...")
        response = requests.get(repo_url, headers=headers)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Saved {file_path}")
        else:
            print(f"Failed to download {file_path}. Status code: {response.status_code}")
    else:
        print(f"{file_path} already exists. Skipping.")

    return file_path


_ensure_settings_file()
CONFIG = _load_settings()


