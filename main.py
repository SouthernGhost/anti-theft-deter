from asyncio import ensure_future
import os
import time

from utils.gui import start_app
from utils.config import _ensure_settings_file, _load_settings

_ensure_settings_file()
CONFIG = _load_settings()

if __name__ == "__main__":
    start_app(CONFIG)