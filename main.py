import os
import time

from utils.gui import start_app
from config import CONFIG

merch_ids = CONFIG["merchandise_classes"]
person_ids = CONFIG["person_classes"]
CONFIG['stream_mode'] = False

def main():
   return


if __name__ == "__main__":
    start_app(CONFIG)