"""Utility functions for the Home GUI."""

from csv import Error
import os
import time
from typing import Dict, List

import tkinter as tk
from tkinter import filedialog

from .monitor import BathroomMonitor
from .gui import Button, Textbox


def get_vid_file_path(button:Button, textbox:Textbox):
    """Open a dialog and select a video file."""
    button.button.config(state='disabled')
    vid_path = filedialog.askopenfilename(title='Select a Video File',
                                            filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
    button.button.config(state='normal')
    textbox.textbox.delete(0, tk.END)
    textbox.textbox.insert(0, vid_path)
    return


def play_video_on_canvas():
    return


def test_video_source(path:str, button:Button):
    """Test the video source."""
    import cv2
    cap = cv2.VideoCapture(path)
    button.button.config(state='disabled')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, "Press Q to exit", 
                        (20,40), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255,255,255),
                        2
                    )
        cv2.imshow("Video Source Test", frame)
        if cv2.waitKey(33) & 0xFF in [ord('q'), ord('Q')]:
            break
    cv2.destroyAllWindows()
    cap.release()
    button.button.config(state='normal')
    return


def draw_roi() -> Dict:
    """Draw and define the region to monitor."""
    return


def get_audio_devices() -> Dict:
    import sounddevice as sd
    audio_devices = {}
    filter_words = ["mapper", "virtual", "mix", "cable", 
                        "loopback", "digital", "stream",
                        "driver", "microphone", "conexant"]
    for index, device in enumerate(sd.query_devices()):
        if device['max_output_channels']>0:
            if not any(word.lower() in device['name'].lower() for word in filter_words):
                audio_devices[device['name']] = index
    return audio_devices


def get_audio_devices_names(devices:dict) -> List:
    return list(devices.keys())


def get_audio_device_index(devices:dict, sel_device:str) -> List:
    return devices[sel_device]


def test_audio_device(devices:dict, sel_device:str, button:Button) -> None:
    """Function to test selected audio output device.
        Args:
            devices (dict): A dictionary of audio output devices.
            sel_device (str): Selected audio output device, taken from Combobox.var.
    """

    import sounddevice as sd
    import soundfile as sf
    button.button.config(state='disabled')
    index = devices[sel_device]
    file = 'audio/speech1.wav'
    try:
        data, samplerate = sf.read(file)
        sd.play(data, samplerate, device=index)
        sd.wait()
    except Exception as e:
        button.button.config(state='normal')
    button.button.config(state='normal')
    return


def start_app(CONFIG):
    """Main function to run the bathroom monitoring system"""

    print("🚀 Initializing Bathroom Monitor...")
    print("📋 Configuration:")
    print(f"   Model: {CONFIG['model_path']}")
    print(f"   Stream Mode: {'✅ Enabled' if CONFIG['stream_mode'] else '❌ Disabled'}")

    if CONFIG['stream_mode']:
        print(f"   Source: {CONFIG['ip_camera_url']} (IP Camera)")
    else:
        print(f"   Source: {CONFIG['video_source']} (Local)")

    zone = CONFIG['bathroom_zone']
    print(f"   Zone: ({zone['x1']:.2f}, {zone['y1']:.2f}) to ({zone['x2']:.2f}, {zone['y2']:.2f})")
    print(f"   Show Stats: {'✅ Yes' if CONFIG['show_stats'] else '❌ No'}")

    # Display annotation toggles
    annotations = CONFIG.get('annotations', {})
    print(f"   Annotation Toggles:")
    print(f"     Bathroom Zone: {'✅ Visible' if annotations.get('bathroom_zone', True) else '❌ Hidden'}")
    print(f"     Person Boxes: {'✅ Visible' if annotations.get('persons', True) else '❌ Hidden'}")
    print(f"     Item Boxes: {'✅ Visible' if annotations.get('items', True) else '❌ Hidden'}")



    # Create and start monitor
    try:
        os.makedirs(CONFIG['images_folder'], exist_ok=True)
        monitor = BathroomMonitor(CONFIG)
    except (ValueError, ConnectionError) as e:
        print(f"❌ Failed to initialize monitor: {e}")
        return

    try:
        print("🎯 Starting monitoring system...")
        monitor.start()

        print("✅ Monitoring system started successfully!")
        print("📋 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to show statistics")
        print("   - Press 'f' to toggle FPS overlay")
        print("   - Close video window to stop")

        if CONFIG['stream_mode']:
            print("🌐 Stream monitoring active - system will auto-reconnect if stream drops")

        # Keep main thread alive
        while monitor.running:
            time.sleep(CONFIG['detection_frequency'])

    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt received...")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        print("🛑 Stopping monitoring system...")
        monitor.stop()
        print("✅ System stopped successfully")
