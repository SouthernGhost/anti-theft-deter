import os
import time

import tkinter as tk
from tkinter import filedialog
from typing import Dict, List
import cv2
import sounddevice as sd
import soundfile as sf

from utils.gui import Button, Textbox
from utils.monitor import BathroomMonitor


def get_vid_file_path(button:Button, textbox:Textbox, config:dict) -> None:
    """Open a dialog and select a video file."""
    button.button.config(state='disabled')
    try:
        vid_path = filedialog.askopenfilename(
            title='Select a Video File',
            filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
        )
        textbox.textbox.delete(0, tk.END)
        if vid_path:
            textbox.textbox.insert(0, vid_path)
    finally:
        config['video_source'] = vid_path
        button.button.config(state='normal')


def play_video_on_canvas() -> None:
    """Play a video inside a Tkinter canvas (to be implemented)."""
    pass


def test_video_source(path: str, button: "Button") -> None:
    """Open a window and test a video source with a quit option."""
    cap = cv2.VideoCapture(path)
    button.button.config(state='disabled')
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, "Press Q to exit",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2)
            cv2.imshow("Video Source Test", frame)
            if cv2.waitKey(33) & 0xFF in (ord('q'), ord('Q')):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        button.button.config(state='normal')


def draw_roi() -> Dict:
    """Draw and define the region of interest (to be implemented)."""
    return {}


def get_audio_devices() -> Dict[str, int]:
    """Return a dictionary of available audio output devices."""
    filter_words = {
        "mapper", "virtual", "mix", "cable",
        "loopback", "digital", "stream",
        "driver", "microphone", "conexant"
    }
    return {
        device['name']: idx
        for idx, device in enumerate(sd.query_devices())
        if device['max_output_channels'] > 0
        and not any(word in device['name'].lower() for word in filter_words)
    }


def get_audio_devices_names(devices: Dict[str, int]) -> List[str]:
    """Return the list of device names from the device dict."""
    return list(devices)


def get_audio_device_index(devices: Dict[str, int], sel_device: str) -> int:
    """Return the index of the selected audio device."""
    return devices[sel_device]


def test_audio_device(devices: Dict[str, int], sel_device: str, button: "Button") -> None:
    """Play a test sound on the selected audio output device."""
    button.button.config(state='disabled')
    try:
        file = 'audio/speech1.wav'
        data, samplerate = sf.read(file)
        sd.play(data, samplerate, device=devices[sel_device])
        sd.wait()
    except Exception:
        pass
    finally:
        button.button.config(state='normal')


def on_checkbox_click(config:dict, key:bool):
    config[key] = not config[key]


def start_app(CONFIG, button:Button):
    """Main function to run the bathroom monitoring system"""

    button.button.config(state='disabled')
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
        button.button.config(state='normal')
        monitor.stop()
        print("✅ System stopped successfully")