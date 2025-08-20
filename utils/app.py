"""Utility functions for the Home GUI."""

import os
import time
from typing import Dict, List

from .monitor import BathroomMonitor
from .gui import Button


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
    """Get a list of audio devices."""
    import pyaudio
    audio_devices = {}
    audio = pyaudio.PyAudio()
    filter_words = ["mapper", "virtual", "mix", "cable", 
                        "loopback", "digital", "stream",
                        "driver"]
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        lower_name = info['name'].lower()
        if info['maxOutputChannels']>=1:
            if not any(k in lower_name for k in filter_words):
                #audio_devices.append(info['name'])
                audio_devices[info['name']] = i-1
    audio.terminate()
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
    import pyaudio
    import wave
    button.button.config(state='disabled')
    index = devices[sel_device]
    file = 'audio/speech1.wav'
    wave_file = wave.open(file, 'rb')
    audio = pyaudio.PyAudio()
    stream = audio.open(format=audio.get_format_from_width(wave_file.getsampwidth()),
                        channels=wave_file.getnchannels(),
                        rate=wave_file.getframerate(),
                        output=True,
                        output_device_index=index)
    chunk = 1024
    data = wave_file.readframes(chunk)
    while data:
        stream.write(data)
        data = wave_file.readframes(chunk)
    stream.stop_stream()
    stream.close()
    wave_file.close()
    audio.terminate()
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


print(get_audio_devices())