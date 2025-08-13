import os
import time

import tkinter
import pyaudio

from .monitor import BathroomMonitor


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
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt received...")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        print("🛑 Stopping monitoring system...")
        monitor.stop()
        print("✅ System stopped successfully")

def create_textbox():
    return
def create_button():
    return
def create_label():
    return
def get_audio_devices():
    audio_devices = []
    audio = pyaudio.PyAudio()
    filter_words = ["mapper", "virtual", "mix", "cable", 
                        "loopback", "digital", "stream",
                        "driver"]
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        lower_name = info['name'].lower()
        if info['maxOutputChannels']>=1:
            if not any(k in lower_name for k in filter_words):
                audio_devices.append(info['name'])
    return audio_devices

#print(get_audio_devices())