import os
import time
from typing import Optional, Callable

import tkinter as tk
from tkinter import ttk
import pyaudio

from .monitor import BathroomMonitor


DEFAULT_FONT = {'Times New Roman': 10}

class Window:
    """Create a Tkinter window.
        Args:
            title(str): Title of the window.
            size(str)
    """
    def __init__(self, title:str, size:tuple[int,int], resizable: Optional[bool]=False):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("x".join(map(str,size)))
        if resizable==True:
            self.root.resizable(width=True, height=True)
        else: self.root.resizable(False, False)

    def launch_window(self):
        self.root.mainloop()

class Button:
    def __init__(self, tk_win:Window, text:str, pos:tuple[int,int], cmd:Callable=None):
        self.button = tk.Button(tk_win.root, text=text, command=cmd)
        self.button.place(x=pos[0], y=pos[1])

class Textbox:
    def __init__(self, tk_win:Window, pos:tuple[int,int], width:int, default_text: Optional[str]=None):
        self.txtbox = tk.Entry(tk_win.root, width=width)
        self.txtbox.pack(padx=10, pady=5)
        if default_text:
            self.txtbox.insert(0, default_text)
        self.txtbox.place(x=pos[0], y=pos[1])

class Label:
    def __init__(self, tk_win:Window, text:str, pos:tuple[int,int]):
        self.label = tk.Label(tk_win.root, text=text)
        self.label.place(x=pos[0], y=pos[1])
        self.label.config(font=DEFAULT_FONT)

class Checkbox:
    def __init__(self, tk_win:Window, text:str, pos:tuple[int,int], cmd: Callable=None):
        self.var = tk.BooleanVar()
        self.checkbox = tk.Checkbutton(tk_win.root, text=text, variable=self.var, command=cmd)
        self.checkbox.pack(pady=20)
        self.checkbox.place(x=pos[0], y=pos[1])

class Combobox:
    def __init__(self, tk_win:Window, pos:tuple[int,int], options:list):
        self.var = tk.StringVar()
        self.combobox = ttk.Combobox(tk_win.root, self.var, options)
        self.combobox.pack(pady=20)
        self.combobox.current(0)
        self.combobox.place(x=pos[0], y=pos[1])
        self.combobox.config(font=DEFAULT_FONT)



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