import os
import sys
import time

import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from typing import Dict, List
import sounddevice as sd
import soundfile as sf

from utils.gui import Button, Textbox, Window
from utils.monitor import BathroomMonitor


def create_roi(video_path, rect_dict, button:Button, parent:Window):
    """Draw ROI for monitoring."""

    button.button.config(state='disabled')
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video.")

        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Use Toplevel instead of Tk()
        win = tk.Toplevel(parent.root)
        win.title("Video with Rectangle Drawing")
        win.geometry(f"{vid_w}x{vid_h+25}")  # +25 for status bar

        canvas = tk.Canvas(win, width=vid_w, height=vid_h)
        canvas.pack()

        status = tk.StringVar()
        ttk.Label(win, textvariable=status).pack(fill="x")

        start = {"x": 0, "y": 0}
        rect_id = {"id": None}
        video_item = {"id": None}

        def canvas_to_video(x, y):
            return x, y

        def on_press(event):
            start["x"], start["y"] = event.x, event.y
            if rect_id["id"]:
                canvas.delete(rect_id["id"])
            rect_id["id"] = canvas.create_rectangle(
                start["x"], start["y"], start["x"], start["y"],
                outline="red", width=2
            )

        def on_drag(event):
            if rect_id["id"]:
                canvas.coords(rect_id["id"], start["x"], start["y"], event.x, event.y)

        def on_release(event):
            if rect_id["id"] is None:
                return
            vx1, vy1 = canvas_to_video(start["x"], start["y"])
            vx2, vy2 = canvas_to_video(event.x, event.y)

            if vx2 < vx1: vx1, vx2 = vx2, vx1
            if vy2 < vy1: vy1, vy2 = vy2, vy1

            norm_x1 = round(vx1 / vid_w, 3)
            norm_y1 = round(vy1 / vid_h, 3)
            norm_x2 = round(vx2 / vid_w, 3)
            norm_y2 = round(vy2 / vid_h, 3)

            rect_dict["x1"] = norm_x1
            rect_dict["y1"] = norm_y1
            rect_dict["x2"] = norm_x2
            rect_dict["y2"] = norm_y2

            print(f"Rectangle saved: {rect_dict}")
            status.set(f"Locked Rect | norm: ({norm_x1},{norm_y1})→({norm_x2},{norm_y2})")

            canvas.tag_raise(rect_id["id"])

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)

        def update_frame():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            if video_item["id"] is None:
                video_item["id"] = canvas.create_image(0, 0, anchor="nw", image=imgtk, tags="video")
            else:
                canvas.itemconfig(video_item["id"], image=imgtk)

            canvas.imgtk = imgtk
            win.after(15, update_frame)

        update_frame()
        win.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), win.destroy()))
    except Exception as e:
        print("ROI error:", e)
    finally:
        button.button.config(state='normal')


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


def test_audio_device(devices: Dict[str, int], sel_device: str, button: "Button", file:str) -> None:
    """Play a test sound on the selected audio output device."""
    button.button.config(state='disabled')
    try:
        file = file
        data, samplerate = sf.read(file)
        sd.play(data, samplerate, device=devices[sel_device])
        sd.wait()
    except Exception:
        pass
    finally:
        button.button.config(state='normal')


def on_checkbox_click(config:dict, key:bool):
    config[key] = not config[key]


def start_app(CONFIG, button:Button, disable:list):
    """Main function to run the bathroom monitoring system"""

    button.button.config(state='disabled')
    for btn in disable:
        btn.button.config(state='disabled')
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
        for btn in disable:
            btn.button.config(state='normal')
        monitor.stop()
        print("✅ System stopped successfully")