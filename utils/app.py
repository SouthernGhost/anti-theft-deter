import tkinter as tk
from tkinter import filedialog
from typing import Dict, List
import cv2
import sounddevice as sd
import soundfile as sf

from utils.gui import Button, Textbox

def get_vid_file_path(button:Button, textbox:Textbox) -> None:
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
