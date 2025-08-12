import os

import tkinter
import pyaudio


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

print(get_audio_devices())