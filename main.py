import threading

from utils import gui
from utils.app import get_audio_devices_names, get_audio_devices, test_audio_device
from utils.app import get_vid_file_path, test_video_source
from utils.config import _ensure_settings_file, _load_settings


_ensure_settings_file()
CONFIG = _load_settings()



win_home = gui.Window("Bathroom Monitor", (640,480), resizable=False)

#Video source GUI elements
lbl_source = gui.Label(tk_win=win_home, 
                        text="Path to video or camera IP address", 
                        pos=(10,20))

txt_source = gui.Textbox(tk_win=win_home, 
                            pos=(10,50), 
                            width=50, 
                            default_text=CONFIG['video_source'])


#Calculate button x position based on textbox's width and font character width
btn_browse = gui.Button(tk_win=win_home, 
                        text="Browse", 
                        pos=((10+(50*6)+10),50),
                        cmd=lambda: threading.Thread(target=get_vid_file_path,
                                                        args=(btn_browse,txt_source),
                                                        daemon=True).start())

btn_test_source = gui.Button(tk_win=win_home, 
                                text="Test", 
                                pos=((10+(50*6)+80),50), 
                                cmd=lambda: threading.Thread(target=test_video_source,
                                                                args=(txt_source.textbox.get(),
                                                                        btn_test_source),
                                                                daemon=True).start())

audio_devices = get_audio_devices() 
lbl_audio_devices = gui.Label(tk_win=win_home, 
                                text="Select audio output device: ", 
                                pos=(10,100))

combo_audio_devices_pos = ((10+7*(len(lbl_audio_devices.text))+10),100)
combo_audio_devices = gui.Combobox(tk_win=win_home, 
                                    pos=combo_audio_devices_pos, 
                                    options=get_audio_devices_names(audio_devices),
                                    width=30)

btn_test_audio = gui.Button(tk_win=win_home, 
                            text="Test", 
                            pos=(520,100),
                            cmd=lambda:threading.Thread(target=test_audio_device,
                                                        args=(audio_devices, 
                                                                combo_audio_devices.var.get(),
                                                                btn_test_audio),
                                                        daemon=True).start())

chk_fps = gui.Checkbox(tk_win=win_home, 
                        text="Show fps", 
                        pos=(10, 180), 
                        state=CONFIG['annotations']['show_fps'])

chk_bathroom_zone = gui.Checkbox(tk_win=win_home, 
                                    text="Show bathroom zone", 
                                    pos=(10,200), 
                                    state=CONFIG['annotations']['bathroom_zone'])

chk_person_bbox = gui.Checkbox(tk_win=win_home, 
                                text="Show person bounding boxes", 
                                pos=(10,220))

chk_items_bbox = gui.Checkbox(tk_win=win_home, 
                                text="Show items bounding boxes", 
                                pos=(10,240))


def start_app():
    win_home.launch_window()

if __name__ == "__main__":
    #start_app(CONFIG)
    start_app()