import threading

from utils import gui
from utils.app import get_audio_device_index, get_audio_devices_names, get_audio_devices, test_audio_device
from utils.app import get_vid_file_path, test_video_source
from utils.app import on_checkbox_click, create_roi
from utils.app import start_app
from utils.config import _ensure_settings_file, _load_settings, _save_settings, _get_asset_files


_get_asset_files()
_ensure_settings_file()
CONFIG = _load_settings()

def on_combo_select(event):
    CONFIG['audio_device'] = get_audio_device_index(audio_devices,
                                                    combo_audio_devices.combobox.get()
                                                    )
    return

win_home = gui.Window("Bathroom Monitor", (640,480), resizable=False)

#Video source GUI elements
lbl_source = gui.Label(tk_win=win_home, 
                        text="Path to video or camera IP address", 
                        pos=(10,20))

txt_source = gui.Textbox(tk_win=win_home, 
                            pos=(10,50), 
                            width=50, 
                        )

#Calculate button x position based on textbox's width and font character width
btn_browse = gui.Button(tk_win=win_home, 
                        text="Browse", 
                        pos=((10+(50*6)+10),50),
                        cmd=lambda: threading.Thread(target=get_vid_file_path,
                                                        args=(btn_browse,txt_source,CONFIG),
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

combo_audio_devices.combobox.bind("<<ComboboxSelected>>", on_combo_select)

btn_test_audio = gui.Button(tk_win=win_home, 
                            text="Test", 
                            pos=(520,100),
                            cmd=lambda:threading.Thread(target=test_audio_device,
                                                        args=(audio_devices, 
                                                                combo_audio_devices.var.get(),
                                                                btn_test_audio,
                                                                CONFIG['audio_file']),
                                                        daemon=True).start())

btn_create_roi = gui.Button(tk_win=win_home,
                            text="Define ROI",
                            pos=(10,140),
                            cmd=lambda:threading.Thread(target=create_roi,
                                                        args=(txt_source.textbox.get(),
                                                                CONFIG['bathroom_zone'],
                                                                btn_create_roi,
                                                                win_home),
                                                                daemon=True).start())

chk_fps = gui.Checkbox(tk_win=win_home, 
                        text="Show fps", 
                        pos=(10, 180), 
                        state=CONFIG['annotations']['show_fps'],
                        cmd=lambda:threading.Thread(target=on_checkbox_click,
                                                args=(CONFIG["annotations"], 'show_fps'),
                                                daemon=True).start())

chk_bathroom_zone = gui.Checkbox(tk_win=win_home, 
                                    text="Show bathroom zone", 
                                    pos=(10,200), 
                                    state=CONFIG['annotations']['bathroom_zone'],
                                    cmd=lambda:threading.Thread(target=on_checkbox_click,
                                                args=(CONFIG["annotations"], 'bathroom_zone'),
                                                daemon=True).start())

chk_person_bbox = gui.Checkbox(tk_win=win_home, 
                                text="Show person bounding boxes", 
                                pos=(10,220),
                                state=CONFIG['annotations']['persons'],
                                cmd=lambda:threading.Thread(target=on_checkbox_click,
                                                args=(CONFIG["annotations"], 'persons'),
                                                daemon=True).start())

chk_items_bbox = gui.Checkbox(tk_win=win_home, 
                                text="Show items bounding boxes", 
                                pos=(10,240),
                                state=CONFIG['annotations']['items'],
                                cmd=lambda:threading.Thread(target=on_checkbox_click,
                                                args=(CONFIG["annotations"], 'items'),
                                                daemon=True).start())

btn_save = gui.Button(tk_win=win_home,
                        text="Save",
                        pos=(400,240),
                        cmd=lambda:threading.Thread(target=_save_settings,
                                                    args=(CONFIG,),
                                                    daemon=True).start())

btn_start = gui.Button(tk_win=win_home,
                        text="Start",
                        pos=(460,240),
                        cmd=lambda:threading.Thread(target=start_app,
                                                    args=(CONFIG, disbale_buttons, txt_source),
                                                    daemon=True).start())

disbale_buttons = [btn_browse, btn_create_roi, btn_test_audio, btn_test_source, btn_save, btn_start]

btn_quit = gui.Button(tk_win=win_home,
                        text="Quit",
                        pos=(520,240),
                        cmd=lambda:win_home.root.quit())
#def start_app():
#    win_home.launch_window()

if __name__ == "__main__":
    #start_app(CONFIG)
    win_home.launch_window()
    #start_app(CONFIG)