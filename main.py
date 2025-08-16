from utils import gui
from utils.config import _ensure_settings_file, _load_settings


_ensure_settings_file()
CONFIG = _load_settings()

def test_video_source(path:str):
    import cv2

    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video Source Test", frame)
        cv2.waitKey(33)
    cv2.destroyAllWindows()
    cap.release()
        

win_home = gui.Window("Bathroom Monitor", (480,480), resizable=True)

#Video source GUI elements
lbl_source = gui.Label(win_home, "Path to video or camera IP address", (10,20))
txt_source = gui.Textbox(win_home, (10,50), 50)
#Calculate button x position based on textbox's width and font character width
btn_browse = gui.Button(win_home, "Browse", ((10+(50*6)+10),50))
btn_test_source = gui.Button(win_home, "Test", ((10+(50*6)+80),50), cmd=lambda: test_video_source(txt_source.txtbox.get()))
btn_test_source.button.config(width=5)

chk_fps = gui.Checkbox(win_home, "Show fps", (10, 110))
chk_bathroom_zone = gui.Checkbox(win_home, "Show bathroom zone", (10,130))
chk_person_bbox = gui.Checkbox(win_home, "Show person bounding boxes", (10,150))
chk_items_bbox = gui.Checkbox(win_home, "Show items bounding boxes", (10,170))


def start_app():
    win_home.launch_window()

if __name__ == "__main__":
    #start_app(CONFIG)
    start_app()