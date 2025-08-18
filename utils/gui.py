import os
import time
from typing import Optional, Callable

import tkinter as tk
from tkinter import ttk
import pyaudio


DEFAULT_FONT = {'Times New Roman': 10}


class Window:
    """Create a Tkinter window.
        Args:
            title(str): Title of the window.
            size(tuple)
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


class Canvas:
    """Create a Tkinter canvas.
        Args:
            Tkinter Window(Window.root): Window class object.
            size(tuple)
    """
    def __init__(self, tk_win:Window, size:tuple[int,int], expand:Optional[bool]=False):
        self.canvas = tk.Canvas(tk_win.root, width=size[0], height=size[1])
        if expand:
            self.canvas.pack(fill='both', expand=True)
        else:
            self.canvas.pack(fill='both', expand=False)


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
    def __init__(self, tk_win:Window, text:str, pos:tuple[int,int], state:Optional[bool]=False, cmd:Callable=None):
        self.var = tk.BooleanVar(value=state)
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





