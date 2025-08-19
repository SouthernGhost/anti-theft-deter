import os
import time
from typing import Optional, Callable

import tkinter as tk
from tkinter import ttk
import pyaudio


DEFAULT_FONT = {'Segoe UI': 7}


class Window:
    """Class to create a Tkinter window.
        Attributes:
            title (str): Title of the window.
            size (tuple): Size of the window.
    """

    def __init__(self, title:str, 
                    size:tuple[int,int], 
                    resizable: Optional[bool]=False
                ):
        """Initialize a tkinter window (Window class object).
        Args:
            title (str): Title of the window.
            size (tuple): Size of the window.
            resizable (bool, optional): Make the window resizable. False by default.
        """

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("x".join(map(str,size)))
        if resizable==True:
            self.root.resizable(width=True, height=True)
        else: self.root.resizable(False, False)

    def launch_window(self):
        """Launch a tkinter window.
        Args:
            None
        """

        self.root.mainloop()


class Canvas:
    """Class to create a Tkinter canvas.
    Attributes:
        tk_win (Window): Window the canvas is to be placed on.
        size (tuple): Size of the canvas.
    """

    def __init__(self, tk_win:Window, 
                    size:tuple[int,int], 
                    expand:Optional[bool]=False
                ):
        """Initialize a Canvas instance.
        Args:
            tk_win (Window): Window the canvas is to be placed on.
            size (tuple): Size of the canvas.
            expand (bool, optional): Expand the canvas with the window.
        """

        self.canvas = tk.Canvas(tk_win.root, width=size[0], height=size[1])
        if expand:
            self.canvas.pack(fill='both', expand=True)
        else:
            self.canvas.pack(fill='both', expand=False)


class Button:
    """Class to create a Tkinter button.
        Attributes:
            button (tkinter.Button): Tkinter Button object.
    """

    def __init__(self, tk_win:Window, 
                    text:str, 
                    pos:tuple[int,int], 
                    size:Optional[tuple[int,int]]=None, 
                    cmd:Optional[Callable]=None
                ):
        """Initialize a Button instance.
        Args:
            text (str): Text to display on button.
            pos (tuple[int,int]): Position of the button on the window.
            size (tuple[int,int], optional): Size of the button in width x height.
            cmd (callable, optional): Function to be called on pressing button.
        """

        self.button = tk.Button(tk_win.root, text=text, command=cmd)
        if size is not None:
            self.button.config(width=size[0], height=size[1])
        self.button.place(x=pos[0], y=pos[1])


class Textbox:
    """
        Class to create a Tkinter textbox.
        Attributes:
            textbox (tkinter.Entry): Tkinter Entry object for storing path to video source.
    """
    def __init__(self, tk_win:Window, 
                    pos:tuple[int,int], 
                    width:int, 
                    default_text: Optional[str]=None
                ):
        """Initialize a Textbox instance.
            Args:
                tk_win (Window): Window the textbox is to be created on.
                pos (tuple[int,int]): Postion of the textbox on the window.
                width (int): Width of the textbox.
                default_text (str, optional): Default text in the textbox.
        """
        self.textbox = tk.Entry(tk_win.root, width=width)
        self.textbox.pack(padx=10, pady=5)
        if default_text:
            self.textbox.insert(0, default_text)
        self.textbox.place(x=pos[0], y=pos[1])


class Label:
    """
        Class to create a Tkinter label.
        Attributes:
            text (str): Text to display as label.
            label (tkinter.Label): Tkinter Label object.
    """
    def __init__(self, tk_win:Window, 
                    text:str, 
                    pos:tuple[int,int]
                ):
        """Initialize a Label instance.
            Args:
                tk_win (Window): Window the label is to be placed on.
                text (str): Text to display as label.
                pos (tuple[int,int]): Position of the label on the window.
        """
        self.text = text
        self.label = tk.Label(tk_win.root, text=text)
        self.label.place(x=pos[0], y=pos[1])
        self.label.config(font=DEFAULT_FONT)


class Checkbox:
    """
        Create a checkbox.
        Attributes:
            var (tkinter.BooleanVar): Tkinter BooleanVar variable to store checkbox state.
            checkbox (tkinter.CheckButton): Tkinter CheckButton object.
    """

    def __init__(self, tk_win:Window, 
                    text:str, 
                    pos:tuple[int,int], 
                    state:Optional[bool]=False, 
                    cmd:Optional[Callable]=None
                ):
        """Initialize a Checkbox instance.
            Args:
            tk_win (Window): Window the checkbox is to be placed on.
            text (str): Text to display for checkbox.
            state (bool, optional): Default state of the checkbox.
            cmd (callable, optional): Function to call when checking the box.
        """

        self.var = tk.BooleanVar(value=state)
        self.checkbox = tk.Checkbutton(tk_win.root, text=text, variable=self.var, command=cmd)
        self.checkbox.pack(pady=20)
        self.checkbox.place(x=pos[0], y=pos[1])


class Combobox:
    """Class to create a Tkinter Combobox.
        Attributes:
            var (tkinter.StringVar): Tkinter string object to store selected item.
            combobox (tkinter.Combobox): Tkinter Combobox object.
    """
    def __init__(self, tk_win:Window, 
                    pos:tuple[int,int], 
                    options:list,
                    width:Optional[int]=None
                ):
        """Initialize a Combobox instance.
            Args:
                tk_win (Window): Window the combobox is to be placed on.
                pos (tuple[int,int]): Position of the combobox on the window.
                options (list): Items to display in the combobox.
                width (int, optional): Width of the combobox.
        """
        self.var = tk.StringVar(master=tk_win.root)
        self.combobox = ttk.Combobox(tk_win.root, textvariable=self.var, values=options)
        if width is not None:
            self.combobox.config(width=width)
        self.combobox.pack(pady=20)
        self.combobox.current(0)
        self.combobox.place(x=pos[0], y=pos[1])
        self.combobox.config(font=DEFAULT_FONT)





