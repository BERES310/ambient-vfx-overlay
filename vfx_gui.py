"""
vfx_gui.py

Desktop app for turning a photo into a looping "live wallpaper" video
(CustomTkinter, dark theme). Designed in the design notes / the design notes:

    Create      pick a photo -> pick a mood -> nudge three sliders -> the preview
                updates by itself -> "Make final video".
    My videos   everything already rendered, with the look that made it.
    Settings    first-run checks (graphics card, ffmpeg, AI models) and downloads.

Design notes:
    - live_wallpaper.process_image() reads every effect toggle straight off
      module-level `config.*` attributes, so the app sets those attributes
      (a dict of overrides) on a worker thread and restores them afterwards.
      gui_system.CONFIG_LOCK makes the live-draft worker and the final-render
      worker take turns, so they can never see each other's overrides.
    - The live draft (fast_preview.py: small, short, no AI passes) re-renders
      ~300 ms after the last change, latest change wins. It pauses while a
      final render runs (both use the same global settings).
    - A final render can be stopped and continued later: the engine keeps a
      checkpoint (render_checkpoint.py), and a small marker in gui_presets/
      lets the next launch offer "Resume".

Run (from inside vfx_overlay/):

    python vfx_gui.py
"""

import json
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageOps, ImageTk

import config
import live_wallpaper
from fast_preview import render_fast_preview
from vfx_engine import error_hints, gui_moods, gui_system, progress
from vfx_engine.effect_params import EFFECT_INTENSITY, EFFECT_SWITCHES, current_intensity_default
from vfx_engine.gui_widgets import EmbeddedPreview, Tooltip, probe_video
from vfx_engine.image_io import find_input_images

try:  # optional: drag a photo onto the window. Without it the Add photo button still works.
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _HAS_DND = True
except ImportError:
    _HAS_DND = False

APPEARANCE_CHOICES = {"Dark": "dark", "Light": "light", "Match Windows": "system"}


def _saved_appearance():
    """The appearance label saved by the last session (read before any window exists)."""
    try:
        with open(gui_system.GUI_PRESETS_DIR / "last_session.json", "r", encoding="utf-8") as f:
            label = json.load(f).get("appearance")
        return label if label in APPEARANCE_CHOICES else "Dark"
    except (OSError, ValueError, AttributeError):
        return "Dark"


ctk.set_appearance_mode(APPEARANCE_CHOICES[_saved_appearance()])
ctk.set_default_color_theme("blue")

GUI_PRESETS_DIR = gui_system.GUI_PRESETS_DIR
SESSION_PATH = GUI_PRESETS_DIR / "last_session.json"

# Visual language: near-black chrome, one accent colour used only
# for the primary action and the active mood; three type sizes; 8-px spacing.
# Colours are (light, dark) pairs so the Light / Dark / Match Windows setting re-themes everything.
BG = ("#f3f4f8", "#101014")
PANEL = ("#e6e8ef", "#17171d")
CARD = ("#d5d9e5", "#20202a")
ACCENT = "#4c8dff"
ACCENT_HOVER = "#6aa0ff"
OK = ("#1c6e30", "#7fd08a")
WARN = ("#8a5200", "#e0b060")
MUTED = ("gray30", "gray75")
TEXT = ("gray10", "white")
FONT_SMALL, FONT_BODY, FONT_TITLE = 12, 14, 22
PAD = 8

_STAGES = [
    ("understand", "Understanding the photo"),
    ("weather", "Building the weather"),
    ("frames", "Rendering frames"),
    ("encoding", "Encoding the video"),
]
_STAGE_OF_EVENT = {"loading": "understand", "depth": "understand", "segmentation": "understand",
                   "weather": "weather", "frames": "frames", "encoding": "encoding"}

# One line + a cost note per effect, shown as tooltips.
EFFECT_HELP = {
    "RAIN_ENABLED": ("Falling rain streaks that pass behind objects in the photo.", "Moderate render cost"),
    "SNOW_ENABLED": ("Soft flakes drifting down, nearer ones larger.", "Moderate render cost"),
    "FOG_ENABLED": ("A drifting mist that thickens with distance.", "Adds roughly 10-20 % render time"),
    "LIGHTNING_ENABLED": ("Occasional bright flashes that light up the whole scene.", "Low render cost"),
    "FLICKER_ENABLED": ("A subtle brightness wobble, like light through moving clouds.", "Low render cost"),
    "GOD_RAYS_ENABLED": ("Beams of sunlight streaming from the brightest area.", "Adds roughly 10 % render time"),
}

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

if _HAS_DND:
    class _AppBase(ctk.CTk, TkinterDnD.DnDWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.TkdndVersion = TkinterDnD._require(self)
else:
    _AppBase = ctk.CTk


def _format_duration(seconds):
    """'2 h 05 min', '7 min 12 s' or '41 s' for a finished-render message."""
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def _video_facts(path):
    """(width, height, fps, seconds) of a video via OpenCV."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    finally:
        capture.release()
    return width, height, fps, (frames / fps if fps else 0)


def _photo_size(path):
    """Display size of a photo (EXIF rotation applied), without decoding all pixels."""
    with Image.open(path) as image:
        width, height = image.size
        try:
            orientation = image.getexif().get(274)
        except Exception:  # noqa: BLE001
            orientation = None
    if orientation in (5, 6, 7, 8):
        width, height = height, width
    return width, height


class _Button(ctk.CTkButton):
    """CTkButton whose outline-style (transparent) variant stays readable in the light theme."""

    def __init__(self, *args, **kwargs):
        if kwargs.get("fg_color") == "transparent":
            kwargs.setdefault("text_color", TEXT)
        super().__init__(*args, **kwargs)


class Collapsible(ctk.CTkFrame):
    """A titled section that folds open/closed (progressive disclosure)."""

    def __init__(self, parent, title, opened=False):
        super().__init__(parent, fg_color="transparent")
        self._title = title
        self._open = opened
        self.header = _Button(self, text="", anchor="w", fg_color="transparent", hover_color=CARD,
                                    text_color=TEXT, font=ctk.CTkFont(size=FONT_BODY, weight="bold"),
                                    height=32, command=self.toggle)
        self.header.pack(fill="x")
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self._refresh()

    def _refresh(self):
        self.header.configure(text=f"{'▾' if self._open else '▸'}  {self._title}")
        if self._open:
            self.body.pack(fill="x", padx=(PAD, 0))
        else:
            self.body.pack_forget()

    def toggle(self):
        self._open = not self._open
        self._refresh()


class VfxApp(_AppBase):
    def __init__(self):
        super().__init__()
        self.title("VFX Overlay - Live Wallpaper Studio")
        # Fit small laptop screens (1366x768) and high Windows scaling: never open bigger than the screen.
        scaling = ctk.ScalingTracker.get_window_scaling(self) or 1.0   # CustomTkinter multiplies sizes by the Windows scale
        screen_w, screen_h = int(self.winfo_screenwidth() / scaling), int(self.winfo_screenheight() / scaling)
        self.geometry(f"{min(1280, screen_w - 40)}x{min(800, screen_h - 90)}")
        self.minsize(min(1040, screen_w - 40), min(680, screen_h - 90))
        self._compact = screen_w < 1500
        self.configure(fg_color=BG)

        self.event_queue = queue.Queue()
        # All GPU work (drafts and final renders) runs on ONE long-lived thread:
        # the shared OpenGL context (vfx_engine/gpu.py) belongs to the thread that made it.
        self.job_queue = queue.Queue()
        threading.Thread(target=self._engine_loop, daemon=True).start()
        self.cancel_event = threading.Event()
        self.rendering = False
        self.draft_running = False
        self.draft_pending = False
        self.draft_generation = 0
        self._draft_after = None
        self._draft_seen_photos = set()
        self._draft_spmf = None          # measured seconds per megapixel-frame of the live draft
        self.last_output_path = None
        self.last_error_details = ""
        self.render_queue = []
        self._undo_stack, self._redo_stack, self._last_look, self._last_look_time = [], [], None, 0.0
        self.showing_original = False
        self._smoothed_eta = None
        self._render_started = 0.0
        self._render_meta = None
        self._resumed = False
        self._last_pending_write = 0.0
        self._loading_state = False
        self._photo_size_cache = {}
        self._thumb_cache = {}
        self._video_thumb_cache = {}
        self.photo_path = None
        self.mood_key = "rain"
        self.system = {"gpu": None, "ffmpeg": True, "missing_models": []}

        self.switch_vars = {}
        self.adv_vars = {}
        self.adv_touched = set()
        self.adv_labels = {}
        self.stage_labels = {}
        self.mood_buttons = {}

        self.intensity_var = tk.DoubleVar(value=0.6)
        self.wind_var = tk.DoubleVar(value=0.4)
        self.darkness_var = tk.DoubleVar(value=0.0)
        self.quality_var = tk.StringVar(value=gui_moods.DEFAULT_QUALITY)
        self.length_var = tk.StringVar(value=gui_moods.DEFAULT_LENGTH)
        self.fps_var = tk.StringVar(value=gui_moods.DEFAULT_FPS)
        self.advanced_var = tk.BooleanVar(value=False)
        self.limit_area_var = tk.BooleanVar(value=False)

        self._build_shell()
        self._build_create_tab()
        self._build_videos_tab()
        self._build_settings_tab()
        self._apply_mood(self.mood_key, schedule=False)
        self._refresh_photo_tiles()
        self._restore_session()
        self._select_default_photo()
        self._show_tab("Create")

        self.bind("<Control-r>", lambda _event: self._start_render())
        self.bind("<Escape>", lambda _event: self._stop_render())
        self.bind("<space>", self._on_space)
        self.bind("<Control-o>", lambda _event: self._add_photo())
        self.bind("<Control-z>", lambda _event: self._undo())
        self.bind("<Control-y>", lambda _event: self._redo())
        self.bind("<Control-Shift-Z>", lambda _event: self._redo())
        if _HAS_DND:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_queue)
        self.after(400, self._startup_checks)

    # ------------------------------------------------------------------
    # Shell: header + tabs
    # ------------------------------------------------------------------
    def _build_shell(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=56)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(header, text="Live Wallpaper Studio", font=ctk.CTkFont(size=FONT_TITLE, weight="bold")).grid(
            row=0, column=0, padx=16, pady=10, sticky="w")
        self.tab_switch = ctk.CTkSegmentedButton(header, values=["Create", "My videos", "Settings"],
                                                 command=self._show_tab, font=ctk.CTkFont(size=FONT_BODY))
        self.tab_switch.grid(row=0, column=1, pady=10)
        self.header_note = ctk.CTkLabel(header, text="", text_color=MUTED, font=ctk.CTkFont(size=FONT_SMALL))
        self.header_note.grid(row=0, column=2, padx=16, sticky="e")

        # Banner for first-run problems / an interrupted render (hidden until needed).
        self.banner = ctk.CTkFrame(self, fg_color="#2a2418", corner_radius=0)
        self.banner.grid(row=1, column=0, sticky="ew")
        self.banner.grid_remove()
        self.banner_label = ctk.CTkLabel(self.banner, text="", wraplength=900, justify="left", anchor="w",
                                         font=ctk.CTkFont(size=FONT_BODY))
        self.banner_label.pack(side="left", padx=16, pady=8)
        self.banner_buttons = ctk.CTkFrame(self.banner, fg_color="transparent")
        self.banner_buttons.pack(side="right", padx=16, pady=6)

        self.tabs = {}
        for name in ("Create", "My videos", "Settings"):
            frame = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
            frame.grid(row=2, column=0, sticky="nsew")
            self.tabs[name] = frame

    def _show_tab(self, name):
        self.tab_switch.set(name)
        self.tabs[name].tkraise()
        if name == "My videos":
            self._refresh_videos()
        elif name == "Settings":
            self._refresh_settings()

    def _show_banner(self, text, buttons, color="#2a2418"):
        for child in self.banner_buttons.winfo_children():
            child.destroy()
        self.banner.configure(fg_color=color)
        self.banner_label.configure(text=text)
        for label, command, primary in buttons:
            _Button(self.banner_buttons, text=label, command=command, width=110, height=32,
                          fg_color=ACCENT if primary else "transparent", border_width=0 if primary else 1,
                          hover_color=ACCENT_HOVER if primary else CARD).pack(side="left", padx=(PAD, 0))
        self.banner.grid()

    def _hide_banner(self):
        self.banner.grid_remove()

    # ------------------------------------------------------------------
    # Create tab
    # ------------------------------------------------------------------
    def _build_create_tab(self):
        tab = self.tabs["Create"]
        tab.grid_columnconfigure(0, weight=0)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(tab, width=340 if self._compact else 400, fg_color=PANEL, corner_radius=0)
        left.grid(row=0, column=0, sticky="nsw")
        self._build_controls(left)

        right = ctk.CTkFrame(tab, fg_color=BG, corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(right, bg=self._plain(BG), highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=16, pady=(16, 8))
        self.preview = EmbeddedPreview(self.canvas, on_error=lambda m: self._set_info(f"Preview problem: {m}"))
        self.canvas.bind("<Configure>", lambda _e: self._draw_canvas_message(), add="+")

        self.bottom = ctk.CTkFrame(right, fg_color=BG, corner_radius=0)
        self.bottom.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 16))
        self.bottom.grid_columnconfigure(0, weight=1)
        self._build_bottom()

    def _section_label(self, parent, text):
        ctk.CTkLabel(parent, text=text.upper(), font=ctk.CTkFont(size=FONT_SMALL, weight="bold"),
                     text_color=MUTED).pack(padx=PAD, pady=(16, 4), anchor="w")

    def _build_controls(self, body):
        # ---- 1. photo ----
        self._section_label(body, "1 · Your photo")
        self.photo_frame = ctk.CTkFrame(body, fg_color="transparent")
        self.photo_frame.pack(fill="x", padx=PAD)
        add_hint = "Drop a photo anywhere on the window, or use Add photo (Ctrl+O). Ctrl+Z / Ctrl+Y undo and redo look changes." if _HAS_DND else "Use Add photo to pick a JPG or PNG."
        self.photo_hint = ctk.CTkLabel(body, text=add_hint, text_color=MUTED, font=ctk.CTkFont(size=FONT_SMALL),
                                       wraplength=360, justify="left")
        self.photo_hint.pack(padx=PAD, pady=(4, 0), anchor="w")

        # ---- 2. mood ----
        self._section_label(body, "2 · Pick a mood")
        cards = ctk.CTkFrame(body, fg_color="transparent")
        cards.pack(fill="x", padx=PAD)
        cards.grid_columnconfigure((0, 1), weight=1)
        for index, mood in enumerate(gui_moods.MOODS):
            button = _Button(
                cards, text=f"{mood['label']}\n{mood['blurb']}", height=76, corner_radius=8, anchor="w",
                fg_color=CARD, hover_color=("#c4c9d8", "#2a2a36"), border_width=2, border_color=CARD,
                font=ctk.CTkFont(size=FONT_SMALL), text_color=TEXT,
                command=lambda key=mood["key"]: self._on_mood_clicked(key))
            button._text_label.configure(wraplength=150)   # long blurbs wrap instead of being cut off
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=4, pady=4)
            self.mood_buttons[mood["key"]] = button
        self.mood_caption = ctk.CTkLabel(body, text="", text_color=MUTED, font=ctk.CTkFont(size=FONT_SMALL))
        self.mood_caption.pack(padx=PAD, anchor="w")

        # ---- 3. three sliders ----
        self._section_label(body, "3 · Fine-tune")
        self.intensity_word = self._word_slider(body, "How much", self.intensity_var, gui_moods.slider_word,
                                                "How heavy the weather is.")
        self.wind_word = self._word_slider(body, "Wind", self.wind_var, gui_moods.slider_word,
                                           "How strongly wind moves the rain, clouds and plants.")
        self.darkness_word = self._word_slider(body, "Darkness", self.darkness_var, gui_moods.darkness_word,
                                               "Makes the whole scene darker and moodier.")

        # ---- 4. size & length ----
        self._section_label(body, "4 · Video size and length")
        grid = ctk.CTkFrame(body, fg_color="transparent")
        grid.pack(fill="x", padx=PAD)
        grid.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(grid, text="Quality", font=ctk.CTkFont(size=FONT_BODY)).grid(row=0, column=0, sticky="w", pady=4)
        quality_menu = ctk.CTkOptionMenu(grid, values=list(gui_moods.QUALITY_PRESETS), variable=self.quality_var,
                                         command=lambda _v: self._on_output_changed())
        quality_menu.grid(row=0, column=1, sticky="ew", padx=(PAD, 0), pady=4)
        Tooltip(quality_menu, "Bigger and sharper videos take much longer to make. Draft is for checking the look.")
        ctk.CTkLabel(grid, text="Length", font=ctk.CTkFont(size=FONT_BODY)).grid(row=1, column=0, sticky="w", pady=4)
        length_menu = ctk.CTkOptionMenu(grid, values=list(gui_moods.LENGTH_CHOICES), variable=self.length_var,
                                        command=lambda _v: self._on_output_changed())
        length_menu.grid(row=1, column=1, sticky="ew", padx=(PAD, 0), pady=4)
        Tooltip(length_menu, "How long before the loop starts over. The end blends into the start, so it repeats without a jump.")
        self.fps_row = ctk.CTkFrame(body, fg_color="transparent")
        self.fps_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.fps_row, text="Smoothness", font=ctk.CTkFont(size=FONT_BODY)).grid(row=0, column=0, sticky="w")
        fps_menu = ctk.CTkOptionMenu(self.fps_row, values=list(gui_moods.FPS_CHOICES), variable=self.fps_var,
                                     command=lambda _v: self._on_output_changed())
        fps_menu.grid(row=0, column=1, sticky="ew", padx=(PAD, 0))
        Tooltip(fps_menu, "Frames per second. More is smoother but takes longer and makes a bigger file.")
        self.quality_caption = ctk.CTkLabel(body, text="", text_color=MUTED, font=ctk.CTkFont(size=FONT_SMALL),
                                            wraplength=360, justify="left")
        self.quality_caption.pack(padx=PAD, anchor="w")

        # ---- 5. more options ----
        self._section_label(body, "More options")
        self.more = Collapsible(body, "Effects, area and advanced", opened=False)
        self.more.pack(fill="x", padx=4)
        self._build_more_options(self.more.body)

        # ---- my looks ----
        self._section_label(body, "Your looks")
        looks = ctk.CTkFrame(body, fg_color="transparent")
        looks.pack(fill="x", padx=PAD, pady=(0, 24))
        looks.grid_columnconfigure((0, 1), weight=1)
        save_button = _Button(looks, text="Save my look", height=32, fg_color="transparent", border_width=1, text_color=TEXT,
                                    command=self._save_look)
        save_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        Tooltip(save_button, "Save the mood, sliders and effects under a name so you can bring them back later.")
        load_button = _Button(looks, text="My looks", height=32, fg_color="transparent", border_width=1, text_color=TEXT,
                                    command=self._load_look)
        load_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        Tooltip(load_button, "Bring back a look you saved.")

    def _word_slider(self, parent, title, variable, word_of, tip):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=PAD, pady=(0, 4))
        row.grid_columnconfigure(1, weight=1)
        title_label = ctk.CTkLabel(row, text=title, width=72, anchor="w", font=ctk.CTkFont(size=FONT_BODY))
        title_label.grid(row=0, column=0, sticky="w")
        word = ctk.CTkLabel(row, text=word_of(variable.get()), width=84, anchor="e", text_color=MUTED,
                            font=ctk.CTkFont(size=FONT_SMALL))
        word.grid(row=0, column=2, sticky="e")
        slider = ctk.CTkSlider(row, from_=0, to=1, variable=variable,
                               command=lambda value: self._on_simple_slider(word, word_of, value))
        slider.grid(row=0, column=1, sticky="ew", padx=PAD)
        Tooltip(slider, tip)
        Tooltip(title_label, tip)
        word.word_of = word_of
        return word

    def _build_more_options(self, body):
        ctk.CTkLabel(body, text="Effects", font=ctk.CTkFont(size=FONT_SMALL, weight="bold"), text_color=MUTED).pack(
            anchor="w", pady=(4, 0))
        for attr, _technical in EFFECT_SWITCHES:
            friendly, technical = gui_moods.FRIENDLY_EFFECT_NAMES[attr]
            var = tk.BooleanVar(value=bool(getattr(config, attr)))
            switch = ctk.CTkSwitch(body, text=friendly, variable=var, font=ctk.CTkFont(size=FONT_BODY),
                                   command=self._on_switch_changed)
            switch.pack(anchor="w", pady=(6, 0), fill="x")
            self.switch_vars[attr] = var
            help_text, cost = EFFECT_HELP[attr]
            technical_note = f" (technical name: {technical})" if technical != friendly else ""
            Tooltip(switch, f"{help_text}{technical_note}\n{cost}.")
            cost_label = ctk.CTkLabel(body, text=cost, text_color=MUTED, font=ctk.CTkFont(size=FONT_SMALL))
            cost_label.pack(anchor="w", padx=(46, 0))

        ctk.CTkLabel(body, text="Limit effects to an area", font=ctk.CTkFont(size=FONT_SMALL, weight="bold"),
                     text_color=MUTED).pack(anchor="w", pady=(16, 0))
        area_row = ctk.CTkFrame(body, fg_color="transparent")
        area_row.pack(fill="x", pady=(4, 0))
        paint_button = _Button(area_row, text="Paint the area…", width=140, height=30, fg_color="transparent",
                                     border_width=1, command=self._paint_area)
        paint_button.pack(side="left")
        Tooltip(paint_button, "Opens a window where you brush over the parts of the photo the effects may appear in.")
        self.limit_switch = ctk.CTkSwitch(area_row, text="Use it", variable=self.limit_area_var,
                                          font=ctk.CTkFont(size=FONT_BODY), command=self._on_look_changed)
        self.limit_switch.pack(side="left", padx=PAD)
        self.area_caption = ctk.CTkLabel(body, text="", text_color=MUTED, font=ctk.CTkFont(size=FONT_SMALL),
                                         wraplength=330, justify="left")
        self.area_caption.pack(anchor="w")

        advanced_row = ctk.CTkFrame(body, fg_color="transparent")
        advanced_row.pack(fill="x", pady=(16, 0))
        advanced_switch = ctk.CTkSwitch(advanced_row, text="Advanced controls", variable=self.advanced_var,
                                        font=ctk.CTkFont(size=FONT_BODY), command=self._on_advanced_toggled)
        advanced_switch.pack(anchor="w")
        Tooltip(advanced_switch, "Shows a separate strength slider for every effect and the smoothness (frames per second) choice.")

        self.advanced_frame = ctk.CTkFrame(body, fg_color="transparent")
        for attr, _t in EFFECT_SWITCHES:
            friendly = gui_moods.FRIENDLY_EFFECT_NAMES[attr][0]
            spec = EFFECT_INTENSITY[attr]
            row = ctk.CTkFrame(self.advanced_frame, fg_color="transparent")
            row.pack(fill="x", pady=(6, 0))
            row.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(row, text=friendly, width=96, anchor="w", font=ctk.CTkFont(size=FONT_SMALL)).grid(row=0, column=0)
            var = tk.DoubleVar(value=current_intensity_default(attr))
            self.adv_vars[attr] = var
            slider = ctk.CTkSlider(row, from_=spec["range"][0], to=spec["range"][1], variable=var,
                                   command=lambda value, a=attr: self._on_adv_slider(a, value))
            slider.grid(row=0, column=1, sticky="ew", padx=PAD)
            Tooltip(slider, f"Exact strength of {friendly.lower()}. Only used while it is switched on.")
            label = ctk.CTkLabel(row, text="", width=48, anchor="e", text_color=MUTED, font=ctk.CTkFont(size=FONT_SMALL))
            label.grid(row=0, column=2)
            self.adv_labels[attr] = label
        self._sync_adv_sliders()

    def _build_bottom(self):
        # Swappable info area: idle / rendering / done / error.
        self.info_frame = ctk.CTkFrame(self.bottom, fg_color=PANEL, corner_radius=8)
        self.info_frame.grid(row=0, column=0, sticky="ew")
        self.info_frame.grid_columnconfigure(0, weight=1)

        self.info_label = ctk.CTkLabel(self.info_frame, text="", text_color=MUTED, anchor="w", justify="left",
                                       wraplength=760, font=ctk.CTkFont(size=FONT_BODY))
        self.info_label.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 8))

        # rendering panel
        self.render_panel = ctk.CTkFrame(self.info_frame, fg_color="transparent")
        self.render_panel.grid_columnconfigure(0, weight=1)
        stages = ctk.CTkFrame(self.render_panel, fg_color="transparent")
        stages.grid(row=0, column=0, sticky="w")
        for key, text in _STAGES:
            label = ctk.CTkLabel(stages, text=f"○ {text}", text_color=MUTED, anchor="w", font=ctk.CTkFont(size=FONT_SMALL))
            label.pack(anchor="w")
            self.stage_labels[key] = label
        self.progress_bar = ctk.CTkProgressBar(self.render_panel)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.stop_button = _Button(self.render_panel, text="Stop — you can continue later", height=34,
                                         fg_color="transparent", border_width=1, command=self._stop_render)
        self.stop_button.grid(row=2, column=0, sticky="w", pady=(8, 0))
        Tooltip(self.stop_button, "Stops now and keeps what is finished. Press Make final video again (same settings) to carry on. Shortcut: Esc.")

        # done panel
        self.done_panel = ctk.CTkFrame(self.info_frame, fg_color="transparent")
        self.done_title = ctk.CTkLabel(self.done_panel, text="", font=ctk.CTkFont(size=FONT_BODY, weight="bold"),
                                       text_color=OK, anchor="w")
        self.done_title.pack(anchor="w")
        self.done_facts = ctk.CTkLabel(self.done_panel, text="", text_color=MUTED, anchor="w",
                                       font=ctk.CTkFont(size=FONT_SMALL))
        self.done_facts.pack(anchor="w")
        done_buttons = ctk.CTkFrame(self.done_panel, fg_color="transparent")
        done_buttons.pack(anchor="w", pady=(8, 0))
        _Button(done_buttons, text="Set as wallpaper", height=34, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      command=self._set_as_wallpaper).pack(side="left")
        _Button(done_buttons, text="Show in folder", height=34, fg_color="transparent", border_width=1, text_color=TEXT,
                      command=self._show_last_in_folder).pack(side="left", padx=PAD)
        _Button(done_buttons, text="Wallpaper Engine folder", height=34, fg_color="transparent", border_width=1, text_color=TEXT,
                      command=self._export_wallpaper_engine).pack(side="left", padx=(0, PAD))
        _Button(done_buttons, text="Make another version", height=34, fg_color="transparent", border_width=1, text_color=TEXT,
                      command=self._make_another).pack(side="left")

        # error panel
        self.error_panel = ctk.CTkFrame(self.info_frame, fg_color="transparent")
        _Button(self.error_panel, text="Copy details", height=30, fg_color="transparent", border_width=1, text_color=TEXT,
                      command=self._copy_error_details).pack(side="left")
        _Button(self.error_panel, text="Show details", height=30, fg_color="transparent", border_width=1,
                      command=self._show_details).pack(side="left", padx=PAD)
        _Button(self.error_panel, text="Open log", height=30, fg_color="transparent", border_width=1,
                      command=self._open_error_log).pack(side="left")

        # primary action
        self.primary_button = _Button(self.bottom, text="Make final video", height=52, corner_radius=8,
                                            fg_color=ACCENT, hover_color=ACCENT_HOVER,
                                            font=ctk.CTkFont(size=16, weight="bold"), command=self._start_render)
        self.primary_button.grid(row=1, column=0, sticky="ew", pady=(PAD, 0))
        Tooltip(self.primary_button, "Makes the full-quality video and saves it in output/live_wallpapers/. Shortcut: Ctrl+R.")
        self.estimate_note = ctk.CTkLabel(self.bottom, text="", text_color=MUTED, font=ctk.CTkFont(size=FONT_SMALL))
        self.estimate_note.grid(row=2, column=0, sticky="w", pady=(4, 0))
        queue_row = ctk.CTkFrame(self.bottom, fg_color="transparent")
        queue_row.grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.queue_add_button = _Button(queue_row, text="Add to queue", height=28, width=110, fg_color="transparent",
                                              border_width=1, text_color=TEXT, command=self._add_to_queue)
        self.queue_add_button.pack(side="left")
        Tooltip(self.queue_add_button, "Saves this photo + look + quality as a job. Queued videos are made one after another, e.g. overnight.")
        self.queue_clear_button = _Button(queue_row, text="Clear queue", height=28, width=90, fg_color="transparent",
                                                border_width=1, text_color=TEXT, command=self._clear_queue)
        self.compare_button = _Button(queue_row, text="Before / After", height=28, width=110, fg_color="transparent",
                                            border_width=1, text_color=TEXT, command=self._toggle_original)
        self.compare_button.pack(side="left", padx=(PAD, 0))
        Tooltip(self.compare_button, "Switch between your original photo and the quick weather preview.")
        self.queue_label = ctk.CTkLabel(queue_row, text="", text_color=MUTED, font=ctk.CTkFont(size=FONT_SMALL))
        self.queue_label.pack(side="left", padx=PAD)
        self._show_panel(None)

    def _show_panel(self, name):
        for panel_name, panel in (("render", self.render_panel), ("done", self.done_panel), ("error", self.error_panel)):
            if panel_name == name:
                panel.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 12))
            else:
                panel.grid_remove()

    def _set_info(self, text, color=MUTED):
        self.info_label.configure(text=text, text_color=color)

    # ------------------------------------------------------------------
    # Photos
    # ------------------------------------------------------------------
    def _photo_names(self):
        return [p.name for p in find_input_images(config.INPUT_DIR)]

    def _thumbnail(self, path, size=(84, 56)):
        key = (str(path), size)
        if key not in self._thumb_cache:
            try:
                with Image.open(path) as image:
                    thumb = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), size, Image.LANCZOS)
            except Exception:  # noqa: BLE001 - an unreadable file gets a grey tile rather than breaking the list
                thumb = Image.new("RGB", size, "#333340")
            self._thumb_cache[key] = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=size)
        return self._thumb_cache[key]

    def _refresh_photo_tiles(self):
        for child in self.photo_frame.winfo_children():
            child.destroy()
        names = self._photo_names()
        columns = 3
        for index, name in enumerate(names):
            path = config.INPUT_DIR / name
            selected = self.photo_path is not None and path == self.photo_path
            tile = _Button(self.photo_frame, text="", image=self._thumbnail(path), width=104, height=68,
                                 fg_color=CARD, hover_color="#2a2a36", border_width=2,
                                 border_color=ACCENT if selected else CARD, command=lambda p=path: self._select_photo(p))
            tile.grid(row=index // columns, column=index % columns, padx=3, pady=3)
            Tooltip(tile, name)
        add_index = len(names)
        add_tile = _Button(self.photo_frame, text="+ Add photo", width=104, height=68, fg_color="transparent",
                                 border_width=1, text_color=TEXT, font=ctk.CTkFont(size=FONT_SMALL), command=self._add_photo)
        add_tile.grid(row=add_index // columns, column=add_index % columns, padx=3, pady=3)
        if not names:
            self.photo_hint.configure(text="No photos yet. Add a JPG or PNG"
                                           + (" (or drop one on this window)" if _HAS_DND else "") + " to begin.")

    def _select_default_photo(self):
        names = self._photo_names()
        if self.photo_path is None or not self.photo_path.exists():
            self.photo_path = config.INPUT_DIR / names[0] if names else None
        self._refresh_photo_tiles()
        self._after_photo_changed()

    def _select_photo(self, path):
        if self.rendering:
            self._set_info("A video is being made - wait for it to finish (or stop it) before changing the photo.", WARN)
            return
        self.photo_path = Path(path)
        self._refresh_photo_tiles()
        self._after_photo_changed()

    def _photo_note(self):
        """One line about the photo's size, warning about tiny or very unusual shapes."""
        try:
            with Image.open(self.photo_path) as image:
                width, height = ImageOps.exif_transpose(image).size
        except (OSError, AttributeError):
            return ""
        note = f"{width}×{height} px."
        if max(width, height) < 1280:
            note += " This photo is small, so the video may look soft. A bigger photo works better."
        ratio = width / height
        if ratio > 2.4 or ratio < 0.7:
            note += " It has an unusual shape. The video keeps this shape, so it may not fill your whole screen as a wallpaper."
        return note

    def _after_photo_changed(self):
        if self.photo_path is not None:
            self.photo_hint.configure(text=self._photo_note())
        self._refresh_area_caption()
        self._refresh_estimate()
        self._draw_canvas_message()
        if self.photo_path is not None:
            self._schedule_draft()

    def _add_photo(self):
        chosen = filedialog.askopenfilenames(
            title="Choose photos", filetypes=[("Photos", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"), ("All files", "*.*")])
        self._import_photos(chosen)

    def _on_drop(self, event):
        self._import_photos(self.tk.splitlist(event.data))

    def _import_photos(self, paths):
        """Copies photos into input_images/ (so everything stays inside the project) and selects the last one."""
        last = None
        for raw in paths:
            source = Path(raw)
            if source.suffix.lower() not in _IMAGE_EXTENSIONS or not source.is_file():
                continue
            target = source
            try:
                if source.parent.resolve() != config.INPUT_DIR.resolve():
                    target = gui_moods.next_free_path(config.INPUT_DIR / source.name)
                    import shutil
                    shutil.copy2(source, target)
            except OSError as exc:
                self._set_info(f"Could not add {source.name}: {exc}", WARN)
                continue
            last = target
        if last is not None:
            self._select_photo(last)
        elif paths:
            self._set_info("That file is not a photo I can use. Try a JPG or PNG.", WARN)

    def _size_of_photo(self):
        if self.photo_path is None:
            return None
        key = str(self.photo_path)
        if key not in self._photo_size_cache:
            try:
                self._photo_size_cache[key] = _photo_size(self.photo_path)
            except Exception:  # noqa: BLE001
                return None
        return self._photo_size_cache[key]

    # ------------------------------------------------------------------
    # Look state (mood, sliders, effects)
    # ------------------------------------------------------------------
    def _switch_states(self):
        return {attr: bool(var.get()) for attr, var in self.switch_vars.items()}

    def _on_mood_clicked(self, key):
        self._apply_mood(key)

    def _apply_mood(self, key, schedule=True):
        mood = gui_moods.MOOD_BY_KEY[key]
        self._loading_state = True
        try:
            for attr, on in gui_moods.effects_for_mood(key).items():
                if attr in self.switch_vars:
                    self.switch_vars[attr].set(on)
            self.intensity_var.set(mood["intensity"])
            self.wind_var.set(mood["wind"])
            self.darkness_var.set(mood["darkness"])
        finally:
            self._loading_state = False
        self.mood_key = key
        self.adv_touched.clear()
        self._refresh_look_labels()
        if schedule:
            self._on_look_changed()

    def _refresh_look_labels(self):
        for key, button in self.mood_buttons.items():
            button.configure(border_color=ACCENT if key == self.mood_key else CARD)
        if self.mood_key is None:
            self.mood_caption.configure(text=f"{gui_moods.CUSTOM_MOOD_LABEL}: your own mix of effects.")
        else:
            self.mood_caption.configure(text="")
        self.intensity_word.configure(text=gui_moods.slider_word(self.intensity_var.get()))
        self.wind_word.configure(text=gui_moods.slider_word(self.wind_var.get()))
        self.darkness_word.configure(text=gui_moods.darkness_word(self.darkness_var.get()))
        self._sync_adv_sliders()

    def _sync_adv_sliders(self):
        for attr, var in self.adv_vars.items():
            if attr in self.adv_touched:
                continue
            spec = EFFECT_INTENSITY[attr]
            var.set(gui_moods.intensity_value(attr, self.intensity_var.get()))
            self.adv_labels[attr].configure(text=spec["fmt"].format(round(var.get(), spec["digits"])))

    def _on_simple_slider(self, word_label, word_of, value):
        word_label.configure(text=word_of(value))
        self.adv_touched.clear()
        self._sync_adv_sliders()
        self._on_look_changed()

    def _on_adv_slider(self, attr, value):
        spec = EFFECT_INTENSITY[attr]
        self.adv_touched.add(attr)
        self.adv_labels[attr].configure(text=spec["fmt"].format(round(float(value), spec["digits"])))
        self._on_look_changed()

    def _on_switch_changed(self):
        if self._loading_state:
            return
        self.mood_key = gui_moods.mood_key_for_switches(self._switch_states())
        self._refresh_look_labels()
        self._on_look_changed()

    def _on_advanced_toggled(self):
        if self.advanced_var.get():
            self.advanced_frame.pack(fill="x", pady=(4, 0))
            self._place_fps_row(True)
        else:
            self.advanced_frame.pack_forget()
            self._place_fps_row(False)
            self.adv_touched.clear()
            self._sync_adv_sliders()
        self._on_look_changed()

    def _place_fps_row(self, show):
        if show:
            self.fps_row.pack(fill="x", padx=PAD, before=self.quality_caption)
        else:
            self.fps_row.pack_forget()
            self.fps_var.set(gui_moods.DEFAULT_FPS)
            self._on_output_changed()

    def _look_overrides(self):
        advanced = None
        if self.advanced_var.get() and self.adv_touched:
            advanced = {attr: self.adv_vars[attr].get() for attr in self.adv_touched}
        overrides = gui_moods.build_look_overrides(
            self._switch_states(), self.intensity_var.get(), self.wind_var.get(), self.darkness_var.get(),
            mood_key=self.mood_key, advanced_intensities=advanced)
        if self.limit_area_var.get() and self._correction_mask_path() is not None and self._correction_mask_path().exists():
            overrides["MASK_MANUAL_CORRECTION"] = True
        return overrides

    def _output_overrides(self):
        return gui_moods.quality_overrides(
            self.quality_var.get(), gui_moods.LENGTH_CHOICES[self.length_var.get()], gui_moods.FPS_CHOICES[self.fps_var.get()])

    def _remember_look(self):
        """Undo history: one step per gesture (a slider drag counts once), not per tiny movement."""
        current = self._look_state()
        if self._last_look is None:
            self._last_look = current
            return
        if current == self._last_look:
            return
        now = time.perf_counter()
        if now - self._last_look_time > 0.8:
            self._undo_stack.append(self._last_look)
            del self._undo_stack[:-50]
            self._redo_stack.clear()
        self._last_look_time = now
        self._last_look = current

    def _undo(self):
        self._step_history(self._undo_stack, self._redo_stack, "Undid the last change.")

    def _redo(self):
        self._step_history(self._redo_stack, self._undo_stack, "Redid the change.")

    def _step_history(self, source, target, message):
        if self.rendering or not source or self.focus_get().__class__.__name__ in ("Entry", "Text"):
            return
        target.append(self._look_state())
        state = source.pop()
        self._apply_look_state(state)
        self._last_look = self._look_state()
        self._last_look_time = 0.0
        self._set_info(message)
        self._on_look_changed(record=False)

    def _toggle_original(self):
        """Before / After: show the untouched photo in place of the weather preview."""
        if self.photo_path is None or self.preview.fast_frames is None:
            self._set_info("Wait for the quick preview first, then compare.", WARN)
            return
        if self.showing_original:
            self.showing_original = False
            self.canvas.delete("original")
            self.compare_button.configure(text="Before / After")
            self.preview.toggle_pause()
            return
        try:
            with Image.open(self.photo_path) as image:
                original = ImageOps.exif_transpose(image).convert("RGB").resize(self.preview.display_size, Image.LANCZOS)
        except OSError as exc:
            self._set_info(f"Could not open the photo to compare: {exc}", WARN)
            return
        if not self.preview.paused:
            self.preview.toggle_pause()
        self._original_tk = ImageTk.PhotoImage(original)
        self.canvas.delete("frame")
        self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2,
                                 anchor="center", image=self._original_tk, tags="original")
        self.showing_original = True
        self.compare_button.configure(text="Back to preview")
        self._set_info("This is your original photo. Press the button again to see the weather.")

    def _on_look_changed(self, record=True):
        if self._loading_state:
            return
        if record:
            self._remember_look()
        self._refresh_estimate()
        self._schedule_draft()

    def _on_output_changed(self):
        max_side, _crf, blurb = gui_moods.QUALITY_PRESETS[self.quality_var.get()]
        size = self._size_of_photo()
        if size is not None:
            width, height = gui_moods.fit_dimension(size, max_side)
            self.quality_caption.configure(text=f"{blurb} Output: {width}×{height}.")
        else:
            self.quality_caption.configure(text=blurb)
        self._refresh_estimate()

    # ------------------------------------------------------------------
    # Area (mask) painting
    # ------------------------------------------------------------------
    def _correction_mask_path(self):
        if self.photo_path is None:
            return None
        return config.MASKS_DIR / f"{self.photo_path.stem}_correction.png"

    def _refresh_area_caption(self):
        path = self._correction_mask_path()
        if path is not None and path.exists():
            self.area_caption.configure(text="An area is saved for this photo. Switch on \"Use it\" to limit effects to it.")
        else:
            self.area_caption.configure(text="No area painted for this photo yet - effects can appear everywhere they fit.")
            if self.limit_area_var.get():
                self.limit_area_var.set(False)

    def _paint_area(self):
        if self.photo_path is None or self.rendering:
            self._set_info("Choose a photo first (and wait for any running video to finish).", WARN)
            return
        from vfx_engine.mask_editor import open_mask_editor

        try:
            with Image.open(self.photo_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            mask = open_mask_editor(image)
        except Exception as exc:  # noqa: BLE001
            self._show_error(exc, self.photo_path.name, "area painting")
            return
        if mask is None:
            return
        config.MASKS_DIR.mkdir(parents=True, exist_ok=True)
        mask.convert("L").save(self._correction_mask_path())
        self.limit_area_var.set(True)
        self._refresh_area_caption()
        self._set_info("Area saved. Effects will now stay inside the part you painted.", OK)
        self._on_look_changed()

    # ------------------------------------------------------------------
    # Time estimate on the primary button
    # ------------------------------------------------------------------
    def _output_dimensions(self):
        size = self._size_of_photo()
        if size is None:
            return None
        return gui_moods.fit_dimension(size, gui_moods.QUALITY_PRESETS[self.quality_var.get()][0])

    def _refresh_estimate(self):
        self._on_output_changed_caption_only()
        if self.rendering:
            return
        dims = self._output_dimensions()
        continuing = self._pending_matches_current()
        label = "Continue where you stopped" if continuing else "Make final video"
        note = ""
        if dims is not None:
            frames = gui_moods.frame_count(gui_moods.LENGTH_CHOICES[self.length_var.get()], gui_moods.FPS_CHOICES[self.fps_var.get()])
            seconds, basis = gui_system.estimate_seconds(dims[0], dims[1], frames, self._draft_spmf)
            if seconds is not None:
                label += f"  ·  {gui_moods.format_estimate(seconds)}"
                note = ("Time based on your earlier videos on this computer."
                        if basis == "history" else
                        "Rough guess from the preview speed - it gets accurate after your first video.")
            else:
                note = "The time estimate appears once the first preview is ready."
        self.primary_button.configure(text=label)
        self.estimate_note.configure(text=note)

    def _on_output_changed_caption_only(self):
        max_side, _crf, blurb = gui_moods.QUALITY_PRESETS[self.quality_var.get()]
        size = self._size_of_photo()
        if size is not None:
            width, height = gui_moods.fit_dimension(size, max_side)
            self.quality_caption.configure(text=f"{blurb} Output: {width}×{height}.")
        else:
            self.quality_caption.configure(text=blurb)

    def _pending_matches_current(self):
        pending = gui_system.read_pending()
        return bool(pending and self.photo_path is not None and pending.get("photo") == self.photo_path.name
                    and pending.get("state") == self._state_for_pending())

    def _submit(self, function, *args):
        self.job_queue.put((function, args))

    def _engine_loop(self):
        while True:
            function, args = self.job_queue.get()
            function(*args)

    # ------------------------------------------------------------------
    # Live draft
    # ------------------------------------------------------------------
    def _draw_canvas_message(self):
        if self.preview.fast_frames is not None or self.preview.video_path is not None and not self.preview._closed:
            return
        self.canvas.delete("message")
        if self.photo_path is None:
            text = "Add a photo to begin.\nYour preview appears here and updates as you choose."
        else:
            text = "Preparing your preview…\nThe first one takes a little longer while the photo is analysed."
        self.canvas.create_text(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, text=text,
                                fill=self._plain(MUTED), font=("Segoe UI", FONT_BODY), justify="center", tags="message")

    @staticmethod
    def _plain(pair):
        """A (light, dark) colour pair -> the one plain colour for the current mode (tk.Canvas takes no pairs).
        Named greys like "gray30" are valid tk colours already."""
        return pair[1] if ctk.get_appearance_mode() == "Dark" else pair[0]

    def _on_appearance_changed(self, label):
        ctk.set_appearance_mode(APPEARANCE_CHOICES[label])
        self.appearance_label = label
        for canvas in (self.canvas, self.videos_canvas):
            canvas.configure(bg=self._plain(BG))
        self._save_session()

    def _schedule_draft(self):
        if self._draft_after is not None:
            self.after_cancel(self._draft_after)
        self._draft_after = self.after(300, self._run_draft)

    def _run_draft(self):
        self._draft_after = None
        if self.rendering or self.photo_path is None or not self.photo_path.exists():
            return
        if self.draft_running:
            self.draft_pending = True
            return
        self.draft_running = True
        self.draft_pending = False
        self.draft_generation += 1
        self._set_info("Updating the preview…")
        overrides = self._look_overrides()
        self._submit(self._draft_worker, self.draft_generation, self.photo_path, overrides)

    def _draft_worker(self, generation, image_path, overrides):
        with gui_system.CONFIG_LOCK:
            originals = {attr: getattr(config, attr, None) for attr in overrides}
            try:
                for attr, value in overrides.items():
                    setattr(config, attr, value)
                start = time.perf_counter()
                result = render_fast_preview(image_path)
                self.event_queue.put(("draft_done", (generation, image_path, result, time.perf_counter() - start), None))
            except Exception as exc:  # noqa: BLE001 - shown in the app instead of a silent dead thread
                self.event_queue.put(("draft_error", exc, image_path.name))
            finally:
                for attr, value in originals.items():
                    setattr(config, attr, value)

    def _on_draft_done(self, payload):
        generation, image_path, (frames, width, height, fps), seconds = payload
        self.draft_running = False
        if image_path in self._draft_seen_photos and frames:
            self._draft_spmf = seconds / (width * height / 1e6 * len(frames))
        self._draft_seen_photos.add(image_path)
        if self.draft_pending or self.rendering or image_path != self.photo_path:
            if not self.rendering:
                self._run_draft()      # newer settings exist - skip showing this stale one
            return
        self.canvas.delete("message")
        self.canvas.delete("original")
        self.showing_original = False
        self.compare_button.configure(text="Before / After")
        self.preview.play_frames(frames, width, height, fps)
        self._set_info("Quick preview — the final video will be larger and sharper.")
        self._refresh_estimate()

    def _on_draft_error(self, exc, photo_name):
        self.draft_running = False
        happened, to_try = error_hints.explain_exception(exc)
        self._set_info(f"The preview could not be made. {happened} {to_try}", WARN)
        try:
            error_hints.log_exception(exc, context=f"live preview {photo_name}")
        except OSError:
            pass
        if self.draft_pending:
            self._run_draft()

    # ------------------------------------------------------------------
    # Final render
    # ------------------------------------------------------------------
    def _state_for_pending(self):
        return {"look": self._look_state(), "output": self._output_state()}

    def _look_state(self):
        return {
            "mood": self.mood_key,
            "effects": self._switch_states(),
            "intensity": round(self.intensity_var.get(), 3),
            "wind": round(self.wind_var.get(), 3),
            "darkness": round(self.darkness_var.get(), 3),
            "advanced": {attr: round(self.adv_vars[attr].get(), 4) for attr in self.adv_touched} if self.advanced_var.get() else {},
            "limit_area": bool(self.limit_area_var.get()),
        }

    def _output_state(self):
        return {"quality": self.quality_var.get(), "length": self.length_var.get(), "fps": self.fps_var.get()}

    def _set_rendering_ui(self, busy):
        self.rendering = busy
        if busy:
            self.primary_button.configure(state="disabled", text="Making your video…")
            self.estimate_note.configure(text="")
            self._show_panel("render")
            self.info_label.configure(text="")
            self.stop_button.configure(state="normal", text="Stop — you can continue later")
        else:
            self.primary_button.configure(state="normal")
            self._refresh_estimate()

    def _mark_stage(self, active_key):
        active_index = [key for key, _ in _STAGES].index(active_key)
        for i, (key, text) in enumerate(_STAGES):
            if i < active_index:
                self.stage_labels[key].configure(text=f"✔ {text}", text_color=OK)
            elif i == active_index:
                self.stage_labels[key].configure(text=f"● {text}", text_color=TEXT)
            else:
                self.stage_labels[key].configure(text=f"○ {text}", text_color=MUTED)

    def _add_to_queue(self):
        if self.photo_path is None or not self.photo_path.exists():
            self._set_info("Add a photo first.", WARN)
            return
        mood_label = gui_moods.MOOD_BY_KEY[self.mood_key]["label"] if self.mood_key else "mix"
        self.render_queue.append({
            "photo": self.photo_path, "slug": mood_label.lower().replace(" ", "_"), "label": f"{self.photo_path.name} + {mood_label}",
            "overrides": {**self._look_overrides(), **self._output_overrides()},
            "dims": self._output_dimensions() or (0, 0),
            "frames": gui_moods.frame_count(gui_moods.LENGTH_CHOICES[self.length_var.get()], gui_moods.FPS_CHOICES[self.fps_var.get()]),
            "state": self._state_for_pending()})
        self._refresh_queue_label()
        if not self.rendering:
            self._run_next_in_queue()
        else:
            self._set_info(f"Added to the queue: {self.render_queue[-1]['label']}.")

    def _clear_queue(self):
        self.render_queue.clear()
        self._refresh_queue_label()

    def _refresh_queue_label(self):
        count = len(self.render_queue)
        self.queue_label.configure(text=f"{count} video{'s' if count != 1 else ''} waiting: {self.render_queue[0]['label']}" if count else "")
        if count:
            self.queue_clear_button.pack(side="left", after=self.queue_add_button, padx=(PAD, 0))
        else:
            self.queue_clear_button.pack_forget()

    def _run_next_in_queue(self):
        if self.rendering or not self.render_queue:
            return
        job = self.render_queue.pop(0)
        self._refresh_queue_label()
        self._start_render(job=job)

    def _start_render(self, resume_out_path=None, job=None):
        if self.rendering:
            return
        if job is not None:
            if not job["photo"].exists():
                self._set_info(f"Skipped {job['label']}: the photo is gone.", WARN)
                self._run_next_in_queue()
                return
        elif self.photo_path is None or not self.photo_path.exists():
            self._set_info("Add a photo first.", WARN)
            return
        photo_path = job["photo"] if job else self.photo_path
        self.cancel_event.clear()
        self._smoothed_eta = None
        self._resumed = False
        self._render_started = time.perf_counter()
        for key, text in _STAGES:
            self.stage_labels[key].configure(text=f"○ {text}", text_color=MUTED)
        self._set_rendering_ui(True)
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        self.info_label.configure(text="Starting… the first video also loads the AI models, which can take a while.")

        pending = gui_system.read_pending()
        if job is not None:
            out_path = gui_moods.next_free_path(config.LIVE_WALLPAPERS_DIR / f"{photo_path.stem}_{job['slug']}.mp4")
            overrides, dims, frames = job["overrides"], job["dims"], job["frames"]
        elif resume_out_path is not None:
            out_path = Path(resume_out_path)
        elif self._pending_matches_current() and pending.get("out_path"):
            out_path = Path(pending["out_path"])
        else:
            mood_label = gui_moods.MOOD_BY_KEY[self.mood_key]["label"] if self.mood_key else "mix"
            slug = mood_label.lower().replace(" ", "_")
            out_path = gui_moods.next_free_path(config.LIVE_WALLPAPERS_DIR / f"{self.photo_path.stem}_{slug}.mp4")
        if job is None:
            overrides = {**self._look_overrides(), **self._output_overrides()}
            dims = self._output_dimensions() or (0, 0)
            frames = gui_moods.frame_count(gui_moods.LENGTH_CHOICES[self.length_var.get()], gui_moods.FPS_CHOICES[self.fps_var.get()])
        self._render_meta = {"photo": photo_path, "out_path": out_path, "dims": dims, "frames": frames}
        gui_system.write_pending({
            "photo": photo_path.name, "out_path": str(out_path), "percent": 0,
            "started": datetime.now().isoformat(timespec="seconds"),
            "state": job["state"] if job else self._state_for_pending()})
        self._submit(self._render_worker, photo_path, overrides, out_path)

    def _stop_render(self):
        if not self.rendering or self.cancel_event.is_set():
            return
        self.cancel_event.set()
        self.stop_button.configure(state="disabled", text="Stopping…")
        self.info_label.configure(text="Stopping — finishing the current step. Your progress is kept.")

    def _render_worker(self, image_path, overrides, out_path):
        with gui_system.CONFIG_LOCK:
            overrides = {**overrides, "LIVE_PREVIEW_WINDOW": False}   # this app IS the preview
            originals = {attr: getattr(config, attr, None) for attr in overrides}
            progress.set_cancel_event(self.cancel_event)
            progress.set_reporter(lambda event: self.event_queue.put(("progress", event, None)))
            try:
                for attr, value in overrides.items():
                    setattr(config, attr, value)
                start = time.perf_counter()
                live_wallpaper.process_image(image_path, out_path=out_path)
                self.event_queue.put(("done", out_path, time.perf_counter() - start))
            except progress.RenderCancelled:
                self.event_queue.put(("cancelled", None, None))
            except Exception as exc:  # noqa: BLE001 - shown in the app instead of a silent dead thread
                self.event_queue.put(("error", exc, image_path.name))
            finally:
                progress.set_reporter(None)
                progress.set_cancel_event(None)
                for attr, value in originals.items():
                    setattr(config, attr, value)

    def _on_progress(self, event):
        if not self.rendering:
            return
        kind = event["kind"]
        if kind == "stage":
            self._mark_stage(_STAGE_OF_EVENT[event["stage"]])
            self.info_label.configure(text=event["label"] + " …")
            return
        if kind == "resume":
            self._resumed = True
            self.info_label.configure(
                text=f"Continuing where it left off — {event['percent']:.0f} % was already done.", text_color=OK)
            return
        if str(self.progress_bar.cget("mode")) != "determinate":
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
        self._mark_stage("frames")
        self.progress_bar.set(event["index"] / event["total"])
        eta = event["eta_seconds"]
        if eta is None:
            eta_text = "working out the time left…"
        else:
            self._smoothed_eta = eta if self._smoothed_eta is None else 0.9 * self._smoothed_eta + 0.1 * eta
            eta_text = progress.format_eta_long(self._smoothed_eta) + " left"
        self.info_label.configure(text=f"{event['percent']:.0f} %  ·  frame {event['index']} of {event['total']}  ·  {eta_text}",
                                  text_color=MUTED)
        now = time.time()
        if now - self._last_pending_write > 10:
            self._last_pending_write = now
            pending = gui_system.read_pending()
            if pending:
                pending["percent"] = round(event["percent"])
                gui_system.write_pending(pending)

    def _on_render_done(self, out_path, duration):
        self.progress_bar.stop()
        self._set_rendering_ui(False)
        gui_system.clear_pending()
        self.last_output_path = out_path
        meta = self._render_meta or {}
        if not self._resumed and meta.get("dims") and meta["dims"][0]:
            gui_system.record_render_timing(duration, meta["dims"][0], meta["dims"][1], meta["frames"])
        try:
            width, height, fps, length = _video_facts(out_path)
            size_mb = out_path.stat().st_size / 1e6
            self.done_facts.configure(
                text=f"{out_path.name}  ·  {width}×{height}  ·  {length:.0f} s loop  ·  {size_mb:.0f} MB  ·  made in {_format_duration(duration)}")
            self.canvas.delete("message")
            self.preview.play(out_path, width, height, fps)
        except Exception as exc:  # noqa: BLE001
            self.done_facts.configure(text=f"{out_path.name} (could not read its details: {exc})")
        self.done_title.configure(text="✔ Your video is ready")
        self._show_panel("done")
        self._refresh_estimate()
        self._notify_finished()
        if self.render_queue:
            self._set_info(f"Done. Starting the next queued video ({len(self.render_queue)} waiting)…")
            self.after(2500, self._run_next_in_queue)

    def _on_render_cancelled(self):
        self.progress_bar.stop()
        self._set_rendering_ui(False)
        self._show_panel(None)
        if self.render_queue:
            self._refresh_queue_label()
            self._set_info(f"Stopped. {len(self.render_queue)} queued video(s) are still waiting - press Add to queue / Clear queue.", WARN)
            self._refresh_estimate()
            return
        self._set_info("Stopped. Your progress is saved — press \"Continue where you stopped\" to carry on.", WARN)
        self._refresh_estimate()

    def _make_another(self):
        self._show_panel(None)
        self._set_info("Change anything you like, then make another version. The new video gets its own name.")
        if self.photo_path is not None:
            self._schedule_draft()

    def _notify_finished(self):
        """Finish alert: a sound and the window raised to the front (not a Windows toast notification)."""
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except (ImportError, RuntimeError):
            self.bell()
        try:
            self.deiconify()
            self.lift()
            self.attributes("-topmost", True)
            self.after(800, lambda: self.attributes("-topmost", False))
        except tk.TclError:
            pass

    def _show_last_in_folder(self):
        if self.last_output_path is not None and self.last_output_path.exists():
            subprocess.Popen(["explorer", "/select,", str(self.last_output_path)])

    def _export_wallpaper_engine(self):
        if self.last_output_path is None or not self.last_output_path.exists():
            return
        try:
            folder = gui_system.export_wallpaper_engine(self.last_output_path)
        except Exception as exc:  # noqa: BLE001
            self._show_error(exc, self.last_output_path.name, "exporting for Wallpaper Engine")
            return
        self.done_title.configure(text="✔ Wallpaper Engine folder ready - copy it into Wallpaper Engine's myprojects folder")
        subprocess.Popen(["explorer", str(folder)])

    def _set_as_wallpaper(self):
        if self.last_output_path is None or not self.last_output_path.exists():
            return
        if not messagebox.askyesno(
                "Set as wallpaper",
                "Windows cannot play a video as the desktop background by itself.\n\n"
                "I can set the first picture of this video as your desktop background now. "
                "To play the video itself, open it in a live-wallpaper app (for example Wallpaper Engine or Lively) "
                "- \"Show in folder\" takes you to the file.\n\nSet the picture now?", parent=self):
            return
        try:
            still = gui_system.extract_first_frame(self.last_output_path, self.last_output_path.with_suffix(".still.png"))
            gui_system.set_desktop_picture(still)
            self.done_title.configure(text="✔ Desktop picture set (the video itself is unchanged)")
        except Exception as exc:  # noqa: BLE001
            self._show_error(exc, self.last_output_path.name, "setting the wallpaper")

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------
    def _show_error(self, exc, photo_name, what):
        happened, to_try = error_hints.explain_exception(exc)
        self.last_error_details = f"{what} of {photo_name} failed\n\n{error_hints.format_details(exc)}"
        try:
            log_path = error_hints.log_exception(exc, context=f"{what} {photo_name}")
            note = f"\n(Full report: output/{log_path.name})"
        except OSError:
            note = ""
        self.info_label.configure(text=f"{happened}\n{to_try}{note}", text_color=WARN)
        self._show_panel("error")

    def _copy_error_details(self):
        self.clipboard_clear()
        self.clipboard_append(self.last_error_details)
        self._set_info("Error details copied to the clipboard.")

    def _on_space(self, event):
        """Space pauses / resumes the quick preview (not while typing in a text box or on a button)."""
        widget_class = event.widget.winfo_class() if event.widget is not None else ""
        if widget_class in ("Entry", "Text", "TEntry", "Button", "Toplevel") or self.rendering:
            return
        paused = self.preview.toggle_pause()
        self._set_info("Preview paused - press Space to resume." if paused else "Quick preview playing.")

    def _show_details_log(self):
        saved, self.last_error_details = self.last_error_details, ""
        try:
            self._show_details()
        finally:
            self.last_error_details = saved

    def _show_details(self):
        """In-app viewer: this error's full technical report, or the tail of the error log."""
        text = self.last_error_details
        if not text:
            try:
                with open(error_hints.ERROR_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()[-20000:]
            except OSError:
                text = ""
            text = text or "Nothing has gone wrong yet - the error log is empty."
        window = ctk.CTkToplevel(self)
        window.title("Details")
        window.geometry("760x480")
        window.transient(self)
        box = ctk.CTkTextbox(window, wrap="word", font=ctk.CTkFont(family="Consolas", size=FONT_SMALL))
        box.pack(fill="both", expand=True, padx=PAD, pady=PAD)
        box.insert("1.0", text)
        box.configure(state="disabled")
        row = ctk.CTkFrame(window, fg_color="transparent")
        row.pack(fill="x", padx=PAD, pady=(0, PAD))

        def copy():
            self.clipboard_clear()
            self.clipboard_append(text)

        _Button(row, text="Copy", width=90, command=copy).pack(side="left")
        _Button(row, text="Close", width=90, fg_color="transparent", border_width=1,
                      command=window.destroy).pack(side="right")
        window.after(150, window.focus)

    def _open_error_log(self):
        if error_hints.ERROR_LOG_PATH.exists():
            os.startfile(str(error_hints.ERROR_LOG_PATH))

    # ------------------------------------------------------------------
    # Queue
    # ------------------------------------------------------------------
    def _poll_queue(self):
        try:
            while True:
                kind, payload, extra = self.event_queue.get_nowait()
                if kind == "progress":
                    self._on_progress(payload)
                elif kind == "done":
                    self._on_render_done(payload, extra)
                elif kind == "cancelled":
                    self._on_render_cancelled()
                elif kind == "error":
                    self.progress_bar.stop()
                    self._set_rendering_ui(False)
                    gui_system.clear_pending()
                    self._show_error(payload, extra, "making the video")
                elif kind == "draft_done":
                    self._on_draft_done(payload)
                elif kind == "draft_error":
                    self._on_draft_error(payload, extra)
                elif kind == "dl_progress":
                    self.download_status.configure(text=f"Downloading {payload[0]} ({payload[1] + 1} of {payload[2]})…")
                elif kind == "dl_done":
                    self._on_download_done(payload)
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)

    # ------------------------------------------------------------------
    # First run: system checks and interrupted renders
    # ------------------------------------------------------------------
    def _startup_checks(self):
        self.system = gui_system.check_system()
        if self.system["gpu"] is None and "quality" not in self._saved_output_keys:
            self.quality_var.set("Draft")
            self._on_output_changed()
        self._refresh_estimate()
        pending = gui_system.read_pending()
        if pending and (config.INPUT_DIR / pending["photo"]).exists():
            self._offer_resume(pending)
        else:
            self._show_setup_banner()

    def _show_setup_banner(self):
        problems = []
        if self.system["missing_models"]:
            problems.append("the AI models are missing (Settings → Download)")
        if not self.system["ffmpeg"]:
            problems.append("ffmpeg is missing, so videos cannot be saved")
        if self.system["gpu"] is None:
            problems.append("no NVIDIA graphics card was found - videos will be very slow, so Quality is set to Draft")
        if problems:
            self._show_banner("Before you start: " + "; ".join(problems) + ".",
                              [("Open Settings", lambda: self._show_tab("Settings"), True)])
        else:
            self._hide_banner()

    def _offer_resume(self, pending):
        percent = pending.get("percent", 0)
        self._show_banner(
            f"Your video of {Path(pending['photo']).stem} was interrupted at {percent} %. Continue where it left off?",
            [("Resume", lambda: self._resume_pending(pending), True), ("Start over", self._discard_pending, False)])

    def _resume_pending(self, pending):
        self._hide_banner()
        state = pending.get("state", {})
        self.photo_path = config.INPUT_DIR / pending["photo"]
        self._apply_look_state(state.get("look", {}))
        self._apply_output_state(state.get("output", {}))
        self._refresh_photo_tiles()
        self._start_render(resume_out_path=pending.get("out_path"))

    def _discard_pending(self):
        gui_system.clear_pending()
        gui_system.discard_render_checkpoints()
        self._show_setup_banner()
        self._refresh_estimate()

    # ------------------------------------------------------------------
    # State (session, saved looks)
    # ------------------------------------------------------------------
    def _apply_look_state(self, look):
        self._loading_state = True
        try:
            for attr, value in look.get("effects", {}).items():
                if attr in self.switch_vars:
                    self.switch_vars[attr].set(bool(value))
            if "intensity" in look:
                self.intensity_var.set(float(look["intensity"]))
            if "wind" in look:
                self.wind_var.set(float(look["wind"]))
            if "darkness" in look:
                self.darkness_var.set(float(look["darkness"]))
            self.adv_touched.clear()
            for attr, value in look.get("advanced", {}).items():
                if attr in self.adv_vars:
                    self.adv_touched.add(attr)
                    self.adv_vars[attr].set(float(value))
                    spec = EFFECT_INTENSITY[attr]
                    self.adv_labels[attr].configure(text=spec["fmt"].format(round(float(value), spec["digits"])))
            self.limit_area_var.set(bool(look.get("limit_area", False)))
        finally:
            self._loading_state = False
        mood = look.get("mood")
        self.mood_key = mood if mood in gui_moods.MOOD_BY_KEY else gui_moods.mood_key_for_switches(self._switch_states())
        self._refresh_look_labels()

    def _apply_output_state(self, output):
        if output.get("quality") in gui_moods.QUALITY_PRESETS:
            self.quality_var.set(output["quality"])
        if output.get("length") in gui_moods.LENGTH_CHOICES:
            self.length_var.set(output["length"])
        if output.get("fps") in gui_moods.FPS_CHOICES:
            self.fps_var.set(output["fps"])
        self._on_output_changed()

    _saved_output_keys = ()

    def _restore_session(self):
        try:
            with open(SESSION_PATH, "r", encoding="utf-8") as f:
                session = json.load(f)
            if not isinstance(session, dict) or session.get("version") != 2:
                return   # the Phase-1 session format is ignored rather than mis-read
            photo = session.get("photo")
            if photo in self._photo_names():
                self.photo_path = config.INPUT_DIR / photo
            self._apply_look_state(session.get("look", {}))
            self._saved_output_keys = tuple(session.get("output", {}))
            self._apply_output_state(session.get("output", {}))
            if session.get("advanced"):
                self.advanced_var.set(True)
                self._on_advanced_toggled()
            geometry = session.get("geometry")
            if isinstance(geometry, str) and geometry:
                size = geometry.split("+")[0]
                width, height = (int(v) for v in size.split("x"))
                if width <= self.winfo_screenwidth() and height <= self.winfo_screenheight():
                    self.geometry(geometry)
        except (OSError, ValueError, TypeError, AttributeError, KeyError, tk.TclError):
            pass  # a missing or stale session file must never stop the app from opening

    def _save_session(self):
        session = {
            "version": 2,
            "photo": self.photo_path.name if self.photo_path else None,
            "look": self._look_state(),
            "output": self._output_state(),
            "advanced": bool(self.advanced_var.get()),
            "geometry": self.geometry(),
            "appearance": self.appearance_label,
        }
        try:
            GUI_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
            with open(SESSION_PATH, "w", encoding="utf-8") as f:
                json.dump(session, f, indent=2)
        except OSError:
            pass

    def _save_look(self):
        name = ctk.CTkInputDialog(text="Name for this look:", title="Save my look").get_input()
        if not name:
            return
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip().replace(" ", "_")
        if not safe:
            self._set_info("A look needs a name with at least one letter or number.", WARN)
            return
        GUI_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        path = GUI_PRESETS_DIR / f"{safe}.json"
        payload = {"version": 2, "name": name, "saved_at": datetime.now().isoformat(timespec="seconds"),
                   "look": self._look_state(), "output": self._output_state()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self._set_info(f"Saved your look \"{name}\".", OK)

    def _load_look(self):
        files = sorted(p for p in GUI_PRESETS_DIR.glob("*.json")
                       if p.name not in (SESSION_PATH.name, gui_system.TIMING_PATH.name, gui_system.PENDING_PATH.name)) \
            if GUI_PRESETS_DIR.exists() else []
        if not files:
            self._set_info("You have not saved any looks yet. Use \"Save my look\" first.", WARN)
            return
        LookPicker(self, files, self._on_look_picked).grab_set()

    def _on_look_picked(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if state.get("version") == 2:
                self._apply_look_state(state.get("look", {}))
                self._apply_output_state(state.get("output", {}))
            else:   # older saved presets only stored the switches
                self._apply_look_state({"effects": state.get("effects", {})})
        except (OSError, ValueError) as exc:
            self._set_info(f"Could not open that look: {exc}", WARN)
            return
        self._set_info(f"Loaded your look \"{Path(path).stem}\".", OK)
        self._on_look_changed()

    # ------------------------------------------------------------------
    # My videos tab
    # ------------------------------------------------------------------
    def _build_videos_tab(self):
        tab = self.tabs["My videos"]
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        self.videos_list = ctk.CTkScrollableFrame(tab, width=440, fg_color=PANEL, corner_radius=0)
        self.videos_list.grid(row=0, column=0, sticky="nsw")
        right = ctk.CTkFrame(tab, fg_color=BG, corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)
        self.videos_canvas = tk.Canvas(right, bg=self._plain(BG), highlightthickness=0)
        self.videos_canvas.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        self.videos_preview = EmbeddedPreview(self.videos_canvas)
        self.videos_note = ctk.CTkLabel(right, text="Pick a video on the left to play it here.", text_color=MUTED,
                                        font=ctk.CTkFont(size=FONT_BODY))
        self.videos_note.grid(row=1, column=0, sticky="w", padx=16, pady=(0, 16))

    def _video_thumbnail(self, path):
        if path not in self._video_thumb_cache:
            try:
                import cv2
                capture = cv2.VideoCapture(str(path))
                ok, frame = capture.read()
                capture.release()
                if not ok:
                    raise RuntimeError("no frame")
                image = Image.fromarray(frame[:, :, ::-1])
                thumb = ImageOps.fit(image, (112, 72), Image.LANCZOS)
            except Exception:  # noqa: BLE001
                thumb = Image.new("RGB", (112, 72), "#333340")
            self._video_thumb_cache[path] = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=(112, 72))
        return self._video_thumb_cache[path]

    def _refresh_videos(self):
        for child in self.videos_list.winfo_children():
            child.destroy()
        videos = gui_system.list_videos()
        if not videos:
            ctk.CTkLabel(self.videos_list, text="No videos yet.\nMake one on the Create tab and it will appear here.",
                         text_color=MUTED, justify="left", font=ctk.CTkFont(size=FONT_BODY)).pack(padx=16, pady=24, anchor="w")
            return
        for video in videos:
            path = video["path"]
            card = ctk.CTkFrame(self.videos_list, fg_color=CARD, corner_radius=8)
            card.pack(fill="x", padx=PAD, pady=4)
            card.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(card, text="", image=self._video_thumbnail(path)).grid(row=0, column=0, rowspan=2, padx=8, pady=8)
            ctk.CTkLabel(card, text=path.stem, anchor="w", font=ctk.CTkFont(size=FONT_BODY, weight="bold")).grid(
                row=0, column=1, sticky="ew", pady=(8, 0), padx=(0, 8))
            details = f"{video['modified']:%d %b %Y %H:%M}  ·  {video['size_mb']:.0f} MB"
            if video["photo"]:
                details = f"{video['photo']}  ·  " + details
            ctk.CTkLabel(card, text=details, anchor="w", text_color=MUTED, font=ctk.CTkFont(size=FONT_SMALL),
                         wraplength=280, justify="left").grid(row=1, column=1, sticky="new", padx=(0, 8))
            buttons = ctk.CTkFrame(card, fg_color="transparent")
            buttons.grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8))
            _Button(buttons, text="Play", width=64, height=28, command=lambda p=path: self._play_video(p)).pack(side="left")
            look_button = _Button(buttons, text="Use this look", width=104, height=28, fg_color="transparent",
                                        border_width=1, command=lambda v=video: self._use_video_look(v))
            look_button.pack(side="left", padx=4)
            if not video["config"]:
                look_button.configure(state="disabled")
                Tooltip(look_button, "This video has no saved settings (it was made before they were recorded).")
            else:
                Tooltip(look_button, "Go back to the Create tab with this video's photo, effects and output settings.")
            _Button(buttons, text="Show in folder", width=104, height=28, fg_color="transparent", border_width=1,
                          command=lambda p=path: subprocess.Popen(["explorer", "/select,", str(p)])).pack(side="left")

    def _play_video(self, path):
        try:
            width, height, fps = probe_video(path)
        except Exception as exc:  # noqa: BLE001
            self.videos_note.configure(text=f"Could not open {path.name}: {exc}")
            return
        self.videos_note.configure(text=path.name)
        self.videos_preview.play(path, width, height, fps)
        self.last_output_path = path

    def _use_video_look(self, video):
        cfg = video["config"]
        self.videos_preview.stop()
        if video["photo"] and (config.INPUT_DIR / video["photo"]).exists():
            self.photo_path = config.INPUT_DIR / video["photo"]
            self._refresh_photo_tiles()
        effects = {attr: bool(cfg[attr]) for attr, _t in EFFECT_SWITCHES if attr in cfg}
        self._apply_look_state({"effects": effects, "intensity": self.intensity_var.get(), "wind": self.wind_var.get(),
                                "darkness": self.darkness_var.get()})
        output = {}
        size = cfg.get("LIVE_WALLPAPER_MAX_DIMENSION")
        for label, (max_side, _crf, _blurb) in gui_moods.QUALITY_PRESETS.items():
            if max_side == size:
                output["quality"] = label
        for label, seconds in gui_moods.LENGTH_CHOICES.items():
            if seconds == cfg.get("LIVE_WALLPAPER_DURATION_SECONDS"):
                output["length"] = label
        for label, fps in gui_moods.FPS_CHOICES.items():
            if fps == cfg.get("FPS"):
                output["fps"] = label
        self._apply_output_state(output)
        self._show_tab("Create")
        self._show_panel(None)
        self._set_info("Brought back the effects and output settings of that video. The strength sliders start from the mood's usual values.")
        self._after_photo_changed()

    # ------------------------------------------------------------------
    # Settings tab
    # ------------------------------------------------------------------
    def _build_settings_tab(self):
        tab = self.tabs["Settings"]
        holder = ctk.CTkFrame(tab, fg_color="transparent")
        holder.pack(fill="both", expand=True, padx=32, pady=24, anchor="nw")
        ctk.CTkLabel(holder, text="Your computer", font=ctk.CTkFont(size=FONT_TITLE, weight="bold")).pack(anchor="w")
        self.check_labels = {}
        for key in ("gpu", "ffmpeg", "models"):
            label = ctk.CTkLabel(holder, text="", anchor="w", justify="left", font=ctk.CTkFont(size=FONT_BODY), wraplength=700)
            label.pack(anchor="w", pady=(8, 0))
            self.check_labels[key] = label
        actions = ctk.CTkFrame(holder, fg_color="transparent")
        actions.pack(anchor="w", pady=(16, 0))
        self.download_button = _Button(actions, text="Download missing AI models", height=36, fg_color=ACCENT,
                                             hover_color=ACCENT_HOVER, command=self._download_models)
        self.download_button.pack(side="left")
        Tooltip(self.download_button, "Downloads into this app's models/ folder (a few GB). Needs an internet connection.")
        _Button(actions, text="Check again", height=36, fg_color="transparent", border_width=1,
                      command=self._recheck).pack(side="left", padx=PAD)
        self.download_status = ctk.CTkLabel(holder, text="", text_color=MUTED, anchor="w", font=ctk.CTkFont(size=FONT_SMALL),
                                            wraplength=700, justify="left")
        self.download_status.pack(anchor="w", pady=(8, 0))
        _Button(holder, text="View error log", width=160, fg_color="transparent", border_width=1,
                      command=lambda: self._show_details_log()).pack(anchor="w", pady=(32, 0))
        ctk.CTkLabel(holder, text="Appearance", font=ctk.CTkFont(size=FONT_TITLE, weight="bold")).pack(anchor="w", pady=(32, 0))
        self.appearance_label = _saved_appearance()
        ctk.CTkSegmentedButton(holder, values=list(APPEARANCE_CHOICES), command=self._on_appearance_changed,
                               font=ctk.CTkFont(size=FONT_BODY)).pack(anchor="w", pady=(8, 0))
        holder.winfo_children()[-1].set(self.appearance_label)
        ctk.CTkLabel(holder, text="Files", font=ctk.CTkFont(size=FONT_TITLE, weight="bold")).pack(anchor="w", pady=(32, 0))
        ctk.CTkLabel(holder, text=f"Photos: {config.INPUT_DIR}\nVideos: {config.LIVE_WALLPAPERS_DIR}\nAI models: {config.MODELS_DIR}",
                     justify="left", anchor="w", text_color=MUTED, font=ctk.CTkFont(size=FONT_SMALL)).pack(anchor="w", pady=(8, 0))
        _Button(holder, text="Open the videos folder", height=32, fg_color="transparent", border_width=1,
                      command=lambda: os.startfile(str(config.LIVE_WALLPAPERS_DIR))).pack(anchor="w", pady=(12, 0))

    def _refresh_settings(self):
        system = self.system
        gpu = system["gpu"]
        self.check_labels["gpu"].configure(
            text=f"{'✔' if gpu else '✖'}  Graphics card: {gpu or 'no NVIDIA card found - videos will be very slow (Draft quality is recommended)'}",
            text_color=OK if gpu else WARN)
        self.check_labels["ffmpeg"].configure(
            text=f"{'✔' if system['ffmpeg'] else '✖'}  Video encoder (ffmpeg): {'ready' if system['ffmpeg'] else 'missing - videos cannot be saved'}",
            text_color=OK if system["ffmpeg"] else WARN)
        missing = system["missing_models"]
        if missing:
            names = ", ".join(f"{repo} ({purpose})" for repo, purpose in missing)
            self.check_labels["models"].configure(text=f"✖  AI models missing: {names}", text_color=WARN)
            self.download_button.configure(state="normal")
        else:
            self.check_labels["models"].configure(text="✔  AI models: all installed", text_color=OK)
            self.download_button.configure(state="disabled")

    def _recheck(self):
        self.system = gui_system.check_system()
        self._refresh_settings()
        self._show_setup_banner()

    def _download_models(self):
        missing = [repo for repo, _purpose in self.system["missing_models"]]
        if not missing:
            return
        self.download_button.configure(state="disabled")
        self.download_status.configure(text="Starting the download…")

        def worker():
            failures = gui_system.download_models(missing, lambda repo, i, n: self.event_queue.put(("dl_progress", (repo, i, n), None)))
            self.event_queue.put(("dl_done", failures, None))

        threading.Thread(target=worker, daemon=True).start()

    def _on_download_done(self, failures):
        if failures:
            self.download_status.configure(text="Some models could not be downloaded: "
                                                + "; ".join(f"{repo} ({message[:80]})" for repo, message in failures), text_color=WARN)
        else:
            self.download_status.configure(text="All models downloaded.", text_color=OK)
        self._recheck()

    # ------------------------------------------------------------------
    # Closing
    # ------------------------------------------------------------------
    def _on_close(self):
        if self.rendering and not self.cancel_event.is_set():
            answer = messagebox.askyesnocancel(
                "A video is being made",
                "Your video is still being made.\n\n"
                "Yes  = stop now and close (you can continue it next time)\n"
                "No   = keep making it: the window just minimises\n"
                "Cancel = go back", parent=self)
            if answer is None:
                return
            if answer is False:
                self.iconify()
                return
            self.cancel_event.set()
        self._save_session()
        self.preview.stop()
        self.videos_preview.stop()
        self.destroy()


class LookPicker(ctk.CTkToplevel):
    def __init__(self, parent, files, on_pick):
        super().__init__(parent)
        self.title("My looks")
        self.geometry("420x360")
        ctk.CTkLabel(self, text="Choose a saved look:", font=ctk.CTkFont(size=FONT_BODY, weight="bold")).pack(
            padx=16, pady=(16, 8), anchor="w")
        scroll = ctk.CTkScrollableFrame(self, width=380, height=260)
        scroll.pack(padx=16, pady=8, fill="both", expand=True)
        for path in files:
            _Button(scroll, text=path.stem, anchor="w",
                          command=lambda p=path: (on_pick(p), self.destroy())).pack(fill="x", pady=3)


if __name__ == "__main__":
    config.make_sure_folders_exist()
    app = VfxApp()
    app.mainloop()
