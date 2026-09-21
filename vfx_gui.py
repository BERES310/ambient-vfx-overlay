"""
vfx_gui.py

A modern desktop GUI (CustomTkinter - dark theme, rounded controls) for
picking a photo + a set of effects, rendering it through live_wallpaper.py's
full pipeline, and previewing the looping result right inside the app -
built in response to: "check that live_preview it works... the gui for that
should be awesome so i can choose different effects and stuff... quite
modern" (2026-08-27).

Design notes:
    - live_wallpaper.process_image() reads every effect toggle straight off
      module-level `config.*` attributes (no per-call overrides exist in the
      pipeline itself) - so this GUI's switches just set those attributes
      before calling process_image(), then restore the previous values
      afterwards. See run_weather_benchmark.py for the same pattern used
      non-interactively.
    - Rendering runs on a background thread (the pipeline can take minutes:
      depth estimation, segmentation, per-frame compositing, two-pass ffmpeg
      encode) so the window stays responsive and shows live status instead
      of freezing.
    - The preview reuses the exact decode approach vfx_engine/preview_window.py
      already uses and proves works (bundled ffmpeg piping raw RGB frames to
      stdout, blitted onto a canvas) - just embedded directly in this
      window's own canvas instead of opening a second separate window, for
      a single cohesive "app" feel rather than a popup.

Run (from inside vfx_overlay/):

    python vfx_gui.py
"""

import json
import queue
import subprocess
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image, ImageTk

import config
import live_wallpaper
from fast_preview import render_fast_preview
from vfx_engine.image_io import find_input_images, get_ffmpeg_path

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_PREVIEW_MAX_DIMENSION = 960
_MAX_CONSECUTIVE_RESTARTS = 3   # give up (and show the ffmpeg error) instead of retrying forever - see vfx_engine/preview_window.py's identical constant

# Where "Save JSON Preset" writes to / "Load JSON Preset" reads from
# (2026-08-28 request: "a button to save a json preset").
GUI_PRESETS_DIR = Path(__file__).resolve().parent / "gui_presets"

# Effect switch/slider/preset definitions now live in
# vfx_engine/effect_params.py so vfx_engine/preview_window.py's standalone
# control panel can share the exact same wiring instead of duplicating it -
# see that module's docstring.
from vfx_engine.effect_params import (
    EFFECT_INTENSITY, EFFECT_SWITCHES, WEATHER_PRESETS, build_rain_intensity_dict,
    current_intensity_default as _current_intensity_default,
    resolve_intensity_override as _resolve_intensity_override,
)


class EmbeddedPreview:
    """Loops an .mp4 on a tk.Canvas by piping raw frames from the bundled
    ffmpeg - the same mechanism vfx_engine/preview_window.py uses, just
    driven from `after()` on this app's own mainloop instead of its own."""

    def __init__(self, canvas, on_error=None):
        self.canvas = canvas
        self.on_error = on_error
        self.process = None
        self.photo = None
        self.display_size = (1, 1)
        self.source_size = (1, 1)
        self.frame_bytes = 0
        self.fps = 30
        self.video_path = None
        self.restart_count = 0
        self._closed = True
        self._after_id = None
        self.fast_frames = None
        self.raw_frames = None
        self.fast_index = 0
        self._resize_after_id = None
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def stop(self):
        self._closed = True
        if self._after_id is not None:
            self.canvas.after_cancel(self._after_id)
            self._after_id = None
        if self.process is not None:
            self.process.terminate()
            self.process = None
        self.fast_frames = None
        self.raw_frames = None
        self.canvas.delete("all")

    def _fit_size(self, source_w, source_h):
        """Scales (source_w, source_h) up or down to fill as much of the
        canvas's current on-screen area as possible while preserving aspect
        ratio - fixes the preview staying pinned to a small fixed size
        (see _PREVIEW_MAX_DIMENSION's old use as a hard cap) instead of
        filling the actual window."""
        container_w = self.canvas.winfo_width()
        container_h = self.canvas.winfo_height()
        if container_w <= 1 or container_h <= 1:
            container_w, container_h = _PREVIEW_MAX_DIMENSION, int(_PREVIEW_MAX_DIMENSION * source_h / source_w)
        scale = min(container_w / source_w, container_h / source_h)
        return max(1, int(source_w * scale)), max(1, int(source_h * scale))

    def _on_canvas_resize(self, event):
        if self._closed:
            return
        # Debounce: dragging a window edge fires many <Configure> events per
        # second, and the video path restarts an ffmpeg decoder on every
        # resize - only act once the size has settled.
        if self._resize_after_id is not None:
            self.canvas.after_cancel(self._resize_after_id)
        self._resize_after_id = self.canvas.after(150, self._apply_resize)

    def _apply_resize(self):
        self._resize_after_id = None
        if self._closed:
            return
        if self.raw_frames is not None:
            self._rescale_fast_frames()
        elif self.video_path is not None:
            new_size = self._fit_size(*self.source_size)
            if new_size != self.display_size:
                self.display_size = new_size
                self.frame_bytes = self.display_size[0] * self.display_size[1] * 3
                if self.process is not None:
                    self.process.terminate()
                self._start_reader()

    def _rescale_fast_frames(self):
        new_size = self._fit_size(*self.source_size)
        self.display_size = new_size
        self.fast_frames = [
            ImageTk.PhotoImage(frame.resize(self.display_size, Image.LANCZOS))
            for frame in self.raw_frames
        ]

    def play_frames(self, frames, width, height, fps):
        """Loops an in-memory list of PIL Images on the canvas instead of
        piping an mp4 through ffmpeg - the fast/draft preview path (see
        fast_preview.py). Mutually exclusive with play(); calling either
        stops the other."""
        self.stop()
        self._closed = False
        self.fps = fps
        self.source_size = (width, height)
        self.raw_frames = frames
        self.display_size = self._fit_size(width, height)
        self.fast_frames = [
            ImageTk.PhotoImage(frame.resize(self.display_size, Image.LANCZOS))
            for frame in frames
        ]
        self.fast_index = 0
        self._show_next_fast_frame()

    def _show_next_fast_frame(self):
        if self._closed or self.fast_frames is None:
            return
        self.photo = self.fast_frames[self.fast_index]
        self.fast_index = (self.fast_index + 1) % len(self.fast_frames)
        self.canvas.delete("frame")
        cx, cy = self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2
        self.canvas.create_image(cx, cy, anchor="center", image=self.photo, tags="frame")
        self._after_id = self.canvas.after(max(1, int(1000 / self.fps)), self._show_next_fast_frame)

    def _start_reader(self):
        command = [
            get_ffmpeg_path(), "-loglevel", "error",
            "-stream_loop", "-1",
            "-i", str(self.video_path),
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.display_size[0]}x{self.display_size[1]}",
            "-",
        ]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _fail(self, message):
        # Surface the problem instead of silently leaving a frozen/blank
        # canvas forever (the previous behavior on a decode failure: set
        # self._closed and just stop, with zero feedback to the user and no
        # indication anything went wrong) - matches
        # vfx_engine/preview_window.py's fail() for the same underlying
        # ffmpeg-piping mechanism.
        self._closed = True
        if self.process is not None:
            self.process.terminate()
            self.process = None
        print(f"  [preview] {message}")
        if self.on_error is not None:
            self.on_error(message)
        try:
            messagebox.showerror("Live Wallpaper Preview", message, parent=self.canvas)
        except tk.TclError:
            pass

    def play(self, video_path, width, height, fps):
        self.stop()
        self._closed = False
        self.video_path = video_path
        self.fps = fps
        self.restart_count = 0
        self.source_size = (width, height)
        self.display_size = self._fit_size(width, height)
        self.frame_bytes = self.display_size[0] * self.display_size[1] * 3

        self._start_reader()
        self._read_frame()

    def _read_frame(self):
        if self._closed or self.process is None:
            return
        raw = self.process.stdout.read(self.frame_bytes)
        if len(raw) < self.frame_bytes:
            self.restart_count += 1
            if self.restart_count > _MAX_CONSECUTIVE_RESTARTS:
                stderr_output = self.process.stderr.read().decode(errors="ignore").strip()
                detail = f":\n\n{stderr_output}" if stderr_output else " (no ffmpeg error output)."
                self._fail(f"Could not decode the preview video after {_MAX_CONSECUTIVE_RESTARTS} attempts{detail}")
                return

            # Shouldn't normally happen (ffmpeg loops on its own), but
            # restart the decoder if it ever does, instead of freezing the
            # preview.
            self.process.terminate()
            self._start_reader()
            self._after_id = self.canvas.after(10, self._read_frame)
            return

        self.restart_count = 0
        frame = Image.frombytes("RGB", self.display_size, raw)
        self.photo = ImageTk.PhotoImage(frame)
        self.canvas.delete("frame")
        cx, cy = self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2
        self.canvas.create_image(cx, cy, anchor="center", image=self.photo, tags="frame")
        self._after_id = self.canvas.after(max(1, int(1000 / self.fps)), self._read_frame)


class VfxApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VFX Overlay - Live Wallpaper Studio")
        self.geometry("1180x760")
        self.minsize(980, 640)

        self.event_queue = queue.Queue()
        self.rendering = False
        self.last_output_path = None
        self.switch_vars = {}
        self.intensity_vars = {}
        self.intensity_value_labels = {}

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_preview_area()

        self.after(100, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- sidebar (controls) ----

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)

        ctk.CTkLabel(sidebar, text="VFX Overlay", font=ctk.CTkFont(size=22, weight="bold")).pack(padx=20, pady=(24, 0), anchor="w")
        ctk.CTkLabel(sidebar, text="Live Wallpaper Studio", font=ctk.CTkFont(size=13), text_color="gray70").pack(padx=20, pady=(0, 20), anchor="w")

        ctk.CTkLabel(sidebar, text="PHOTO", font=ctk.CTkFont(size=11, weight="bold"), text_color="gray60").pack(padx=20, anchor="w")
        image_names = [p.name for p in find_input_images(config.INPUT_DIR)] or ["(no photos in input_images/)"]
        self.photo_var = tk.StringVar(value=image_names[0])
        ctk.CTkOptionMenu(sidebar, values=image_names, variable=self.photo_var, width=280).pack(padx=20, pady=(6, 20))

        ctk.CTkLabel(sidebar, text="WEATHER PRESET", font=ctk.CTkFont(size=11, weight="bold"), text_color="gray60").pack(padx=20, anchor="w")
        self.preset_var = tk.StringVar(value="Custom")
        ctk.CTkOptionMenu(sidebar, values=list(WEATHER_PRESETS.keys()), variable=self.preset_var,
                           command=self._apply_preset_to_switches, width=280).pack(padx=20, pady=(6, 20))

        ctk.CTkLabel(sidebar, text="EFFECTS", font=ctk.CTkFont(size=11, weight="bold"), text_color="gray60").pack(padx=20, anchor="w")
        effects_frame = ctk.CTkScrollableFrame(sidebar, fg_color="transparent", height=260)
        effects_frame.pack(padx=20, pady=(6, 16), fill="x")
        for attr, label in EFFECT_SWITCHES:
            var = tk.BooleanVar(value=bool(getattr(config, attr)))
            switch = ctk.CTkSwitch(effects_frame, text=label, variable=var,
                                    command=lambda: self.preset_var.set("Custom"))
            switch.pack(anchor="w", pady=(8, 0), fill="x")
            self.switch_vars[attr] = var

            # Per-effect intensity slider ("how heavy") - always visible
            # (not just when the switch is on) so it can be dialed in
            # ahead of time; it's a no-op in the render whenever that
            # effect's own switch is off, same as any other config value.
            spec = EFFECT_INTENSITY[attr]
            default_value = _current_intensity_default(attr)
            row = ctk.CTkFrame(effects_frame, fg_color="transparent")
            row.pack(fill="x", pady=(2, 6))
            value_label = ctk.CTkLabel(row, text=spec["fmt"].format(round(default_value, spec["digits"])), text_color="gray60", font=ctk.CTkFont(size=11))
            value_label.pack(anchor="e")
            self.intensity_value_labels[attr] = value_label

            slider_var = tk.DoubleVar(value=default_value)
            self.intensity_vars[attr] = slider_var
            slider = ctk.CTkSlider(
                row, from_=spec["range"][0], to=spec["range"][1], variable=slider_var,
                command=lambda value, a=attr: self._on_intensity_change(a, value),
            )
            slider.pack(fill="x")

        self.render_button = ctk.CTkButton(sidebar, text="Render + Preview", height=42,
                                            font=ctk.CTkFont(size=14, weight="bold"), command=self._start_render)
        self.render_button.pack(padx=20, pady=(10, 6), fill="x")

        # Fast, low-res/short/AI-skipping draft render (see fast_preview.py) -
        # for quickly eyeballing a parameter change without waiting for the
        # full-quality render's depth/segmentation/two-pass encode. 2026-09-04
        # request: "is it possible to render it in real time? just for
        # preview purposes of course quality could be much lower".
        self.fast_preview_button = ctk.CTkButton(sidebar, text="Fast Preview (draft)", height=36,
                                                  fg_color="transparent", border_width=1, command=self._start_fast_preview)
        self.fast_preview_button.pack(padx=20, pady=(0, 6), fill="x")

        self.load_existing_button = ctk.CTkButton(sidebar, text="Preview Existing Render", height=36,
                                                    fg_color="transparent", border_width=1, command=self._preview_existing)
        self.load_existing_button.pack(padx=20, pady=(0, 6), fill="x")

        preset_io_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        preset_io_frame.pack(padx=20, pady=(0, 20), fill="x")
        preset_io_frame.grid_columnconfigure((0, 1), weight=1)
        self.save_preset_button = ctk.CTkButton(preset_io_frame, text="Save JSON Preset", height=32,
                                                 fg_color="transparent", border_width=1, command=self._save_json_preset)
        self.save_preset_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.load_preset_button = ctk.CTkButton(preset_io_frame, text="Load JSON Preset", height=32,
                                                 fg_color="transparent", border_width=1, command=self._load_json_preset)
        self.load_preset_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.status_label = ctk.CTkLabel(sidebar, text="Ready.", text_color="gray70", wraplength=280, justify="left")
        self.status_label.pack(padx=20, pady=(0, 10), anchor="w")

        self.progress_bar = ctk.CTkProgressBar(sidebar, width=280)
        self.progress_bar.set(0)
        self.progress_bar.pack(padx=20, pady=(0, 10))

    def _apply_preset_to_switches(self, preset_name):
        overrides = WEATHER_PRESETS.get(preset_name)
        if overrides is None:
            return
        for attr, value in overrides.items():
            if attr in self.switch_vars:
                self.switch_vars[attr].set(value)

    def _on_intensity_change(self, attr, value):
        spec = EFFECT_INTENSITY[attr]
        self.intensity_value_labels[attr].configure(text=spec["fmt"].format(round(float(value), spec["digits"])))
        self.preset_var.set("Custom")

    # ---- preview area ----

    def _build_preview_area(self):
        preview_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#101014")
        preview_frame.grid(row=0, column=1, sticky="nsew")
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(preview_frame, bg="#101014", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.create_text(
            10, 10, anchor="nw", tags="placeholder", fill="gray50",
            text="Pick a photo + effects, then click \"Render + Preview\".\n"
                 "(Or click \"Preview Existing Render\" to loop a file already on disk.)",
            font=("Segoe UI", 13),
        )
        self.preview = EmbeddedPreview(self.canvas, on_error=lambda message: self.status_label.configure(text=f"Preview error: {message}"))

    # ---- render flow ----

    def _start_render(self):
        if self.rendering:
            return
        image_names = [p.name for p in find_input_images(config.INPUT_DIR)]
        if self.photo_var.get() not in image_names:
            self.status_label.configure(text="No valid photo selected.")
            return

        # vfx_engine.mask._load_or_draw_mask (behind MASK_MODE = "manual"
        # and/or MASK_MANUAL_CORRECTION) opens its own tk.Tk() drawing
        # window (mask_editor.py) the first time a given photo needs one -
        # but process_image() runs on this GUI's background render thread
        # (see _render_worker below), and Tkinter is not thread-safe: a
        # second Tk() root created from a non-main thread can crash or
        # silently hang the whole app instead of showing the drawing
        # window. Refuse up front with a clear message rather than let
        # that happen partway through a render - the mask files it would
        # produce can still be created by running main.py/live_wallpaper.py
        # directly (or the standalone mask_editor.py) from a terminal,
        # which run on the main thread.
        needs_manual_mask_editor = config.MASK_MODE == "manual" or config.MASK_MANUAL_CORRECTION
        if needs_manual_mask_editor:
            mask_stem = Path(self.photo_var.get()).stem
            suffixes = []
            if config.MASK_MODE == "manual" and not (config.MASKS_DIR / f"{mask_stem}.png").exists():
                suffixes.append(f"{mask_stem}.png")
            if config.MASK_MANUAL_CORRECTION and not (config.MASKS_DIR / f"{mask_stem}_correction.png").exists():
                suffixes.append(f"{mask_stem}_correction.png")
            if suffixes:
                self.status_label.configure(
                    text="Manual mask drawing isn't supported from this GUI (needs "
                         + ", ".join(suffixes)
                         + f" in {config.MASKS_DIR} - draw it first via mask_editor.py or main.py/live_wallpaper.py from a terminal)."
                )
                return

        self.rendering = True
        self.render_button.configure(state="disabled", text="Rendering...")
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        self.status_label.configure(text="Rendering - this can take a few minutes (depth + segmentation + compositing + encode) ...")
        self.canvas.delete("placeholder")

        image_path = config.INPUT_DIR / self.photo_var.get()
        overrides = {attr: var.get() for attr, var in self.switch_vars.items()}
        overrides.update(self._collect_intensity_overrides())
        preset_label = self.preset_var.get().lower().replace(" / ", "_").replace(" ", "_") if self.preset_var.get() != "Custom" else None

        thread = threading.Thread(target=self._render_worker, args=(image_path, overrides, preset_label), daemon=True)
        thread.start()

    def _collect_intensity_overrides(self):
        """Turns every slider's current raw value into real config
        attribute overrides (see _resolve_intensity_override) - the same
        {config_attr: value} shape _render_worker already knows how to
        apply-and-restore for the boolean switches."""
        result = {}
        for attr, var in self.intensity_vars.items():
            config_attr, value = _resolve_intensity_override(attr, var.get())
            result[config_attr] = value
        return result

    def _render_worker(self, image_path, overrides, preset_label):
        original_values = {attr: getattr(config, attr) for attr in overrides}
        original_preview = config.LIVE_PREVIEW_WINDOW
        config.LIVE_PREVIEW_WINDOW = False  # this app IS the preview - never also pop the old standalone window
        try:
            for attr, value in overrides.items():
                setattr(config, attr, value)
            start = time.perf_counter()
            live_wallpaper.process_image(image_path, preset_label=preset_label)
            duration = time.perf_counter() - start
            out_path = config.LIVE_WALLPAPERS_DIR / f"{image_path.stem}{('_' + preset_label) if preset_label else live_wallpaper.resolve_effect_suffix()}.mp4"
            self.event_queue.put(("done", out_path, duration))
        except Exception as exc:  # noqa: BLE001 - surface any render failure to the GUI instead of a silent dead thread
            self.event_queue.put(("error", str(exc), None))
        finally:
            for attr, value in original_values.items():
                setattr(config, attr, value)
            config.LIVE_PREVIEW_WINDOW = original_preview

    def _start_fast_preview(self):
        if self.rendering:
            return
        image_names = [p.name for p in find_input_images(config.INPUT_DIR)]
        if self.photo_var.get() not in image_names:
            self.status_label.configure(text="No valid photo selected.")
            return

        self.rendering = True
        self.render_button.configure(state="disabled")
        self.fast_preview_button.configure(state="disabled", text="Rendering draft...")
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        self.status_label.configure(text="Rendering fast draft preview (low-res, ~2.5s, AI passes skipped) ...")
        self.canvas.delete("placeholder")

        image_path = config.INPUT_DIR / self.photo_var.get()
        overrides = {attr: var.get() for attr, var in self.switch_vars.items()}
        overrides.update(self._collect_intensity_overrides())

        thread = threading.Thread(target=self._fast_preview_worker, args=(image_path, overrides), daemon=True)
        thread.start()

    def _fast_preview_worker(self, image_path, overrides):
        original_values = {attr: getattr(config, attr) for attr in overrides}
        try:
            for attr, value in overrides.items():
                setattr(config, attr, value)
            result = render_fast_preview(image_path)
            self.event_queue.put(("fast_done", result, None))
        except Exception as exc:  # noqa: BLE001 - surface any draft-render failure to the GUI instead of a silent dead thread
            self.event_queue.put(("fast_error", str(exc), None))
        finally:
            for attr, value in original_values.items():
                setattr(config, attr, value)

    def _preview_existing(self):
        out_dir = config.LIVE_WALLPAPERS_DIR
        files = sorted(out_dir.glob("*.mp4")) if out_dir.exists() else []
        if not files:
            self.status_label.configure(text=f"No .mp4 files yet in {out_dir}")
            return
        picker = ExistingRenderPicker(self, files, self._play_path)
        picker.grab_set()

    def _play_path(self, path):
        self.canvas.delete("placeholder")
        try:
            width, height, fps = _probe_video(path)
        except Exception as exc:
            self.status_label.configure(text=f"Could not read video info for {path.name}: {exc}")
            return
        self.status_label.configure(text=f"Previewing: {path.name}")
        self.preview.play(path, width, height, fps)

    def _poll_queue(self):
        try:
            while True:
                kind, payload, extra = self.event_queue.get_nowait()
                if kind == "done":
                    out_path, duration = payload, extra
                    self.last_output_path = out_path
                    self.status_label.configure(text=f"Done in {duration:.1f}s -> {out_path.name}")
                    try:
                        width, height, fps = _probe_video(out_path)
                        self.preview.play(out_path, width, height, fps)
                    except Exception as exc:
                        self.status_label.configure(text=f"Rendered {out_path.name}, but could not read video info for preview: {exc}")
                    self._finish_render_ui()
                elif kind == "error":
                    self.status_label.configure(text=f"Render failed: {payload}")
                    self._finish_render_ui()
                elif kind == "fast_done":
                    frames, width, height, fps = payload
                    self.status_label.configure(text=f"Fast draft preview ({len(frames)} frames @ {width}x{height}) - not final quality.")
                    self.preview.play_frames(frames, width, height, fps)
                    self._finish_render_ui()
                elif kind == "fast_error":
                    self.status_label.configure(text=f"Fast preview failed: {payload}")
                    self._finish_render_ui()
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)

    def _finish_render_ui(self):
        self.rendering = False
        self.render_button.configure(state="normal", text="Render + Preview")
        self.fast_preview_button.configure(state="normal", text="Fast Preview (draft)")
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(1)

    def _on_close(self):
        self.preview.stop()
        self.destroy()

    # ---- JSON presets (2026-08-28 request: "a button to save a json preset") ----

    def _current_preset_state(self):
        return {
            "effects": {attr: bool(var.get()) for attr, var in self.switch_vars.items()},
            "intensity": {attr: var.get() for attr, var in self.intensity_vars.items()},
        }

    def _apply_preset_state(self, state):
        for attr, value in state.get("effects", {}).items():
            if attr in self.switch_vars:
                self.switch_vars[attr].set(bool(value))
        for attr, value in state.get("intensity", {}).items():
            if attr in self.intensity_vars:
                self.intensity_vars[attr].set(float(value))
                self._on_intensity_change(attr, float(value))
        self.preset_var.set("Custom")

    def _save_json_preset(self):
        dialog = ctk.CTkInputDialog(text="Preset name:", title="Save JSON Preset")
        name = dialog.get_input()
        if not name:
            return
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip().replace(" ", "_")
        if not safe_name:
            self.status_label.configure(text="Preset name must contain at least one letter/number.")
            return

        GUI_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        preset_path = GUI_PRESETS_DIR / f"{safe_name}.json"
        payload = {
            "name": name,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            **self._current_preset_state(),
        }
        with open(preset_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.status_label.configure(text=f"Saved preset: {preset_path.name}")

    def _load_json_preset(self):
        files = sorted(GUI_PRESETS_DIR.glob("*.json")) if GUI_PRESETS_DIR.exists() else []
        if not files:
            self.status_label.configure(text=f"No saved presets yet in {GUI_PRESETS_DIR}")
            return
        picker = JsonPresetPicker(self, files, self._on_preset_picked)
        picker.grab_set()

    def _on_preset_picked(self, path):
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self._apply_preset_state(state)
        self.status_label.configure(text=f"Loaded preset: {path.name}")


class ExistingRenderPicker(ctk.CTkToplevel):
    def __init__(self, parent, files, on_pick):
        super().__init__(parent)
        self.title("Preview Existing Render")
        self.geometry("420x360")
        ctk.CTkLabel(self, text="Choose a rendered .mp4:", font=ctk.CTkFont(size=14, weight="bold")).pack(padx=16, pady=(16, 8), anchor="w")
        scroll = ctk.CTkScrollableFrame(self, width=380, height=260)
        scroll.pack(padx=16, pady=8, fill="both", expand=True)
        for path in files:
            btn = ctk.CTkButton(scroll, text=path.name, anchor="w",
                                 command=lambda p=path: (on_pick(p), self.destroy()))
            btn.pack(fill="x", pady=3)


class JsonPresetPicker(ctk.CTkToplevel):
    def __init__(self, parent, files, on_pick):
        super().__init__(parent)
        self.title("Load JSON Preset")
        self.geometry("420x360")
        ctk.CTkLabel(self, text="Choose a saved preset:", font=ctk.CTkFont(size=14, weight="bold")).pack(padx=16, pady=(16, 8), anchor="w")
        scroll = ctk.CTkScrollableFrame(self, width=380, height=260)
        scroll.pack(padx=16, pady=8, fill="both", expand=True)
        for path in files:
            btn = ctk.CTkButton(scroll, text=path.stem, anchor="w",
                                 command=lambda p=path: (on_pick(p), self.destroy()))
            btn.pack(fill="x", pady=3)


def _probe_video(path):
    """Reads width/height/fps back out of the mp4 itself via OpenCV's
    VideoCapture (cv2 is already a hard project dependency - see
    vfx_engine/segmentation.py - and, unlike ffprobe, is guaranteed present:
    imageio-ffmpeg only bundles the ffmpeg binary itself, not ffprobe, so a
    path derived by swapping "ffmpeg" for "ffprobe" in get_ffmpeg_path()'s
    result never actually exists and silently fell through to hardcoded
    1920x1080 defaults for every file, misaligning/stretching the preview
    for any render at a different resolution). Avoids trusting a filename or
    hardcoding config.FPS for files that may have been rendered under a
    different config."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video file: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if width <= 0 or height <= 0 or fps <= 0:
            raise RuntimeError(f"OpenCV reported invalid video properties for {path}: {width}x{height}@{fps}")
        return width, height, fps
    finally:
        capture.release()


if __name__ == "__main__":
    config.make_sure_folders_exist()
    app = VfxApp()
    app.mainloop()
