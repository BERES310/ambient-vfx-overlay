"""
preview_window.py

Opens a small tkinter window that plays an exported live-wallpaper .mp4
file on a continuous loop, decoded with our own bundled ffmpeg binary - so
you can preview the loop without leaving the program or opening a separate
video player. Controlled by config.LIVE_PREVIEW_WINDOW (see
live_wallpaper.py). Closes on Escape or the window's close button.

When `on_rerender` is supplied (live_wallpaper.py always passes one), the
window also shows a side panel with the same weather-preset dropdown +
effect switches + intensity sliders as vfx_gui.py (see
vfx_engine/effect_params.py) - so running `python live_wallpaper.py`
directly still lets you choose parameters for the final render, the same
way the GUI does, instead of only being able to play back whatever config.py
happened to have set already. Added in response to: "in the live preview
why cannot i choose parameters i was supposed to be choosing parameters for
the final render" (2026-09-04) - previously this window had zero controls.

`on_fast_preview`, if also supplied, wires up a second "Fast Preview (draft)"
button that renders fast_preview.render_fast_preview()'s small/short/cached
in-memory frame list (see fast_preview.py) and loops THAT on the same canvas
instead of the full-quality mp4 - a much quicker (seconds, not minutes) way
to see roughly what a parameter change looks like, at lower resolution/
duration/quality. A "Back to full preview" button returns to the normal mp4
loop. Added in response to: "is it possible to render it in real time? just
for preview purposes of course quality could be much lower" (2026-09-04).
"""

import subprocess
import threading
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk

import config
from vfx_engine.effect_params import (
    EFFECT_INTENSITY, EFFECT_SWITCHES, WEATHER_PRESETS, current_intensity_default,
    resolve_intensity_override,
)
from vfx_engine.image_io import get_ffmpeg_path

_PREVIEW_MAX_DIMENSION = 1280   # downscale big renders for a snappier, lighter preview window
_MAX_CONSECUTIVE_RESTARTS = 3   # give up (and show the ffmpeg error) instead of retrying forever


def play_loop(video_path, width, height, fps, window_title="Live Wallpaper Preview", on_rerender=None, on_fast_preview=None):
    """
    Opens a window that loops `video_path` (a video file already on disk,
    e.g. the one live_wallpaper.py just exported) until it's closed.
    `width`/`height`/`fps` must match the video's own properties.

    `on_rerender`, if given, is called as on_rerender(overrides) -> (new_path,
    new_width, new_height, new_fps) whenever the user changes a control and
    clicks "Re-render" - `overrides` is a dict of {config_attr: value} to
    apply for that render (same shape vfx_gui.py's _start_render collects).
    Runs on a background thread so the window stays responsive during the
    (potentially minutes-long) render.

    `on_fast_preview`, if given, is called as on_fast_preview(overrides) ->
    (frames, width, height, fps) where `frames` is a list of in-memory PIL
    Images (see fast_preview.render_fast_preview) - looped directly on the
    canvas as a quick low-quality draft instead of decoding another mp4.
    """
    scale = min(1.0, _PREVIEW_MAX_DIMENSION / max(width, height))
    display_size = [max(1, int(width * scale)), max(1, int(height * scale))]
    state = {
        "process": None, "photo": None, "closed": False, "restart_count": 0,
        "frame_bytes": display_size[0] * display_size[1] * 3, "fps": fps, "rendering": False,
        "fast_frames": None, "fast_index": 0, "fast_after_id": None,
    }

    root = tk.Tk()
    root.title(window_title)

    canvas = tk.Canvas(root, width=display_size[0], height=display_size[1])
    canvas.pack(side="left")

    # A freshly-created Tk window can otherwise open behind the console/IDE
    # window with no visible indication it exists at all, which looks
    # identical to "nothing happened". Force it to the front once, then drop
    # the always-on-top flag so it behaves like a normal window afterwards.
    root.lift()
    root.attributes("-topmost", True)
    root.after(200, lambda: root.attributes("-topmost", False))
    root.focus_force()

    def start_reader():
        command = [
            get_ffmpeg_path(), "-loglevel", "error",
            "-stream_loop", "-1",   # loop the input file indefinitely at the demuxer level
            "-i", str(video_path),
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{display_size[0]}x{display_size[1]}",
            "-",
        ]
        state["process"] = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def fail(message):
        # Surface the problem instead of silently leaving a blank window
        # open forever (the previous behavior on a decode failure: retry
        # start_reader() in an unbounded ~10ms loop with stderr thrown away,
        # so a genuine ffmpeg failure just looked like a frozen/broken
        # preview with zero feedback).
        state["closed"] = True
        if state["process"] is not None:
            state["process"].terminate()
        print(f"  [preview] {message}")
        try:
            messagebox.showerror("Live Wallpaper Preview", message, parent=root)
        except tk.TclError:
            pass
        root.destroy()

    def read_and_show_frame():
        if state["closed"] or state["rendering"] or state["fast_frames"] is not None:
            return

        raw = state["process"].stdout.read(state["frame_bytes"])
        if len(raw) < state["frame_bytes"]:
            state["restart_count"] += 1
            if state["restart_count"] > _MAX_CONSECUTIVE_RESTARTS:
                stderr_output = state["process"].stderr.read().decode(errors="ignore").strip()
                state["process"].terminate()
                detail = f":\n\n{stderr_output}" if stderr_output else " (no ffmpeg error output)."
                fail(f"Could not decode the preview video after {_MAX_CONSECUTIVE_RESTARTS} attempts{detail}")
                return

            # Shouldn't normally happen (ffmpeg loops on its own), but
            # restart the decoder if it ever does, instead of freezing the
            # preview window.
            state["process"].terminate()
            start_reader()
            root.after(10, read_and_show_frame)
            return

        state["restart_count"] = 0
        frame = Image.frombytes("RGB", display_size, raw)
        state["photo"] = ImageTk.PhotoImage(frame)   # kept in `state` so it isn't garbage-collected before tkinter draws it
        canvas.create_image(0, 0, anchor="nw", image=state["photo"])
        root.after(max(1, int(1000 / state["fps"])), read_and_show_frame)

    def on_close():
        state["closed"] = True
        if state["process"] is not None:
            state["process"].terminate()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<Escape>", lambda event: on_close())

    if on_rerender is not None:
        _build_control_panel(root, canvas, state, on_rerender, on_fast_preview, display_size, start_reader, read_and_show_frame)

    start_reader()
    root.after(10, read_and_show_frame)
    root.mainloop()


def _build_control_panel(root, canvas, state, on_rerender, on_fast_preview, display_size, start_reader, read_and_show_frame):
    """Adds the weather-preset dropdown + effect switches + intensity
    sliders (see vfx_engine/effect_params.py) alongside the canvas, plus a
    "Re-render" button that swaps in the new output once it's ready."""
    panel = tk.Frame(root, padx=10, pady=10)
    panel.pack(side="right", fill="y")

    tk.Label(panel, text="Weather preset").pack(anchor="w")
    preset_var = tk.StringVar(value="Custom")
    tk.OptionMenu(panel, preset_var, *WEATHER_PRESETS.keys()).pack(anchor="w", fill="x", pady=(0, 10))

    switch_vars = {}
    intensity_vars = {}

    def apply_preset(*_args):
        overrides = WEATHER_PRESETS.get(preset_var.get())
        if overrides is None:
            return
        for attr, value in overrides.items():
            switch_vars[attr].set(value)

    preset_var.trace_add("write", apply_preset)

    for attr, label in EFFECT_SWITCHES:
        row = tk.Frame(panel)
        row.pack(anchor="w", fill="x", pady=(4, 0))
        var = tk.BooleanVar(value=bool(getattr(config, attr)))
        switch_vars[attr] = var
        tk.Checkbutton(row, text=label, variable=var).pack(side="left")

        spec = EFFECT_INTENSITY[attr]
        lo, hi = spec["range"]
        ivar = tk.DoubleVar(value=current_intensity_default(attr))
        intensity_vars[attr] = ivar
        tk.Scale(
            panel, from_=lo, to=hi, resolution=(hi - lo) / 200.0, orient="horizontal",
            variable=ivar, showvalue=False, length=180,
        ).pack(anchor="w", fill="x")

    status_var = tk.StringVar(value="")
    status_label = tk.Label(panel, textvariable=status_var, wraplength=200, justify="left")
    status_label.pack(anchor="w", pady=(10, 4))

    def collect_overrides():
        overrides = {}
        for attr, _label in EFFECT_SWITCHES:
            overrides[attr] = switch_vars[attr].get()
        for attr, _label in EFFECT_SWITCHES:
            config_attr, value = resolve_intensity_override(attr, intensity_vars[attr].get())
            overrides[config_attr] = value
        return overrides

    def rerender_worker(overrides):
        try:
            result = on_rerender(overrides)
        except Exception as exc:  # surface pipeline errors instead of freezing the panel
            root.after(0, lambda: rerender_failed(exc))
            return
        root.after(0, lambda: rerender_done(result))

    def rerender_failed(exc):
        state["rendering"] = False
        rerender_button.configure(state="normal")
        status_var.set(f"Render failed: {exc}")

    def rerender_done(result):
        new_path, new_width, new_height, new_fps = result
        new_scale = min(1.0, _PREVIEW_MAX_DIMENSION / max(new_width, new_height))
        display_size[0] = max(1, int(new_width * new_scale))
        display_size[1] = max(1, int(new_height * new_scale))
        state["frame_bytes"] = display_size[0] * display_size[1] * 3
        state["fps"] = new_fps
        canvas.configure(width=display_size[0], height=display_size[1])
        if state["process"] is not None:
            state["process"].terminate()
        if state["fast_after_id"] is not None:
            root.after_cancel(state["fast_after_id"])
            state["fast_after_id"] = None
        state["fast_frames"] = None   # a fresh full render supersedes any fast draft that was showing
        state["restart_count"] = 0
        start_reader()
        state["rendering"] = False
        rerender_button.configure(state="normal")
        status_var.set(f"Now previewing: {new_path.name}")
        read_and_show_frame()

    def on_rerender_click():
        if state["rendering"] or state["closed"]:
            return
        state["rendering"] = True
        rerender_button.configure(state="disabled")
        status_var.set("Rendering... this can take a while.")
        threading.Thread(target=rerender_worker, args=(collect_overrides(),), daemon=True).start()

    rerender_button = tk.Button(panel, text="Re-render with these settings", command=on_rerender_click)
    rerender_button.pack(anchor="w", fill="x", pady=(0, 4))

    if on_fast_preview is not None:
        _add_fast_preview_controls(root, canvas, state, on_fast_preview, display_size, collect_overrides, status_var, rerender_button, start_reader, read_and_show_frame)


def _add_fast_preview_controls(root, canvas, state, on_fast_preview, display_size, collect_overrides, status_var, rerender_button, start_reader, read_and_show_frame):
    """Adds the "Fast Preview (draft)" / "Back to full preview" buttons -
    see play_loop's `on_fast_preview` docstring for what this does."""
    panel = rerender_button.master

    def show_next_fast_frame():
        frames = state["fast_frames"]
        if state["closed"] or frames is None:
            return
        state["fast_index"] = (state["fast_index"] + 1) % len(frames)
        state["photo"] = frames[state["fast_index"]]
        canvas.create_image(0, 0, anchor="nw", image=state["photo"])
        state["fast_after_id"] = root.after(max(1, int(1000 / state["fps"])), show_next_fast_frame)

    def fast_preview_worker(overrides):
        try:
            result = on_fast_preview(overrides)
        except Exception as exc:  # surface pipeline errors instead of freezing the panel
            root.after(0, lambda: fast_preview_failed(exc))
            return
        root.after(0, lambda: fast_preview_done(result))

    def fast_preview_failed(exc):
        state["rendering"] = False
        fast_preview_button.configure(state="normal")
        rerender_button.configure(state="normal")
        status_var.set(f"Fast preview failed: {exc}")

    def fast_preview_done(result):
        frames, new_width, new_height, new_fps = result
        new_scale = min(1.0, _PREVIEW_MAX_DIMENSION / max(new_width, new_height))
        display_size[0] = max(1, int(new_width * new_scale))
        display_size[1] = max(1, int(new_height * new_scale))
        canvas.configure(width=display_size[0], height=display_size[1])

        if state["process"] is not None:
            state["process"].terminate()
        state["fps"] = new_fps
        state["fast_index"] = -1
        state["fast_frames"] = [ImageTk.PhotoImage(frame.resize(display_size, Image.LANCZOS)) for frame in frames]

        state["rendering"] = False
        fast_preview_button.configure(state="normal")
        rerender_button.configure(state="normal")
        back_button.configure(state="normal")
        status_var.set("Showing fast draft preview (low quality).")
        show_next_fast_frame()

    def on_fast_preview_click():
        if state["rendering"] or state["closed"]:
            return
        state["rendering"] = True
        fast_preview_button.configure(state="disabled")
        rerender_button.configure(state="disabled")
        status_var.set("Rendering fast draft preview...")
        threading.Thread(target=fast_preview_worker, args=(collect_overrides(),), daemon=True).start()

    def on_back_to_full_click():
        if state["fast_frames"] is None:
            return
        if state["fast_after_id"] is not None:
            root.after_cancel(state["fast_after_id"])
            state["fast_after_id"] = None
        state["fast_frames"] = None
        back_button.configure(state="disabled")
        status_var.set("Back to full preview.")
        state["restart_count"] = 0
        start_reader()
        read_and_show_frame()

    fast_preview_button = tk.Button(panel, text="Fast Preview (draft)", command=on_fast_preview_click)
    fast_preview_button.pack(anchor="w", fill="x", pady=(4, 4))

    back_button = tk.Button(panel, text="Back to full preview", command=on_back_to_full_click, state="disabled")
    back_button.pack(anchor="w", fill="x")
