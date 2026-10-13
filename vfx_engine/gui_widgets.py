"""
gui_widgets.py

Reusable widgets for vfx_gui.py: the looping preview canvas, hover tooltips
and a video probe. (Moved out of vfx_gui.py unchanged in behaviour.)
"""

import subprocess
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk

from vfx_engine.image_io import get_ffmpeg_path

_PREVIEW_MAX_DIMENSION = 960
_MAX_CONSECUTIVE_RESTARTS = 3   # give up (and show the ffmpeg error) instead of retrying forever - see vfx_engine/preview_window.py's identical constant


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

    def toggle_pause(self):
        """Space bar: freeze / resume the draft preview. Returns True when now paused."""
        if self.fast_frames is None or self._closed:
            return False
        self.paused = not getattr(self, "paused", False)
        if self.paused:
            if self._after_id is not None:
                self.canvas.after_cancel(self._after_id)
                self._after_id = None
        else:
            self._show_next_fast_frame()
        return self.paused

    def _show_next_fast_frame(self):
        if self._closed or self.fast_frames is None:
            return
        self.paused = False
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


class Tooltip:
    """One-sentence hover help for any widget."""

    def __init__(self, widget, text, delay_ms=450):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id = None
        self._window = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self):
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        self._after_id = None
        if self._window is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._window = tk.Toplevel(self.widget)
        self._window.wm_overrideredirect(True)
        self._window.wm_geometry(f"+{x}+{y}")
        tk.Label(self._window, text=self.text, justify="left", wraplength=280, bg="#26262e", fg="#f0f0f4",
                 relief="solid", borderwidth=1, padx=8, pady=5, font=("Segoe UI", 10)).pack()

    def _hide(self, _event=None):
        self._cancel()
        if self._window is not None:
            self._window.destroy()
            self._window = None


def probe_video(path):
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
