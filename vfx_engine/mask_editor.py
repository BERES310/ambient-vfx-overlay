"""
mask_editor.py

A window where you can hand-draw a mask with the mouse - used with
MASK_MODE = "manual" in config.py. Opened separately for each photo (unless
a mask for that photo is already saved in masks/ - in that case the window
doesn't open, the program just uses the saved mask).

Window controls:
    left mouse button + move  -> draws the mask (white = effect visible)
    right mouse button + move -> eraser (clears the selection)
    mouse wheel                -> changes brush size
    Enter or the "Save" button -> saves the mask and closes the window
    Escape                     -> cancels (the effect will apply to the whole image)

Uses only tkinter (built into Python), Pillow, and numpy - no extra
libraries.

the brush used to be a flat, fully-opaque circle stamp
(PIL's ImageDraw.ellipse/line, `fill=255`/`fill=0`, no soft edge at all) -
"the same 'hard shape now, blur uniformly later' pattern as 10.1, just
via a human's mouse movements instead of a numeric threshold." Any real
paint tool offers at minimum a soft/feathered brush as a baseline
expectation for exactly this kind of task. Now uses a radially-symmetric
smoothstep falloff brush (see _make_soft_brush_stamp) - the same
smoothstep-based softness technique already established elsewhere in
this codebase (vfx_engine/mask.py's own soft-threshold masks) - giving
real per-stroke edge-softness control instead of relying entirely on one
global post-hoc MASK_FEATHER blur constant to soften a mask assembled
from hard shapes.
"""

import math
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

_MAX_WINDOW_SIZE = 900   # photos larger than this are downscaled in the window so they fit on screen

# The brush's soft edge starts this many pixels in from its own outer
# radius (as a FRACTION of the radius, not a fixed pixel count, so a
# large and a small brush both keep a proportionally similar feel) -
# the original design asks for "actual control over the mask's edge softness
# per-stroke," so leaving a solid, fully-opaque core (not fading the
# ENTIRE brush from center to edge) matters: a brush that's soft
# everywhere, even dead center, would make it hard to ever paint a
# confidently fully-opaque area at all - real soft brushes keep a solid
# core and only taper near the rim.
_BRUSH_HARD_CORE_FRACTION = 0.55


def open_mask_editor(image, brush_size=35):
    """
    Opens a window for hand-drawing a mask for `image` (a PIL image).
    Returns the finished mask as a PIL image in "L" mode, at the photo's
    original size - or None if the user cancelled (closed the window
    without saving / pressed Escape).
    """
    original_size = image.size
    display_image = _scaled_for_display(image)
    display_width, display_height = display_image.size

    # The mask is kept as a float32 0.0-1.0 numpy array (not a PIL image
    # mutated via ImageDraw) so soft-brush strokes can blend smoothly
    # (screen-blend for drawing, multiplicative fade for erasing - see
    # _stamp_brush) instead of a hard overwrite - converted to a PIL
    # image only for display/return.
    mask_array = np.zeros((display_height, display_width), dtype="float32")

    editor_state = {"brush_size": brush_size, "last_point": None, "saved": False}

    root = tk.Tk()
    root.title("Draw the mask (soft brush)  |  left click = draw, right click = eraser, wheel = brush size, Enter = save")

    canvas = tk.Canvas(root, width=display_width, height=display_height)
    canvas.pack()

    # Tkinter needs a "live" reference to images, otherwise they'd
    # disappear right away (garbage collector) - we keep them in this dict.
    photos = {
        "background": ImageTk.PhotoImage(display_image),
        "overlay": ImageTk.PhotoImage(_preview_with_mask(display_image, mask_array)),
    }
    canvas.create_image(0, 0, anchor="nw", image=photos["background"])
    overlay_id = canvas.create_image(0, 0, anchor="nw", image=photos["overlay"])

    def redraw_overlay():
        photos["overlay"] = ImageTk.PhotoImage(_preview_with_mask(display_image, mask_array))
        canvas.itemconfig(overlay_id, image=photos["overlay"])

    def paint(event, is_white):
        x, y = event.x, event.y
        radius = editor_state["brush_size"]
        stamp = _make_soft_brush_stamp(radius)

        last_point = editor_state["last_point"]
        if last_point is None:
            _stamp_brush(mask_array, stamp, radius, x, y, is_white)
        else:
            _stamp_line(mask_array, stamp, radius, last_point[0], last_point[1], x, y, is_white)

        editor_state["last_point"] = (x, y)
        redraw_overlay()

    def stop_stroke(_event):
        editor_state["last_point"] = None

    def change_brush_size(event):
        step = 3 if event.delta > 0 else -3
        editor_state["brush_size"] = max(3, min(150, editor_state["brush_size"] + step))

    def save_and_close(_event=None):
        editor_state["saved"] = True
        root.destroy()

    def cancel(_event=None):
        root.destroy()

    canvas.bind("<Button-1>", lambda e: paint(e, True))
    canvas.bind("<B1-Motion>", lambda e: paint(e, True))
    canvas.bind("<ButtonRelease-1>", stop_stroke)

    canvas.bind("<Button-3>", lambda e: paint(e, False))
    canvas.bind("<B3-Motion>", lambda e: paint(e, False))
    canvas.bind("<ButtonRelease-3>", stop_stroke)

    canvas.bind("<MouseWheel>", change_brush_size)   # Windows
    canvas.bind("<Button-4>", lambda e: editor_state.__setitem__("brush_size", min(150, editor_state["brush_size"] + 3)))
    canvas.bind("<Button-5>", lambda e: editor_state.__setitem__("brush_size", max(3, editor_state["brush_size"] - 3)))

    root.bind("<Return>", save_and_close)
    root.bind("<Escape>", cancel)

    button_bar = tk.Frame(root)
    button_bar.pack(fill="x")
    tk.Button(button_bar, text="Save and close (Enter)", command=save_and_close).pack(side="left", fill="x", expand=True)
    tk.Button(button_bar, text="Cancel (Esc)", command=cancel).pack(side="left", fill="x", expand=True)

    root.mainloop()

    if not editor_state["saved"]:
        return None

    mask_layer = Image.fromarray(np.clip(mask_array * 255.0, 0, 255).astype("uint8"), mode="L")
    return mask_layer.resize(original_size)


def _make_soft_brush_stamp(radius):
    """
    Returns a (2*radius+1, 2*radius+1) float32 array, 0.0-1.0 - a
    radially-symmetric soft brush: a solid, fully-opaque
    core out to _BRUSH_HARD_CORE_FRACTION of the radius, then a
    smoothstep falloff (the same 3t^2-2t^3 ease already used for
    softness elsewhere in this codebase, e.g. vfx_engine/mask.py's
    soft-threshold masks) from there out to the brush's outer edge,
    where it reaches exactly 0. A real soft/feathered brush, not the old
    flat, fully-opaque hard-edged circle stamp.
    """
    diameter = 2 * radius + 1
    yy, xx = np.mgrid[0:diameter, 0:diameter].astype("float32")
    distance = np.sqrt((xx - radius) ** 2 + (yy - radius) ** 2)
    normalized_distance = np.clip(distance / max(radius, 1e-6), 0.0, 1.0)

    core = _BRUSH_HARD_CORE_FRACTION
    # Map [0, core] -> 1.0 (solid), [core, 1.0] -> smoothstep(1 -> 0).
    falloff_t = np.clip((normalized_distance - core) / max(1.0 - core, 1e-6), 0.0, 1.0)
    smoothstep = falloff_t * falloff_t * (3.0 - 2.0 * falloff_t)
    return (1.0 - smoothstep).astype("float32")


def _stamp_brush(mask_array, stamp, radius, x, y, is_white):
    """
    Blends one soft-brush stamp (see _make_soft_brush_stamp) into
    mask_array at (x, y), clipped to the array's own bounds (a stroke
    near the canvas edge only partially overlaps it). Drawing
    (is_white=True) uses a SCREEN blend (`1 - (1-a)*(1-b)`) so repeated
    passes naturally build up toward full opacity without ever
    overshooting past 1.0, the same screen-blend technique
    vfx_engine/lighting.py's brighten_frame already uses for an
    analogous "combine without overshoot" need; erasing (is_white=False)
    multiplicatively fades existing coverage DOWN toward 0
    (`a * (1 - b)`) instead of a hard cutoff, so a soft eraser stroke
    genuinely feathers off rather than punching a hard-edged hole.
    """
    height, width = mask_array.shape
    diameter = stamp.shape[0]

    left, top = x - radius, y - radius
    right, bottom = left + diameter, top + diameter

    stamp_left = max(0, -left)
    stamp_top = max(0, -top)
    clipped_left = max(0, left)
    clipped_top = max(0, top)
    clipped_right = min(width, right)
    clipped_bottom = min(height, bottom)

    if clipped_right <= clipped_left or clipped_bottom <= clipped_top:
        return   # the brush stamp doesn't overlap the canvas at all here

    stamp_right = stamp_left + (clipped_right - clipped_left)
    stamp_bottom = stamp_top + (clipped_bottom - clipped_top)
    stamp_region = stamp[stamp_top:stamp_bottom, stamp_left:stamp_right]

    region = mask_array[clipped_top:clipped_bottom, clipped_left:clipped_right]
    if is_white:
        region[:] = 1.0 - (1.0 - region) * (1.0 - stamp_region)
    else:
        region[:] = region * (1.0 - stamp_region)


def _stamp_line(mask_array, stamp, radius, x0, y0, x1, y1, is_white):
    """
    Stamps the soft brush at evenly-spaced points along the segment from
    (x0, y0) to (x1, y1), close enough together that a fast mouse stroke
    doesn't leave visible gaps between individual stamps - replaces the
    old hard-edged PIL `ImageDraw.line(..., width=radius*2)` primitive,
    which doesn't have a soft-brush equivalent (a "line" drawn with flat
    end caps and a soft round brush's shape aren't the same thing) - a
    dense enough series of round stamps approximates a real brush stroke
    correctly instead.
    """
    distance = math.hypot(x1 - x0, y1 - y0)
    step = max(1, radius // 4)   # dense enough to avoid gaps even for a small brush
    steps = max(1, int(distance / step))

    for i in range(1, steps + 1):
        t = i / steps
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        _stamp_brush(mask_array, stamp, radius, int(round(x)), int(round(y)), is_white)


def _scaled_for_display(image):
    """Downscales large photos so the editor window fits on screen."""
    width, height = image.size
    scale = min(1.0, _MAX_WINDOW_SIZE / max(width, height))
    if scale >= 1.0:
        return image.copy()
    return image.resize((int(width * scale), int(height * scale)))


def _preview_with_mask(display_image, mask_array):
    """Live preview: the photo with a translucent red overlay wherever the mask has already been drawn."""
    red_layer = Image.new("RGBA", display_image.size, (255, 60, 60, 0))
    alpha = np.clip(mask_array * 0.55 * 255.0, 0, 255).astype("uint8")
    red_layer.putalpha(Image.fromarray(alpha, mode="L"))
    return Image.alpha_composite(display_image.convert("RGBA"), red_layer)
