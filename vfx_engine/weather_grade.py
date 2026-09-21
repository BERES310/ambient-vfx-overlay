"""
weather_grade.py

Weather-specific color grade applied to the base photo (after the
color-grade unification pass, before any effect samples the image), so a
"sunny" render actually reads as warm, clear, high-contrast daylight and a
"storm" render reads as dark, cool, desaturated and heavy - instead of both
being the same neutral photo with different particles on top.

Styles (config.WEATHER_GRADE_STYLE):
    None      - no change.
    "sunny"   - dehaze (removes the grey airlight veil), warm white balance,
                exposure lift, S-curve contrast, saturation boost, blue-
                shifted upper sky, warm highlight bloom toward the top.
    "snow"    - lifted shadows, desaturated cool blue-white cast, luminous
                grey overcast sky.
    "storm"   - exposure drop, cool blue-grey cast, desaturation, flattened
                highlights, darker sky vignette toward the top.
"""

import numpy as np
from PIL import Image, ImageFilter


def _blur(channel, radius):
    image = Image.fromarray(np.clip(channel * 255.0, 0, 255).astype("uint8"), mode="L")
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), dtype="float32") / 255.0


def _luma(rgb):
    return rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722


def _saturate(rgb, amount):
    luma = _luma(rgb)[:, :, None]
    return luma + (rgb - luma) * amount


def _s_curve(rgb, amount):
    # Smoothstep-based S-curve blended in by `amount` (0 = identity).
    smooth = rgb * rgb * (3.0 - 2.0 * rgb)
    return rgb * (1.0 - amount) + smooth * amount


def _dehaze(rgb, strength, radius):
    # Dark-channel-style airlight removal: the local minimum across channels
    # (blurred) approximates the haze veil; subtracting a portion of it and
    # renormalizing restores contrast and colour in hazy photos.
    dark = np.min(rgb, axis=2)
    airlight = _blur(dark, radius)
    veil = airlight * strength
    return np.clip((rgb - veil[:, :, None]) / np.maximum(1.0 - veil[:, :, None], 1e-3), 0.0, 1.0)


def _local_contrast(rgb, amount, radius):
    luma = _luma(rgb)
    detail = luma - _blur(luma, radius)
    return rgb + (detail * amount)[:, :, None]


def _top_gradient(height, width, power=1.5):
    rows = 1.0 - np.arange(height, dtype="float32") / max(height - 1, 1)  # 1.0 at top
    return np.power(rows, power).reshape(-1, 1, 1) * np.ones((1, width, 1), dtype="float32")


def _sky_mask(rgb, top, radius, luma_lo, luma_span):
    """Soft 0-1 mask of open sky. Colour/luma are smoothed first so single
    foliage pixels in hazy midground can't punch holes in it (that showed as
    dark blocks), and the final blur is tightened so it doesn't bleed onto
    the dark ridgeline (that showed as a halo)."""
    smooth = np.stack([_blur(rgb[:, :, c], radius * 0.5) for c in range(3)], axis=2)
    luma = _luma(smooth)
    not_foliage = np.clip(1.0 - (smooth[:, :, 1] - np.maximum(smooth[:, :, 0], smooth[:, :, 2])) * 8.0, 0.0, 1.0)
    bright = np.clip((luma - luma_lo) / luma_span, 0.0, 1.0) * not_foliage
    weight = _blur(top * bright, radius * 0.6)
    return np.clip((weight - 0.35) / 0.5, 0.0, 1.0)


def apply_weather_grade(image, style, strength=1.0):
    """
    image: RGB PIL Image. style: None/"sunny"/"storm"/"snow". strength: 0.0-1.0+
    blend between the ungraded photo and the fully graded one. Returns a
    new RGB PIL Image of the same size.
    """
    if style is None or strength <= 0.0:
        return image

    original = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
    height, width = original.shape[:2]
    radius = max(4, int(min(height, width) * 0.03))
    rgb = original.copy()

    if style == "sunny":
        # Midday, not sunset: the source photos are often dusk/dawn shots, so
        # the grade neutralizes the orange cast, opens shadows, and turns the
        # sky blue instead of just boosting what is already there.
        rgb = _dehaze(rgb, strength=0.5, radius=radius)
        rgb = np.power(np.clip(rgb, 0.0, 1.0), 0.66)  # open shadows (daylight fill)
        rgb = rgb * np.array([1.03, 1.01, 0.96], dtype="float32")  # slightly warm sunlight
        rgb = np.clip(rgb * 1.26, 0.0, 1.0)
        rgb = _local_contrast(rgb, amount=0.2, radius=radius * 2)
        rgb = _s_curve(np.clip(rgb, 0.0, 1.0), 0.25)
        rgb = _saturate(rgb, 1.12)
        # Sky: upper, bright regions become a blue daylight sky that keeps the
        # cloud structure (luma) but drops the sunset colour.
        luma = _luma(rgb)
        top = _top_gradient(height, width, 1.6)[:, :, 0]
        sky_weight = _sky_mask(rgb, top, radius, 0.10, 0.18) * 0.97
        sky_tone = np.clip(0.68 + luma * 0.25, 0.0, 1.0)[:, :, None] * np.array([0.55, 0.78, 1.0], dtype="float32")
        sky_tone = np.clip(sky_tone + (luma[:, :, None] - 0.85).clip(0, 1) * 1.0, 0.0, 1.0)  # clouds stay white
        rgb = rgb * (1.0 - sky_weight[:, :, None]) + sky_tone * sky_weight[:, :, None]
    elif style == "storm":
        rgb = rgb * 0.78  # exposure drop
        rgb = _saturate(rgb, 0.55)
        rgb = rgb * np.array([0.86, 0.94, 1.08], dtype="float32")  # cool cast
        # Flatten highlights so the sky reads as a heavy overcast ceiling.
        rgb = np.where(rgb > 0.55, 0.55 + (rgb - 0.55) * 0.45, rgb)
        rgb = _s_curve(np.clip(rgb / 0.8, 0.0, 1.0), 0.25) * 0.8
        top = _top_gradient(height, width, 1.3)
        rgb = rgb * (1.0 - 0.38 * top) + np.array([0.10, 0.12, 0.16], dtype="float32") * (0.38 * top)
    elif style == "snow":
        # Cold, bright, flat overcast: lifted shadows (snow light bounces
        # everywhere), pale desaturated palette, cool blue-white cast and a
        # sky pulled toward luminous grey so white flakes read against it.
        rgb = np.power(np.clip(rgb, 0.0, 1.0), 0.82)
        rgb = _saturate(rgb, 0.6)
        rgb = rgb * np.array([0.93, 0.98, 1.06], dtype="float32")
        luma = _luma(rgb)
        top = _top_gradient(height, width, 1.2)[:, :, 0]
        sky_weight = _sky_mask(rgb, top, radius, 0.12, 0.2) * 0.75
        sky_tone = np.clip(0.52 + luma * 0.4, 0.0, 1.0)[:, :, None] * np.array([0.9, 0.95, 1.02], dtype="float32")
        rgb = rgb * (1.0 - sky_weight[:, :, None]) + sky_tone * sky_weight[:, :, None]
        rgb = _s_curve(np.clip(rgb, 0.0, 1.0), 0.12)
    else:
        raise ValueError(f"unknown WEATHER_GRADE_STYLE: {style!r}")

    rgb = np.clip(rgb, 0.0, 1.0)
    blended = original * (1.0 - strength) + rgb * strength
    return Image.fromarray((np.clip(blended, 0.0, 1.0) * 255.0 + 0.5).astype("uint8"), mode="RGB")
