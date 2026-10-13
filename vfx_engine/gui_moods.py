"""
gui_moods.py

Plain-language layer between vfx_gui.py's simple controls (mood cards, the
Intensity / Wind / Darkness sliders, quality and output choices - see
the design notes) and the technical config.* knobs the pipeline actually reads.

Pure logic, no Tk: everything here returns dicts of config overrides
({config attribute: value}) that the GUI applies with setattr on the render
worker thread and restores afterwards - the same apply-and-restore pattern
vfx_engine/effect_params.py documents. Keeping it Tk-free makes it testable
on its own and lets the fast draft preview and the final render share the
exact same mapping, so what the draft shows is what the final render makes.
"""

import re
from pathlib import Path

import config
from vfx_engine import effect_params

# Plain names shown in the app -> the technical name kept for tooltips.
FRIENDLY_EFFECT_NAMES = {
    "RAIN_ENABLED": ("Rain", "Rain"),
    "SNOW_ENABLED": ("Snow", "Snow"),
    "FOG_ENABLED": ("Mist", "Fog"),
    "LIGHTNING_ENABLED": ("Lightning", "Lightning"),
    "FLICKER_ENABLED": ("Light flicker", "Flicker"),
    "GOD_RAYS_ENABLED": ("Sunbeams", "God rays"),
}

# Mood cards. `effects` lists the switches that are ON; every other switch is
# off. `grade` is config.WEATHER_GRADE_STYLE for the mood (None = leave the
# photo's own colours alone). `intensity`/`wind`/`darkness` are the slider
# positions (0-1) the mood starts at.
MOODS = [
    dict(key="rain", label="Rainy", blurb="Steady rain and a soft mist.",
         effects=("RAIN_ENABLED", "FOG_ENABLED"), grade=None, intensity=0.6, wind=0.4, darkness=0.0,
         preset="Rain + Fog"),
    dict(key="snow", label="Snowy", blurb="Quiet, drifting snowfall.",
         effects=("SNOW_ENABLED",), grade="snow", intensity=0.5, wind=0.2, darkness=0.0,
         preset="Snow"),
    dict(key="sunny", label="Sunny", blurb="Warm light with sunbeams.",
         effects=("GOD_RAYS_ENABLED",), grade="sunny", intensity=0.6, wind=0.2, darkness=0.0,
         preset="Clear / Sunny"),
    dict(key="storm", label="Stormy", blurb="Heavy rain, lightning and flickering light.",
         effects=("RAIN_ENABLED", "LIGHTNING_ENABLED", "FLICKER_ENABLED"), grade="storm", intensity=0.85, wind=0.8,
         darkness=0.6, preset="Storm"),
    dict(key="mist", label="Misty dawn", blurb="Low mist and gentle sunbeams.",
         effects=("FOG_ENABLED", "GOD_RAYS_ENABLED"), grade=None, intensity=0.6, wind=0.1, darkness=0.0,
         preset=None),
]
MOOD_BY_KEY = {mood["key"]: mood for mood in MOODS}
CUSTOM_MOOD_LABEL = "My mix"

# Words shown next to a 0-1 slider position.
_WORD_SCALE = ("None", "Very light", "Light", "Medium", "Strong", "Very strong")


def slider_word(position):
    """0-1 slider position -> the word shown beside it."""
    position = max(0.0, min(1.0, position))
    return _WORD_SCALE[min(len(_WORD_SCALE) - 1, int(position * (len(_WORD_SCALE) - 1) + 0.5))]


def darkness_word(position):
    position = max(0.0, min(1.0, position))
    return ("Natural", "Slightly dim", "Dim", "Dark", "Very dark", "Night-like")[min(5, int(position * 5 + 0.5))]


def effects_for_mood(mood_key):
    """{switch attr: bool} for every effect switch under this mood."""
    on = set(MOOD_BY_KEY[mood_key]["effects"])
    return {attr: attr in on for attr, _label in effect_params.EFFECT_SWITCHES}


def mood_key_for_switches(states):
    """Best-matching mood key for a {switch attr: bool} dict, or None ("My mix")."""
    for mood in MOODS:
        if effects_for_mood(mood["key"]) == {attr: bool(states.get(attr)) for attr, _l in effect_params.EFFECT_SWITCHES}:
            return mood["key"]
    return None


# Intensity 0-1 -> each effect's real knob. (low, high) per switch; the GUI's
# Intensity slider interpolates inside these bounds. Bounds sit inside
# effect_params.EFFECT_INTENSITY's ranges so the Advanced sliders can still go further.
_INTENSITY_SPANS = {
    "RAIN_ENABLED": (10.0, 100.0),
    "SNOW_ENABLED": (0.1, 0.95),
    "FOG_ENABLED": (0.08, 0.6),
    "LIGHTNING_ENABLED": (0.03, 0.35),
    "FLICKER_ENABLED": (0.05, 0.5),
    "GOD_RAYS_ENABLED": (0.25, 1.3),
}


def intensity_value(switch_attr, position):
    """Slider position (0-1) -> the numeric value for that effect's own knob."""
    low, high = _INTENSITY_SPANS[switch_attr]
    return low + (high - low) * max(0.0, min(1.0, position))


def build_look_overrides(switch_states, intensity, wind, darkness, mood_key=None, advanced_intensities=None):
    """
    The full {config attribute: value} override dict for one look.

    switch_states: {switch attr: bool}.
    intensity/wind/darkness: 0-1 simple-slider positions.
    advanced_intensities: optional {switch attr: raw value} that wins over the
        Intensity slider for that one effect (the per-effect Advanced sliders).
    """
    overrides = dict(switch_states)
    advanced_intensities = advanced_intensities or {}
    for attr, enabled in switch_states.items():
        if not enabled or attr not in effect_params.EFFECT_INTENSITY:
            continue
        value = advanced_intensities.get(attr, intensity_value(attr, intensity))
        config_attr, new_value = effect_params.resolve_intensity_override(attr, value)
        # Two effects never share a config attribute, so plain assignment is safe.
        overrides[config_attr] = new_value

    wind = max(0.0, min(1.0, wind))
    overrides["WEATHER_GUST_STRENGTH"] = 0.9 * wind
    overrides["CLOUD_DRIFT_STRENGTH"] = 0.15 + 0.85 * wind
    overrides["VEGETATION_SWAY_STRENGTH"] = 0.15 + 0.85 * wind

    grade = MOOD_BY_KEY[mood_key]["grade"] if mood_key in MOOD_BY_KEY else None
    darkness = max(0.0, min(1.0, darkness))
    mood_darkness = MOOD_BY_KEY[mood_key]["darkness"] if mood_key in MOOD_BY_KEY else 0.0
    if darkness > mood_darkness + 0.05 or (grade is None and darkness > 0.05):
        # Slid darker than the mood's own default: use the dark grade, scaled.
        overrides["WEATHER_GRADE_STYLE"] = "storm"
        overrides["WEATHER_GRADE_STRENGTH"] = darkness
    else:
        overrides["WEATHER_GRADE_STYLE"] = grade
        overrides["WEATHER_GRADE_STRENGTH"] = 1.0
    return overrides


# ---------------------------------------------------------------------------
# Output + quality choices
# ---------------------------------------------------------------------------
QUALITY_PRESETS = {
    # label: (max longest side in px or None = original, x264 CRF, blurb)
    "Draft": (640, 30, "Small and fast - to check the look."),
    "Balanced": (1280, 21, "Good for most screens."),
    "High": (1920, 16, "Full HD, sharp."),
    "Maximum": (None, 14, "Original photo size, best quality. Slowest."),
}
DEFAULT_QUALITY = "High"

# label: seconds
LENGTH_CHOICES = {"10 seconds": 10, "30 seconds": 30, "1 minute": 60, "2 minutes": 120, "5 minutes": 300}
DEFAULT_LENGTH = "30 seconds"
FPS_CHOICES = {"24 frames/s (smooth)": 24, "30 frames/s": 30, "15 frames/s (small file)": 15}
DEFAULT_FPS = "24 frames/s (smooth)"


def quality_overrides(quality_label, length_seconds, fps):
    """Config overrides for the Output + Quality choices."""
    max_side, crf, _blurb = QUALITY_PRESETS[quality_label]
    return {
        "LIVE_WALLPAPER_MAX_DIMENSION": max_side,
        "LIVE_WALLPAPER_CRF": crf,
        "LIVE_WALLPAPER_DURATION_SECONDS": length_seconds,
        "FPS": fps,
    }


def frame_count(length_seconds, fps):
    return round(length_seconds * fps)


def fit_dimension(size, max_side):
    """(w, h) scaled down so the longest side is <= max_side; never up, always even (x264)."""
    width, height = size
    if max_side is not None and max(width, height) > max_side:
        scale = max_side / max(width, height)
        width, height = width * scale, height * scale
    return max(2, int(width) // 2 * 2), max(2, int(height) // 2 * 2)


def estimate_render_seconds(seconds_per_frame, frames, setup_seconds=0.0, encode_factor=0.08):
    """Total time estimate: measured per-frame cost x frames, plus setup and a
    proportional allowance for the encode passes."""
    return setup_seconds + seconds_per_frame * frames * (1.0 + encode_factor)


def format_estimate(seconds):
    seconds = int(seconds)
    if seconds < 90:
        return "about a minute" if seconds >= 45 else "under a minute"
    minutes = round(seconds / 60)
    hours, minutes = divmod(minutes, 60)
    if hours >= 24:
        return f"about {hours // 24} d {hours % 24} h"
    if hours:
        return f"about {hours} h {minutes:02d} min"
    return f"about {minutes} min"


# ---------------------------------------------------------------------------
# Auto-versioned output names
# ---------------------------------------------------------------------------
_VERSION_RE = re.compile(r"^(?P<stem>.*?)(?:_(?P<n>\d+))?$")


def next_free_path(path):
    """`name.mp4` if free, else `name_2.mp4`, `name_3.mp4`, ... - never overwrites."""
    path = Path(path)
    if not path.exists():
        return path
    match = _VERSION_RE.match(path.stem)
    base = match.group("stem") if match.group("n") else path.stem
    counter = 2
    while True:
        candidate = path.with_name(f"{base}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1
