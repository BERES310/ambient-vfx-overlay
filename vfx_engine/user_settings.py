"""
user_settings.py

the design notes items 30 and 32: the handful of options a person actually
wants to change live in a small `settings.json` next to the program, so nobody
has to edit config.py (~68 KB of engine constants) to change the look or the
output size. Every value is checked when the file is loaded and a mistake is
reported as "which setting, what was found, what is allowed" instead of
surfacing as a crash deep inside the pipeline.

Pure logic (no Tk, no GPU); shared by vfx_cli.py.
"""

import json
from pathlib import Path

from vfx_engine import gui_moods

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "settings.json"


class SettingsError(ValueError):
    """Carries every problem found, one readable line each."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


# key: (kind, allowed, default, description). kind: "choice" | "number" | "unit"
SCHEMA = {
    "mood": ("choice", [m["key"] for m in gui_moods.MOODS], "rain",
             "Which weather mood to apply."),
    "quality": ("choice", list(gui_moods.QUALITY_PRESETS), gui_moods.DEFAULT_QUALITY,
                "Draft is fast and small, Maximum keeps the photo's own size."),
    "length_seconds": ("number", (5, 600), 30, "Length of the looping video, in seconds."),
    "fps": ("number", (10, 60), 24, "Frames per second."),
    "intensity": ("unit", (0.0, 1.0), None, "How much weather (0 none - 1 very strong). Empty = the mood's own."),
    "wind": ("unit", (0.0, 1.0), None, "How windy (0 - 1). Empty = the mood's own."),
    "darkness": ("unit", (0.0, 1.0), None, "How dark (0 - 1). Empty = the mood's own."),
}


def defaults():
    return {key: spec[2] for key, spec in SCHEMA.items()}


def validate(data):
    """Returns a cleaned dict (defaults filled in) or raises SettingsError."""
    problems = []
    if not isinstance(data, dict):
        raise SettingsError(['settings.json must contain one JSON object like {"mood": "rain"}.'])
    cleaned = defaults()
    for key, value in data.items():
        if key not in SCHEMA:
            problems.append(f"Unknown setting '{key}'. Known settings: {', '.join(SCHEMA)}.")
            continue
        kind, allowed, _default, _desc = SCHEMA[key]
        if value is None and kind == "unit":
            cleaned[key] = None
        elif kind == "choice":
            if value not in allowed:
                problems.append(f"'{key}' is {value!r}; it must be one of: {', '.join(allowed)}.")
            else:
                cleaned[key] = value
        else:
            low, high = allowed
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(f"'{key}' is {value!r}; it must be a number between {low} and {high}.")
            elif not low <= value <= high:
                problems.append(f"'{key}' is {value}; it must be between {low} and {high}.")
            else:
                cleaned[key] = value
    if problems:
        raise SettingsError(problems)
    return cleaned


def load(path=None):
    """Reads settings.json (creating it with the defaults when missing) and validates it."""
    path = Path(path or SETTINGS_PATH)
    if not path.exists():
        save(defaults(), path)
        return defaults()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except ValueError as exc:
        raise SettingsError([f"{path.name} is not valid JSON ({exc}). Fix it, or delete it to get a fresh one."])
    return validate(data)


def save(data, path=None):
    path = Path(path or SETTINGS_PATH)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path
