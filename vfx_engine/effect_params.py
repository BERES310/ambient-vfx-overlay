"""
effect_params.py

Shared effect-toggle / intensity-slider definitions used by every UI that
lets a user pick weather effects before rendering (vfx_gui.py's sidebar and
vfx_engine/preview_window.py's control panel) - kept in one place so both
stay in sync and neither has to duplicate the config.* wiring.

live_wallpaper.process_image() reads every effect toggle straight off
module-level config.* attributes (no per-call overrides exist in the
pipeline itself) - so a UI's switches/sliders just set those attributes
before calling process_image(), then restore the previous values
afterwards. See run_weather_benchmark.py for the same pattern used
non-interactively.
"""

import config

# (config attribute, switch label) - the same 6 flags run_weather_benchmark.py's
# presets toggle.
EFFECT_SWITCHES = [
    ("RAIN_ENABLED", "Rain"),
    ("SNOW_ENABLED", "Snow"),
    ("FOG_ENABLED", "Fog"),
    ("LIGHTNING_ENABLED", "Lightning"),
    ("FLICKER_ENABLED", "Flicker"),
    ("GOD_RAYS_ENABLED", "God Rays"),
]

# Rain's config.RAIN_PARAMS["intensity"] is normally just one of the named
# strings "light"/"medium"/"storm" (see vfx_engine/particles_rain.py's
# _INTENSITY_PRESETS) - but render_rain() also accepts a raw dict with the
# same keys instead, specifically so callers can fully customize a preset
# without editing that file. That's exactly the hook a continuous "heavier /
# less heavy" rain slider needs: interpolate every field between the
# "light" and "storm" presets by the slider's 0-100 position instead of
# snapping between only 3 fixed steps.
_RAIN_LIGHT_PRESET = dict(count=100, speed_range=(8.0, 11.0), size_range=(2, 4), drift=0.1)
_RAIN_STORM_PRESET = dict(count=420, speed_range=(16.0, 26.0), size_range=(2, 5), drift=0.22)


def _lerp(a, b, t):
    return a + (b - a) * t


def build_rain_intensity_dict(percent):
    """0-100 -> a full _INTENSITY_PRESETS-shaped dict, linearly interpolated
    between the "light" and "storm" built-in presets."""
    t = max(0.0, min(1.0, percent / 100.0))
    return dict(
        count=round(_lerp(_RAIN_LIGHT_PRESET["count"], _RAIN_STORM_PRESET["count"], t)),
        speed_range=(
            round(_lerp(_RAIN_LIGHT_PRESET["speed_range"][0], _RAIN_STORM_PRESET["speed_range"][0], t), 1),
            round(_lerp(_RAIN_LIGHT_PRESET["speed_range"][1], _RAIN_STORM_PRESET["speed_range"][1], t), 1),
        ),
        size_range=(
            round(_lerp(_RAIN_LIGHT_PRESET["size_range"][0], _RAIN_STORM_PRESET["size_range"][0], t)),
            round(_lerp(_RAIN_LIGHT_PRESET["size_range"][1], _RAIN_STORM_PRESET["size_range"][1], t)),
        ),
        drift=round(_lerp(_RAIN_LIGHT_PRESET["drift"], _RAIN_STORM_PRESET["drift"], t), 3),
    )


# Per-effect "how heavy" slider wiring (2026-08-28 request: "a slider that
# will make rain heavier or less heavy"). Each entry maps the SAME
# EFFECT_SWITCHES attr (e.g. "RAIN_ENABLED") to the real numeric config
# knob that actually controls that effect's density/strength - not a fake
# cosmetic slider, an actual parameter each effect module reads:
#   - "dict_key": the config attribute is a dict (e.g. config.SNOW_PARAMS) -
#     override just one numeric key in a copy of it.
#   - "scalar": the config attribute is itself the number (e.g.
#     config.GOD_RAYS_STRENGTH) - override it directly.
#   - "rain": special-cased - see build_rain_intensity_dict above.
EFFECT_INTENSITY = {
    "RAIN_ENABLED": dict(kind="rain", attr="RAIN_PARAMS", range=(0, 100), default=70, fmt="{:.0f}% heavy", digits=0),
    "SNOW_ENABLED": dict(kind="dict_key", attr="SNOW_PARAMS", key="density", range=(0.0, 1.0), default=0.6, fmt="density {:.2f}", digits=2),
    "FOG_ENABLED": dict(kind="dict_key", attr="FOG_PARAMS", key="density", range=(0.0, 1.0), default=0.28, fmt="density {:.2f}", digits=2),
    "LIGHTNING_ENABLED": dict(kind="dict_key", attr="LIGHTNING_PARAMS", key="frequency", range=(0.0, 0.5), default=0.15, fmt="freq {:.2f}", digits=2),
    "FLICKER_ENABLED": dict(kind="dict_key", attr="FLICKER_PARAMS", key="intensity", range=(0.0, 1.0), default=0.15, fmt="strength {:.2f}", digits=2),
    "GOD_RAYS_ENABLED": dict(kind="scalar", attr="GOD_RAYS_STRENGTH", range=(0.0, 1.5), default=0.7, fmt="strength {:.2f}", digits=2),
}


_RAIN_NAMED_INTENSITY_TO_PERCENT = {"light": 0, "medium": 50, "storm": 100}


def current_intensity_default(switch_attr):
    """Reads the slider's starting position straight off the CURRENT
    config.* value, the same way EFFECT_SWITCHES already seeds its switch
    state from getattr(config, attr) - so opening a UI shows what config.py
    actually has configured right now instead of an unrelated hardcoded
    guess."""
    spec = EFFECT_INTENSITY[switch_attr]
    if spec["kind"] == "rain":
        current = config.RAIN_PARAMS.get("intensity", "storm")
        if isinstance(current, str):
            return _RAIN_NAMED_INTENSITY_TO_PERCENT.get(current, spec["default"])
        return spec["default"]  # already a custom dict - no single percent reconstructs it exactly
    if spec["kind"] == "dict_key":
        return getattr(config, spec["attr"]).get(spec["key"], spec["default"])
    return getattr(config, spec["attr"], spec["default"])  # "scalar"


def resolve_intensity_override(switch_attr, value):
    """Given a slider's raw value for one EFFECT_SWITCHES entry, returns
    (config_attr_name, new_value) ready for setattr(config, ...) - a whole-
    dict replacement for "dict_key"/"rain" kinds (same restore-after
    pattern callers already use for the boolean switches), or the raw
    number for "scalar"."""
    spec = EFFECT_INTENSITY[switch_attr]
    if spec["kind"] == "rain":
        base = dict(config.RAIN_PARAMS)
        base["intensity"] = build_rain_intensity_dict(value)
        return spec["attr"], base
    if spec["kind"] == "dict_key":
        base = dict(getattr(config, spec["attr"]))
        base[spec["key"]] = value
        return spec["attr"], base
    # "scalar"
    return spec["attr"], value


WEATHER_PRESETS = {
    "Custom": None,
    "Rain + Fog": dict(RAIN_ENABLED=True, SNOW_ENABLED=False, FOG_ENABLED=True, LIGHTNING_ENABLED=False, FLICKER_ENABLED=False, GOD_RAYS_ENABLED=False),
    "Snow": dict(RAIN_ENABLED=False, SNOW_ENABLED=True, FOG_ENABLED=False, LIGHTNING_ENABLED=False, FLICKER_ENABLED=False, GOD_RAYS_ENABLED=False),
    "Clear / Sunny": dict(RAIN_ENABLED=False, SNOW_ENABLED=False, FOG_ENABLED=False, LIGHTNING_ENABLED=False, FLICKER_ENABLED=False, GOD_RAYS_ENABLED=True),
    "Storm": dict(RAIN_ENABLED=True, SNOW_ENABLED=False, FOG_ENABLED=False, LIGHTNING_ENABLED=True, FLICKER_ENABLED=True, GOD_RAYS_ENABLED=False),
}
