"""
presets.py

"a named preset library, once the config surface is
this large." With 20+ Parts' worth of tunable parameters now in
config.py, a small library of named starting presets ("storm," "gentle
snowfall," "misty dawn") bundling sensible combinations of the many
individual knobs is meaningfully more usable than starting every photo
from a blank slate of 80+ raw values.

The design notes's own write-up frames this as tying "into [Part 12]'s
slider-based tuning workflow as a starting point" - Part 12's GUI itself
doesn't exist yet, so this intentionally does NOT build one; what it
builds is the part that doesn't depend on a GUI existing first: a plain
named dict of config.py overrides, selected via config.ACTIVE_PRESET and
applied at the very start of a run (see main.py/live_wallpaper.py's
`main()`/module entry point). A future Part 12 GUI would read/offer
PRESETS the same way a config-file user does today - this is the data
layer that workflow needs either way, not a throwaway that gets replaced
once a GUI exists.

Each preset only lists the settings IT actually wants to override - every
setting not mentioned in a given preset keeps whatever config.py's own
top-level default already says, so a preset is a targeted "start from
these defaults, then change just this handful of knobs" bundle, not a
from-scratch replacement of the entire config surface.
"""

import config

PRESETS = {
    "storm": {
        "RAIN_ENABLED": True,
        "SNOW_ENABLED": False,
        "FOG_ENABLED": True,
        "LIGHTNING_ENABLED": True,
        "RAIN_PARAMS": dict(
            intensity="storm", blur=True,
            motion_blur_samples=3, shutter_fraction=0.5,
        ),
        "FOG_PARAMS": dict(density=0.35, movement=0.5, color=(180, 185, 195)),
        "LIGHTNING_PARAMS": dict(
            frequency=0.25, intensity=1.0, color=(230, 235, 255),
            branch_chance=0.45, strike_frames=3,
            flash_falloff_radius_frac=0.85, flash_min_intensity_frac=0.25,
        ),
        "WEATHER_WIND_DIRECTION_DEG": 200.0,
        "WEATHER_GUST_STRENGTH": 0.6,
        "CLOUD_DRIFT_STRENGTH": 0.8,
        "WATER_RIPPLE_STRENGTH": 0.8,
        "VEGETATION_SWAY_STRENGTH": 0.75,
        "AMBIENT_DRIFT_STRENGTH": 0.1,
    },
    "gentle_snowfall": {
        "RAIN_ENABLED": False,
        "SNOW_ENABLED": True,
        "FOG_ENABLED": True,
        "LIGHTNING_ENABLED": False,
        "SNOW_PARAMS": dict(
            density=0.35, wind=0.15, depth=True,
            motion_blur_samples=2, shutter_fraction=0.5,
        ),
        "FOG_PARAMS": dict(density=0.15, movement=0.15, color=(230, 232, 238)),
        "WEATHER_WIND_DIRECTION_DEG": 10.0,
        "WEATHER_GUST_STRENGTH": 0.15,
        "CLOUD_DRIFT_STRENGTH": 0.2,
        "WATER_RIPPLE_STRENGTH": 0.15,
        "VEGETATION_SWAY_STRENGTH": 0.25,
        "AMBIENT_DRIFT_STRENGTH": 0.03,
    },
    "misty_dawn": {
        "RAIN_ENABLED": False,
        "SNOW_ENABLED": False,
        "FOG_ENABLED": True,
        "LIGHTNING_ENABLED": False,
        "FOG_PARAMS": dict(density=0.55, movement=0.25, color=(235, 225, 210)),
        "FOG_DEPTH_POWER": 1.3,
        "GOD_RAYS_ENABLED": True,
        "GOD_RAYS_STRENGTH": 0.9,
        "GOD_RAYS_COLOR": (255, 236, 200),
        "WEATHER_WIND_DIRECTION_DEG": 5.0,
        "WEATHER_GUST_STRENGTH": 0.1,
        "CLOUD_DRIFT_STRENGTH": 0.15,
        "WATER_RIPPLE_STRENGTH": 0.1,
        "VEGETATION_SWAY_STRENGTH": 0.2,
        "AMBIENT_DRIFT_STRENGTH": 0.08,
    },
}


def apply_preset(name):
    """
    Applies PRESETS[name]'s overrides directly onto the config module -
    every setting the preset mentions is set on `config` (so every
    existing call site, which all read `config.SOMETHING` today, picks
    up the override with no changes needed anywhere else); every setting
    it doesn't mention is left completely untouched. A no-op when `name`
    is None (config.ACTIVE_PRESET's default - "no preset, use config.py's
    own top-level values exactly as written").

    Raises ValueError for an unknown preset name, rather than silently
    ignoring a typo in config.py and rendering with the wrong settings.
    """
    if name is None:
        return

    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{name}' in config.ACTIVE_PRESET - available presets: {available}")

    for key, value in PRESETS[name].items():
        setattr(config, key, value)
