"""
run_weather_benchmark.py

Runs live_wallpaper.py's full render pipeline once per (photo, weather
preset) combination, saving a real .mp4 for each to output/live_wallpapers/
and appending one timed row per run to output/live_wallpaper_benchmark.csv
(see live_wallpaper.process_image's built-in timing/CSV logging).

Four weather presets (2026-08-27 clarification with the user - "Weather-
themed" was the chosen option over a custom or auto-picked set):
    - rain_fog:    rain + fog, no lightning/snow/god-rays
    - snow:        falling snow only
    - clear_sunny: no precipitation - god rays + ambient only
    - storm:       rain + lightning + stronger wind (gust strength), with
                    flicker on since real storm light visibly flickers with
                    each strike

Every input photo in input_images/ (config.INPUT_DIR) x every preset above
is rendered - with exactly castle_by_the_lake.png and photo_of_shire.JPG
present (the only two photos in this project as of this writing), that's
the requested 2 photos x 4 presets = 8 total runs.

Live preview is force-disabled for the whole batch (LIVE_PREVIEW_WINDOW is
per-run interactive and would block this script 8 times waiting for a
window to be closed) - use the new CustomTkinter GUI (vfx_gui.py) or a
plain `python live_wallpaper.py` run afterwards to actually preview any of
these renders.

Run (from inside vfx_overlay/):

    python run_weather_benchmark.py
"""

import copy

import config
from vfx_engine import image_io

import live_wallpaper

def _rain_params(intensity):
    params = copy.deepcopy(config.RAIN_PARAMS)
    params["intensity"] = intensity
    return params


def _storm_lightning():
    params = copy.deepcopy(config.LIGHTNING_PARAMS)
    params["frequency"] = 0.35
    return params


WEATHER_PRESETS = {
    # Steady rain: medium density, calm wind, ordinary grade.
    "rain_fog": dict(
        RAIN_ENABLED=True, SNOW_ENABLED=False, FOG_ENABLED=True,
        LIGHTNING_ENABLED=False, FLICKER_ENABLED=False, GOD_RAYS_ENABLED=False,
        RAIN_PARAMS=_rain_params("medium"),
        WEATHER_GUST_STRENGTH=0.35,
        WEATHER_GRADE_STYLE=None,
    ),
    "snow": dict(
        RAIN_ENABLED=False, SNOW_ENABLED=True, FOG_ENABLED=False,
        LIGHTNING_ENABLED=False, FLICKER_ENABLED=False, GOD_RAYS_ENABLED=False,
        # Rain-only effects must not run under snow: lens rain drops, splash
        # ground FX and water ripples showed up as dark blotches over it.
        LENS_RAIN_ENABLED=False, LENS_DIRT_ENABLED=False, GROUND_FX_ENABLED=False,
        WATER_RIPPLE_ENABLED=False,
        WEATHER_GRADE_STYLE="snow",
    ),
    # Genuinely sunny: dehazed, warm, saturated, blue-sky grade (god rays washed the sky out to milky white, so they stay off).
    "clear_sunny": dict(
        RAIN_ENABLED=False, SNOW_ENABLED=False, FOG_ENABLED=False,
        LIGHTNING_ENABLED=False, FLICKER_ENABLED=False, GOD_RAYS_ENABLED=False,
        LENS_DIRT_ENABLED=False,
        WEATHER_GRADE_STYLE="sunny",
    ),
    # Storm: dense fast wind-driven rain, dark desaturated cool grade,
    # frequent lightning, heavy lens rain.
    "storm": dict(
        RAIN_ENABLED=True, SNOW_ENABLED=False, FOG_ENABLED=False,
        LIGHTNING_ENABLED=True, FLICKER_ENABLED=False, GOD_RAYS_ENABLED=False,
        RAIN_PARAMS=_rain_params("storm"),
        LIGHTNING_PARAMS=_storm_lightning(),
        WEATHER_GUST_STRENGTH=0.9,
        WEATHER_GRADE_STYLE="storm",
    ),
}

# Every attribute any preset above touches, plus the ones live_wallpaper.py
# reads unconditionally - restored to these original values after each run
# so presets never leak into one another or into a later normal run of
# live_wallpaper.py in the same process.
_RESTORE_ATTRS = sorted({key for preset in WEATHER_PRESETS.values() for key in preset})


def _apply_preset(preset_overrides):
    for key, value in preset_overrides.items():
        setattr(config, key, value)


def main():
    config.make_sure_folders_exist()
    original_values = {key: copy.deepcopy(getattr(config, key)) for key in _RESTORE_ATTRS}
    original_preview = config.LIVE_PREVIEW_WINDOW
    config.LIVE_PREVIEW_WINDOW = False  # never block the batch on an interactive preview window

    images = image_io.find_input_images(config.INPUT_DIR)
    if not images:
        print(f"No photos found in: {config.INPUT_DIR}")
        return

    already_done = set()
    if live_wallpaper.BENCHMARK_CSV_PATH.exists():
        import csv
        with open(live_wallpaper.BENCHMARK_CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                already_done.add((row["photo"], row["preset_label"]))

    total = len(images) * len(WEATHER_PRESETS)
    done = 0
    try:
        for image_path in images:
            for preset_name, preset_overrides in WEATHER_PRESETS.items():
                done += 1
                if (image_path.name, preset_name) in already_done:
                    print(f"\n=== [{done}/{total}] {image_path.name} :: {preset_name} === SKIPPED (already in {live_wallpaper.BENCHMARK_CSV_PATH.name})")
                    continue
                print(f"\n=== [{done}/{total}] {image_path.name} :: {preset_name} ===")
                try:
                    _apply_preset(preset_overrides)
                    live_wallpaper.process_image(image_path, preset_label=preset_name)
                finally:
                    # Restore after every individual run - not just once at
                    # the end of the whole batch - so a preset that sets a
                    # key another preset doesn't mention (e.g. "storm"'s
                    # WEATHER_GUST_STRENGTH) can never leak into the next
                    # preset/photo's run.
                    for key, value in original_values.items():
                        setattr(config, key, value)
    finally:
        for key, value in original_values.items():
            setattr(config, key, value)
        config.LIVE_PREVIEW_WINDOW = original_preview

    print(f"\nDone! {total} renders complete -> {config.LIVE_WALLPAPERS_DIR}")
    print(f"Timing log: {live_wallpaper.BENCHMARK_CSV_PATH}")


if __name__ == "__main__":
    main()
