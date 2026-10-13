"""
vfx_cli.py

Command-line entry: the same looks and
output choices as the app, scriptable and batchable.

    python vfx_cli.py photo.jpg --mood storm --quality draft --length 10
    python vfx_cli.py a.jpg b.png --mood rain snow      # 4 videos, one summary
    python vfx_cli.py --all-photos --mood mist          # everything in input_images/
    python vfx_cli.py --list-moods

A batch keeps going when one job fails and prints a summary at the end. A job
that was interrupted (Ctrl+C, power cut) continues from its last frame when
the same command is run again.
"""

import argparse
import sys
import time
from pathlib import Path

import config
from vfx_engine import error_hints, gui_moods, user_settings


def build_parser():
    moods = ", ".join(m["key"] for m in gui_moods.MOODS)
    parser = argparse.ArgumentParser(
        prog="vfx_cli.py",
        description="Turn photos into looping weather videos.",
        epilog="Examples:\n"
               "  python vfx_cli.py photo.jpg --mood storm --quality draft --length 10\n"
               "  python vfx_cli.py a.jpg b.png --mood rain snow\n"
               "  python vfx_cli.py --all-photos --mood mist --quality balanced\n"
               "Defaults come from settings.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photos", nargs="*", help="photo files (or names inside input_images/)")
    parser.add_argument("--all-photos", action="store_true", help="use every photo in input_images/")
    parser.add_argument("--mood", nargs="+", metavar="MOOD", help=f"one or more of: {moods}")
    parser.add_argument("--quality", help="draft, balanced, high or maximum")
    parser.add_argument("--length", type=float, metavar="SECONDS", help="video length in seconds")
    parser.add_argument("--fps", type=int, help="frames per second")
    parser.add_argument("--intensity", type=float, help="0 (none) to 1 (very strong)")
    parser.add_argument("--wind", type=float, help="0 to 1")
    parser.add_argument("--darkness", type=float, help="0 to 1")
    parser.add_argument("--list-moods", action="store_true", help="show the available moods and exit")
    parser.add_argument("--dry-run", action="store_true", help="show what would be made, render nothing")
    return parser


def resolve_photo(text):
    path = Path(text)
    for candidate in (path, config.INPUT_DIR / path.name):
        if candidate.is_file():
            return candidate
    raise user_settings.SettingsError([f"Photo not found: {text} (looked here and in {config.INPUT_DIR})."])


def merged_settings(args):
    """settings.json defaults, overridden by the command line, then validated."""
    data = dict(user_settings.load())
    if args.mood:
        data["mood"] = args.mood[0]
    if args.quality:
        data["quality"] = args.quality.capitalize()
    for name, key in (("length", "length_seconds"), ("fps", "fps"), ("intensity", "intensity"),
                      ("wind", "wind"), ("darkness", "darkness")):
        value = getattr(args, name)
        if value is not None:
            data[key] = value
    return user_settings.validate(data)


def overrides_for(settings, mood_key):
    mood = gui_moods.MOOD_BY_KEY[mood_key]
    overrides = gui_moods.build_look_overrides(
        gui_moods.effects_for_mood(mood_key),
        mood["intensity"] if settings["intensity"] is None else settings["intensity"],
        mood["wind"] if settings["wind"] is None else settings["wind"],
        mood["darkness"] if settings["darkness"] is None else settings["darkness"],
        mood_key=mood_key)
    overrides.update(gui_moods.quality_overrides(settings["quality"], settings["length_seconds"], settings["fps"]))
    overrides["LIVE_PREVIEW_WINDOW"] = False
    return overrides


def run_job(photo, mood_key, settings):
    import live_wallpaper
    overrides = overrides_for(settings, mood_key)
    originals = {attr: getattr(config, attr, None) for attr in overrides}
    out_path = gui_moods.next_free_path(
        config.LIVE_WALLPAPERS_DIR / f"{photo.stem}_{mood_key}_{settings['quality'].lower()}.mp4")
    try:
        for attr, value in overrides.items():
            setattr(config, attr, value)
        live_wallpaper.process_image(photo, out_path=out_path)
    finally:
        for attr, value in originals.items():
            setattr(config, attr, value)
    return out_path


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_moods:
        for mood in gui_moods.MOODS:
            print(f"  {mood['key']:<6} {mood['label']:<12} {mood['blurb']}")
        return 0
    try:
        settings = merged_settings(args)
        moods = args.mood or [settings["mood"]]
        bad = [m for m in moods if m not in gui_moods.MOOD_BY_KEY]
        if bad:
            raise user_settings.SettingsError(
                [f"Unknown mood {b!r}; choose from: {', '.join(gui_moods.MOOD_BY_KEY)}." for b in bad])
        config.make_sure_folders_exist()
        if args.all_photos:
            from vfx_engine import image_io
            photos = image_io.find_input_images(config.INPUT_DIR)
        else:
            photos = [resolve_photo(p) for p in args.photos]
    except user_settings.SettingsError as exc:
        print("Cannot start:\n  " + "\n  ".join(exc.problems), file=sys.stderr)
        return 2
    if not photos:
        parser.print_usage(sys.stderr)
        print("Give at least one photo, or use --all-photos.", file=sys.stderr)
        return 2

    jobs = [(photo, mood) for photo in photos for mood in moods]
    print(f"{len(jobs)} video(s) to make - quality {settings['quality']}, "
          f"{settings['length_seconds']:g} s, {settings['fps']} fps.")
    if args.dry_run:
        for photo, mood in jobs:
            print(f"  would make: {photo.name} + {mood}")
        return 0

    results = []
    started = time.perf_counter()
    for number, (photo, mood) in enumerate(jobs, 1):
        print(f"\n[{number}/{len(jobs)}] {photo.name} + {mood}")
        try:
            results.append((photo, mood, run_job(photo, mood, settings), None))
        except KeyboardInterrupt:
            print("\nStopped. Run the same command again to continue where it left off.")
            results.append((photo, mood, None, "stopped"))
            break
        except Exception as exc:  # noqa: BLE001 - one bad job must not sink the batch
            what, fix = error_hints.explain_exception(exc)
            error_hints.log_exception(exc, context=f"{photo.name} + {mood}")
            results.append((photo, mood, None, f"{what} {fix}"))
    print(f"\nSummary ({(time.perf_counter() - started) / 60:.1f} min)")
    failed = 0
    for photo, mood, path, error in results:
        if path:
            print(f"  OK      {photo.name} + {mood} -> {path}")
        else:
            failed += 1
            print(f"  FAILED  {photo.name} + {mood}: {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
