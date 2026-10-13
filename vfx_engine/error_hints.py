"""
error_hints.py

Turns the raw exception a render died with into something a non-developer can
act on: a one-sentence "what happened", a
"what to try" line, and the full traceback for a log file / "Copy details".
"""

import traceback
from datetime import datetime

import config

ERROR_LOG_PATH = config.OUTPUT_DIR / "gui_error.log"


def _has(text, *needles):
    return any(n in text for n in needles)


def explain_exception(exc):
    """Returns (what_happened, what_to_try) for `exc`."""
    text = f"{type(exc).__name__}: {exc}".lower()

    if _has(text, "out of memory", "cuda error: out of memory", "outofmemoryerror", "failed to allocate", "memoryerror"):
        return ("The graphics card (or memory) ran out of space.",
                "Try a smaller photo, turn off some effects, or lower the supersampling / resolution in the settings.")
    if _has(text, "no space left", "disk full", "errno 28"):
        return ("The disk is full, so the video could not be written.",
                "Free up some disk space (a render needs several GB of temporary room) and try again.")
    if isinstance(exc, FileNotFoundError) and _has(text, "ffmpeg"):
        return ("The video encoder (ffmpeg) could not be found.",
                "Reinstall the requirements (pip install -r requirements.txt) so the bundled ffmpeg is present.")
    if _has(text, "ffmpeg failed", "ffmpeg"):
        return ("The video encoder (ffmpeg) reported an error.",
                "Check there is enough disk space and that the output file is not open in another program, then retry.")
    if _has(text, "models/", "huggingface", "local_files_only", "offline mode", "couldn't connect", "we couldn't connect",
            "does not appear to have a file named", "is not a local folder", "cannot find the requested files"):
        return ("The AI models are missing from the models/ folder.",
                "Run  python share/download_models.py  once (about 20 GB), then start the render again.")
    if _has(text, "cannot identify image file", "unidentifiedimageerror", "image file is truncated", "cannot open image"):
        return ("The photo could not be read.",
                "Make sure it is a normal JPG or PNG that opens in another program, or re-save it and try again.")
    if _has(text, "opengl", "moderngl", "glsl", "context"):
        return ("Your graphics driver does not support what the effects need (OpenGL 4.3).",
                "Update your graphics driver. If that does not help, use Fast Preview (draft), which is lighter.")
    if _has(text, "permission denied", "permissionerror", "being used by another process"):
        return ("A file could not be written because another program is using it.",
                "Close any player that has the previous video open and try again.")
    return ("Something unexpected went wrong.",
            "Copy the details below and check output/gui_error.log for the full technical report.")


def format_details(exc):
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def log_exception(exc, context=""):
    """Appends the full traceback to output/gui_error.log and returns that path."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} {context} =====\n")
        f.write(format_details(exc))
    return ERROR_LOG_PATH

