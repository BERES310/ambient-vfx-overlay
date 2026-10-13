"""
progress.py

Console progress bar - rendering long animations (see
config.VIDEO_DURATION_SECONDS) can take a while, so main.py wraps the
stream of finished frames with this right before saving, so you can see how
many frames have been produced and how much longer it'll take.
"""

import sys
import threading
import time


class RenderCancelled(Exception):
    """Raised inside a render when the user pressed Cancel (see set_cancel_event)."""


# GUI hooks. A front-end (vfx_gui.py) installs
# these before starting a render on its worker thread; command-line runs leave
# them unset and behave exactly as before.
_reporter = None       # callable(event: dict) - receives stage + frame progress
_cancel_event = None   # threading.Event - set() means "stop as soon as possible"
_hooks_lock = threading.Lock()


def set_reporter(reporter):
    """Installs (or, with None, removes) the callback that receives
    {"kind": "stage", "stage": key, "label": text} and
    {"kind": "frames", "index", "total", "percent", "eta_seconds", "eta_text"}
    events."""
    global _reporter
    with _hooks_lock:
        _reporter = reporter


def set_cancel_event(event):
    """Installs (or, with None, removes) the threading.Event that with_progress
    and check_cancelled() poll."""
    global _cancel_event
    with _hooks_lock:
        _cancel_event = event


def check_cancelled():
    """Raises RenderCancelled if the installed cancel event has been set."""
    event = _cancel_event
    if event is not None and event.is_set():
        raise RenderCancelled("Render cancelled.")


def report_stage(stage, label):
    """Tells the front-end which pipeline phase is starting (and honours a
    pending cancel request at the phase boundary)."""
    check_cancelled()
    reporter = _reporter
    if reporter is not None:
        reporter({"kind": "stage", "stage": stage, "label": label})

def report_resume(start_frame, total):
    """Tells the front-end that an interrupted render is being continued from `start_frame`."""
    reporter = _reporter
    if reporter is not None:
        reporter({"kind": "resume", "index": start_frame, "total": total, "percent": 100 * start_frame / total})


# Every N frames, return PyTorch's cached-but-unused CUDA blocks to the
# driver (see torch.cuda.empty_cache() below). postprocess.py allocates a
# fresh set of tensors (glow, motion blur accumulator, tonemap, film grain)
# every single frame and never frees its cache itself - depth.py/
# illumination.py/surface_normals.py/sam_segmentation.py all call
# empty_cache() after their one-time per-image CUDA work, but nothing did
# for the per-frame path. On a small (e.g. 6GB) card that let the caching
# allocator's reserved-but-idle memory grow across hundreds of frames until
# it neared the card's ceiling, at which point Windows' WDDM driver began
# silently paging GPU memory to system RAM over PCIe instead of raising
# OOM - a render that looked "stuck" (progress bar creeping from seconds-
# per-frame to minutes-per-frame) rather than crashing. This is the one
# place every render loop's frames already pass through regardless of
# which effects are enabled, so it's clearing the cache for the whole
# pipeline, not just one effect module.
_EMPTY_CACHE_EVERY_N_FRAMES = 20


def report_frames(frames, total, start_frame=0):
    """
    Generator - passes `frames` through unchanged while checking for Cancel
    and sending "frames" events to the front-end. Use it on the RAW frame
    stream, upstream of anything that buffers frames (make_seamless_loop
    holds back its whole crossfade head before yielding one): counting
    frames downstream of such a buffer makes the progress bar sit at 0 % -
    and Cancel go unnoticed - until that many frames have already rendered.
    Pair with with_progress(..., emit_events=False) so events aren't doubled.

    start_frame: how many frames a resumed render already had finished - the
    stream then holds only the remaining frames, but the events still count
    from start_frame + 1 so the bar continues where the earlier run stopped.
    """
    start_time = time.time()
    first_frame_elapsed = None
    for done, frame in enumerate(frames, start=1):
        check_cancelled()
        index = start_frame + done
        elapsed = time.time() - start_time
        steady_eta = None
        if done == 1:
            first_frame_elapsed = elapsed   # frame 1 also pays one-time setup; see with_progress
        else:
            steady_elapsed = elapsed - first_frame_elapsed
            if steady_elapsed > 0:
                steady_eta = (total - index) * steady_elapsed / (done - 1)
        reporter = _reporter
        if reporter is not None:
            reporter({
                "kind": "frames", "index": index, "total": total, "percent": 100 * index / total,
                "eta_seconds": steady_eta,
                "eta_text": format_eta_long(steady_eta) if steady_eta is not None else "calculating...",
            })
        yield frame


def with_progress(frames, total, label="Rendering", emit_events=True):
    """
    Generator - passes frames from `frames` through unchanged, while
    printing a progress bar to the console along the way (frame number,
    percentage, estimated time remaining). `total` is the frame count known
    up front (needed to compute percent/ETA before the stream ends).
    """
    start_time = time.time()
    first_frame_elapsed = None
    bar_width = 30
    torch_cuda = _torch_cuda_if_available()

    for index, frame in enumerate(frames, start=1):
        check_cancelled()
        elapsed = time.time() - start_time

        if index == 1:
            # Frame 1 also pays for one-time setup this lazy generator chain hasn't
            # forced yet (model loads, procedural texture builds, first-touch
            # caches) that will NOT recur on later frames - averaging that inflated
            # cost in as if it were the steady-state per-frame rate is what produces
            # a wildly overstated ETA on the very first update. Steady-state timing
            # starts counting from frame 2 instead.
            first_frame_elapsed = elapsed
            frames_per_second = 0
        else:
            steady_elapsed = elapsed - first_frame_elapsed
            frames_per_second = (index - 1) / steady_elapsed if steady_elapsed > 0 else 0

        eta_seconds = (total - index) / frames_per_second if frames_per_second > 0 else 0
        eta_display = _format_seconds(eta_seconds) if frames_per_second > 0 else "calculating..."

        filled = int(bar_width * index / total)
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = 100 * index / total

        reporter = _reporter if emit_events else None
        if reporter is not None:
            reporter({
                "kind": "frames", "index": index, "total": total, "percent": percent,
                "eta_seconds": eta_seconds if frames_per_second > 0 else None,
                "eta_text": format_eta_long(eta_seconds) if frames_per_second > 0 else "calculating...",
            })

        sys.stdout.write(
            f"\r  {label}: [{bar}] {index}/{total} frames ({percent:5.1f}%) "
            f"- ETA ~{eta_display}   "
        )
        sys.stdout.flush()

        yield frame

        if torch_cuda is not None and index % _EMPTY_CACHE_EVERY_N_FRAMES == 0:
            torch_cuda.empty_cache()

    sys.stdout.write("\n")
    sys.stdout.flush()


def _torch_cuda_if_available():
    """
    Returns the torch.cuda module if torch is importable and a CUDA device
    is actually available, else None - progress.py otherwise has no reason
    to depend on torch at all (plenty of render configs are CPU-only or
    don't hit any of the per-frame torch effects), so the import stays
    local and failure-tolerant instead of a hard top-of-file dependency.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda


def _format_seconds(seconds):
    """Formats a number of seconds as 'Xm Ys' (or just 'Ys' under a minute), for display in the progress bar."""
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def format_eta_long(seconds):
    """Human wording for a remaining time: 'about 8 h 10 min', 'about 12 min', 'under a minute'."""
    seconds = int(seconds)
    if seconds < 60:
        return "under a minute"
    minutes = round(seconds / 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"about {hours} h {minutes:02d} min"
    return f"about {minutes} min"
