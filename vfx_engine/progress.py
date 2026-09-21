"""
progress.py

Console progress bar - rendering long animations (see
config.VIDEO_DURATION_SECONDS) can take a while, so main.py wraps the
stream of finished frames with this right before saving, so you can see how
many frames have been produced and how much longer it'll take.
"""

import sys
import time

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


def with_progress(frames, total, label="Rendering"):
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
