"""
frame_cache.py

Per-frame disk cache backing the frame-level render checkpoint/resume
feature (see render_checkpoint.py, which tracks ONLY how many frames were
already written to the output encoder - a bare frame COUNT is not enough
to actually resume from, because the whole point of resuming is to skip
re-doing the work that PRODUCED those frames' pixels, not just to skip
re-running the GPU effect generators that would (deterministically)
recreate them bit-identically. Something still has to hand those exact
already-finished frames back to image_io.save_result/save_mp4, in order,
before the newly-rendered frames continue after them - this module is
that "something": it caches each fully composited, post-processed,
ALREADY-ENCODED (gamma-encoded 8-bit PIL Image - see image_io.
encode_frames_to_images, the very last step before this point in the
pipeline) output frame to a lossless PNG on disk under config.CACHE_DIR,
keyed by the same fingerprint render_checkpoint.py uses, so a resumed
render can read back exactly the same bytes an uninterrupted render would
have produced for those frames instead of re-deriving them.

Deleted (clear_frame_cache) as soon as its render finishes successfully -
the same lifecycle contract as render_checkpoint.delete_checkpoint (see
that module's docstring and notes_for_me.txt item 1: "cache should be
deleted permanently after the program finished working").
"""

import os
import shutil

from PIL import Image, UnidentifiedImageError

import config
from vfx_engine import render_checkpoint


def _frame_cache_dir(fingerprint):
    """
    One subdirectory per fingerprint, next to render_checkpoint.py's own
    JSON checkpoint files (same config.CACHE_DIR/"render_checkpoints"
    parent, same _fingerprint_key naming - see that module) so both halves
    of one interrupted render's state live side by side and are trivial to
    correlate by eye if you ever need to inspect config.CACHE_DIR by hand.
    """
    return config.CACHE_DIR / "render_checkpoints" / f"{render_checkpoint._fingerprint_key(fingerprint)}_frames"


def save_frame(fingerprint, frame_index, image):
    """
    Atomically (temp-file + os.replace, matching this codebase's usual
    cache-write pattern - see render_checkpoint.save_checkpoint,
    provenance.save_provenance, cache_store.cache_put) writes `image` (an
    already gamma-encoded 8-bit PIL Image - the exact output of
    image_io.encode_frames_to_images, not a raw linear-light array) to
    this fingerprint's frame cache as a lossless PNG, named by its
    ABSOLUTE frame_index (not its position within any particular
    truncated stream) - safe to call after every frame, same as
    render_checkpoint.save_checkpoint.
    """
    cache_dir = _frame_cache_dir(fingerprint)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{frame_index:08d}.png"
    temp_path = cache_dir / f"{frame_index:08d}.tmp{os.getpid()}.png"
    image.save(temp_path, format="PNG")
    os.replace(temp_path, path)


def load_frame(fingerprint, frame_index):
    """
    Reads back a previously cached frame (see save_frame) as a fresh PIL
    Image, or returns None if it isn't there or can't be read - callers
    must treat that as "nothing usable to resume from," never as "resume
    from a blank/placeholder frame."
    """
    path = _frame_cache_dir(fingerprint) / f"{frame_index:08d}.png"
    if not path.exists():
        return None
    try:
        with Image.open(path) as loaded:
            loaded.load()
            return loaded.convert(loaded.mode)
    except (OSError, UnidentifiedImageError):
        return None


def clear_frame_cache(fingerprint):
    """
    Permanently deletes this fingerprint's ENTIRE frame-cache directory -
    must be called once the render it belongs to finishes successfully,
    alongside render_checkpoint.delete_checkpoint (notes_for_me.txt: "cache
    should be deleted permanently after the program finished working"). A
    missing directory (nothing to clear, or already cleared) is not an
    error.
    """
    shutil.rmtree(_frame_cache_dir(fingerprint), ignore_errors=True)


def resume_or_checkpoint_frames(new_frames, fingerprint, start_frame, num_frames):
    """
    Generator - the checkpoint/resume wiring point, meant to sit right
    before with_progress/image_io.save_result in main.process_image's
    pipeline:

        final_frames = image_io.encode_frames_to_images(final_frames)  # (num_frames - start_frame) frames
        final_frames = frame_cache.resume_or_checkpoint_frames(final_frames, fingerprint, start_frame, num_frames)
        final_frames = with_progress(final_frames, num_frames)
        out_path = image_io.save_result(final_frames, image_path, output_mode, ...)

    First yields the `start_frame` already-finished frames straight back
    from disk (see load_frame) - cheap, no GPU/CPU render work, so a
    resumed render's progress bar starts moving immediately instead of
    silently stalling while those frames are skipped underneath it. This
    also means the combined stream this function yields is always exactly
    `num_frames` frames long, regardless of start_frame - so with_progress
    (total=num_frames) and image_io.save_result need no special-casing at
    all for a resumed render; they see the same shape of input either way.

    Then yields each of `new_frames`'s (num_frames - start_frame) freshly
    rendered frames one at a time, first caching a copy of each to disk
    (save_frame) and advancing the checkpoint (render_checkpoint.
    save_checkpoint) to that frame's absolute index + 1 - "safe to call
    after every frame" (see save_checkpoint's docstring) - so an
    interruption at any point after this leaves a checkpoint/frame-cache
    pair that resumes from exactly where it left off, never re-doing (or
    silently dropping) a single frame.

    Deliberately does NOT delete the checkpoint or frame cache itself -
    that only happens once the caller has confirmed image_io.save_result
    fully consumed this entire generator without raising (see
    process_image) - see render_checkpoint.delete_checkpoint's docstring
    for why doing it any earlier would be wrong.

    If a cached frame that render_checkpoint.load_checkpoint says should
    exist turns out to be missing or unreadable (see load_frame), this
    raises a plain RuntimeError rather than silently resuming from frame 0
    or skipping the frame - a resumed render should either be exactly
    right or fail loudly, never quietly produce a shorter or corrupted
    output.
    """
    for frame_index in range(start_frame):
        cached = load_frame(fingerprint, frame_index)
        if cached is None:
            raise RuntimeError(
                f"render_checkpoint claims frame {frame_index} of {num_frames} was already "
                f"rendered for this fingerprint, but its cached copy is missing or unreadable "
                f"under {_frame_cache_dir(fingerprint)} - refusing to silently resume with a "
                f"gap in the output. Delete this render's checkpoint (render_checkpoint."
                f"delete_checkpoint) and frame cache (clear_frame_cache) to force a full "
                f"re-render from frame 0 instead."
            )
        yield cached

    for offset, frame in enumerate(new_frames):
        frame_index = start_frame + offset
        save_frame(fingerprint, frame_index, frame)
        render_checkpoint.save_checkpoint(fingerprint, frame_index + 1)
        yield frame
