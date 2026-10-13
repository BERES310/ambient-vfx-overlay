"""
render_checkpoint.py

Frame-level render checkpoint/resume. Directly answers the design notes
item 1: "there should be a cache for every render in case rendering gets
interrupted so there would be no need to regenerate everything from
scratch - cache should be deleted permanently after the program finished
working."

This is distinct from vfx_engine/cache_store.py (per-computation caching
of depth/segmentation/surface-normals results, keyed by image content) and
from main.py's resolve_resume_skip (whole-photo resume: skips an already-
FINISHED photo entirely). Neither helps a photo whose render was
interrupted PARTWAY through - this module closes that specific gap.

How it works: every GPU-effect generator in this codebase (fog, rain,
snow, ground effects, god rays, lens dirt/rain, scene warp) computes its
per-frame output as a pure function of (frame_i, num_frames, a handful of
setup arrays derived once from a content-derived seed) - see the
RNG-determinism audit in the design notes Progress Log. That means
re-running the SAME effect generator with the SAME seed/config and asking
it to start its frame loop at `start_frame` instead of 0 produces
bit-identical output for every frame from `start_frame` onward, without
ever having to touch (or re-render) frames 0..start_frame-1. So resuming
an interrupted render costs nothing beyond skipping GPU work for already-
completed frames - it is not a true "suspend/restore a live generator"
(impossible after a process death; there is no OpenGL context, no Python
generator state, nothing left to restore), it is "cheaply re-derive the
same deterministic setup, then fast-forward straight past the frames that
were already written to the output encoder."

A checkpoint is a small JSON file recording: how many frames were already
written to the output encoder for a specific (source photo content, full
config.py snapshot, output mode, mask file contents, num_frames) fingerprint
- the exact same fingerprint main.py's resolve_resume_skip already uses to
decide whether a whole photo can be skipped. Any mismatch in that
fingerprint (different config, different photo content, a changed manual
mask, etc.) invalidates the checkpoint - it is never reused across a
config change, exactly like resolve_resume_skip's own strictness.

Checkpoints live under config.CACHE_DIR/"render_checkpoints" (a cache, not
an output - see config.py's own OUTPUT_DIR/CACHE_DIR distinction), one
JSON file per fingerprint, and MUST be deleted (delete_checkpoint) as soon
as the render they belong to finishes successfully - a checkpoint left on
disk after a successful render would incorrectly resume a *future*,
unrelated render of the same photo/config from the middle.
"""

import hashlib
import json
import os

import config


def _checkpoint_dir():
    checkpoint_dir = config.CACHE_DIR / "render_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def _fingerprint_key(fingerprint):
    """
    Turns a fingerprint dict into a short, filesystem-safe hex string -
    the checkpoint's file stem. The full fingerprint is also stored
    inside the checkpoint file itself (see save_checkpoint/load_checkpoint)
    so a hash collision can never cause a mismatched checkpoint to be
    silently accepted - the stored fingerprint is always compared
    field-by-field before a checkpoint is trusted.
    """
    encoded = json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


def build_fingerprint(image_path, source_image_hash, output_mode, config_snapshot, mask_hashes, num_frames):
    """
    Assembles the fingerprint a checkpoint is keyed and validated against.
    Callers should pass the exact same values already computed for
    provenance.save_provenance/main.resolve_resume_skip for this render,
    so a checkpoint can never silently apply across a source photo,
    config, mask, or frame-count change it wasn't actually produced under.
    """
    return {
        "source_image": image_path.name,
        "source_image_hash": source_image_hash,
        "output_mode": output_mode,
        "config": config_snapshot,
        "mask_hashes": mask_hashes,
        "num_frames": num_frames,
    }


def checkpoint_path(fingerprint):
    return _checkpoint_dir() / f"{_fingerprint_key(fingerprint)}.json"


def load_checkpoint(fingerprint):
    """
    Returns how many frames were already written to the output encoder
    the last time this exact fingerprint was rendered (0 if there is no
    checkpoint, it can't be read, it doesn't match this fingerprint, or it
    claims a frame count that isn't actually less than num_frames - i.e.
    nothing usable to resume from).
    """
    path = checkpoint_path(fingerprint)
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as checkpoint_file:
            data = json.load(checkpoint_file)
    except (OSError, json.JSONDecodeError):
        return 0

    if data.get("fingerprint") != fingerprint:
        return 0

    frames_written = data.get("frames_written")
    num_frames = fingerprint.get("num_frames")
    if not isinstance(frames_written, int) or frames_written <= 0:
        return 0
    if num_frames is not None and frames_written >= num_frames:
        return 0
    return frames_written


def save_checkpoint(fingerprint, frames_written):
    """
    Atomically (temp-file + os.replace, same pattern as provenance.py's
    save_provenance and cache_store.py's cache_put) records that
    `frames_written` frames have been produced and handed to the output
    encoder for this fingerprint. Safe to call after every frame - it's a
    few hundred bytes of JSON, not a meaningful cost next to a GPU render.
    """
    path = checkpoint_path(fingerprint)
    temp_path = path.with_name(f"{path.stem}.tmp{os.getpid()}.json")
    payload = {"fingerprint": fingerprint, "frames_written": frames_written}
    with open(temp_path, "w", encoding="utf-8") as checkpoint_file:
        json.dump(payload, checkpoint_file, indent=2, sort_keys=True, default=str)
    os.replace(temp_path, path)


def delete_checkpoint(fingerprint):
    """
    Permanently removes this fingerprint's checkpoint - must be called
    once the render it belongs to finishes successfully. A missing checkpoint (nothing to resume, or already
    deleted) is not an error.
    """
    path = checkpoint_path(fingerprint)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
