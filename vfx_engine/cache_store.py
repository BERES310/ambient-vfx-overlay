"""
cache_store.py

Resumable on-disk cache for the per-photo, run-once, expensive
model-inference results that gate every render - depth.py's
estimate_depth() and segmentation.py's estimate_segmentation(). Both are
pure functions of (photo pixel content, the handful of config.py settings
that actually change their output), so their results are safe to persist
to disk and reload on a later run of the SAME photo under the SAME
relevant config - letting an interrupted or re-run batch skip straight
back to frame rendering without redoing multi-minute model inference it
has already done once before.

Kept inside vfx_overlay/ (config.CACHE_DIR, next to models/ and output/ -
see config.py) per this project's "everything stays inside vfx_overlay/"
rule - never a global/user-home cache location.

Each cache entry is one .npz file (numpy's own compressed multi-array
container - a natural fit, since every cached value here is already a
numpy array) named by a cache key that mixes the source photo's own pixel
content (so a differently-edited/replaced photo reusing an old filename
never serves back a stale entry) with every config setting that can
change the cached function's output (so flipping e.g. DEPTH_MODE can
never accidentally serve back a result computed under the old mode) and a
per-call-site version number (so a future code change to what's stored or
how it's interpreted can invalidate every existing entry at once just by
bumping that number, instead of ever risking loading an old, differently-
shaped result back into new code).

Writes are atomic (write to a temp file in the same directory, then
os.replace onto the final name) - os.replace is atomic on both POSIX and
Windows, so a process interrupted mid-write (Ctrl+C, crash, power loss)
can never leave behind a half-written .npz that a later run would try
(and fail, or worse, silently partially-load) to read back. Call sites
write to this cache immediately after each expensive computation
finishes (not batched/deferred to end-of-run) - the whole point is that
an interrupted run still leaves every already-finished computation safely
cached on disk, per the "save as often as possible" requirement this
module was built for.
"""

import hashlib
import os

import numpy as np

import config

CACHE_DIR = config.CACHE_DIR


def _image_content_hash(image):
    """
    Returns a short, stable hex digest of a PIL image's actual pixel
    content (mode + size + raw bytes) - two different photos, or the same
    filename re-exported/re-edited to different pixel content, always
    hash differently; the exact same pixel content always hashes
    identically regardless of filename, so renaming a photo doesn't
    invalidate its cache, and replacing a photo's content under a
    previously-used filename can never silently reuse a stale entry.
    """
    hasher = hashlib.sha256()
    hasher.update(image.mode.encode("utf-8"))
    hasher.update(f"{image.size[0]}x{image.size[1]}".encode("utf-8"))
    hasher.update(image.tobytes())
    return hasher.hexdigest()[:32]


def build_cache_key(image, category, version=1, **config_fields):
    """
    Builds a filesystem-safe cache key string for one (image, category)
    pair.

    category: names the kind of computation ("depth", "segmentation", ...).
    version: lets a call site invalidate every existing entry for that
        category at once if it ever changes what it stores or how the
        stored arrays are interpreted - bump the constant the call site
        passes in, and old entries are simply never matched again (a
        clean cache miss, not a stale-format crash).
    config_fields: whatever named config values actually change that
        category's output (e.g. depth.py passes mode=config.DEPTH_MODE,
        variant=config.DEPTH_METRIC_VARIANT) - included directly in the
        key so a config change is itself enough to miss the old cache
        entry, instead of ever serving back a result computed under
        different settings. Fields whose value is None are omitted (a
        call site passes None for a field that doesn't apply under the
        current config, e.g. DEPTH_METRIC_VARIANT when DEPTH_MODE isn't
        "metric" - it shouldn't affect the key in that case).
    """
    image_hash = _image_content_hash(image)
    field_part = "_".join(
        f"{name}-{value}" for name, value in sorted(config_fields.items()) if value is not None
    )
    key = f"{category}_v{version}_{image_hash}"
    if field_part:
        key += f"_{field_part}"
    return key


def _entry_path(category, cache_key):
    category_dir = CACHE_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)
    return category_dir / f"{cache_key}.npz"


def cache_get(category, cache_key):
    """
    Returns a dict of numpy arrays previously saved under this exact
    (category, cache_key) via cache_put(), or None if there's no entry
    yet, or if the entry on disk is unreadable (e.g. disk corruption, or a
    file left over from an older, incompatible on-disk layout) - a cache
    miss always just means "recompute it", never a hard failure, since
    this cache is purely a resume/speed optimization and never the only
    copy of anything.
    """
    entry_path = _entry_path(category, cache_key)
    if not entry_path.exists():
        return None
    try:
        with np.load(entry_path) as data:
            return {name: data[name] for name in data.files}
    except Exception as error:
        print(f"  [cache] Ignoring unreadable cache entry {entry_path.name} ({error}) - recomputing.")
        return None


def cache_put(category, cache_key, **arrays):
    """
    Saves `arrays` (numpy arrays) to disk under this exact (category,
    cache_key), atomically (see this module's docstring) - safe to call
    immediately after the expensive computation finishes, with no risk of
    a concurrent read (e.g. from a separately-resumed process) ever
    observing a half-written file.
    """
    entry_path = _entry_path(category, cache_key)
    temp_path = entry_path.with_name(f"{entry_path.stem}.tmp{os.getpid()}.npz")
    np.savez_compressed(temp_path, **arrays)
    os.replace(temp_path, entry_path)
