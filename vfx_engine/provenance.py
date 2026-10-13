"""
provenance.py

the design notes Part 20.4: "render provenance - save the exact parameter set
alongside every output." With 20+ Parts' worth of interacting tunable
parameters now in config.py, being able to look back at any past render
and know precisely what produced it (and diff two renders' parameter
sets against each other) becomes increasingly valuable as the system
grows - and costs almost nothing to add, since the parameters are all
already assembled in memory (as plain config.py module attributes) at
render time.

save_provenance() writes a small JSON sidecar file next to every
rendered output - not just "approved" profiles (that's a separate,
not-yet-built Part 12 concept), every render, test, draft, or final.
"""

import datetime
import hashlib
import json
import os
from pathlib import Path

import config


def _json_safe(value):
    """
    Recursively converts a config.py value into something json.dump can
    serialize directly - config.py's settings are almost entirely plain
    JSON-safe types already (numbers, strings, bools, tuples/lists,
    dicts, None), the two exceptions being pathlib.Path objects (several
    *_DIR settings) and tuples (JSON has no tuple type, encoded as a
    plain list, which is lossless for this purely-informational sidecar's
    purpose - nothing reads this file back into the running program).
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # A genuinely unexpected type (e.g. a function reference accidentally
    # left as a module-level config value) - stringify rather than crash
    # a run over a purely-informational sidecar file.
    return str(value)


def _snapshot_config():
    """
    Returns a dict of every "public setting" in config.py - by this
    codebase's own established convention (see config.py throughout),
    that means every module-level UPPERCASE name that isn't itself a
    module (config.py imports pathlib/os at module scope, which are
    lowercase and already excluded by the naming check alone, but the
    isinstance guard below is a second, explicit safeguard against ever
    accidentally serializing an imported module object).
    """
    snapshot = {}
    for name in dir(config):
        if not name.isupper():
            continue
        value = getattr(config, name)
        if isinstance(value, type(config)):   # a ModuleType, e.g. an accidental `import X` that happens to be all-caps
            continue
        snapshot[name] = _json_safe(value)
    return snapshot


def _file_content_hash(path):
    """
    Returns a short, stable hex digest of a file's raw bytes on disk -
    used for the source photo and any manual mask file(s) so
    resolve_resume_skip (main.py) can detect "same filename, different
    content" (a photo or hand-drawn mask re-exported/re-edited in place)
    instead of trusting the filename alone, which previously let a
    changed file silently reuse a stale rendered output.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as source_file:
        hasher.update(source_file.read())
    return hasher.hexdigest()[:32]


def resolve_mask_paths(image_path):
    """
    Returns the manual mask file(s) (see vfx_engine/mask.py's
    _load_or_draw_mask, and vfx_gui.py's identical MASKS_DIR/suffix
    convention) that actually affect this photo's render under the
    CURRENT config.py settings, if any exist yet on disk. Empty when
    MASK_MODE isn't "manual" and MASK_MANUAL_CORRECTION is off, since no
    hand-drawn file influences the render in that case.
    """
    stem = image_path.stem
    candidates = []
    if config.MASK_MODE == "manual":
        candidates.append(config.MASKS_DIR / f"{stem}.png")
    if config.MASK_MANUAL_CORRECTION:
        candidates.append(config.MASKS_DIR / f"{stem}_correction.png")
    return [path for path in candidates if path.exists()]


def save_provenance(out_path, image_path, num_frames, output_mode):
    """
    Writes a JSON sidecar file next to `out_path` (e.g.
    output/castle_by_the_lake_rain_fog.mp4 ->
    output/castle_by_the_lake_rain_fog.mp4.provenance.json) recording:
    the exact config.py parameter set active for this render, the source
    photo (name AND content hash - see _file_content_hash), any manual
    mask file(s) actually used (by content hash too), resolved output
    mode/frame count, and a timestamp - everything needed to know
    precisely what produced this specific output file, or to diff two
    renders' parameter sets against each other later.

    out_path: the actual rendered file's path (from image_io.save_result)
        - the sidecar is named by appending ".provenance.json" to it, so
        it sorts/groups next to its output in a file listing and survives
        the output's own suffix-disambiguation (resolve_effect_suffix)
        without needing to duplicate that logic here.
    """
    mask_paths = resolve_mask_paths(image_path)
    provenance = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "output_file": out_path.name,
        "source_image": image_path.name,
        "source_image_hash": _file_content_hash(image_path),
        "mask_hashes": {path.name: _file_content_hash(path) for path in mask_paths},
        "output_mode": output_mode,
        "num_frames": num_frames,
        "config": _snapshot_config(),
    }

    # Atomic temp-file + os.replace write (same pattern as, and for the
    # same reason as, cache_store.py's cache_put: os.replace is atomic on
    # both POSIX and Windows, so a process interrupted mid-write - Ctrl+C,
    # crash, power loss - can never leave a half-written .provenance.json
    # behind. That matters here specifically because resolve_resume_skip
    # (main.py) trusts this exact file's presence and content to decide
    # whether a photo's render is already done - a truncated/corrupt
    # sidecar read back as valid JSON with missing fields would otherwise
    # risk a wrong resume decision.
    sidecar_path = out_path.with_name(out_path.name + ".provenance.json")
    temp_path = sidecar_path.with_name(f"{sidecar_path.stem}.tmp{os.getpid()}.json")
    with open(temp_path, "w", encoding="utf-8") as sidecar_file:
        json.dump(provenance, sidecar_file, indent=2, sort_keys=True)
    os.replace(temp_path, sidecar_path)

    return sidecar_path
