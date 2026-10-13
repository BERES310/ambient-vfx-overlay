"""
surface_normals.py

monocular surface-normal estimation - a NEW model added to
the pipeline, not a swap of an existing one (contrast with depth.py's
DEPTH_MODE = "marigold" branch, the design notes, which swaps the
architecture behind an ALREADY-existing contract every caller depends on).

Named in the design notes as "the strongest new-model candidate in
this document" - it directly closes a gap that's surfaced repeatedly and
never had a real fix: ground_fx.build_ground_mask's own docstring already
admits it has no actual surface-flatness/normal detection,
which is exactly why a splash or wet-sheen highlight can land on a raised,
angled surface that only *looks* like flat ground when judged from
depth+segmentation alone. A real normal-estimation pass gives "is this
surface actually horizontal and flat" data directly, instead of inferring
it indirectly through stacked depth+position+segmentation heuristics.

Uses diffusers.MarigoldNormalsPipeline (checkpoint
prs-eth/marigold-normals-v1-1) - the same diffusion-based Marigold family
already wired into this project for depth.py's DEPTH_MODE = "marigold"
(item 46), but a DIFFERENT checkpoint fine-tuned for a different output
modality; the two are not interchangeable. API verified against the real
diffusers docs () rather than assumed:

    pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-v1-1"
    ).to(device)
    normals = pipe(image)
    # normals.prediction: unit-length 3D vectors in the screen-space
    # camera frame of reference - X = right, Y = up (top of frame),
    # Z = toward the camera/observer - each component in [-1.0, 1.0].
    # Shape: numimages x height x width x 3 for the default np output
    # type. match_input_resolution defaults to True (already resized to
    # the input image's resolution, same as the depth pipelines).

Output contract of estimate_surface_normals() below: this passes Marigold's
own native XYZ/[-1,1] convention through UNCHANGED rather than remapping it
onto some other convention. Unlike depth.py's estimate_depth (which has to
reconcile THREE different source conventions onto one pre-existing
0.0=far/1.0=near contract several already-written callers depend on), there
is no pre-existing surface-normal consumer in this codebase yet to match -
so inventing a remapped convention here would just be a made-up contract
nothing has actually validated. Wiring this into ground_fx (or any other
consumer) is separate follow-on work, not part of adding the model itself.

Model weights and the HuggingFace/PyTorch cache are kept inside
vfx_overlay/models/ (not the user's global AppData/home-folder caches),
and this module forces fully-offline HuggingFace mode, identically to
depth.py - see that module's header comment for the rationale.

Dormant by default: nothing in the render pipeline calls
estimate_surface_normals() yet, and even a direct call raises a clear
RuntimeError unless config.ENABLE_SURFACE_NORMALS = True. Enabling it
requires the 'diffusers' package (not installed in this project's venv)
and the prs-eth/marigold-normals-v1-1 checkpoint (not downloaded) - see
the guard errors below for the exact one-time setup each requires.
"""

import os

import config

_MODELS_DIR = config.MODELS_DIR
os.environ.setdefault("HF_HOME", str(_MODELS_DIR / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_MODELS_DIR / "huggingface" / "transformers"))
os.environ.setdefault("TORCH_HOME", str(_MODELS_DIR / "torch"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

from vfx_engine import cache_store

_NORMALS_MODEL_NAME = "prs-eth/marigold-normals-v1-1"

_pipeline = None


def _raise_missing_diffusers_dependency(original_error):
    raise RuntimeError(
        "config.ENABLE_SURFACE_NORMALS is True but the 'diffusers' package "
        "is not installed in this project's venv. Surface-normal estimation "
        " uses diffusers.MarigoldNormalsPipeline - the same "
        "package depth.py's DEPTH_MODE = 'marigold' (item 46) needs. "
        "Install it once with a real network connection available: "
        "'./venv/Scripts/python.exe -m pip install diffusers'. Set "
        "config.ENABLE_SURFACE_NORMALS back to False to disable this "
        "feature in the meantime."
    ) from original_error


def _raise_missing_normals_checkpoint(original_error):
    raise RuntimeError(
        f"config.ENABLE_SURFACE_NORMALS is True but the checkpoint "
        f"'{_NORMALS_MODEL_NAME}' is not present in this project's local "
        f"model cache ({_MODELS_DIR / 'huggingface'}). This module forces "
        "fully-offline HuggingFace mode (HF_HUB_OFFLINE=1) so it never "
        "makes a silent network call, so this checkpoint (a full "
        "Stable-Diffusion-2-sized pipeline - UNet, VAE, and CLIP text "
        "encoder bundled together, several GB - and NOT the same "
        "checkpoint as depth.py's marigold-depth-v1-1, even though both "
        "are Marigold models) must be downloaded ONCE first, with a real "
        "network connection available and HF_HUB_OFFLINE temporarily "
        "unset, before config.ENABLE_SURFACE_NORMALS can be used. Once "
        "downloaded it is cached locally under models/huggingface/ (never "
        "re-downloaded again) and every future run works fully offline "
        "again. Set config.ENABLE_SURFACE_NORMALS back to False to disable "
        "this feature in the meantime."
    ) from original_error


def _get_pipeline():
    """
    Lazily loads and caches the MarigoldNormalsPipeline, mirroring
    depth.py's _get_pipeline lazy-singleton pattern. Raises an actionable
    RuntimeError (via the two guard functions above) instead of letting a
    bare ModuleNotFoundError/OSError propagate, if the 'diffusers' package
    or the checkpoint aren't available yet.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import torch

    try:
        import diffusers
    except ModuleNotFoundError as error:
        _raise_missing_diffusers_dependency(error)

    try:
        pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(_NORMALS_MODEL_NAME)
    except OSError as error:
        _raise_missing_normals_checkpoint(error)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _pipeline = pipe.to(device)
    return _pipeline


def _release_pipeline():
    """
    Frees GPU memory by moving the cached pipeline back to CPU and dropping
    it. gc.collect() right after dropping the reference (see depth.py's
    _release_pipeline docstring for the measured reason) - without it, this
    large diffusion pipeline can stay alive on the CPU heap past this
    function returning, stacking on top of the NEXT model's load instead of
    actually being freed first.
    """
    global _pipeline
    if _pipeline is not None:
        _pipeline.to("cpu")
        _pipeline = None
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def estimate_surface_normals(image):
    """
    Runs monocular surface-normal estimation on a single photo.

    Args:
        image: PIL.Image.Image, the input photo (any size/mode - converted
            to RGB internally).

    Returns:
        float32 numpy array, shape (height, width, 3). Each pixel is a
        unit-length 3D vector in Marigold's own screen-space camera frame
        of reference: X = right, Y = up (top of frame), Z = toward the
        camera/observer, each component in [-1.0, 1.0]. Passed through
        unchanged from the model's native output - see this module's
        header comment for why no remapping is applied.

    Raises:
        RuntimeError: if config.ENABLE_SURFACE_NORMALS is False (this is a
            dormant, opt-in feature - see config.py), or if the required
            'diffusers' package / checkpoint aren't available yet (see the
            guard functions above for the exact one-time setup needed).
    """
    if not getattr(config, "ENABLE_SURFACE_NORMALS", False):
        raise RuntimeError(
            "estimate_surface_normals() was called but "
            "config.ENABLE_SURFACE_NORMALS is False. This is a dormant, "
            "opt-in feature - set "
            "config.ENABLE_SURFACE_NORMALS = True to enable it (see this "
            "module's header comment for the one-time setup required "
            "first)."
        )

    # Resumable on-disk cache (see vfx_engine/cache_store.py) - same
    # once-per-photo, expensive-model-call shape as depth.py/segmentation.py,
    # cached the same way, so if this dormant feature is ever enabled the
    # same interrupted-run-resume benefit applies to it without any extra
    # work. Output depends only on the photo's pixel content (no config
    # setting changes what this returns, beyond the ENABLE_SURFACE_NORMALS
    # gate already checked above), so the cache key is the image content
    # hash alone.
    cache_key = cache_store.build_cache_key(image, "surface_normals", version=1)
    cached = cache_store.cache_get("surface_normals", cache_key)
    if cached is not None:
        return cached["normals"]

    pipe = _get_pipeline()
    rgb_image = image.convert("RGB")
    result = pipe(rgb_image)
    normals = np.asarray(result.prediction, dtype="float32")
    normals = np.squeeze(normals, axis=0)

    # Release the model's GPU memory now (see _release_pipeline's docstring)
    # - everything below this point is plain numpy, the GPU is never
    # touched again for this photo. Matches depth.py/segmentation.py's
    # identical placement right after their own last GPU call.
    _release_pipeline()

    cache_store.cache_put("surface_normals", cache_key, normals=normals)
    return normals
