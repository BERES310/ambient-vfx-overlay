"""
illumination.py

learned illumination estimation - the design notes own text
scopes this item as "an exploratory spike... before committing to a
specific checkpoint", NOT a green-lit implementation the way items 44/46/48
(depth.py's DEPTH_MODE = "marigold"/"metric", surface_normals.py) were. An
earlier pass over this same document already ran that spike and logged a
"do not implement yet" decision for two reasons: (a) the design notes explicitly asks for
evaluation first, not code, and (b) this project's own numeric/synthetic-
only verification protocol cannot judge whether the candidate model's
output actually proxies "where is the light source and what color is it" -
that requires looking at real output on a real photo, which this session's
verification protocol is explicitly not doing until the user says
otherwise.

This module exists anyway, because the user was shown that exact caveat and
explicitly asked for it to be built despite it (see config.py's
ENABLE_LEARNED_ILLUMINATION comment and the design notes second item-49 entry).
It is dormant by default and its light-source-position/color outputs below
are clearly documented as an unverified derived heuristic, not a validated
fix for the four gaps the spike named (3.5 wet-sheen's hardcoded white
highlight, 5.4 fog not knowing light color, 15.1 lens dirt having no light
coupling, 16.5 lens flare only detecting one light source via
god_rays.detect_light_sources's brightest-pixel approach) - wiring this
in as an actual replacement/supplement for god_rays.sample_light_color is
separate follow-on work, contingent on someone actually judging this
module's real output against real photos first.

Uses diffusers.MarigoldIntrinsicsPipeline, "lighting" checkpoint variant
(prs-eth/marigold-iid-lighting-v1-1) - part of the same Marigold diffusion
family already wired into this project for depth.py's DEPTH_MODE =
"marigold" (item 46) and surface_normals.py (item 48), but yet another
separate checkpoint (NOT interchangeable with either). API verified
directly against the installed package's own source
(diffusers/pipelines/marigold/pipeline_marigold_intrinsics.py - not merely
assumed by analogy with the depth/normals pipelines):

    pipe = diffusers.MarigoldIntrinsicsPipeline.from_pretrained(
        "prs-eth/marigold-iid-lighting-v1-1"
    ).to(device)
    result = pipe(image)
    # result.prediction: np.ndarray, shape
    # (num_images * num_targets, height, width, 3), values in [0, 1].
    # pipe.target_properties["target_names"] gives the target order for
    # this checkpoint: ["albedo", "shading", "residual"] - i.e.
    # I (the original photo) = Albedo * Shading + Residual. "shading" is
    # the diffuse-illumination component; "residual" captures whatever the
    # diffuse model doesn't explain (specular highlights, inter-reflection),
    # which is this module's only real per-pixel signal for a light's own
    # COLOR (as opposed to its overall diffuse intensity).

Same lazy-load/local-cache/offline pattern as depth.py and
surface_normals.py: model weights and the HuggingFace/PyTorch cache stay
inside vfx_overlay/models/ (never the user's global AppData/home-folder
caches), and the process runs fully offline once the checkpoint has been
downloaded once - see the environment variables below, set before
torch/transformers/diffusers get imported.

Dormant by default: nothing in the render pipeline calls
estimate_illumination() yet, and even a direct call raises a clear
RuntimeError unless config.ENABLE_LEARNED_ILLUMINATION = True.
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

_ILLUMINATION_MODEL_NAME = "prs-eth/marigold-iid-lighting-v1-1"

_pipeline = None

# How far the light-position heuristic below biases its search toward the
# top of the frame, and how big a neighborhood it averages for the derived
# light color - kept identical to god_rays.detect_light_source's own
# top_bias=0.7 default and sample_light_color's radius_frac=0.05 default
# specifically so this module's derived estimate can be compared apples-
# to-apples against the existing heuristic it might someday replace, not
# because these particular numbers were independently tuned for this
# module.
_LIGHT_POS_TOP_BIAS = 0.7
_LIGHT_COLOR_RADIUS_FRAC = 0.05


def _raise_missing_diffusers_dependency(original_error):
    raise RuntimeError(
        "config.ENABLE_LEARNED_ILLUMINATION is True but the 'diffusers' "
        "package is not installed in this project's venv. Illumination "
        "estimation uses diffusers.MarigoldIntrinsicsPipeline "
        "- the same package depth.py's DEPTH_MODE = 'marigold' (item 46) "
        "and surface_normals.py (item 48) need. Install it once with a real "
        "network connection available: "
        "'./venv/Scripts/python.exe -m pip install diffusers'. Set "
        "config.ENABLE_LEARNED_ILLUMINATION back to False to disable this "
        "feature in the meantime."
    ) from original_error


def _raise_missing_illumination_checkpoint(original_error):
    raise RuntimeError(
        f"config.ENABLE_LEARNED_ILLUMINATION is True but the checkpoint "
        f"'{_ILLUMINATION_MODEL_NAME}' is not present in this project's "
        f"local model cache ({_MODELS_DIR / 'huggingface'}). This module "
        "forces fully-offline HuggingFace mode (HF_HUB_OFFLINE=1) so it "
        "never makes a silent network call, so this checkpoint (a full "
        "Stable-Diffusion-2-sized pipeline - UNet, VAE, and CLIP text "
        "encoder bundled together, several GB - and NOT the same "
        "checkpoint as marigold-depth-v1-1 or marigold-normals-v1-1, even "
        "though all three are Marigold models) must be downloaded ONCE "
        "first, with a real network connection available and "
        "HF_HUB_OFFLINE temporarily unset, before "
        "config.ENABLE_LEARNED_ILLUMINATION can be used. Once downloaded it "
        "is cached locally under models/huggingface/ (never re-downloaded "
        "again) and every future run works fully offline again. Set "
        "config.ENABLE_LEARNED_ILLUMINATION back to False to disable this "
        "feature in the meantime."
    ) from original_error


def _get_pipeline():
    """
    Lazily loads and caches the MarigoldIntrinsicsPipeline (lighting
    variant), mirroring depth.py/surface_normals.py's lazy-singleton
    pattern. Raises an actionable RuntimeError (via the two guard functions
    above) instead of letting a bare ModuleNotFoundError/OSError propagate,
    if the 'diffusers' package or the checkpoint aren't available yet.
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
        pipe = diffusers.MarigoldIntrinsicsPipeline.from_pretrained(_ILLUMINATION_MODEL_NAME)
    except OSError as error:
        _raise_missing_illumination_checkpoint(error)

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


def _estimate_light_position(shading):
    """
    Returns (x, y) pixel coordinates of the brightest region of the
    per-pixel diffuse-shading map, biased toward the top of the frame -
    deliberately the SAME top-biased-argmax approach as
    god_rays.detect_light_source, applied to the decomposition's shading
    channel instead of the raw photo's luminance, so the two can be
    compared directly. This is a derived heuristic, not something the
    model itself outputs or was trained to predict - see this module's
    header comment for why its usefulness as an actual light-source
    localizer is unverified.
    """
    height, width = shading.shape[:2]
    luminance = shading.mean(axis=2) if shading.ndim == 3 else shading

    thumb_size = 96
    # Match god_rays.detect_light_source's use of a small thumbnail rather
    # than the full-resolution map - same reasoning (cheap, and a coarse
    # search is all a single dominant-region heuristic needs).
    row_idx = np.linspace(0, height - 1, thumb_size).astype("int64")
    col_idx = np.linspace(0, width - 1, thumb_size).astype("int64")
    thumb = luminance[np.ix_(row_idx, col_idx)]

    rows = np.arange(thumb_size, dtype="float32").reshape(-1, 1) / max(thumb_size - 1, 1)
    vertical_weight = 1.0 - _LIGHT_POS_TOP_BIAS * rows
    weighted = thumb * vertical_weight

    flat_index = int(np.argmax(weighted))
    row, col = divmod(flat_index, thumb_size)
    return (col / thumb_size) * width, (row / thumb_size) * height


def _estimate_light_color(residual, albedo, shading, light_pos):
    """
    Returns an approximate (r, g, b) 0-255 light color, sampled from a small
    neighborhood around `light_pos` in the RESIDUAL map (the "non-diffuse"
    component the decomposition can't explain via Albedo*Shading alone -
    the closest signal this decomposition has to genuine specular/light-tint
    information). Falls back to sampling the SHADING map itself (tinted by
    ALBEDO, undoing the decomposition to approximate the original photo's
    own local color) when the residual in that neighborhood is negligible
    (a purely diffuse-lit region has essentially nothing left in the
    residual to sample) - this fallback path is equivalent to reconstructing
    Albedo*Shading locally, i.e. approximately re-deriving the original
    photo's own color at that point, the same information
    god_rays.sample_light_color already reads directly from the photo.
    """
    height, width = residual.shape[:2]
    radius = max(2, int(min(width, height) * _LIGHT_COLOR_RADIUS_FRAC))
    x, y = light_pos
    left, right = max(0, int(x - radius)), min(width, int(x + radius))
    top, bottom = max(0, int(y - radius)), min(height, int(y + radius))
    if right <= left or bottom <= top:
        return (255, 255, 255)

    residual_patch = residual[top:bottom, left:right].reshape(-1, 3)
    residual_strength = float(residual_patch.mean())

    if residual_strength > 1e-3:
        color = residual_patch.mean(axis=0)
    else:
        albedo_patch = albedo[top:bottom, left:right].reshape(-1, 3)
        shading_patch = shading[top:bottom, left:right]
        shading_patch = shading_patch.reshape(-1, 3) if shading_patch.ndim == 3 else shading_patch.reshape(-1, 1)
        color = (albedo_patch * shading_patch).mean(axis=0)

    color = np.clip(color, 0.0, 1.0) * 255.0
    return tuple(int(channel) for channel in color)


def estimate_illumination(image):
    """
    Runs Marigold's lighting-variant Intrinsic Image Decomposition on a
    single photo.

    Args:
        image: PIL.Image.Image, the input photo (any size/mode - converted
            to RGB internally).

    Returns:
        dict with:
            "albedo":   float32 (height, width, 3), values in [0, 1].
            "shading":  float32 (height, width, 3), values in [0, 1] - the
                        diffuse-illumination component.
            "residual": float32 (height, width, 3), values in [0, 1] - the
                        non-diffuse component (I = Albedo*Shading + Residual).
            "light_pos": (x, y) pixel coordinates - a DERIVED heuristic
                        estimate (see _estimate_light_position), not a
                        model output. Directly comparable to
                        god_rays.detect_light_source's return value.
            "light_color": (r, g, b) 0-255 - a DERIVED heuristic estimate
                        (see _estimate_light_color), not a model output.
                        Directly comparable to god_rays.sample_light_color's
                        return value.

    Raises:
        RuntimeError: if config.ENABLE_LEARNED_ILLUMINATION is False (this
            is a dormant, opt-in feature - see config.py), or if the
            required 'diffusers' package / checkpoint aren't available yet
            (see the guard functions above for the exact one-time setup
            needed).
    """
    if not getattr(config, "ENABLE_LEARNED_ILLUMINATION", False):
        raise RuntimeError(
            "estimate_illumination() was called but "
            "config.ENABLE_LEARNED_ILLUMINATION is False. This is a "
            "dormant, opt-in feature - set "
            "config.ENABLE_LEARNED_ILLUMINATION = True to enable it (see "
            "this module's header comment for the one-time setup required "
            "first, and the important caveat about its light-source "
            "outputs being an unverified derived heuristic)."
        )

    # Resumable on-disk cache (see vfx_engine/cache_store.py) - same
    # once-per-photo, expensive-model-call shape as depth.py/
    # surface_normals.py. Output depends only on the photo's pixel content
    # (no config setting changes what this returns, beyond the
    # ENABLE_LEARNED_ILLUMINATION gate already checked above), so the cache
    # key is the image content hash alone.
    cache_key = cache_store.build_cache_key(image, "illumination", version=1)
    cached = cache_store.cache_get("illumination", cache_key)
    if cached is not None:
        return {
            "albedo": cached["albedo"],
            "shading": cached["shading"],
            "residual": cached["residual"],
            "light_pos": (float(cached["light_pos_x"]), float(cached["light_pos_y"])),
            "light_color": tuple(int(v) for v in cached["light_color"]),
        }

    pipe = _get_pipeline()
    rgb_image = image.convert("RGB")
    result = pipe(rgb_image)
    prediction = np.asarray(result.prediction, dtype="float32")   # (num_targets, H, W, 3)

    target_names = pipe.target_properties["target_names"]
    targets = {name: prediction[index] for index, name in enumerate(target_names)}
    albedo = targets["albedo"]
    shading = targets["shading"]
    residual = targets["residual"]

    # Release the model's GPU memory now (see _release_pipeline's docstring)
    # - everything below this point is plain numpy, the GPU is never
    # touched again for this photo. Matches depth.py/surface_normals.py's
    # identical placement right after their own last GPU call.
    _release_pipeline()

    light_pos = _estimate_light_position(shading)
    light_color = _estimate_light_color(residual, albedo, shading, light_pos)

    cache_store.cache_put(
        "illumination", cache_key,
        albedo=albedo, shading=shading, residual=residual,
        light_pos_x=np.float32(light_pos[0]), light_pos_y=np.float32(light_pos[1]),
        light_color=np.asarray(light_color, dtype="int32"),
    )
    return {
        "albedo": albedo,
        "shading": shading,
        "residual": residual,
        "light_pos": light_pos,
        "light_color": light_color,
    }
