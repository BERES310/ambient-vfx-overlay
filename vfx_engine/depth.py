"""
depth.py

Monocular depth estimation - guesses, from a single 2D photo, how near or
far every pixel is. This is what lets effects like snow/rain/fog respect
the scene's geometry: falling *behind* a foreground tree or railing
instead of as a flat layer painted over the whole image, which is the
single biggest visual difference between "an overlay slapped on a photo"
and "a convincing animated scene".

Used via MASK_MODE = "depth" in config.py - see mask.py's _build_depth_mask,
which turns the raw depth map from estimate_depth() below into a mask.

Model weights and the HuggingFace/PyTorch cache are kept inside
vfx_overlay/models/ (not the user's global AppData/home-folder caches) -
the environment variables below are set before torch/transformers get
imported anywhere in the process, so this holds for every effect that
ends up needing them.

estimate_depth() also runs a guided-filter edge-refinement pass (see
_guided_filter_refine_depth below) on the model's raw output before
returning it: monocular depth models produce noisy, sometimes-wrong edges
around fine detail (thin branches, chain-link fences, distant power
lines/rooflines) - pulling the depth map's boundaries back toward the
PHOTO's own real edges sharpens exactly the boundaries every downstream
depth-driven system (occlusion masking, ground_fx's ground mask, fog's
depth falloff, the focus-plane blur) relies on for a clean result.
"""

import os

import config

_MODELS_DIR = config.MODELS_DIR
os.environ.setdefault("HF_HOME", str(_MODELS_DIR / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_MODELS_DIR / "huggingface" / "transformers"))
os.environ.setdefault("TORCH_HOME", str(_MODELS_DIR / "torch"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# The model is expected to already be cached locally (see MODELS_DIR above) -
# force fully offline mode so no network call is ever made (not even a
# metadata/rate-limit check), keeping this strictly local as required.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

from vfx_engine import cache_store

# The Large checkpoint (~1.3GB, cached locally under MODELS_DIR - see
# above - never re-downloaded after the first run) - meaningfully sharper
# depth boundaries than the Small variant this used to load, which
# directly limits how clean an occlusion mask (MASK_MODE = "depth") or
# ground_fx's ground mask can ever be, since both are built by
# thresholding this map. depth estimation runs exactly ONCE per photo
# (the result is cached and reused for every frame of the animation, see
# main.py's process_image), so the extra inference cost of the bigger
# checkpoint is a one-time-per-photo cost, not a per-frame one - and
# generation time is explicitly not a constraint for this project, so there's no real reason to keep using
# the smaller/faster/blurrier checkpoint.
_RELATIVE_MODEL_NAME = "depth-anything/Depth-Anything-V2-Large-hf"

# Metric checkpoints - same architecture/parameter count as
# the relative Large checkpoint above (so the VRAM/compute budget discussed
# in 18.1 doesn't change), fine-tuned to output real-world metric distance
# instead of a purely relative 0..1 spread. "Outdoor" was fine-tuned on
# Virtual KITTI (driving/outdoor scenery); "indoor" on Hypersim. See
# config.DEPTH_MODE / config.DEPTH_METRIC_VARIANT.
_METRIC_MODEL_NAMES = {
    "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}

# Marigold - a diffusion-based depth model (repurposes a
# Stable-Diffusion-2 image generator's own learned prior for depth
# prediction, via a diffusers pipeline, not the transformers
# "depth-estimation" pipeline the two checkpoints above use) - see
# _get_pipeline's marigold branch and estimate_depth's docstring for the
# full rationale (18.3 names this as the option to reach for specifically
# when depth artifacts recur in hazy/low-texture regions, e.g. bug 5.4a's
# fog-seam issue - Depth-Anything's regression architecture has to commit
# to a single confident guess per pixel even where the photo itself gives
# almost no signal, whereas Marigold's generative/ensemble-capable design
# can express that ambiguity instead of confidently guessing wrong).
_MARIGOLD_MODEL_NAME = "prs-eth/marigold-depth-v1-1"

_pipeline = None
_pipeline_model_name = None
_pipeline_kind = "relative"


def _resolve_model_name():
    """
    Picks which checkpoint to load based on config.DEPTH_MODE - "relative"
    (default, unchanged behavior), "metric", or "marigold"
. Returns (model_name, kind) where kind is one of
    "relative", "metric", "marigold".
    """
    mode = getattr(config, "DEPTH_MODE", "relative")
    if mode == "relative":
        return _RELATIVE_MODEL_NAME, "relative"
    if mode == "metric":
        variant = getattr(config, "DEPTH_METRIC_VARIANT", "outdoor")
        if variant not in _METRIC_MODEL_NAMES:
            raise ValueError(
                f"Unknown config.DEPTH_METRIC_VARIANT: '{variant}'. "
                f"Use one of: {sorted(_METRIC_MODEL_NAMES)}."
            )
        return _METRIC_MODEL_NAMES[variant], "metric"
    if mode == "marigold":
        return _MARIGOLD_MODEL_NAME, "marigold"
    raise ValueError(
        f"Unknown config.DEPTH_MODE: '{mode}'. Use 'relative', 'metric', or 'marigold'."
    )


def _raise_missing_metric_checkpoint(model_name, original_error):
    """
    depth.py forces fully-offline HuggingFace mode (see this module's
    header comment) so a photo render never silently blocks on or performs
    a network call. The metric checkpoints aren't part of this project's
    local model cache yet (only the relative Small/Large checkpoints are -
    confirmed by listing models/huggingface/hub). Rather than letting the
    cryptic underlying "couldn't connect to hf.co, and couldn't find it
    locally either" OSError surface, this gives a precise, actionable
    explanation of exactly what's missing and what the one-time fix is.
    """
    raise RuntimeError(
        f"config.DEPTH_MODE is 'metric' but the checkpoint '{model_name}' "
        f"is not present in this project's local model cache "
        f"({_MODELS_DIR / 'huggingface'}). depth.py forces fully-offline "
        "HuggingFace mode (HF_HUB_OFFLINE=1) so it never makes a silent "
        "network call, so this checkpoint must be downloaded ONCE first, "
        "with a real network connection available and HF_HUB_OFFLINE "
        "temporarily unset, before DEPTH_MODE = 'metric' can be used. "
        "Once downloaded it is cached locally under models/huggingface/ "
        "(never re-downloaded again, same as the relative checkpoints) "
        "and every future run works fully offline again. Set "
        "config.DEPTH_MODE back to 'relative' to keep using the existing "
        "cached checkpoint in the meantime."
    ) from original_error


def _raise_missing_marigold_dependency(original_error):
    """
    Marigold is a diffusers pipeline, not a transformers
    "depth-estimation" pipeline - it needs the `diffusers` package, which
    this project does not otherwise depend on (confirmed absent: plain
    `import diffusers` raises ModuleNotFoundError in this project's venv).
    Surfaces that plainly, with the exact one-time fix, instead of letting
    a bare ModuleNotFoundError surface with no context.
    """
    raise RuntimeError(
        "config.DEPTH_MODE is 'marigold' but the 'diffusers' package is not "
        "installed in this project's venv. Marigold uses "
        "diffusers.MarigoldDepthPipeline, not the transformers "
        "'depth-estimation' pipeline the relative/metric modes use. Install "
        "it once with a real network connection available: "
        "'./venv/Scripts/python.exe -m pip install diffusers'. Set "
        "config.DEPTH_MODE back to 'relative' (or 'metric') to keep using "
        "the existing checkpoints in the meantime."
    ) from original_error


def _raise_missing_marigold_checkpoint(model_name, original_error):
    """
    Same rationale as _raise_missing_metric_checkpoint above: depth.py
    forces fully-offline HuggingFace mode, so the multi-gigabyte Marigold
    checkpoint (a full Stable-Diffusion-2-sized pipeline: UNet + VAE +
    CLIP text encoder, all bundled under this one repo id) must be
    downloaded ONCE, with a real network connection available and
    HF_HUB_OFFLINE temporarily unset, before config.DEPTH_MODE = "marigold"
    can be used.
    """
    raise RuntimeError(
        f"config.DEPTH_MODE is 'marigold' but the checkpoint '{model_name}' "
        f"is not present in this project's local model cache "
        f"({_MODELS_DIR / 'huggingface'}). depth.py forces fully-offline "
        "HuggingFace mode (HF_HUB_OFFLINE=1) so it never makes a silent "
        "network call, so this checkpoint (a full Stable-Diffusion-2-sized "
        "pipeline - UNet, VAE, and CLIP text encoder all bundled together, "
        "several GB) must be downloaded ONCE first, with a real network "
        "connection available and HF_HUB_OFFLINE temporarily unset, before "
        "DEPTH_MODE = 'marigold' can be used. Once downloaded it is cached "
        "locally under models/huggingface/ (never re-downloaded again, same "
        "as the other checkpoints) and every future run works fully offline "
        "again. Set config.DEPTH_MODE back to 'relative' (or 'metric') to "
        "keep using the existing cached checkpoint in the meantime."
    ) from original_error

# Guided-filter edge-refinement parameters (see _guided_filter_refine_depth) -
# eps is a fixed regularization constant (small relative to the guide's
# 0..1 luminance variance); the filter RADIUS instead scales with the
# photo's own resolution (via _edge_refine_radius below) so the
# neighborhood the filter reasons over stays a sensible size in actual
# pixels across very different photo resolutions, the same way e.g.
# ground_fx's feather radius does.
_EDGE_REFINE_EPS = 1e-3


def _edge_refine_radius(image_size):
    """Box-filter window radius (pixels) for the guided-filter depth refinement below - scales with the photo's shorter side, so the neighborhood stays meaningful whether the photo is 720p or 8K."""
    return max(3, round(min(image_size) / 220.0))


def _get_pipeline():
    """Lazily loads (and caches, both on disk in models/ and in memory for the rest of the run) the depth-estimation model - see _resolve_model_name for how config.DEPTH_MODE picks relative vs metric vs marigold."""
    global _pipeline, _pipeline_model_name, _pipeline_kind
    model_name, kind = _resolve_model_name()
    if _pipeline is None or _pipeline_model_name != model_name:
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [depth] Loading depth-estimation model ({model_name}) on {device} (first run downloads it into {_MODELS_DIR}) ...")

        if kind == "marigold":
            try:
                import diffusers
            except ModuleNotFoundError as error:
                _raise_missing_marigold_dependency(error)
            try:
                _pipeline = diffusers.MarigoldDepthPipeline.from_pretrained(model_name)
            except OSError as error:
                _raise_missing_marigold_checkpoint(model_name, error)
            _pipeline = _pipeline.to(device)
        else:
            from transformers import pipeline

            try:
                _pipeline = pipeline(task="depth-estimation", model=model_name, device=device)
            except OSError as error:
                if kind == "metric":
                    _raise_missing_metric_checkpoint(model_name, error)
                raise
        _pipeline_model_name = model_name
        _pipeline_kind = kind
    return _pipeline


def _release_pipeline():
    """
    Frees the depth-estimation model's GPU memory (moves it back to CPU
    and drops the cached pipeline object) immediately after inference,
    instead of leaving a large model GPU-resident for the rest of the
    render even though depth's own result, from here on, is nothing but
    a plain numpy array.

    Confirmed root cause of a real, user-reported multi-minute stall
: segmentation.py's
    SegFormer-B5 model, loaded and run right after this one on the same
    GPU, measured ~10x slower per-tile with this model still resident
    (~370s for a real photo's full tile set) than with a clean GPU
    (~38s for the identical tile set) - and this was NOT simple VRAM
    capacity exhaustion (several GB stayed free either way, confirmed
    via nvidia-smi). Two large models sharing a single small (6GB-class
    laptop) GPU's memory pool fragments PyTorch's CUDA caching allocator
    badly enough that segmentation's many small per-tile allocations
    can't find a large contiguous free block and fall back to slow,
    effectively-synchronous cudaMalloc calls on nearly every tile,
    instead of being served instantly from the allocator's cache.

    Safe to call unconditionally: estimate_depth() is only called once
    per photo (its result is cached and reused for every frame - see
    main.py's process_image), so the model is never needed again for
    THIS photo; if a later photo in the same run needs depth again,
    _get_pipeline() reloads it from the on-disk cache (a few seconds,
    not a re-download) - a trivial cost next to the 300+ seconds of GPU
    memory contention this avoids.
    """
    global _pipeline
    if _pipeline is not None:
        import gc

        import torch
        if _pipeline_kind == "marigold":
            _pipeline.to("cpu")   # diffusers DiffusionPipeline - no .model wrapper
        else:
            _pipeline.model.to("cpu")   # transformers pipeline
        _pipeline = None
        # gc.collect() BEFORE empty_cache(): dropping the reference above
        # doesn't guarantee CPython frees the model immediately - large
        # PyTorch modules commonly hold internal reference cycles (autograd
        # metadata, parametrization/hook registries), so without an
        # explicit collection pass the model can sit alive, on the CPU heap,
        # until Python's next natural GC cycle - which may not happen before
        # the NEXT model (illumination/surface_normals/segmentation) loads
        # on top of it. Confirmed via direct measurement: sequential model
        # loads without this accumulated CPU RSS well past 14GB before any
        # per-frame rendering even began, on a machine with only ~32GB
        # total RAM - collecting immediately here is what makes each
        # module's release actually free memory back to the process instead
        # of just formally dropping a reference.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_DEPTH_NORMALIZATION_LOW_PERCENTILE = 1.0
_DEPTH_NORMALIZATION_HIGH_PERCENTILE = 99.0


def _robust_normalize(array, low_percentile=_DEPTH_NORMALIZATION_LOW_PERCENTILE,
                       high_percentile=_DEPTH_NORMALIZATION_HIGH_PERCENTILE):
    """
    Normalizes `array` to 0.0-1.0 using a ROBUST percentile range instead
    of the literal min/max - monocular depth models
    occasionally mis-estimate a stray pixel (known to happen especially at
    frame edges or on reflective/transparent surfaces), and a single such
    outlier being treated as "the single nearest/farthest point in the
    whole scene" silently compresses the EFFECTIVE depth range every
    OTHER pixel gets stretched across, distorting every depth-dependent
    system downstream (occlusion threshold, ground masking, fog falloff,
    focus-plane blur, camera parallax layer weights).

    Values outside the [low_percentile, high_percentile] range are
    CLAMPED to 0.0/1.0 (via the final clip), not left to silently stretch
    the scale for everyone else - a genuine outlier just saturates to
    "very far"/"very near" instead of quietly warping the whole scene's
    relative depth spread.
    """
    low = np.percentile(array, low_percentile)
    high = np.percentile(array, high_percentile)
    if high - low < 1e-6:
        return np.zeros_like(array, dtype="float32")
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype("float32")


def _depth_confidence(guide, radius):
    """
    Returns a (height, width) float32 array, 0.0-1.0 - a proxy for how
    much to trust the guided filter's edge-snapped depth at each pixel
. Built from the guide (photo luminance) image's own
    LOCAL VARIANCE, reusing _box_filter, the same local-statistics
    building block _guided_filter_refine_depth itself is built from - a
    real monocular depth model is LEAST confident, and the guided filter
    has LITERALLY NOTHING to snap an edge to, in flat/textureless regions:
    sky, calm water, blank walls, reflective/transparent surfaces (exactly
    the surfaces this module's docstring names as the model's known weak
    points). High local variance (real nearby texture/edges) = high
    confidence; low variance = low confidence.

    Normalized by a high percentile of ITS OWN distribution (this photo's
    own texture range), not a fixed absolute threshold - what counts as
    "textured" varies hugely between a detailed close-up and a soft
    landscape photo.
    """
    mean = _box_filter(guide, radius)
    mean_of_squares = _box_filter(guide * guide, radius)
    variance = np.clip(mean_of_squares - mean * mean, 0.0, None)
    normalizer = max(1e-6, float(np.percentile(variance, 90)))
    return np.clip(variance / normalizer, 0.0, 1.0).astype("float32")


def estimate_depth(image, return_confidence=False):
    """
    Returns a numpy float32 array, shape (height, width), normalized
    0.0 (farthest pixel in the photo) .. 1.0 (nearest pixel), same
    orientation/size as `image` (a PIL image). The raw model output is
    passed through a guided-filter edge-refinement pass (see
    _guided_filter_refine_depth) before being returned, using the photo's
    own luminance as a guide - see this module's docstring for why.

    Metric-vs-relative: with the
    default relative checkpoint, "nearness" is only ever this ONE photo's
    own relative spread renormalized to 0..1. With config.DEPTH_MODE =
    "metric", a checkpoint fine-tuned to output real-world distance is
    used instead - the raw metric distance (larger = FARTHER, the opposite
    sign convention from the relative model's disparity-like raw output)
    is read from the pipeline's `predicted_depth` tensor, then flipped
    after normalization so this function's own return contract ("0.0 =
    far, 1.0 = near") stays identical either way - every existing caller
    keeps working unmodified regardless of which mode is active.

    Marigold: a completely
    different architecture - a diffusion model repurposed for depth,
    rather than Depth-Anything's single-pass regression network - loaded
    via diffusers.MarigoldDepthPipeline instead of the transformers
    "depth-estimation" pipeline the other two modes use (see
    _get_pipeline's marigold branch). Its raw `prediction` output is
    affine-invariant depth in [0, 1] with 0 = the NEAREST plane and 1 =
    the FARTHEST plane (per the model's own documented convention) - the
    same "larger = farther" direction as the metric checkpoints above (for
    an unrelated reason: metric mode is farther-as-larger because it's
    real-world distance, Marigold is farther-as-larger because that's just
    this model's own convention), so it gets the identical flip-after-
    normalization treatment to reach this function's "0.0 = far, 1.0 =
    near" contract. Requires the `diffusers` package and the Marigold
    checkpoint to be present locally (see _raise_missing_marigold_* above)
    - neither is installed/downloaded by default.

    Confidence-aware blending: the guided filter improves
    edge PLACEMENT by snapping depth boundaries back toward the photo's
    own luminance edges, but that snap is only trustworthy where the
    photo actually HAS local texture/edges to snap to - forcing an edge-
    snap in a flat, textureless region (sky, calm water, a blank wall)
    risks introducing a boundary that doesn't really exist there. The
    final depth is now blended, per pixel, between the fully-refined
    value and the smoother pre-refinement estimate, weighted by
    `_depth_confidence` - high-confidence (textured) regions keep the
    full edge-snap, low-confidence (flat) regions lean back toward the
    less aggressively-corrected estimate instead.

    return_confidence: False (default, unchanged return type - every
        existing caller keeps working exactly as before) or True to get
        `(depth_array, confidence_array)` instead - for callers that want
        to use the confidence signal themselves (e.g. to be more
        conservative in low-confidence regions), not currently wired into
        every depth consumer in this pipeline (a larger, separate scope -
        see the design notes own framing of 7.4 as "not a large gap on its
        own, but worth naming").
    """
    # Resumable on-disk cache (see vfx_engine/cache_store.py): estimate_depth
    # is a pure function of (this photo's pixel content, DEPTH_MODE, and -
    # only when DEPTH_MODE is "metric" - DEPTH_METRIC_VARIANT), and is
    # expensive enough (a full model inference pass) that it's worth
    # skipping entirely on a re-run of the same photo under the same
    # settings, e.g. after an earlier run was interrupted partway through
    # a multi-photo batch. Both the depth and confidence arrays are always
    # cached together (confidence is already computed internally below
    # regardless of return_confidence - see this function's docstring), so
    # a later call with a DIFFERENT return_confidence value than the one
    # that first populated the cache still hits it correctly.
    mode = getattr(config, "DEPTH_MODE", "relative")
    variant = getattr(config, "DEPTH_METRIC_VARIANT", "outdoor") if mode == "metric" else None
    cache_key = cache_store.build_cache_key(image, "depth", version=1, mode=mode, variant=variant)
    cached = cache_store.cache_get("depth", cache_key)
    if cached is not None:
        return (cached["depth"], cached["confidence"]) if return_confidence else cached["depth"]

    pipe = _get_pipeline()
    kind = _pipeline_kind
    is_metric = (kind == "metric")
    is_marigold = (kind == "marigold")

    if is_marigold:
        # MarigoldDepthPipeline's own calling convention (verified against
        # the diffusers docs, not guessed - see this project's Progress Log
        # for item 46/18.3): pass the PIL image directly, read `.prediction`
        # back. With the default match_input_resolution=True, the
        # prediction is already resized to the input image's own
        # resolution, and with output_type="np" (the default) its shape is
        # (num_images, height, width, 1) - squeeze down to (height, width).
        depth = pipe(image)
        depth_array = np.asarray(depth.prediction, dtype="float32").squeeze()
    else:
        result = pipe(image)
        depth_image = result["depth"].resize(image.size)   # the pipeline can return a slightly different size

        if is_metric:
            # Metric checkpoints' `result["predicted_depth"]` carries the raw
            # real-world distance values (meters) that `depth_image` above was
            # only quantized/rescaled-to-a-viewable-image from - reading the
            # tensor directly instead of the rendered image avoids losing that
            # precision to an intermediate 0-255 PIL image.
            depth_array = result["predicted_depth"].squeeze().detach().cpu().numpy().astype("float32")
            target_shape = (image.size[1], image.size[0])   # (height, width) - PIL's .size is (width, height)
            if depth_array.shape != target_shape:
                from PIL import Image
                resized = Image.fromarray(depth_array, mode="F").resize(image.size, Image.BILINEAR)
                depth_array = np.asarray(resized, dtype="float32")
        else:
            depth_array = np.asarray(depth_image, dtype="float32")

    # Release the model's GPU memory now (see _release_pipeline's
    # docstring) - everything below this point is plain numpy/PIL, the
    # GPU is never touched again for this photo.
    _release_pipeline()

    if depth_array.max() - depth_array.min() < 1e-6:
        flat = np.zeros((image.size[1], image.size[0]), dtype="float32")
        cache_store.cache_put("depth", cache_key, depth=flat, confidence=flat.copy())
        return (flat, flat.copy()) if return_confidence else flat
    normalized = _robust_normalize(depth_array)
    if is_metric or is_marigold:
        # Metric depth is real-world DISTANCE: larger value = farther away.
        # Marigold's raw `prediction` is 0 = near plane, 1 = far plane -
        # also larger = farther, for an unrelated reason (see this
        # function's docstring). Either way, every consumer downstream of
        # this module (occlusion masking, ground_fx, fog falloff,
        # focus-plane blur, camera parallax) was built against the
        # opposite, relative-model convention (this module's own
        # docstring: "0.0 = farthest .. 1.0 = nearest", since the relative
        # model's raw output is disparity-like - larger value = nearer).
        # Flipping here is the ONLY change either non-default mode makes
        # to the rest of the pipeline's contract - everything past this
        # line keeps operating on "0.0 = far, 1.0 = near" exactly as
        # before regardless of which of the three modes is active.
        normalized = 1.0 - normalized

    guide = np.asarray(image.convert("L"), dtype="float32") / 255.0
    radius = _edge_refine_radius(image.size)
    refined = _guided_filter_refine_depth(normalized, guide, radius=radius, eps=_EDGE_REFINE_EPS)

    # The guided filter is a locally-linear fit, not a clamp - its output
    # isn't guaranteed to stay exactly within the input's 0..1 range at
    # every pixel (a small overshoot near strong edges is normal for this
    # filter). Renormalize so downstream code's "0.0 = far, 1.0 = near"
    # assumption keeps holding exactly, the same way the raw-model
    # normalization above does - robustly again, for the same reason
    # (the guided filter's own overshoot is itself a form of outlier that
    # a literal min/max renormalization would be equally vulnerable to).
    confidence = _depth_confidence(guide, radius)
    if refined.max() - refined.min() < 1e-6:
        result_depth = normalized
    else:
        refined_normalized = _robust_normalize(refined)
        result_depth = confidence * refined_normalized + (1.0 - confidence) * normalized

    cache_store.cache_put("depth", cache_key, depth=result_depth, confidence=confidence)
    return (result_depth, confidence) if return_confidence else result_depth


def _box_filter(array, radius):
    """
    Fast box (mean) filter via an integral image (cumulative sum) -
    O(H*W) total regardless of radius, the building block
    _guided_filter_refine_depth needs (local means/correlations),
    reimplemented here with plain numpy instead of pulling in scipy as a
    new dependency for a single filter. Handles the image border
    correctly by dividing each pixel's window sum by its ACTUAL (clipped)
    window area, not a fixed (2*radius+1)^2 - a plain fixed-divisor box
    filter would incorrectly darken/fade every edge and corner pixel,
    whose window is smaller than the interior's.
    """
    array = np.asarray(array, dtype="float64")
    height, width = array.shape

    integral = np.zeros((height + 1, width + 1), dtype="float64")
    integral[1:, 1:] = np.cumsum(np.cumsum(array, axis=0), axis=1)

    rows = np.arange(height)
    row_lo = np.clip(rows - radius, 0, height)
    row_hi = np.clip(rows + radius + 1, 0, height)
    cols = np.arange(width)
    col_lo = np.clip(cols - radius, 0, width)
    col_hi = np.clip(cols + radius + 1, 0, width)

    # integral[row_hi, col_hi] etc., vectorized via fancy row-then-column
    # indexing - equivalent to a nested loop over every pixel's own
    # (row_lo..row_hi, col_lo..col_hi) rectangle, without actually looping.
    sum_hi_hi = integral[row_hi][:, col_hi]
    sum_lo_hi = integral[row_lo][:, col_hi]
    sum_hi_lo = integral[row_hi][:, col_lo]
    sum_lo_lo = integral[row_lo][:, col_lo]
    window_sum = sum_hi_hi - sum_lo_hi - sum_hi_lo + sum_lo_lo

    window_area = (row_hi - row_lo)[:, None].astype("float64") * (col_hi - col_lo)[None, :].astype("float64")
    return (window_sum / window_area).astype("float32")


def _guided_filter_refine_depth(depth_array, guide, radius=8, eps=1e-3):
    """
    Edge-aware refinement of a raw depth map using the base photo's own
    luminance as a guide - He et al.'s guided filter, the standard fast,
    edge-preserving smoothing technique (also used for e.g. image matting
    and dehazing). Fits a locally-linear model (depth ~= a*guide + b) in
    small windows across the image, which has the effect of snapping the
    depth map's boundaries back onto the PHOTO's own real edges instead of
    the depth model's own blurrier estimate of where they are - exactly
    what a thin branch, a railing, or a distant roofline needs to end up
    with a clean edge in the depth map, instead of blurring into its
    background the way raw monocular depth output tends to.

    radius: box-filter window radius in pixels (see _box_filter) - the
        neighborhood size the filter reasons about local structure over.
    eps: regularization constant - larger values smooth more / preserve
        edges less, smaller values track the guide's edges more
        aggressively (and can amplify noise already present in the guide
        image if pushed too low).
    """
    guide = guide.astype("float32")
    p = depth_array.astype("float32")

    mean_I = _box_filter(guide, radius)
    mean_p = _box_filter(p, radius)
    corr_I = _box_filter(guide * guide, radius)
    corr_Ip = _box_filter(guide * p, radius)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)

    return mean_a * guide + mean_b
