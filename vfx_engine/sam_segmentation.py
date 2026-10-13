"""
sam_segmentation.py

SAM-based precise mask-boundary segmentation. Described
in the design notes own prose as "the largest architecture change in this
entire document," worded repeatedly as conditional on WHETHER to build it
at all ("only pursue once mask boundary precision specifically is the
confirmed bottleneck, not preemptively"). Built anyway at the user's own
direct, repeated request naming SAM segmentation specifically - see
the design notes Progress Log entry for item 47.

This is a genuine TWO-STAGE pipeline, not a swap of segmentation.py's
existing SegFormer model:

  Stage 1 (this module) - transformers.SamModel + transformers.SamProcessor
  (checkpoint facebook/sam-vit-huge) propose class-agnostic, pixel-precise
  mask boundaries. SAM itself produces NO semantic labels at all - it only
  answers "what is the precise boundary of the object at/around this
  point". transformers==5.15.1 does not ship the original SAM repo's
  SamAutomaticMaskGenerator (that class lives in the separate
  'segment_anything' PyPI package, which is not a dependency of this
  project), so this module hand-implements the same idea directly against
  SamModel/SamProcessor's lower-level point-prompt API:
    - a grid of point prompts is laid across the image
    - each point is run through the model to get several candidate masks
      plus the model's own predicted IoU ("this mask is probably N%
      accurate") and a derived stability score (does the mask's area
      change much if the logit threshold is nudged up/down - an unstable
      mask is evidence of an ambiguous/low-quality proposal, the same
      signal the original SAM paper's automatic mask generator uses)
    - low-IoU and low-stability proposals are discarded
    - remaining proposals are deduplicated with greedy IoU-based
      non-maximum suppression (highest predicted-IoU mask kept first;
      any later proposal overlapping an already-kept mask above
      _NMS_IOU_THRESHOLD is dropped as a near-duplicate of the same
      region), mirroring the original SAM paper's approach.

  Stage 2 (still this module) - each surviving SAM region is assigned one
  of segmentation.py's existing 9 category labels by reusing
  segmentation.estimate_segmentation(image)'s existing per-pixel category
  probability arrays directly: for each region, the mean probability per
  category over that region's pixels is computed, and the region is
  assigned the argmax category if its mean probability clears
  _CLASSIFICATION_CONFIDENCE_THRESHOLD (left unassigned/dropped
  otherwise). All regions assigned to the same category are unioned into
  one final boolean mask per category. This reuses segmentation.py's
  existing classifier directly rather than adding a second, separate
  CLIP-based zero-shot classifier - segmentation.py already solves "what category is this," so
  SAM only needs to contribute better boundaries for it, not a second
  independent labeling system.

Model weights and the HuggingFace/PyTorch cache are kept inside
vfx_overlay/models/ (never the user's global AppData/home-folder caches),
and this module forces fully-offline HuggingFace mode, identically to
depth.py/surface_normals.py - see those modules' header comments for the
rationale.

Dormant by default: nothing in the render pipeline calls
estimate_sam_segmentation() yet, and even a direct call raises a clear
RuntimeError unless config.ENABLE_SAM_SEGMENTATION = True. Enabling it
requires the facebook/sam-vit-huge checkpoint (not downloaded in this
project yet) - see the guard error below for the exact one-time setup
required. Wiring these precise per-category masks into ground_fx (or any
other consumer) as a replacement/refinement for segmentation.py's soft
probability masks is separate follow-on work, not part of adding the
model itself.
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
from vfx_engine import segmentation

_SAM_MODEL_NAME = "facebook/sam-vit-huge"

# How many points to lay across each axis of the image for the grid-of-
# points automatic mask proposal loop - e.g. 16 means a
# 16x16 = 256-point grid. Matches the original SAM paper's default
# automatic-mask-generator grid density (points_per_side=32 is the paper's
# highest-recall default; 16 is used here as a lighter-weight default
# since this project's "no shortcuts" protocol governs code completeness,
# not runtime-cost tuning of a dormant, not-yet-wired feature).
_POINTS_PER_SIDE = 16

# A proposed mask is discarded if the model's own predicted IoU score
# (outputs.iou_scores - "how accurate do I think this mask is") is below
# this threshold. Mirrors the original SAM paper's pred_iou_thresh=0.88
# default.
_PRED_IOU_THRESHOLD = 0.88

# A proposed mask is discarded if its stability score (area overlap
# between thresholding the mask logits at -1 and +1 around the standard
# 0 cutoff, divided by their union - a mask whose area barely changes
# when the cutoff is nudged is "stable"/confident) is below this
# threshold. Mirrors the original SAM paper's stability_score_thresh=0.95
# default.
_STABILITY_SCORE_THRESHOLD = 0.95

# The logit offset used to compute the stability score above. Mirrors the
# original SAM paper's stability_score_offset=1.0 default.
_STABILITY_SCORE_OFFSET = 1.0

# Greedy IoU-based deduplication threshold: a later mask proposal is
# dropped as a near-duplicate of an already-kept proposal if their IoU
# exceeds this value. Mirrors the original SAM paper's box_nms_thresh
# analogue applied directly to masks (this module has no boxes, only
# masks, so NMS is computed on mask IoU rather than box IoU).
_NMS_IOU_THRESHOLD = 0.7

# A SAM-proposed region is assigned to a segmentation.py category only if
# that category's mean per-pixel probability over the region clears this
# threshold - otherwise the region is left unclassified and contributes
# to no output category (better to omit a region than mislabel it).
_CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.5

_model = None
_processor = None


def _raise_missing_sam_checkpoint(original_error):
    raise RuntimeError(
        f"config.ENABLE_SAM_SEGMENTATION is True but the checkpoint "
        f"'{_SAM_MODEL_NAME}' is not present in this project's local model "
        f"cache ({_MODELS_DIR / 'huggingface'}). This module forces "
        "fully-offline HuggingFace mode (HF_HUB_OFFLINE=1) so it never "
        "makes a silent network call, so this checkpoint (~2.4GB) must be "
        "downloaded ONCE first, with a real network connection available "
        "and HF_HUB_OFFLINE temporarily unset, before "
        "config.ENABLE_SAM_SEGMENTATION can be used. Once downloaded it is "
        "cached locally under models/huggingface/ (never re-downloaded "
        "again) and every future run works fully offline again. Set "
        "config.ENABLE_SAM_SEGMENTATION back to False to disable this "
        "feature in the meantime."
    ) from original_error


def _get_model_and_processor():
    """
    Lazily loads and caches transformers.SamModel/SamProcessor, mirroring
    surface_normals.py's _get_pipeline lazy-singleton pattern. Raises an
    actionable RuntimeError (via the guard function above) instead of
    letting a bare OSError propagate, if the checkpoint isn't available
    yet.
    """
    global _model, _processor
    if _model is not None:
        return _model, _processor

    import torch
    from transformers import SamModel, SamProcessor

    try:
        model = SamModel.from_pretrained(_SAM_MODEL_NAME)
        processor = SamProcessor.from_pretrained(_SAM_MODEL_NAME)
    except OSError as error:
        _raise_missing_sam_checkpoint(error)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    _model = model
    _processor = processor
    return _model, _processor


def _release_model():
    """
    Frees GPU memory by moving the cached model back to CPU and dropping it.
    gc.collect() right after dropping the reference (see depth.py's
    _release_pipeline docstring for the measured reason) - without it, this
    model can stay alive on the CPU heap past this function returning,
    stacking on top of the NEXT model's load instead of actually being
    freed first.
    """
    global _model, _processor
    if _model is not None:
        _model.to("cpu")
        _model = None
        _processor = None
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _build_point_grid(width, height, points_per_side):
    """
    Returns an (points_per_side * points_per_side, 2) float array of
    (x, y) pixel coordinates evenly spaced across the image, each point
    inset from the edges by half a grid cell - the same "grid of points"
    prompting strategy the original SAM paper's automatic mask generator
    uses in place of manual point/box prompts.
    """
    xs = (np.arange(points_per_side, dtype="float64") + 0.5) * (width / points_per_side)
    ys = (np.arange(points_per_side, dtype="float64") + 0.5) * (height / points_per_side)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    return points


def _compute_stability_score(mask_logits, offset):
    """
    Fraction of pixels that agree between thresholding `mask_logits` at
    +offset vs -offset around the standard 0.0 cutoff (intersection over
    union of the two binary masks) - a low value means the mask's extent
    is highly sensitive to the exact logit cutoff, i.e. an unstable/
    ambiguous proposal. Mirrors the original SAM paper's stability-score
    definition.

    Args:
        mask_logits: float array, any shape (one mask's raw logits).
        offset: float, the +/- offset applied around the 0.0 cutoff.

    Returns:
        float in [0.0, 1.0].
    """
    high = mask_logits > offset
    low = mask_logits > -offset
    intersection = np.logical_and(high, low).sum()
    union = np.logical_or(high, low).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def _mask_iou(mask_a, mask_b):
    """Intersection-over-union of two boolean masks of the same shape."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def _generate_mask_proposals(image):
    """
    Runs the grid-of-points automatic mask proposal loop
    against a single PIL image, using transformers.SamModel/SamProcessor's
    point-prompt API directly since transformers does not ship the
    original SAM repo's SamAutomaticMaskGenerator.

    Args:
        image: PIL.Image.Image, RGB.

    Returns:
        list of (boolean numpy array shape (height, width), float
        predicted_iou) tuples, one per surviving proposal after
        IoU/stability-score filtering and greedy IoU-based deduplication,
        sorted by predicted_iou descending.
    """
    import torch

    model, processor = _get_model_and_processor()
    device = next(model.parameters()).device

    width, height = image.size
    points = _build_point_grid(width, height, _POINTS_PER_SIDE)

    candidates = []

    # SAM's point-prompt API expects a batch of points-per-image, each
    # with a batch of points-per-mask - one point per call here, run one
    # point at a time to keep memory bounded regardless of grid density
    # (this is dormant, not-yet-wired code; simplicity is preferred over
    # a batched-inference optimization nothing yet depends on).
    for point in points:
        input_points = [[[float(point[0]), float(point[1])]]]
        inputs = processor(image, input_points=input_points, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=True)

        # post_process_masks resizes SAM's low-resolution logits back up
        # to the original image resolution and undoes the processor's
        # own resize/padding - required before any pixel-level IoU/
        # stability-score math against the original image geometry.
        upscaled_masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
            binarize=False,
        )
        mask_logits = upscaled_masks[0][0].numpy()
        iou_scores = outputs.iou_scores[0][0].detach().cpu().numpy()

        for mask_index in range(mask_logits.shape[0]):
            predicted_iou = float(iou_scores[mask_index])
            if predicted_iou < _PRED_IOU_THRESHOLD:
                continue

            logits_for_mask = mask_logits[mask_index]
            stability_score = _compute_stability_score(logits_for_mask, _STABILITY_SCORE_OFFSET)
            if stability_score < _STABILITY_SCORE_THRESHOLD:
                continue

            binary_mask = logits_for_mask > 0.0
            if not binary_mask.any():
                continue

            candidates.append((binary_mask, predicted_iou))

    # Greedy IoU-based deduplication (non-maximum suppression on masks):
    # process candidates from highest predicted IoU to lowest, keeping a
    # candidate only if it does not overlap any already-kept mask above
    # _NMS_IOU_THRESHOLD.
    candidates.sort(key=lambda item: item[1], reverse=True)
    kept = []
    for mask, predicted_iou in candidates:
        is_duplicate = False
        for kept_mask, _ in kept:
            if _mask_iou(mask, kept_mask) > _NMS_IOU_THRESHOLD:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append((mask, predicted_iou))

    return kept


def _classify_proposals(image, proposals):
    """
    Stage 2 of the pipeline: assigns each SAM-proposed region (from
    `proposals`) one of segmentation.py's existing category labels, by
    reusing segmentation.estimate_segmentation(image)'s per-pixel
    probability arrays directly (mean probability per category over the
    region's pixels, argmax above _CLASSIFICATION_CONFIDENCE_THRESHOLD),
    then unions same-category regions into one final boolean mask per
    category.

    Args:
        image: PIL.Image.Image, RGB.
        proposals: list of (boolean mask, predicted_iou) as returned by
            _generate_mask_proposals.

    Returns:
        dict of category_name -> boolean numpy array, shape
        (height, width), one entry per category in
        segmentation._CATEGORY_LABELS - a category with no assigned
        region is an all-False mask.
    """
    category_probabilities = segmentation.estimate_segmentation(image)
    category_names = list(category_probabilities.keys())

    width, height = image.size
    result = {
        category: np.zeros((height, width), dtype=bool)
        for category in category_names
    }

    for mask, _predicted_iou in proposals:
        pixel_count = mask.sum()
        if pixel_count == 0:
            continue

        best_category = None
        best_mean_probability = _CLASSIFICATION_CONFIDENCE_THRESHOLD
        for category in category_names:
            mean_probability = float(category_probabilities[category][mask].mean())
            if mean_probability > best_mean_probability:
                best_mean_probability = mean_probability
                best_category = category

        if best_category is not None:
            result[best_category] |= mask

    return result


def estimate_sam_segmentation(image):
    """
    Runs the full two-stage SAM segmentation pipeline on a single photo:
    SAM proposes precise, class-agnostic mask boundaries (Stage 1, this
    module), then each proposed region is assigned one of
    segmentation.py's existing category labels by reusing its per-pixel
    probability output directly (Stage 2, this module).

    Args:
        image: PIL.Image.Image, the input photo (any size/mode -
            converted to RGB internally).

    Returns:
        dict of category_name -> boolean numpy array, shape
        (height, width) - True where that category's precise, SAM-
        derived boundary was assigned. Same 9 category names as
        segmentation.estimate_segmentation's output ("sky", "water",
        "vegetation_tree", "vegetation_palm", "vegetation_flower",
        "ground_hard", "ground_soft", "rock_terrain", "building"), but
        hard boolean masks with SAM-precise boundaries instead of
        per-pixel float probabilities.

    Raises:
        RuntimeError: if config.ENABLE_SAM_SEGMENTATION is False (this is
            a dormant, opt-in feature - see config.py), or if the
            required facebook/sam-vit-huge checkpoint isn't available yet
            (see the guard function above for the exact one-time setup
            needed).
    """
    if not getattr(config, "ENABLE_SAM_SEGMENTATION", False):
        raise RuntimeError(
            "estimate_sam_segmentation() was called but "
            "config.ENABLE_SAM_SEGMENTATION is False. This is a dormant, "
            "opt-in feature - set "
            "config.ENABLE_SAM_SEGMENTATION = True to enable it (see this "
            "module's header comment for the one-time setup required "
            "first)."
        )

    # Resumable on-disk cache (see vfx_engine/cache_store.py), same
    # once-per-photo, expensive-model-call shape as depth.py/
    # segmentation.py/surface_normals.py. Output depends only on the
    # photo's pixel content (no config setting changes what this returns,
    # beyond the ENABLE_SAM_SEGMENTATION gate already checked above), so
    # the cache key is the image content hash alone.
    cache_key = cache_store.build_cache_key(image, "sam_segmentation", version=1)
    cached = cache_store.cache_get("sam_segmentation", cache_key)
    if cached is not None:
        return dict(cached)

    rgb_image = image.convert("RGB")
    proposals = _generate_mask_proposals(rgb_image)

    # Release the model's GPU memory now (see _release_model's docstring) -
    # _classify_proposals below reuses segmentation.py's own probability
    # output, not the SAM model, so the GPU is never touched again for
    # this photo. Matches depth.py/surface_normals.py/illumination.py's
    # identical placement right after their own last GPU call.
    _release_model()

    result = _classify_proposals(rgb_image, proposals)
    cache_store.cache_put("sam_segmentation", cache_key, **result)
    return result
