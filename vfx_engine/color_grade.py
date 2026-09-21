"""
color_grade.py

Base-photo color-grade unification pass - pulls a source photo's sky and foreground/ground regions toward a
single shared color cast BEFORE any effect layers on top, instead of
leaving inherently mismatched lighting untouched.

Root cause this addresses: this pipeline only ever ADDS effect layers on top of
the base photo - there has never been a global color-grade unification
pass anywhere. A source photo with a warm sunset sky over a cool-toned
foreground (or the reverse), or a golden sky over a dark-green forest,
stays exactly that inconsistent; nothing corrects source-photo lighting
mismatches. Worse, every effect that samples the photo's own color for
its own tint (sample_ambient_tint for rain drop color, splash refraction's
fill_color, god_rays/lens-flare highlight color, etc.) inherits whichever
mismatch is already baked in. Correcting the base photo itself, once,
before anything else runs, means every downstream consumer automatically
benefits with no changes needed anywhere else - the same "fix it once,
upstream" principle main.py's SUPERSAMPLE_FACTOR step already uses.

Design: this is deliberately a gentle, PARTIAL pull toward a shared
compromise cast, not a full white-balance reset - a photo already lit
coherently comes out of this pass visually unchanged (the math below
naturally produces a ~1.0 gain when the two regions already agree, with
no explicit "skip if already coherent" branch needed), and even a badly
mismatched photo keeps its own overall character, just with the harshest
cast conflict softened. All math happens in LINEAR light (see color.py's
own rationale for why grading/blending belongs there, not in gamma-
encoded sRGB), and the sky/ground split is blended per-pixel using the
mask's own soft/continuous value (not a hard threshold), so there is no
visible seam at the boundary.
"""

import numpy as np
from PIL import Image

from vfx_engine import color


def _region_cast(pixels_linear, weights):
    """
    Returns a (3,) linear-light color "cast" vector for a region: its
    weighted-average color, normalized so the vector's own mean is 1.0.
    Dividing out the region's average brightness this way isolates HUE/
    color-temperature (what "warm" or "cool" actually means) from overall
    exposure, so two regions lit by the same color of light at different
    brightness levels don't get flagged as mismatched.

    `weights` is a soft (H, W) 0.0-1.0 mask, so a region with only a
    sliver of pixels (a thin strip of sky over the treeline) still gets a
    sensible weighted average instead of being dominated by a handful of
    outlier pixels.

    Returns None when the region is negligible (near-zero total weight,
    or near-black average) - nothing coherent to measure.
    """
    total_weight = float(weights.sum())
    if total_weight < 1.0:
        return None
    avg = (pixels_linear * weights[:, :, None]).sum(axis=(0, 1)) / total_weight
    mean_channel = float(avg.mean())
    if mean_channel < 1e-4:
        return None
    return avg / mean_channel


def _fallback_sky_mask(width, height):
    """
    A smooth top-to-bottom gradient mask used when no real segmentation
    sky mask is available: fully "sky" in the top 10% of rows, fully
    "foreground/ground" from 55% of the way down, linearly blended in
    between. Coarse compared to real AI segmentation (see
    vfx_engine/segmentation.py), but it costs nothing to compute and
    still gives this pass a sensible region split to work with on photos
    where segmentation wasn't already computed for another effect.
    """
    row = np.linspace(0.0, 1.0, height, dtype="float32")
    gradient = np.clip(1.0 - (row - 0.10) / 0.45, 0.0, 1.0)
    return np.repeat(gradient[:, None], width, axis=1)


def unify_color_grade(image, segmentation=None, strength=0.5, max_gain=1.8, min_gain=0.55):
    """
    Returns a new PIL Image (same size/mode as `image`) with its sky and
    foreground/ground regions pulled toward a shared color cast.

    `segmentation`: the dict returned by main.py's resolve_segmentation
        (i.e. vfx_engine.segmentation.estimate_segmentation), or None.
        `segmentation["sky"]` is used as the soft sky mask when present -
        a real AI-segmented boundary following actual object edges
        (clouds, rooflines, treetops), not a crude horizontal split.
        Falls back to `_fallback_sky_mask` when no segmentation is
        available, or it has no "sky" entry.

    `strength`: 0.0 (no correction) - 1.0 (fully match the two regions'
        casts to their shared weighted-average target). See the module
        docstring for why partial correction is the intended default.

    `max_gain`/`min_gain`: hard per-channel clamp on the computed
        correction - keeps a badly-mismatched outlier photo from being
        pushed into an implausible over-corrected look (crushing one
        channel toward zero, or blowing another one out).
    """
    if strength <= 0.0:
        return image

    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    srgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    linear = color.srgb_to_linear(srgb)

    sky_mask = None
    if segmentation is not None:
        sky_mask = segmentation.get("sky")
    if sky_mask is None:
        sky_mask = _fallback_sky_mask(width, height)
    else:
        sky_mask = np.asarray(sky_mask, dtype=np.float32)
        if sky_mask.ndim == 3:
            sky_mask = sky_mask[:, :, 0]
        sky_mask = np.clip(sky_mask, 0.0, 1.0)
        if sky_mask.shape[:2] != (height, width):
            sky_mask = np.asarray(
                Image.fromarray((sky_mask * 255.0).astype(np.uint8)).resize((width, height), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0

    ground_mask = 1.0 - sky_mask

    sky_cast = _region_cast(linear, sky_mask)
    ground_cast = _region_cast(linear, ground_mask)
    if sky_cast is None or ground_cast is None:
        # One region is negligible (e.g. an extreme close-up with no
        # visible sky, or a photo that's nothing but sky) - there is
        # nothing coherent to unify against, so leave the photo untouched
        # rather than "correcting" against noise.
        return image

    sky_weight = float(sky_mask.sum())
    ground_weight = float(ground_mask.sum())
    target_cast = (sky_cast * sky_weight + ground_cast * ground_weight) / (sky_weight + ground_weight)

    safe_sky_cast = np.maximum(sky_cast, 1e-4)
    safe_ground_cast = np.maximum(ground_cast, 1e-4)
    gain_sky = np.clip((target_cast / safe_sky_cast) ** strength, min_gain, max_gain).astype(np.float32)
    gain_ground = np.clip((target_cast / safe_ground_cast) ** strength, min_gain, max_gain).astype(np.float32)

    # Per-pixel gain blended by the mask's own continuous value (not a
    # hard sky/ground split) - this is what keeps the sky/ground boundary
    # from showing up as a visible seam after grading.
    per_pixel_gain = gain_ground[None, None, :] + sky_mask[:, :, None] * (gain_sky - gain_ground)[None, None, :]
    graded_linear = np.clip(linear * per_pixel_gain, 0.0, 1.0)

    graded_srgb = color.linear_to_srgb(graded_linear)
    graded_uint8 = np.clip(graded_srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    graded_image = Image.fromarray(graded_uint8, mode="RGB")

    if image.mode != "RGB":
        graded_image = graded_image.convert(image.mode)
    return graded_image
