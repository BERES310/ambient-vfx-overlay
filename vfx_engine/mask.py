"""
mask.py

Handles masks - i.e. which AREA of the photo a given effect should be
applied to, when it's not meant to be the whole image (e.g. only the sky,
only a window, only the area around a lamp).

A mask is simply a grayscale image the same size as the photo: white (1.0)
= effect fully visible here, black (0.0) = effect completely invisible
here, shades in between = partially visible (a soft, blurred transition at
the edge).

How the mask is built is chosen in config.py via MASK_MODE:
    "none"      - no mask, effect on the whole image (build_mask returns None)
    "file"      - mask loaded from a file in the masks/ folder
    "threshold" - mask computed automatically from the photo's pixel brightness
    "manual"    - mask hand-drawn with the mouse in a separate window (see
                  mask_editor.py); once drawn, the mask is saved to masks/,
                  so the next time the program runs for the same photo the
                  window won't open again
    "depth"     - mask computed automatically from the photo's estimated
                  scene depth (see vfx_engine/depth.py): the effect is
                  hidden behind whatever's in the near foreground (a tree,
                  a railing, a person) instead of painted flatly over it

MASK_MODE picks exactly one PRIMARY source above, but two
further, independently-optional SECONDARY layers can restrict it further
at the same time - config.MASK_SEGMENTATION_CATEGORY (restrict to one AI-
detected region, e.g. "sky"/"water") and config.MASK_MANUAL_CORRECTION
(restrict by a separate hand-drawn correction layer) - see
_build_secondary_mask. Either, both, or neither can be active regardless
of which primary MASK_MODE is chosen, e.g. MASK_MODE="depth" +
MASK_SEGMENTATION_CATEGORY="sky" gives "depth occlusion, but only within
the detected sky region."
"""

import numpy as np
from PIL import Image, ImageChops, ImageFilter

import config
from vfx_engine.mask_editor import open_mask_editor

# "a systemic pattern across this whole codebase: hard
# cutoff first, blur second - instead of a naturally soft edge." Both
# _build_threshold_mask (brightness) and _build_depth_mask/
# _build_depth_mask_frame_stream (depth) used to make a hard binary
# per-pixel decision FIRST (np.where(value >= threshold, 255, 0)), then
# soften the already-discarded-to-a-single-bit result with a single
# fixed-radius Gaussian blur (config.MASK_FEATHER) applied AFTER. Both
# underlying signals (brightness, depth) are genuinely continuous - a
# pixel just barely past the threshold and one far past it were treated
# identically (both "fully in"), throwing away exactly the information a
# real soft edge needs. Same fix direction already used for segmentation
# in Part 9 : derive the softness from the continuous signal
# itself, via a smoothstep transition band, BEFORE any decision is
# hardened - not a blur bolted on afterward.
#
# "Sized to the LOCAL value distribution" (the write-up's own framing,
# not just a fixed global blur radius) - see _local_value_scale: a region
# with a sharp local gradient (a real content edge - a lamp's silhouette
# against a dark room, a foreground railing's crisp depth discontinuity)
# gets a NARROW transition band, since the real content itself changes
# quickly there; a smooth/flat region (a softly overcast sky, a gently
# receding lawn) gets a WIDER one, since the real underlying signal
# itself varies gradually there too - a single fixed blur radius is
# correct for neither case at once, which is exactly this item's
# complaint about MASK_FEATHER's "one radius for a lamp and for the
# whole sky" behavior.


def _smoothstep(t):
    """Classic smoothstep (3t^2 - 2t^3), clamped to the [0, 1] domain first - a smooth 0->1 ease, zero-and-flat at both ends, used to turn a linear 0-1 ramp into a naturally-graduated transition instead of a straight line."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _local_value_scale(values, radius, band_min, band_max, band_multiplier=1.25):
    """
    Returns a (height, width) float32 array: each pixel's local
    transition BAND HALF-WIDTH, in the same units as `values` - derived
    from the local windowed standard deviation of `values` (the same
    mean/mean-of-squares-via-two-blurs technique
    vegetation._local_detail_weight and depth.py's confidence estimate
    already use elsewhere in this codebase - two Gaussian blurs, not a
    real per-pixel sliding-window loop).

    band_multiplier scales local stdev into a band half-width (a pixel
    whose local neighborhood varies a lot gets more transition room);
    band_min/band_max clamp the result - band_min prevents a perfectly
    flat local neighborhood from collapsing to an infinitely sharp
    (zero-width) transition, band_max keeps an unusually noisy region
    from blowing the transition out to something absurdly wide.
    """
    values = values.astype("float32")
    mean = _box_blur(values, radius)
    mean_of_squares = _box_blur(values * values, radius)
    variance = np.clip(mean_of_squares - mean * mean, 0.0, None)
    local_stdev = np.sqrt(variance)
    return np.clip(local_stdev * band_multiplier, band_min, band_max)


def _box_blur(array, radius):
    """A cheap, edge-safe box-ish blur via PIL's GaussianBlur, used only inside _local_value_scale's variance estimate - not exposed further."""
    # Full float32 precision (three separable box passes ~ Gaussian, the same
    # approximation PIL uses). The old 8-bit-quantized PIL blur made
    # mean-of-squares - mean^2 noisy, inflating the local band in bright
    # flat regions (e.g. band ~19 instead of the 3 minimum on a value-250 area).
    array = np.asarray(array, dtype="float64")
    sigma = max(float(radius), 1e-6)
    ideal = np.sqrt(12.0 * sigma * sigma / 3.0 + 1.0)
    width = int(np.floor(ideal))
    if width % 2 == 0:
        width -= 1
    width = max(width, 1)
    half = width // 2
    result = array
    for axis in (0, 1):
        for _ in range(3):
            padded = np.pad(
                result, [(half, half) if a == axis else (0, 0) for a in (0, 1)], mode="edge"
            )
            csum = np.cumsum(padded, axis=axis, dtype="float64")
            zero_shape = list(csum.shape)
            zero_shape[axis] = 1
            csum = np.concatenate([np.zeros(zero_shape), csum], axis=axis)
            length = result.shape[axis]
            upper = np.take(csum, np.arange(width, width + length), axis=axis)
            lower = np.take(csum, np.arange(0, length), axis=axis)
            result = (upper - lower) / width
    return result.astype("float32")


def _soft_threshold_mask(values, threshold, invert, local_radius, band_min, band_max):
    """
    Returns a (height, width) float32 0.0-1.0 soft mask array directly
    from a continuous `values` field and a scalar `threshold` - the original design's replacement for "hard np.where cutoff, then blur afterward".
    Builds a per-pixel smoothstep transition band around the threshold,
    sized per-pixel by _local_value_scale (see above), instead of a
    single hard yes/no decision softened by one fixed-radius blur applied
    to the whole image regardless of local content.

    invert: False = "at or above threshold" counts as in (1.0) - matches
        THRESHOLD_MODE="bright"/the old `>= threshold` depth-mask
        direction (near/foreground = excluded=0, see _build_depth_mask,
        which passes invert=True since ITS "in" direction is the
        opposite: far/background stays visible).
    """
    band = _local_value_scale(values, local_radius, band_min, band_max)
    t = (values - (threshold - band)) / np.clip(2.0 * band, 1e-6, None)
    result = _smoothstep(t)
    return (1.0 - result) if invert else result


def build_mask(image, image_path, depth_array=None, depth_frames=None, segmentation=None):
    """
    Builds a mask for the given photo, according to MASK_MODE in config.py
    (the PRIMARY source), optionally further restricted by one or both of
    MASK_SEGMENTATION_CATEGORY/MASK_MANUAL_CORRECTION. Returns a
    numpy array (height x width) with values 0.0-1.0, or None when
    MASK_MODE = "none" AND neither secondary layer is configured (i.e.
    truly "whole image, no masking at all") - EXCEPT when depth_frames is
    given (see below) and/or a secondary layer IS active, in which case
    this returns a GENERATOR of such arrays, one per frame, instead of a
    single one.

    depth_array: optional, only used when MASK_MODE = "depth" - a
        precomputed depth map (from vfx_engine.depth.estimate_depth) to
        reuse instead of running depth estimation again. Callers that also
        want the raw depth map for particle realism (see particles.py's
        depth_array param) should compute it once and pass it both here
        and to render_snow/render_rain, instead of estimating it twice.
    depth_frames: optional, only used when MASK_MODE = "depth" - an
        iterable of PER-FRAME depth arrays (see
        vfx_engine.camera.apply_camera_drift_to_depth), one already
        resampled through that frame's own camera-drift zoom/pan crop.
        When given, this function returns a GENERATOR yielding one mask
        array per frame instead of a single static array, so the
        occlusion mask (foreground objects hiding rain/snow behind them)
        tracks the same drift the base photo goes through underneath it,
        instead of staying locked to frame 0's original framing - see
        main.py's resolve_mask_frames and camera.py's docstring. Every
        other MASK_MODE ignores this parameter (their masks have no
        per-frame camera-drift dependency to begin with).
    segmentation: optional dict from vfx_engine.segmentation.estimate_segmentation
        - only needed when config.MASK_SEGMENTATION_CATEGORY is set (see
        _build_secondary_mask); ignored otherwise.
    """
    mode = config.MASK_MODE

    if mode == "none":
        primary = None
    elif mode == "depth" and depth_frames is not None:
        primary = _build_depth_mask_frame_stream(depth_frames)
    elif mode == "threshold":
        # already a soft, continuous 0.0-1.0 numpy array
        # (see _soft_threshold_mask) - no further MASK_FEATHER blur
        # needed or wanted here, the same way segmentation's
        # masks intrinsically soft rather than blurred after the fact.
        primary = _build_threshold_mask(image)
    elif mode == "depth":
        primary = _build_depth_mask(image, depth_array)
    elif mode == "file":
        mask_image = _load_mask_from_file(image_path, image.size)
        primary = _feather_hard_mask(mask_image)
    elif mode == "manual":
        mask_image = _get_manual_mask(image, image_path)
        primary = _feather_hard_mask(mask_image)
    else:
        raise ValueError(f"Unknown MASK_MODE in config.py: '{mode}'")

    secondary = _build_secondary_mask(image, image_path, segmentation)
    return _combine_masks(primary, secondary)


def _feather_hard_mask(mask_image):
    """
    "file"/"manual" masks are genuinely hard-edged with no continuous
    underlying signal to derive a smoothstep transition from (a hand-
    drawn brush stroke or a loaded file mask isn't a sampled brightness/
    depth field) - MASK_FEATHER's post-hoc blur is still the right tool
    for these two, unlike threshold/depth. Returns a
    float32 0.0-1.0 numpy array.
    """
    if config.MASK_FEATHER > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(config.MASK_FEATHER))
    return np.asarray(mask_image, dtype="float32") / 255.0


def _build_secondary_mask(image, image_path, segmentation):
    """
    "MASK_MODE is a single exclusive switch - masking
    sources can't be combined." There was no way to combine, e.g., depth
    occlusion AND "only within the segmentation-detected sky region," or
    depth occlusion WITH a hand-painted correction layered on top, even
    though ground_fx.build_ground_mask already does exactly this kind of
    multiplicative mask-gates-mask combination internally. Exposes that
    same capability at the general MASK_MODE level via two independent,
    OPTIONAL layers (see config.py's MASK_SEGMENTATION_CATEGORY/
    MASK_MANUAL_CORRECTION) that further restrict whatever the primary
    MASK_MODE already produced - either, both, or neither can be active.

    Returns None (no secondary restriction at all) when neither is
    configured, a single static (height, width) float32 array otherwise -
    a segmentation category and a hand-drawn correction are both static
    per-photo layers, not per-frame animated ones, so this is always a
    static array/None even when the PRIMARY mask (from build_mask) turns
    out to be a per-frame generator.
    """
    layers = []

    if config.MASK_SEGMENTATION_CATEGORY is not None:
        if segmentation is None:
            raise ValueError(
                "config.MASK_SEGMENTATION_CATEGORY is set but no segmentation data was "
                "available to build_mask() - is config.SEGMENTATION_ENABLED on?"
            )
        category_mask = segmentation.get(config.MASK_SEGMENTATION_CATEGORY)
        if category_mask is None:
            raise ValueError(
                f"config.MASK_SEGMENTATION_CATEGORY = '{config.MASK_SEGMENTATION_CATEGORY}' is not "
                f"a real segmentation category - see vfx_engine/segmentation.py's _CATEGORY_LABELS "
                f"for the full list (e.g. 'sky', 'water', 'vegetation_tree', 'ground_hard')."
            )
        layers.append(category_mask)

    if config.MASK_MANUAL_CORRECTION:
        correction_image = _get_manual_correction_mask(image, image_path)
        layers.append(_feather_hard_mask(correction_image))

    if not layers:
        return None

    combined = layers[0]
    for layer in layers[1:]:
        combined = combined * layer
    return combined


def _combine_masks(primary, secondary):
    """
    Multiplies build_mask's primary source by its optional secondary
    layer (see _build_secondary_mask) - handles every combination of
    "primary is None/a static array/a per-frame generator" x "secondary
    is None/a static array" that build_mask's own docstring promises.
    """
    if secondary is None:
        return primary
    if primary is None:
        return secondary
    if isinstance(primary, np.ndarray):
        return primary * secondary
    # primary is a per-frame generator (MASK_MODE = "depth" with
    # depth_frames) - the secondary layer is always static, so it's
    # reapplied identically to every yielded frame.
    return (frame * secondary for frame in primary)


def apply_mask_to_frames(frames, mask):
    """
    Generator - limits an effect's visibility to the mask's area by
    multiplying each effect frame's alpha (transparency) channel by the
    mask (outside the mask the effect becomes invisible), yielding frames
    one at a time.

    mask: either a single static numpy array (0.0-1.0, applied unchanged
        to every frame - the common case), or an iterable/generator
        yielding one such array PER FRAME (see build_mask's depth_frames
        param) - camera-drift-aware depth occlusion masking uses the
        latter, everything else the former. None = no masking at all.
    """
    if mask is None:
        yield from frames
        return

    if isinstance(mask, np.ndarray):
        mask_image = Image.fromarray((mask * 255).astype("uint8"), mode="L")
        for frame in frames:
            red, green, blue, alpha = frame.split()
            masked_alpha = ImageChops.multiply(alpha, mask_image)
            yield Image.merge("RGBA", (red, green, blue, masked_alpha))
        return

    # A per-frame mask stream (see build_mask's depth_frames param) - each
    # frame gets its OWN mask array instead of one shared/reused image.
    for frame, mask_array in zip(frames, mask):
        mask_image = Image.fromarray((mask_array * 255).astype("uint8"), mode="L")
        red, green, blue, alpha = frame.split()
        masked_alpha = ImageChops.multiply(alpha, mask_image)
        yield Image.merge("RGBA", (red, green, blue, masked_alpha))


def _load_mask_from_file(image_path, size):
    """Loads a mask from the masks/ folder - the file must have the same name as the photo (e.g. winter1.jpg -> masks/winter1.png)."""
    mask_path = config.MASKS_DIR / f"{image_path.stem}.png"
    if not mask_path.exists():
        print(f"  [mask] No file {mask_path.name} in the masks/ folder - the effect will be applied to the whole image.")
        return Image.new("L", size, 255)
    return Image.open(mask_path).convert("L").resize(size)


# Local-band tuning for the brightness threshold mask -
# grayscale values are 0-255, so these are in that same domain: a flat
# local neighborhood still gets at least a 3-value-wide transition (never
# a razor-sharp single-pixel step), an unusually noisy/high-contrast
# neighborhood is capped at a 45-value-wide band (still meaningfully
# narrower than MASK_FEATHER's old one-size-fits-all 15px BLUR RADIUS
# applied in pixel-space, not value-space, to every region regardless of
# local content).
_THRESHOLD_LOCAL_RADIUS = 8
_THRESHOLD_BAND_MIN = 3.0
_THRESHOLD_BAND_MAX = 45.0

# Same idea for the depth mask - depth_array is normalized to 0.0-1.0
# (see depth.py's estimate_depth), so these bounds are in that domain.
_DEPTH_LOCAL_RADIUS = 8
_DEPTH_BAND_MIN = 0.01
_DEPTH_BAND_MAX = 0.15

# _depth_prominence's broad baseline-blur radius, as a fraction of the
# image's larger dimension - see that function's docstring for why raw
# depth alone can't tell "near because it's a real foreground object" from
# "near because it's the ground plane receding toward the bottom of the
# frame under ordinary perspective." Wide enough to average away that
# perspective gradient (and a large gently-sloped surface like a lawn or a
# lake) but still narrow enough that a real foreground object - a tree
# trunk, a person, a railing, a rooftop edge - stands out sharply against
# it rather than getting smoothed away itself. Tuned empirically against
# both project photos (a dense forest-canopy aerial view with no near
# object at all, and a lake/castle scene with real near stonework) - see
# the original design for the verification renders.
_DEPTH_PROMINENCE_RADIUS_FRACTION = 0.08


def _build_threshold_mask(image):
    """
    Builds a mask automatically from the photo's pixel brightness (e.g.
    "only bright sky") - returns an already-soft float32
    0.0-1.0 numpy array directly (see _soft_threshold_mask), not a hard
    binary PIL mask for build_mask to blur afterward.
    """
    grayscale = np.asarray(image.convert("L"), dtype="float32")

    if config.THRESHOLD_MODE == "bright":
        invert = False
    elif config.THRESHOLD_MODE == "dark":
        invert = True
    else:
        raise ValueError(f"Unknown THRESHOLD_MODE in config.py: '{config.THRESHOLD_MODE}'")

    return _soft_threshold_mask(
        grayscale, config.THRESHOLD_VALUE, invert,
        _THRESHOLD_LOCAL_RADIUS, _THRESHOLD_BAND_MIN, _THRESHOLD_BAND_MAX,
    )


def _depth_prominence(depth_array):
    """
    Turns a raw depth field into a PROMINENCE field: how much nearer each
    pixel is than its own broad local surroundings (see
    _DEPTH_PROMINENCE_RADIUS_FRACTION's comment for the full reasoning).

    Raw depth alone can't distinguish a real foreground occluder from
    ordinary camera perspective: in almost any photo, the bottom rows are
    simply nearer than the top rows because the ground/floor/water plane
    recedes into the distance as it rises toward the horizon - that's true
    whether or not there's an actual solid object anywhere near the
    camera. Thresholding on raw "nearness" (the old behavior) therefore
    always caught most of the lower frame regardless of content, hiding
    rain/snow across the bottom of the image and making it fade out right
    where it should be visibly reaching the ground, instead of only
    hiding it behind genuine near objects (a tree, a railing, a person).

    Subtracting a wide local blur of the depth field cancels out that
    smooth perspective gradient (and any other large, gently-sloped
    surface) while leaving a real discrete near object's sharp depth
    discontinuity against its background intact - a tree trunk or a
    person still stands out as strongly "nearer than its surroundings"
    after this subtraction; the ground plane, having no such local
    discontinuity, mostly cancels down to near zero even where its raw
    depth value is high.
    """
    radius = max(depth_array.shape) * _DEPTH_PROMINENCE_RADIUS_FRACTION
    baseline = _box_blur(depth_array, radius)
    return depth_array - baseline


def _build_depth_mask(image, depth_array=None):
    """
    Builds a mask from the photo's estimated scene depth (see
    vfx_engine/depth.py): pixels at or above config.DEPTH_NEAR_PERCENTILE
    of estimated PROMINENCE (see _depth_prominence - not raw "nearness",
    see that function's docstring for why) are treated as foreground
    occluders (mask near 0.0, effect hidden), everything else stays
    visible (mask near 1.0) - still a soft, per-pixel-local-
    band smoothstep transition around that threshold (see
    _soft_threshold_mask), not a hard np.where cutoff blurred afterward by
    build_mask. Imports vfx_engine.depth lazily - it pulls in
    torch/transformers, which is only worth loading when depth masking is
    actually used.

    depth_array: reuses a precomputed depth map when given, instead of
        running depth estimation again - see build_mask's docstring.
    """
    if depth_array is None:
        from vfx_engine import depth
        depth_array = depth.estimate_depth(image)

    prominence = _depth_prominence(depth_array)
    threshold = np.percentile(prominence, config.DEPTH_NEAR_PERCENTILE)
    return _soft_threshold_mask(
        prominence, threshold, True,
        _DEPTH_LOCAL_RADIUS, _DEPTH_BAND_MIN, _DEPTH_BAND_MAX,
    )


def _build_depth_mask_frame_stream(depth_frames):
    """
    Generator version of _build_depth_mask - see build_mask's depth_frames
    param. Yields one 0.0-1.0 numpy mask array per frame, built from that
    frame's own already camera-drift-resampled depth array.

    The near/far THRESHOLD itself is computed once, from the first frame's
    PROMINENCE distribution (frame 0's, i.e. the original undrifted
    framing's - see _depth_prominence for what prominence means and why
    _build_depth_mask uses it instead of raw depth) and reused for every
    subsequent frame, rather than recomputed per frame from each frame's
    own (drifted) prominence distribution - what counts as "the near
    foreground" is a property of the SCENE, not of which fraction of it
    happens to be in frame at a given moment, so a fixed threshold is both
    more correct and avoids the mask's occlusion cutoff subtly jittering
    frame to frame as the camera pans - only the mask's ALIGNMENT should
    track the drift, not its criteria.
    """
    threshold = None
    for depth_frame in depth_frames:
        prominence = _depth_prominence(depth_frame)
        if threshold is None:
            threshold = np.percentile(prominence, config.DEPTH_NEAR_PERCENTILE)

        # the same soft, locally-sized smoothstep transition
        # _build_depth_mask uses for the static case - the fixed threshold
        # from frame 0 is reused every frame (see this function's own
        # docstring above for why), but the LOCAL band itself is
        # recomputed fresh from each frame's own (camera-drift-resampled)
        # prominence field, since which pixels sit near a sharp depth edge
        # shifts as the "camera" pans/zooms.
        yield _soft_threshold_mask(
            prominence, threshold, True,
            _DEPTH_LOCAL_RADIUS, _DEPTH_BAND_MIN, _DEPTH_BAND_MAX,
        )


def _load_or_draw_mask(image, image_path, suffix, label):
    """
    Shared mechanism behind _get_manual_mask (MASK_MODE = "manual") and
    _get_manual_correction_mask - loads a previously hand-drawn mask from
    masks/{stem}{suffix}.png if one exists, otherwise opens the drawing
    window (mask_editor.py) and saves the result there so it doesn't have
    to be drawn again next time. `suffix` is what keeps the two use cases
    from colliding on the same file: "" for the primary manual mask,
    "_correction" for the secondary correction layer - a photo can have
    BOTH, independently drawn, at once.
    """
    mask_path = config.MASKS_DIR / f"{image_path.stem}{suffix}.png"
    if mask_path.exists():
        print(f"  [mask] Using previously drawn {label}: {mask_path.name} (delete this file to draw it again).")
        return Image.open(mask_path).convert("L").resize(image.size)

    print(f"  [mask] Opening the {label}-drawing window - left click = draw, right click = eraser, Enter = save, Esc = cancel.")
    drawn_mask = open_mask_editor(image)

    if drawn_mask is None:
        print(f"  [mask] {label.capitalize()} drawing cancelled - treating it as fully permissive (no restriction from this layer).")
        return Image.new("L", image.size, 255)

    config.MASKS_DIR.mkdir(parents=True, exist_ok=True)
    drawn_mask.save(mask_path)
    print(f"  [mask] Saved {label}: {mask_path.name}")
    return drawn_mask


def _get_manual_mask(image, image_path):
    """Returns a hand-drawn mask for the given photo (MASK_MODE = "manual") - see _load_or_draw_mask."""
    return _load_or_draw_mask(image, image_path, suffix="", label="mask")


def _get_manual_correction_mask(image, image_path):
    """
    Returns a SEPARATE hand-drawn correction layer for the given photo
 - saved under
    masks/{stem}_correction.png, a different file than _get_manual_mask's
    own masks/{stem}.png, so the two stay independent even when both are
    active at once (e.g. MASK_MODE = "depth" as the primary occlusion
    source, plus a hand-drawn correction fixing one spot the automatic
    depth mask got wrong). See _load_or_draw_mask.
    """
    return _load_or_draw_mask(image, image_path, suffix="_correction", label="correction mask")
