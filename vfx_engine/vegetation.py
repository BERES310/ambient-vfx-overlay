"""
vegetation.py

Wind-driven vegetation sway - image-space displacement warping of the
photo's OWN tree/plant/foliage pixels (a single photo has no 3D mesh to
actually simulate wind-blown motion on), driven by the same curl-noise
wind field family the particle system uses, so trees sway in the same
gusts the rain/snow is already blowing through - see
the original design's Vegetation animation section and
vfx_engine/scene_warp.py for the shared GPU engine this and cloud.py's
sky drift are both built on.

Before AI scene segmentation (vfx_engine/segmentation.py) existed, "where
is the vegetation" would have had to be a color/texture heuristic
(green-dominant, high local texture frequency) - this uses the model's
real per-species classification instead, so it only wobbles pixels the
model actually thinks are foliage. tree/plant, palm, and
flower get three genuinely DIFFERENT motion models here (see
render_vegetation_sway/render_palm_sway/render_grass_sway respectively),
not one undifferentiated "vegetation" sway - a palm frond whips very
differently in wind than an oak canopy sways, and a flower head bobs on a
flexible stem more like grass than like a rigid branch.

Trees, and vegetation animation in general, is the single most obvious
remaining "this is a static photo with effects on it" tell without this -
see the original design: motion is what human vision is most
sensitive to, more than any static-detail improvement.
"""

import numpy as np
from PIL import Image, ImageFilter

from vfx_engine import camera, weather
from vfx_engine.scene_warp import render_region_warp

# How much of the shared prevailing wind direction (see
# vfx_engine/weather.py) leans into vegetation's own localized curl-noise
# gusts - deliberately small: real trees mostly sway/rustle IN PLACE from
# turbulent local gusts, they don't lean hard in one constant direction
# the way a wind-blown flag or drifting cloud layer does. Just enough that
# the canopy visibly leans slightly downwind on top of its own gusting,
# tying it to the same prevailing wind rain/fog/clouds are also blowing in.
_WIND_BIAS_FRACTION = 0.4

# A small edge feather so the warp fades in/out smoothly at the
# vegetation mask's boundary (a tree silhouette against the sky, say),
# instead of a hard, visibly-cut-out warped region.
_MASK_FEATHER = 3

# Distant/fine vegetation compensation: a real cap on how
# much amplitude a low-detail (distant/fine treeline) pixel gets boosted
# relative to a high-detail (near/sharp) one - this is a PARTIAL
# mitigation, not a full fix: a shifted
# sample of a low-contrast, repetitive-texture region will always look
# less different from an unshifted one than the same shift applied to a
# sharp, high-contrast region, no matter how much the raw displacement
# amplitude is boosted. Capped well short of "no visible difference at
# all" to avoid absurd over-amplification turning a distant treeline into
# an obviously-fake shimmering blur.
_DETAIL_COMPENSATION_MAX_BOOST = 2.2

# Caps how many screen-heights "worth" of contiguous foliage run-length
# counts as "full canopy" sway - see _canopy_sway_weight. A photo with a
# single very tall tree filling most of the frame shouldn't get a
# different sway character than one with several smaller trees; this
# keeps the weighting curve meaningful regardless of how tall any one
# vegetation mass happens to be in a given photo.
_MAX_CANOPY_RUN = 420.0

# the green/brown/yellow-green hue band (in degrees,
# PIL "HSV" convention) real foliage reads as, with a soft ramp on either
# side rather than a hard cutoff - see _foliage_hue_gate.
_HUE_GATE_LOW_DEG = 15.0
_HUE_GATE_HIGH_DEG = 175.0
_HUE_GATE_RAMP_DEG = 20.0


def _local_detail_weight(base_image, size, radius=3):
    """
    Returns a (height, width) float32 array, 0.0 (flat/low-detail, e.g. a
    distant hazy treeline) - 1.0 (sharp/high-detail, e.g. a near backlit
    leaf edge) - a cheap local-contrast proxy (a windowed standard
    deviation of luminance, via the mean/mean-of-squares identity so it
    only needs two blurs, not a real per-pixel sliding-window stdev loop).

    See _DETAIL_COMPENSATION_MAX_BOOST's comment: used to
    partially compensate low-detail regions' displacement amplitude, since
    the same raw pixel shift is far less visually legible there than on
    high-detail texture.
    """
    gray = np.asarray(base_image.convert("L").resize(size), dtype="float32") / 255.0

    def blur(array):
        image = Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype("uint8"), mode="L")
        image = image.filter(ImageFilter.GaussianBlur(radius))
        return np.asarray(image, dtype="float32") / 255.0

    mean = blur(gray)
    mean_of_squares = blur(gray * gray)
    variance = np.clip(mean_of_squares - mean * mean, 0.0, None)
    local_contrast = np.sqrt(variance)

    # Normalize by its own high percentile (not a fixed constant) so this
    # adapts to whatever contrast range THIS photo actually has, rather
    # than assuming a universal absolute scale.
    normalizer = max(1e-6, np.percentile(local_contrast, 95))
    return np.clip(local_contrast / normalizer, 0.0, 1.0)


def _foliage_hue_gate(base_image, size):
    """
    Returns a (height, width) float32 array, 0.0-1.0 - gates _build_canopy_region_rg's detail_compensation BOOST (not the
    base sway amplitude) so only pixels whose hue actually falls in the
    green/brown foliage band get the full high-contrast sway boost.
    Without this, ANY high-contrast content inside a vegetation mask - a
    chain-link fence glimpsed through branches, dappled sun/shadow speckle
    on bark, a patterned trunk - gets the same up-to-2.2x amplitude boost
    as genuine sharp foliage, since _local_detail_weight is a pure
    contrast proxy with no idea what color the detail actually is. 1.0 =
    confidently green/brown (full boost allowed), 0.0 = confidently some
    other hue (fence, sky sliver seen through branches, etc - boost fully
    suppressed).

    Hue alone is meaningless/noisy for near-grey, near-white, or
    near-black pixels (deep shadow, a blown highlight, desaturated haze) -
    a stray hue reading on a pixel that's actually just dark grey bark in
    shadow would wrongly zero out its gate. For those, this falls back to
    a saturation-based passthrough instead (low saturation = gate stays
    open) rather than trusting a meaningless hue angle.

    This is deliberately just a per-pixel color heuristic, not a deeper
    fix cross-checking segmentation's own per-pixel vegetation confidence -
    the original design explicitly defers that to when segmentation (segmentation.py
    review) is reached in the follow-up part.
    """
    hsv = np.asarray(base_image.convert("HSV").resize(size), dtype="float32")
    hue_deg = hsv[..., 0] * (360.0 / 255.0)
    saturation = hsv[..., 1] / 255.0

    # Soft ramps at both edges of the green/brown band instead of a hard
    # cutoff, so a pixel right at the boundary doesn't flip discontinuously
    # between full boost and none.
    low_ramp = np.clip((hue_deg - (_HUE_GATE_LOW_DEG - _HUE_GATE_RAMP_DEG)) / _HUE_GATE_RAMP_DEG, 0.0, 1.0)
    high_ramp = np.clip(((_HUE_GATE_HIGH_DEG + _HUE_GATE_RAMP_DEG) - hue_deg) / _HUE_GATE_RAMP_DEG, 0.0, 1.0)
    hue_gate = low_ramp * high_ramp

    # Desaturation passthrough: below roughly saturation 0.25 the hue
    # angle is increasingly just noise (a near-grey pixel's "hue" is
    # essentially arbitrary), so the gate blends back toward fully open as
    # saturation drops instead of trusting that noisy angle.
    desaturation_passthrough = np.clip(1.0 - saturation * 4.0, 0.0, 1.0)
    return np.clip(np.maximum(hue_gate, desaturation_passthrough), 0.0, 1.0)


def _canopy_sway_weight(vegetation_mask, threshold=0.4):
    """
    Returns a (height, width) float32 array, 0.0-1.0, approximating each
    vegetation pixel's height within its own local mass of foliage - near
    0 at the base of a tree/bush (trunks/roots barely move in real wind),
    ramping toward 1 higher up within the same contiguous blob (canopies
    sway the most) - see the original design's Vegetation animation
    section: "movement increases toward the top of each tree mass ...
    rather than uniform per-pixel displacement."

    Implementation: for each column, walks from the bottom of the frame
    upward, counting how many CONSECUTIVE mask-true pixels have been seen
    so far in that column (resetting to 0 the moment the mask drops out) -
    a cheap per-column "distance from this blob's own base" measure that
    needs no connected-component labeling (this project has no scipy
    dependency to do that with) and is naturally correct for multiple
    separate trees stacked in the same column, since the counter resets
    between them. Row-by-row (not fully vectorized: each row's result
    depends on the row below it) but this only runs ONCE per photo at
    setup, on a single already-small mask array, not per frame.
    """
    mask = vegetation_mask >= threshold
    height, width = mask.shape

    run = np.zeros(width, dtype="float32")
    weight = np.zeros((height, width), dtype="float32")
    for y in range(height - 1, -1, -1):
        row_mask = mask[y]
        run = np.where(row_mask, run + 1.0, 0.0)
        weight[y] = run

    weight = np.clip(weight / _MAX_CANOPY_RUN, 0.0, 1.0)
    return weight * vegetation_mask


def _build_canopy_region_rg(mask, base_image, size):
    """
    Shared canopy-height-weighted region_rg builder - used
    by BOTH render_vegetation_sway (tree/plant) and render_palm_sway
    (a palm's own silhouette has the same rough trunk-at-bottom/
    crown-at-top structure a tree's does, even though the two need very
    different render_region_warp MOTION parameters afterward - broad
    turbulent gusting vs. rapid frond flutter - see each function's own
    docstring for why they diverge past this shared shape).

    Feathers the mask edge, builds the height-graded trunk-to-canopy
    amplitude weighting (_canopy_sway_weight, smoothed to avoid column-
    independent "combing"), applies the distant/fine-detail amplitude
    compensation, and packs the result into a region_rg
    array (see scene_warp.render_region_warp's docstring for the format).
    """
    feathered = mask
    if _MASK_FEATHER > 0:
        feathered_image = Image.fromarray((mask * 255).astype("uint8"), mode="L")
        feathered_image = feathered_image.filter(ImageFilter.GaussianBlur(_MASK_FEATHER))
        feathered = np.asarray(feathered_image, dtype="float32") / 255.0

    canopy_weight = _canopy_sway_weight(feathered)

    # _canopy_sway_weight's column-by-column scan is independent PER
    # COLUMN - a real mask's edge is noisy/irregular enough that adjacent
    # columns can land on meaningfully different run-lengths (one column's
    # contiguous foliage run breaks a pixel earlier/later than its
    # neighbor's), which, left unsmoothed, shows up as visible vertical
    # "combing"/tearing once it's driving a real pixel displacement -
    # confirmed empirically (a first version without this produced exactly
    # that: jagged, torn-looking foliage edges). A real tree's canopy sways
    # as a broad, smoothly-varying mass, not a column-independent stripe
    # pattern, so blurring this back into that broad shape is a correction,
    # not just a cosmetic smoothing pass.
    weight_radius = max(6, round(min(size) * 0.01))
    weight_image = Image.fromarray((canopy_weight * 255.0).astype("uint8"), mode="L")
    weight_image = weight_image.filter(ImageFilter.GaussianBlur(weight_radius))
    canopy_weight = np.asarray(weight_image, dtype="float32") / 255.0

    # Even trunk-level pixels (canopy_weight near 0) get a small baseline
    # amplitude rather than exactly 0 - a real tree trunk isn't perfectly
    # rigid either, and a hard 0 at the base would read as a visible
    # "hinge" where motion suddenly starts partway up the tree.
    amplitude = feathered * (0.15 + 0.85 * canopy_weight)

    # Distant/fine vegetation compensation: a
    # low-detail region (a hazy distant treeline) shows far less visible
    # motion than a high-detail one (a sharp near bush) at the SAME raw
    # displacement amplitude, since a shifted sample of near-uniform
    # texture looks almost identical to an unshifted one. Boosting
    # amplitude specifically where local detail is low compensates for
    # (not eliminates - an inherent property of image-space resampling)
    # that gap.
    # the boost portion of detail_compensation is
    # additionally gated by hue - see _foliage_hue_gate's docstring. Only
    # the BOOST (the "+ ... * hue_gate" term) is gated, not the flat 1.0
    # baseline, so a non-foliage-colored high-contrast pixel still gets
    # ordinary (unboosted) amplitude, not zero.
    detail_weight = _local_detail_weight(base_image, size)
    hue_gate = _foliage_hue_gate(base_image, size)
    detail_compensation = 1.0 + (1.0 - detail_weight) * (_DETAIL_COMPENSATION_MAX_BOOST - 1.0) * hue_gate
    amplitude = amplitude * detail_compensation

    return np.stack([
        (feathered * 255.0).astype("uint8"),
        (np.clip(amplitude, 0.0, 1.0) * 255.0).astype("uint8"),
    ], axis=-1)


def render_vegetation_sway(size, num_frames, base_image, vegetation_mask,
                            wind_strength=0.5, gust_scale=1.0, gap_flicker_strength=0.18,
                            base_image_frames=None, camera_drift_params=None, start_frame=0):
    """
    Generator - yields transparent RGBA frames of the photo's own
    vegetation, resampled through a wind-driven displacement field (see
    vfx_engine.scene_warp.render_region_warp) - composite this OVER the
    base image and UNDER every particle effect (see main.py's
    render_enabled_effects), so trees visibly sway in the same wind
    rain/snow is already blowing through, with the particles still
    falling in front of them.

    vegetation_mask: (height, width) float32 array, 0.0-1.0 - see
        vfx_engine.segmentation.estimate_segmentation's "vegetation_tree"
        category.
        When this photo has no detected vegetation at all (None, or an
        all-zero/near-zero mask), yields an empty (fully transparent)
        stream instead of warping nothing.
    wind_strength: 0.0-1.0+, overall sway amplitude.
    gust_scale: >0, spatial scale of the wind field - smaller values give
        more localized, independently-timed-looking gusts across the
        canopy (different trees/branches catching the wind at slightly
        different moments); larger values sway broader areas together.
    gap_flicker_strength: 0.0-1.0 - fakes the light/gap
        flicker real foliage shows as individual leaf clusters move
        relative to each other (an APPROXIMATION - a flat image-space
        warp has no per-leaf-cluster depth to structurally reveal actual
        new "behind the leaves" content, see region_warp.glsl's matching
        comment). 0.0 disables it.
    base_image_frames: optional iterable yielding one per-frame drifted
        PIL Image (e.g. vfx_engine.camera.apply_camera_drift) - when
        given, THIS (not the static base_image) is what the swaying
        canopy actually resamples from, so it stays in
        registration with a simultaneously camera-drifting photo instead
        of always sampling the original, undrifted framing. base_image
        itself is still used for the one-time _local_detail_weight call
        below regardless (a static per-photo analysis, not a per-pixel
        texture).
    camera_drift_params: optional dict {"zoom_amplitude", "pan_amplitude",
        "pan_phase_deg"} - when given, the vegetation MASK
        itself (not just the base texture above) is also resampled
        per-frame through the same camera-drift path (see
        vfx_engine.camera.apply_camera_drift_to_region_rg), so which
        pixels are even eligible to sway stays registered with the
        drifting photo too. None (default) = the old behavior, one
        static mask reused for every frame.
    start_frame: render-checkpoint/resume support (see render_checkpoint.py) -
        skips yielding frames 0..start_frame-1. The wind field/region_rg
        setup above is a pure function of the photo/mask content, not of
        any frame index, so resuming at start_frame is safe; forwarded to
        render_region_warp (which does the actual skipping) and, when an
        animated camera-drift-synced mask is in play, to
        camera.apply_camera_drift_to_region_rg so that source's first
        yielded item also lines up with frame start_frame instead of 0.
    """
    if vegetation_mask is None or vegetation_mask.max() < 0.05:
        for _ in range(start_frame, num_frames):
            yield Image.new("RGBA", size, (0, 0, 0, 0))
        return

    region_rg = _build_canopy_region_rg(vegetation_mask, base_image, size)

    wind_direction = weather.wind_direction_vector()
    wind_bias = (wind_direction[0] * _WIND_BIAS_FRACTION, wind_direction[1] * _WIND_BIAS_FRACTION)

    region_source = region_rg
    if camera_drift_params is not None:
        region_source = camera.apply_camera_drift_to_region_rg(region_rg, num_frames, start_frame=start_frame, **camera_drift_params)

    yield from render_region_warp(
        size, num_frames, base_image_frames if base_image_frames is not None else base_image, region_source,
        drift_strength=wind_strength * 9.0,
        noise_scale=0.0035 / max(gust_scale, 1e-3),
        wind_bias=wind_bias,
        noise_loop_radius=3.0,
        flutter_strength=wind_strength * 2.2,
        flutter_scale=0.02,
        flutter_cycles=5.0,
        gap_flicker_strength=gap_flicker_strength,
        gap_flicker_scale=0.09,
        # Elastic (lagged/damped) gust response: the raw
        # gust envelope applies to drift/flutter strength INSTANTLY, the
        # same frame the noise says it should - a real branch is a spring,
        # not a value directly copied from a driving function, and keeps
        # oscillating slightly after a gust passes rather than snapping
        # back to neutral the instant the wind does.
        gust_kind="elastic",
        start_frame=start_frame,
    )


def render_palm_sway(size, num_frames, base_image, palm_mask,
                      wind_strength=0.5, gust_scale=1.0,
                      base_image_frames=None, camera_drift_params=None, start_frame=0):
    """
    Generator - like render_vegetation_sway, but for palm trees
.

    Shares the same trunk-at-bottom/crown-at-top height weighting a tree
    silhouette has (see _build_canopy_region_rg - a palm's own silhouette
    has the same rough structure, so the SHAPE of the weighting still
    applies), but with deliberately different render_region_warp MOTION
    parameters: substantially reduced broad drift_strength (fronds don't
    shift position as a whole mass the way a canopy gusting does) paired
    with much stronger, faster flutter (flutter_strength/flutter_cycles
    well above render_vegetation_sway's tree defaults) for the
    characteristic rapid frond whip. gap_flicker is left OFF entirely
    (unlike tree sway) - that effect approximates light flickering
    through dense, overlapping LEAF clusters as they shift relative to
    each other, which doesn't describe a palm's sparse, separated frond
    structure.

    base_image_frames/camera_drift_params: see render_vegetation_sway's
    matching params - identical the original design's mechanism, reused as-is.
    start_frame: render-checkpoint/resume support - see
        render_vegetation_sway's matching param for the full explanation;
        identical mechanism reused as-is.
    """
    if palm_mask is None or palm_mask.max() < 0.05:
        for _ in range(start_frame, num_frames):
            yield Image.new("RGBA", size, (0, 0, 0, 0))
        return

    region_rg = _build_canopy_region_rg(palm_mask, base_image, size)

    wind_direction = weather.wind_direction_vector()
    wind_bias = (wind_direction[0] * _WIND_BIAS_FRACTION, wind_direction[1] * _WIND_BIAS_FRACTION)

    region_source = region_rg
    if camera_drift_params is not None:
        region_source = camera.apply_camera_drift_to_region_rg(region_rg, num_frames, start_frame=start_frame, **camera_drift_params)

    yield from render_region_warp(
        size, num_frames, base_image_frames if base_image_frames is not None else base_image, region_source,
        # Deliberately LOWER than tree sway's 9.0x - a palm crown whips
        # and flexes far more than it broadly translates as one mass.
        drift_strength=wind_strength * 4.0,
        noise_scale=0.0035 / max(gust_scale, 1e-3),
        wind_bias=wind_bias,
        noise_loop_radius=3.0,
        # Deliberately HIGHER and FINER than tree sway's 2.2x/5.0 cycles -
        # the fast, fine frond-whip is the dominant, characteristic palm
        # motion, the way the traveling wave is grass's.
        flutter_strength=wind_strength * 5.0,
        flutter_scale=0.035,
        flutter_cycles=9.0,
        gap_flicker_strength=0.0,
        gap_flicker_scale=0.09,
        # Fronds are, if anything, an even springier structure than a
        # rigid branch - same elastic (lagged/damped) gust response as
        # tree sway.
        gust_kind="elastic",
        start_frame=start_frame,
    )


def render_grass_sway(size, num_frames, base_image, grass_mask,
                       wind_strength=0.5, gust_scale=1.0, base_image_frames=None,
                       camera_drift_params=None, depth_array=None, start_frame=0):
    """
    Generator - like render_vegetation_sway, but for grass/field terrain
    (segmentation's "ground_soft" category - see main.py's
    render_vegetation_effects) instead of tree/plant canopy.

    Fixes the confirmed gap in grass has far LOWER stiffness
    than a tree canopy and should sway MORE readily to the same gust, not
    less. Investigating the actual current code turned up something more
    fundamental than the original design's own framing, worth flagging: grass was
    never routed through _canopy_sway_weight's tree-tuned model AT ALL -
    segmentation.py classifies "grass"/"field" under its own "ground_soft"
    category, entirely separate from vegetation, and main.py's
    render_vegetation_effects only ever read the (then-single, now split
    three ways - the original design) vegetation mask. So grass previously
    received NO sway whatsoever, not merely an under-weighted one. The
    fix the original design actually asked for - "give grass its own low-
    stiffness, higher-amplitude, traveling-wave motion instead of routing
    it through the tree-tuned canopy-height weighting" - is still exactly
    the right one; it just needed grass connected to a sway effect in the
    first place, not just reweighted within the existing one.

    Uses region_warp.glsl's new wave_strength/wave_direction, a genuinely different motion type from tree sway's in-place
    curl-noise turbulence - a coherent ripple crossing the field in the
    wind direction, the classic "wind visibly crossing a lawn" look -
    plus a uniform (not height-graded) high amplitude across the whole
    mask, since real grass has no rigid trunk section the way a tree does.

    Also reused directly for segmentation's "vegetation_flower" category
: "a flower head
    bobs on a flexible stem more like a blade of grass than like a rigid
    branch" is the original design's own framing, and this function's motion
    model - uniform low-stiffness amplitude, no trunk-to-canopy height
    grading, a traveling wave rather than tree sway's in-place turbulence -
    already IS exactly that description, with nothing about its
    implementation actually specific to grass/field content over flower
    content. Reusing it directly avoids maintaining two copies of
    identical motion logic under different names.

    base_image_frames: optional iterable yielding one per-frame drifted
        PIL Image - see render_vegetation_sway's matching
        param for the full explanation. When given, used instead of the
        static base_image for the actual per-pixel texture sampling.
    camera_drift_params: optional dict - see
        render_vegetation_sway's matching param for the full explanation.
    depth_array: optional numpy float32 array (height, width), 0.0 (far)
        to 1.0 (near), from vfx_engine.depth.estimate_depth() - a field of grass often spans a real range of camera
        distance within one mask (a lawn stretching from the foreground to
        a distant treeline), so without this the traveling wave's spacing/
        speed is identical everywhere in the mask regardless of how far
        away each blade actually is. When given, passed straight through
        to render_region_warp's own depth_array param (see that function
        and shaders/region_warp.glsl's wave block) so far grass gets a
        tighter, slower-reading wave and near grass a wider, faster one.
        None (default) = the old flat, depth-independent wave.
    start_frame: render-checkpoint/resume support - see
        render_vegetation_sway's matching param for the full explanation;
        identical mechanism reused as-is.
    """
    if grass_mask is None or grass_mask.max() < 0.05:
        for _ in range(start_frame, num_frames):
            yield Image.new("RGBA", size, (0, 0, 0, 0))
        return

    feathered = grass_mask
    if _MASK_FEATHER > 0:
        feathered_image = Image.fromarray((grass_mask * 255).astype("uint8"), mode="L")
        feathered_image = feathered_image.filter(ImageFilter.GaussianBlur(_MASK_FEATHER))
        feathered = np.asarray(feathered_image, dtype="float32") / 255.0

    # Uniform, high amplitude (no canopy-height gradient) - every blade in
    # a grass mat is similarly low-stiffness, unlike a tree's rigid trunk.
    amplitude = feathered * 0.95

    region_rg = np.stack([
        (feathered * 255.0).astype("uint8"),
        (np.clip(amplitude, 0.0, 1.0) * 255.0).astype("uint8"),
    ], axis=-1)

    wind_direction = weather.wind_direction_vector()

    region_source = region_rg
    if camera_drift_params is not None:
        region_source = camera.apply_camera_drift_to_region_rg(region_rg, num_frames, start_frame=start_frame, **camera_drift_params)

    yield from render_region_warp(
        size, num_frames, base_image_frames if base_image_frames is not None else base_image, region_source,
        # A small turbulent component on top of the wave - real grass
        # isn't a perfectly clean sine wave either - but the traveling
        # wave below is the dominant, characteristic motion, unlike tree
        # sway where the curl-noise turbulence IS the whole motion.
        drift_strength=wind_strength * 3.0,
        noise_scale=0.004 / max(gust_scale, 1e-3),
        noise_loop_radius=3.0,
        flutter_strength=wind_strength * 1.2,
        flutter_scale=0.05,
        flutter_cycles=7.0,
        wave_strength=wind_strength * 7.0,
        wave_direction=wind_direction,
        wave_frequency=0.015,
        wave_cycles=4.0,
        # Same elastic gust response as tree sway - grass
        # blades are also a spring, not a value directly copied from the
        # driving gust signal.
        gust_kind="elastic",
        depth_array=depth_array,
        start_frame=start_frame,
    )
