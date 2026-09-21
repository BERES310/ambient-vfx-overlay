"""
camera.py

Adds a slow, cyclic camera drift/zoom (Ken Burns effect) to the base
photo before compositing effects on top - the design notes'
"Camera motion" section. This is likely the single highest-impact "does
this feel alive" upgrade outside of the particles themselves: a live
wallpaper with dense rain/snow falling over a perfectly static, pixel-
locked photo still reads as "particles pasted on a photo," not a filmed
scene. A few percent of slow zoom/pan changes that entirely.

Built to close the loop exactly, the same way particles.py's own motion
does: the pan/zoom path is a closed ellipse in zoom/pan-space, parameterized
purely by loop phase (theta = 2*pi*cycles*frame_i/num_frames, cycles a whole
number - see _resolve_drift_cycles) using sin/cos terms - so frame
`num_frames` reproduces frame 0's exact camera framing bit for bit, with no
crossfade needed to hide a seam (see loop.py and the design notes' "Loop
generation" section).

Vegetation sway, cloud drift, and particle (rain/snow) atmosphere/
backlight/refraction sampling can ALSO be kept in sync with this drift via apply_camera_drift's own per-frame generator passed
through as their `base_image_frames` param (see main.py's
_resolve_scene_warp_drift_kwargs, vfx_engine/scene_warp.py,
vfx_engine/particles.py) - each re-uploads a fresh GPU texture per frame
instead of one static upload. This is opt-in per caller, not automatic:
main.py wires it up whenever config.CAMERA_DRIFT_ENABLED is on; a caller
that doesn't pass `base_image_frames` still samples the static original,
which remains correct/harmless at this module's default subtle drift
amplitudes (single-digit percent) even when unsynced.

The scene DEPTH MAP (see vfx_engine/depth.py) and segmentation-derived
region masks are similarly kept in sync via
apply_camera_drift_to_depth()/apply_camera_drift_to_region_rg() below -
see those functions and apply_camera_drift_to_array's docstring. All of
these walk the exact same per-frame zoom/pan path (_camera_path() below),
so anything wired up this way tracks where the "camera" has actually
drifted to at that instant, instead of staying locked to frame 0's
original framing while the photo underneath it pans/zooms away from it.
"""

import math

import numpy as np
from PIL import Image

import config


_PAN_BIAS_SAFETY_FACTOR = 1.4142135623730951  # sqrt(2) - see _effective_pan_amplitude's docstring


def _resolve_drift_cycles(num_frames):
    """
    config.CAMERA_DRIFT_SECONDS_PER_CYCLE is a RATE (seconds
    per zoom/pan cycle), not a fixed "exactly one cycle over the entire
    render" - converts it into a whole cycle count (whole numbers only, so
    _camera_path's sin/cos terms still close the loop exactly) sized to
    THIS render's actual duration (num_frames/config.FPS), so a long
    render gets proportionally more, naturally-paced drift cycles instead
    of one lone zoom/pan stretched across the whole thing. max(1, ...)
    keeps a very short render from collapsing to 0 whole cycles (which
    would flatten the drift to a static frame).
    """
    duration_seconds = max(num_frames, 1) / config.FPS
    return max(1, round(duration_seconds / config.CAMERA_DRIFT_SECONDS_PER_CYCLE))


def _effective_pan_amplitude(pan_amplitude, pan_bias_angle_deg):
    """
    rotating the pan ellipse toward a composition-aware bias
    angle (see _camera_path) can push an individual axis's excursion
    ABOVE pan_amplitude - rotating (ex, ey) = (sin(theta), sin(theta+phase))
    by angle b gives rotated_x = ex*cos(b) - ey*sin(b), whose magnitude is
    bounded (triangle inequality, true for ANY phase/theta) by
    |cos(b)| + |sin(b)| <= sqrt(2), not by 1.0 the way the unrotated axes
    are. Every caller that sizes its working canvas/padding from
    pan_amplitude (_working_size, _pad_amount) needs this inflated value
    instead whenever pan_bias_angle_deg is actually in use, or the rotated
    pan could exceed the margin built for it and overrun the crop box.
    Returns pan_amplitude UNCHANGED when pan_bias_angle_deg is None, so
    every existing caller that doesn't use this new feature gets an
    IDENTICAL padding size to before - no behavior change unless the new
    feature is actually turned on.
    """
    if pan_bias_angle_deg is None:
        return pan_amplitude
    return pan_amplitude * _PAN_BIAS_SAFETY_FACTOR


def _camera_path(num_frames, zoom_amplitude, pan_amplitude, pan_phase_deg, zoom_pattern="breathe", pan_bias_angle_deg=None):
    """
    Yields (zoom, pan_x_frac, pan_y_frac) for every frame_i in
    0..num_frames-1 - the shared Ken Burns path math used by BOTH
    apply_camera_drift (the photo itself) and apply_camera_drift_to_depth
    (the scene depth map), so the two stay pixel-for-pixel in sync every
    single frame instead of risking two independently-written copies of
    the same formula silently drifting apart after a future edit to only
    one of them. pan_x_frac/pan_y_frac are fractions of the image's
    width/height respectively - the caller multiplies by its own
    width/height to get pixel offsets.

    zoom_pattern: selects the zoom envelope's SHAPE, all still closed
    loops (identical value at theta=0 and 2*pi) built purely from theta,
    so loop-closure is preserved automatically regardless of which is
    picked:
      "breathe" (default, unchanged from before this option existed):
          1 -> 1+za -> 1, eased with a raised cosine.
      "breathe_out": the same raised-cosine shape, phase-inverted - starts
          and ends at the PEAK zoom (1+za) and eases OUT to the baseline
          (1.0) at the loop's midpoint instead - "zoom out and back" vs.
          "breathe"'s "zoom in and back".
      "ease_hold": the same 1 -> 1+za -> 1 envelope, but reshaped through
          a smoothstep curve (3x^2-2x^3) instead of a plain raised cosine -
          more dwell time near the loop's rest/peak framing (flatter
          derivative there) and a snappier sweep through the middle,
          instead of "breathe"'s constant-feeling sinusoidal motion
          throughout.
    pan_bias_angle_deg: when given (degrees, image pixel convention -
    0 = due right, 90 = straight down, matching how pan_x/pan_y are
    added directly to center_x/center_y in _crop_box), ROTATES the base
    pan ellipse's (sin(theta), sin(theta+phase)) point by this angle
    before scaling by pan_amplitude - the ellipse's SHAPE/dynamics (and
    loop-closure guarantee) are unchanged, only its orientation is, so
    the camera's dominant drift direction can be biased toward actual
    photo content (e.g. main.py biases it toward the photo's own
    detected light position - see resolve_camera_composition_kwargs)
    instead of always tracing the same content-agnostic diagonal
    regardless of what's in the frame. None (default) = no rotation,
    identical to this function's behavior before this option existed.
    """
    pan_phase = math.radians(pan_phase_deg)
    bias_angle = None if pan_bias_angle_deg is None else math.radians(pan_bias_angle_deg)
    drift_cycles = _resolve_drift_cycles(num_frames)

    for frame_i in range(num_frames):
        theta = 2.0 * math.pi * drift_cycles * (frame_i / max(num_frames, 1))

        # (1 - cos(theta)) / 2 ramps 0 -> 1 -> 0 across exactly one loop,
        # starting and ending at the original (unzoomed) framing.
        raw_envelope = 0.5 * (1.0 - math.cos(theta))
        if zoom_pattern == "breathe":
            envelope = raw_envelope
        elif zoom_pattern == "breathe_out":
            envelope = 1.0 - raw_envelope
        elif zoom_pattern == "ease_hold":
            envelope = raw_envelope * raw_envelope * (3.0 - 2.0 * raw_envelope)
        else:
            raise ValueError(f"Unknown zoom_pattern: '{zoom_pattern}' (expected 'breathe', 'breathe_out', or 'ease_hold')")
        zoom = 1.0 + zoom_amplitude * envelope

        # Plain sin()/cos() terms - identical value at theta=0 and
        # theta=2*pi by construction, so the pan closes the loop exactly
        # the same way the zoom does above.
        ellipse_x = math.sin(theta)
        ellipse_y = math.sin(theta + pan_phase)

        if bias_angle is not None:
            cos_b, sin_b = math.cos(bias_angle), math.sin(bias_angle)
            ellipse_x, ellipse_y = (
                ellipse_x * cos_b - ellipse_y * sin_b,
                ellipse_x * sin_b + ellipse_y * cos_b,
            )

        pan_x_frac = pan_amplitude * ellipse_x
        pan_y_frac = pan_amplitude * ellipse_y

        yield zoom, pan_x_frac, pan_y_frac


def compute_global_drift_pixel_vectors(num_frames, width, height, zoom_amplitude, pan_amplitude, pan_phase_deg,
                                        zoom_pattern="breathe", pan_bias_angle_deg=None):
    """
    derives each frame's own global camera-motion pixel
    displacement vector (dx, dy) - how far the visible scene content
    itself appears to have shifted on screen since the PREVIOUS frame,
    purely from this same _camera_path the rest of this module already
    walks - for postprocess.py's directional, velocity-aware motion blur
    to consume (see _apply_motion_blur there), replacing the old flat,
    non-directional whole-frame blend.

    Only covers the camera-driven global drift case (the "spatially
    uniform, cheap, no per-pixel vector needed" case the design notes itself
    calls out) - NOT scene-content motion (vegetation sway, cloud drift
    via scene_warp.py), which the design notes frames as a much bigger, separate
    effort (a per-pixel vector field would need its own render target/
    uniform readback) and is explicitly out of scope for this fix. Fast,
    independently-moving content (rain/snow) already has its own correct,
    dedicated mechanism - see particles.py's motion_blur_samples/
    shutter_fraction - this function does not touch that.

    Derivation: the crop window's center in `_crop_box` is always
    (working_w/2 + pan_x, working_h/2 + pan_y) where pan_x = pan_x_frac *
    width, pan_y = pan_y_frac * height, and the crop is then resized back
    up to (width, height) by a factor of `zoom`. A crop window that has
    moved by (delta_pan_x, delta_pan_y) pixels in SOURCE space, once
    resized by `zoom`, moves the visible content by zoom * delta_pan in
    OUTPUT space - and it moves the OPPOSITE direction the crop window
    itself moved (panning the crop window right makes the same static
    content appear to slide left across the output frame), hence the
    negative sign. `zoom_avg` (this frame's and the previous frame's zoom,
    averaged) approximates the zoom in effect over that single-frame
    step; the second-order radial displacement term that would vary with
    distance from frame center (from the zoom itself changing) is
    deliberately omitted as a documented simplification matching the
    design notes's "spatially uniform... cheap" framing for this sub-case -
    this pipeline's characteristically subtle, single-digit-percent drift
    amplitudes keep that omitted term small.

    Treats the path as a CLOSED LOOP (frame 0's vector is measured against
    frame num_frames-1, not against a fictitious (0,0) rest state) via
    Python's negative-index wraparound, matching every other per-frame
    effect in this pipeline that assumes _camera_path closes exactly -
    otherwise frame 0 would show a spurious, discontinuous blur "snap"
    relative to the true, seamless loop boundary.

    Returns a list of (dx, dy) pixel-offset tuples, one per frame, in the
    same units _bilinear_sample_channel already expects (postprocess.py's
    _apply_motion_blur samples along -motion_vector directly).
    """
    path = list(_camera_path(num_frames, zoom_amplitude, pan_amplitude, pan_phase_deg, zoom_pattern, pan_bias_angle_deg))
    if num_frames <= 1:
        return [(0.0, 0.0) for _ in range(num_frames)]

    vectors = []
    for frame_i in range(num_frames):
        zoom_curr, pan_x_curr, pan_y_curr = path[frame_i]
        zoom_prev, pan_x_prev, pan_y_prev = path[frame_i - 1]  # -1 wraps to the last frame, closing the loop
        zoom_avg = 0.5 * (zoom_curr + zoom_prev)
        dx = -zoom_avg * (pan_x_curr - pan_x_prev) * width
        dy = -zoom_avg * (pan_y_curr - pan_y_prev) * height
        vectors.append((dx, dy))
    return vectors


def _working_size(width, height, zoom_amplitude, pan_amplitude, pan_bias_angle_deg=None):
    """
    The working canvas has to be big enough to contain every possible crop
    window (largest at zoom=1.0, i.e. the full original size) even when
    panned to its maximum offset in either direction - otherwise the crop
    box would run off the edge of the source image/array. Solved for
    below: worst case is zoom=1 (crop size = full width/height) combined
    with max pan offset, so the working canvas needs (1 + 2*pan_amplitude)
    headroom on top of the zoom headroom. Shared by apply_camera_drift and
    apply_camera_drift_to_depth so both build a working canvas of the
    exact same size/scale.

    Uses math.ceil, not round(): at exactly the theoretical peak pan
    offset (sin(theta) hitting exactly +-1.0, which real num_frames values
    do land on), the exact required size can have a fractional part under
    0.5 - round() would then round DOWN, leaving the canvas a fraction of
    a pixel too small and making the crop box overrun it (confirmed:
    PIL's resize() raises "box can't exceed original image size" in
    exactly this case). ceil() guarantees enough headroom always.

    pan_bias_angle_deg: see _camera_path/_effective_pan_amplitude -
   .4's pan-ellipse rotation can push an individual axis's
    excursion above pan_amplitude, so the margin uses the inflated
    effective value whenever a bias angle is actually in use. None
    (default) = pan_amplitude unchanged, identical sizing to before this
    option existed.
    """
    effective_pan_amplitude = _effective_pan_amplitude(pan_amplitude, pan_bias_angle_deg)
    working_scale = 1.0 + zoom_amplitude + 2.0 * effective_pan_amplitude
    return (math.ceil(width * working_scale), math.ceil(height * working_scale))


def _crop_box(zoom, pan_x_frac, pan_y_frac, width, height, working_w, working_h):
    """Turns one frame's (zoom, pan_x_frac, pan_y_frac) - see _camera_path - into a source `box` for PIL's resize(), shared by apply_camera_drift and apply_camera_drift_to_depth."""
    crop_w = width / zoom
    crop_h = height / zoom
    pan_x = pan_x_frac * width
    pan_y = pan_y_frac * height
    center_x = working_w / 2.0 + pan_x
    center_y = working_h / 2.0 + pan_y
    left = center_x - crop_w / 2.0
    top = center_y - crop_h / 2.0
    return (left, top, left + crop_w, top + crop_h)


def _pad_amount(width, height, zoom_amplitude, pan_amplitude, pan_bias_angle_deg=None):
    """
    fix: how much thin, symmetric border padding the ORIGINAL
    full-resolution image/array needs on each side so every possible
    pan/zoom crop box (see _crop_box) stays within bounds, WITHOUT
    pre-resampling the entire photo through an extra upscale pass first
    (the old "working canvas" approach - see apply_camera_drift's
    docstring for why that was a real, avoidable softening).

    Deliberately reuses _working_size's EXACT existing total-canvas-size
    formula (not a smaller/tighter one) - only HOW that canvas's pixel
    content is produced changes here (mirror-padding the true-resolution
    original instead of resizing/stretching it), not the canvas's size or
    the crop math built on top of it, so the actual zoom/pan FRAMING at
    every point in the loop stays exactly what zoom_amplitude/
    pan_amplitude already meant before this fix (both here and in
    apply_layered_camera_drift's per-layer canvases, which still call
    _working_size directly) - this fix's scope is the resampling quality,
    not a redefinition of what these parameters visually mean.

    pan_bias_angle_deg: passed straight through to _working_size - see
    its docstring and _effective_pan_amplitude.
    """
    working_w, working_h = _working_size(width, height, zoom_amplitude, pan_amplitude, pan_bias_angle_deg)
    pad_x = (working_w - width + 1) // 2
    pad_y = (working_h - height + 1) // 2
    return pad_x, pad_y


def _pad_edge_mirror(array, pad_x, pad_y):
    """
    Pads a numpy array (2D scalar field like depth/mask, or 3D image data
    like RGB/RGBA) by REFLECTING its own edge content outward - design notes
    8.2: this synthesizes only the thin new border region needed for the
    maximum pan offset, at true source resolution (no resampling
    involved at all, just index reflection), instead of resampling the
    ENTIRE image through an extra LANCZOS/BILINEAR upscale pass the way
    the old shared "working canvas" did. mode='reflect' (not 'edge') to
    avoid a visible flat repeated-pixel band right at the original
    photo's border - a mirrored continuation reads far less like an
    artificial seam.
    """
    if array.ndim == 2:
        return np.pad(array, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    return np.pad(array, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="reflect")


def apply_camera_drift(base_image, num_frames, zoom_amplitude=0.05, pan_amplitude=0.045, pan_phase_deg=100.0,
                        zoom_pattern="breathe", pan_bias_angle_deg=None, start_frame=0):
    """
    Generator - yields num_frames copies of base_image, each resampled
    through a slow, cyclic zoom+pan (Ken Burns) path, at the original
    image's resolution.

    The "camera" starts at the original framing (zoom=1.0, no pan),
    drifts to its most zoomed-in/panned point at the midpoint of the
    loop, and eases back to the exact original framing by the end -  a
    closed loop, not a one-way pan, so it's automatically seamless across
    the loop boundary with no crossfade needed (a linear pan would need
    one, and would still show a visible "snap back" at the seam).

    zoom_amplitude: 0.0-1.0, how much the "camera" zooms in at the peak
        of the cycle (e.g. 0.05 = 5% zoom-in at the midpoint of the loop).
    pan_amplitude: 0.0-1.0, how far the "camera" drifts at the peak of the
        cycle, as a fraction of the image's width/height.
    pan_phase_deg: phase offset between the horizontal and vertical pan
        sine waves - 90-180 degrees gives a genuine looping ELLIPSE path
        instead of the camera sliding back and forth along one straight
        line (which reads as far more mechanical/repetitive).
    zoom_pattern/pan_bias_angle_deg: see _camera_path.
        Defaults ("breathe", None) reproduce this function's exact
        behavior from before these options existed.
    start_frame: skips yielding frames 0..start_frame-1 (their setup cost
        is still paid, but no resize work happens for them) - see
        render_checkpoint.py. Every frame's content is a pure function of
        its own frame_i, so resuming at start_frame produces bit-identical
        output to an uninterrupted run from frame 0.

    this used to build one shared "working canvas" up front
    via `base_image.resize(working_size, LANCZOS)` - a first, whole-photo
    resample pass every frame's own crop-resize was then stacked on top
    of, softening detail across essentially the entire frame twice per
    frame instead of once. Fixed by padding the ORIGINAL full-resolution
    image with a thin mirrored border (see _pad_amount/_pad_edge_mirror -
    no resampling involved, just reflecting existing edge pixels outward)
    just wide enough to cover the maximum possible pan offset, then
    resampling directly from THAT padded original once per frame - the
    vast majority of every output frame (everything except the thin
    synthesized border, which is only ever visible at the extreme edge of
    the maximum pan offset) is now true single-resample quality.
    """
    width, height = base_image.size
    pad_x, pad_y = _pad_amount(width, height, zoom_amplitude, pan_amplitude, pan_bias_angle_deg)
    padded_array = _pad_edge_mirror(np.asarray(base_image), pad_x, pad_y)
    padded_image = Image.fromarray(padded_array, mode=base_image.mode)
    padded_w, padded_h = padded_image.size

    for frame_i, (zoom, pan_x_frac, pan_y_frac) in enumerate(_camera_path(num_frames, zoom_amplitude, pan_amplitude, pan_phase_deg, zoom_pattern, pan_bias_angle_deg)):
        if frame_i < start_frame:
            continue
        box = _crop_box(zoom, pan_x_frac, pan_y_frac, width, height, padded_w, padded_h)
        # PIL's resize() accepts a source `box` directly - one resample
        # pass instead of a separate crop() + resize().
        yield padded_image.resize((width, height), Image.LANCZOS, box=box)


_LAYER_DEPTH_CENTERS = (0.15, 0.5, 0.85)   # far, mid, near - depth values (0=far, 1=near) each layer is centered on
_LAYER_DEPTH_SIGMA = 0.28                  # width of each layer's soft membership bump - wide enough to overlap
                                             # neighbors smoothly, so there's no hard depth cutoff/seam anywhere
_LAYER_AMPLITUDE_SCALES = (0.45, 1.0, 1.6)  # far, mid, near - how much MORE (near) or LESS (far) each layer's
                                             # effective zoom/pan amplitude is vs. the base amplitude (mid = unchanged)


def _weighted_percentile(values, weights, percentiles):
    """
    the confidence-weighted equivalent of np.percentile:
    each value's contribution to the percentile ranking is scaled by its
    own `weights` entry instead of counting every value equally. Standard
    interpolated weighted-percentile formula: sort by value, build each
    value's cumulative weight fraction (using the midpoint of its own
    weight interval, the usual convention so the result reduces to
    np.percentile's own answer when every weight is equal), then linearly
    interpolate `percentiles` (0..100) against that cumulative fraction.
    """
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative /= np.sum(sorted_weights)
    return np.interp(np.asarray(percentiles) / 100.0, cumulative, sorted_values)


def _compute_layer_depth_centers(depth_array, confidence=None):
    """
    Gap A fix - returns this photo's own 15th/50th/85th depth
    percentiles as the three layer centers, instead of always using the
    fixed _LAYER_DEPTH_CENTERS = (0.15, 0.5, 0.85), which silently assumes
    a roughly-uniform depth distribution. A photo that is almost entirely
    midground content (depth clustered tightly around 0.5) would have its
    far/near layers contribute almost nothing under the fixed centers,
    collapsing the layered parallax to a single flat plane for exactly the
    photos where a well-tuned partition would matter least anyway - while
    a genuinely bimodal photo (a near foreground branch against a far sky,
    almost nothing in between) would waste the "mid" layer's amplitude
    differentiation on a nearly-empty gap in the real distribution, and
    compress both populated extremes toward the edges of their Gaussian's
    effective range instead of centering on them. Re-centering on the
    photo's own percentiles fixes both: a flat-depth photo naturally gets
    three closely-clustered centers (correctly degrading toward one
    averaged amplitude, since there's little to parallax-differentiate),
    and a bimodal photo gets centers that land on its actual near/far
    clusters instead of one sitting in empty space between them.

    confidence: optional, same-shape array from
    depth.py's estimate_depth(..., return_confidence=True) - when given,
    the percentiles are WEIGHTED by per-pixel depth confidence instead of
    counting every pixel equally, so a large low-confidence region (sky,
    calm water, a blank wall - see depth.py's _depth_confidence, the same
    local-texture signal the guided-filter refinement itself already
    trusts less) can't drag a layer center toward its own, less reliable
    depth estimate as easily as a well-textured, high-confidence region
    can. None (the default) keeps the exact unweighted behavior above.
    """
    finite_mask = np.isfinite(depth_array)
    finite_depths = depth_array[finite_mask]
    if finite_depths.size == 0:
        return _LAYER_DEPTH_CENTERS
    if confidence is not None:
        finite_confidence = confidence[finite_mask].astype("float64")
        if finite_confidence.sum() > 1e-6:
            p15, p50, p85 = _weighted_percentile(finite_depths, finite_confidence, [15.0, 50.0, 85.0])
            return (float(p15), float(p50), float(p85))
    p15, p50, p85 = np.percentile(finite_depths, [15.0, 50.0, 85.0])
    return (float(p15), float(p50), float(p85))


def _layer_depth_weights(depth_array, centers=_LAYER_DEPTH_CENTERS):
    """
    Returns a list of 3 (height, width) float32 arrays (far, mid, near) -
    a smooth PARTITION OF UNITY (they sum to exactly 1.0 at every pixel)
    built from Gaussian bumps centered on `centers`. Soft,
    not hard-thresholded, layer membership -.1's multi-layer
    parallax needs each layer's content to fade smoothly into its
    neighbors' where they overlap in depth, not snap at a hard cutoff
    (which would show a visible seam the instant the layers start moving
    at different rates during the drift).
    """
    weights = [np.exp(-((depth_array - center) ** 2) / (2.0 * _LAYER_DEPTH_SIGMA ** 2)) for center in centers]
    total = sum(weights) + 1e-6
    return [weight / total for weight in weights]


def _layered_camera_drift_frames(base_image, depth_array, num_frames, zoom_amplitude, pan_amplitude, pan_phase_deg,
                                  zoom_pattern="breathe", pan_bias_angle_deg=None, depth_confidence=None):
    """
    Shared generator behind apply_layered_camera_drift/
    apply_layered_camera_drift_to_depth - yields
    (photo_frame: PIL RGB Image, depth_frame: float32 array) tuples, both
    warped and re-composited from the SAME 3 depth-based layers so they
    stay in exact per-pixel registration with each other (an occlusion
    mask built from a depth frame that drifted differently than the photo
    it's supposed to mask would be wrong).

    The core idea: split the photo into near/mid/far layers by depth,
    centered on this photo's own 15th/50th/85th depth percentiles (see
    _compute_layer_depth_centers/_layer_depth_weights Gap A),
    and let EACH layer's effective camera zoom/pan
    amplitude scale differently (_LAYER_AMPLITUDE_SCALES) - the near layer
    pans/zooms further per frame than the far layer, exactly like real
    parallax (near objects slide across frame faster than distant ones
    during an actual camera pan), instead of one flat image warped by a
    single rigid transform (the old apply_camera_drift/
    apply_camera_drift_to_depth, still used when no depth_array is
    available - see main.py's fallback).

    Each layer is a FULL warp of the ENTIRE original photo (not a cutout) -
    only its per-pixel WEIGHT mask limits where it visually contributes.
    This is what avoids literal "gaps"/holes where a near layer shifts
    away from a background it would, in reality, have been occluding:
    since the far layer is also a full-frame warp of the same photo (just
    at a smaller effective amplitude), whatever the near layer's shift
    reveals is the far layer's own (correctly, if only approximately,
    since there's no true 3D reconstruction) drifting version of that
    same area - the standard, accepted approximation for single-photo
    layered/2.5D parallax, not a new artifact this introduces.

    zoom_pattern/pan_bias_angle_deg: see _camera_path -
    applied identically to every layer (each layer's own zoom/pan
    AMPLITUDE is already scaled independently via _LAYER_AMPLITUDE_SCALES;
    the SHAPE/bias these select is a property of the overall camera move,
    not of any one depth layer, so all layers share it).

    depth_confidence: optional - see
    _compute_layer_depth_centers's own `confidence` param; passed straight
    through so the layer split this function computes uses the same
    confidence-weighted percentiles. Callers that warp both the photo
    (apply_layered_camera_drift) and its depth map
    (apply_layered_camera_drift_to_depth) for the SAME render must pass
    the identical `depth_confidence` (or both leave it None) to both
    calls - the layer centers are otherwise nondeterministic-per-call-site
    relative to each other and the two outputs would fall out of registration.
    """
    width, height = base_image.size
    layer_centers = _compute_layer_depth_centers(depth_array, confidence=depth_confidence)
    layer_weights_original = _layer_depth_weights(depth_array, layer_centers)
    depth_image_full = Image.fromarray(depth_array, mode="F")

    layer_data = []
    for weight_original, scale in zip(layer_weights_original, _LAYER_AMPLITUDE_SCALES):
        layer_zoom = zoom_amplitude * scale
        layer_pan = pan_amplitude * scale
        working_size = _working_size(width, height, layer_zoom, layer_pan, pan_bias_angle_deg)

        working_photo = base_image.resize(working_size, Image.LANCZOS)
        working_depth = depth_image_full.resize(working_size, Image.BILINEAR)
        weight_image = Image.fromarray((np.clip(weight_original, 0.0, 1.0) * 255.0).astype("uint8"), mode="L")
        working_weight = weight_image.resize(working_size, Image.BILINEAR)

        camera_path = list(_camera_path(num_frames, layer_zoom, layer_pan, pan_phase_deg, zoom_pattern, pan_bias_angle_deg))
        layer_data.append((working_photo, working_depth, working_weight, working_size, camera_path))

    for frame_i in range(num_frames):
        numerator_rgb = np.zeros((height, width, 3), dtype="float32")
        numerator_depth = np.zeros((height, width), dtype="float32")
        denominator = np.zeros((height, width), dtype="float32")

        for working_photo, working_depth, working_weight, working_size, camera_path in layer_data:
            zoom, pan_x_frac, pan_y_frac = camera_path[frame_i]
            working_w, working_h = working_size
            box = _crop_box(zoom, pan_x_frac, pan_y_frac, width, height, working_w, working_h)

            photo_sample = working_photo.resize((width, height), Image.LANCZOS, box=box)
            depth_sample = working_depth.resize((width, height), Image.BILINEAR, box=box)
            weight_sample = working_weight.resize((width, height), Image.BILINEAR, box=box)

            weight_arr = np.asarray(weight_sample, dtype="float32") / 255.0
            numerator_rgb += np.asarray(photo_sample, dtype="float32") * weight_arr[:, :, None]
            numerator_depth += np.asarray(depth_sample, dtype="float32") * weight_arr
            denominator += weight_arr

        safe_denominator = np.clip(denominator, 1e-6, None)
        composited_rgb = np.clip(numerator_rgb / safe_denominator[:, :, None], 0, 255).astype("uint8")
        composited_depth = numerator_depth / safe_denominator

        yield Image.fromarray(composited_rgb, mode="RGB"), composited_depth


def apply_layered_camera_drift(base_image, depth_array, num_frames, zoom_amplitude=0.05, pan_amplitude=0.045, pan_phase_deg=100.0,
                                zoom_pattern="breathe", pan_bias_angle_deg=None, depth_confidence=None):
    """
    Generator - like apply_camera_drift, but with real depth-driven
    parallax: splits the photo into near/mid/far depth
    layers and lets each drift at its own effective amplitude, so
    foreground content visibly slides across frame faster than background
    content during the "camera" pan - see _layered_camera_drift_frames's
    docstring for the full mechanism. Same parameters/loop-closure
    guarantees as apply_camera_drift (the underlying _camera_path is
    identical, just scaled per layer). zoom_pattern/pan_bias_angle_deg:
    see _camera_path. depth_confidence: optional - see _layered_camera_drift_frames's docstring; must
    match whatever apply_layered_camera_drift_to_depth is called with for
    this same render, if that is also being used.
    """
    for photo_frame, _ in _layered_camera_drift_frames(base_image, depth_array, num_frames, zoom_amplitude, pan_amplitude, pan_phase_deg, zoom_pattern, pan_bias_angle_deg, depth_confidence=depth_confidence):
        yield photo_frame


def apply_layered_camera_drift_to_depth(base_image, depth_array, num_frames, zoom_amplitude=0.05, pan_amplitude=0.045, pan_phase_deg=100.0,
                                         zoom_pattern="breathe", pan_bias_angle_deg=None, depth_confidence=None):
    """
    Generator - the depth-map equivalent of apply_layered_camera_drift,
    yielding the SAME multi-layer parallax's depth field per frame (not
    the flat single-transform depth warp apply_camera_drift_to_depth
    produces) - so a depth-derived occlusion/ground/fog mask stays
    correctly registered with the multi-layer photo instead of tracking a
    different (flat) motion than what the photo actually shows.
    zoom_pattern/pan_bias_angle_deg: see _camera_path -
    MUST match whatever apply_layered_camera_drift was called with for
    the same photo, or the depth frame drifts differently than the photo
    it's supposed to mask. depth_confidence: optional - see
    _layered_camera_drift_frames's docstring; MUST also match whatever
    apply_layered_camera_drift was called with for the same reason.
    """
    for _, depth_frame in _layered_camera_drift_frames(base_image, depth_array, num_frames, zoom_amplitude, pan_amplitude, pan_phase_deg, zoom_pattern, pan_bias_angle_deg, depth_confidence=depth_confidence):
        yield depth_frame


def apply_camera_drift_to_array(array, num_frames, zoom_amplitude=0.05, pan_amplitude=0.045, pan_phase_deg=100.0,
                                 zoom_pattern="breathe", pan_bias_angle_deg=None, start_frame=0):
    """
    Generator - the generic per-pixel-scalar-field equivalent of
    apply_camera_drift: resamples ANY (height, width) float32 array
    through the EXACT SAME per-frame zoom/pan crop path (see
    _camera_path/_crop_box above), yielding one (height, width) float32
    numpy array per frame. Originally built for depth.py's depth arrays
    (see apply_camera_drift_to_depth, now a thin alias for this), and
    reused as-is for segmentation masks - both are just a (height, width) float32 scalar field that
    needs to track wherever the "camera" has actually drifted to at each
    moment, so there was no reason to write this twice.

    Bilinear resampling (not the photo path's LANCZOS): a depth/mask
    value is a smooth scalar field, not photographic detail with fine
    texture worth preserving sharply - bilinear is enough, and avoids
    LANCZOS's overshoot/ringing artifacts right at value discontinuities
    (which would fabricate fake intermediate values exactly at object/
    mask edges - precisely where accuracy matters most for a clean
    occlusion/effect boundary).

    the old shared "working canvas"
    double-resample is replaced the same way - pad the ORIGINAL array
    with a thin mirrored border (see _pad_amount/_pad_edge_mirror, no
    resampling involved) just wide enough for the maximum pan offset,
    then resample directly from that padded original once per frame.

    zoom_pattern/pan_bias_angle_deg: see _camera_path -
    MUST match whatever apply_camera_drift/apply_layered_camera_drift was
    called with for the same photo, or this scalar field drifts
    differently than the photo it's meant to stay registered with.

    start_frame: skips yielding frames 0..start_frame-1. CONTRACT for
    every caller that pairs this animated array with a start_frame-aware
    consumer (e.g. scene_warp.render_region_warp, god_rays.render_light_overlay/
    render_lens_flare_overlay): those functions expect the FIRST item they
    pull from an animated source to already correspond to frame
    start_frame, not frame 0 - passing the same start_frame here is what
    keeps this generator's first yielded item aligned with theirs.
    """
    height, width = array.shape
    pad_x, pad_y = _pad_amount(width, height, zoom_amplitude, pan_amplitude, pan_bias_angle_deg)
    padded_array = _pad_edge_mirror(array, pad_x, pad_y)
    padded_image = Image.fromarray(padded_array, mode="F")
    padded_w, padded_h = padded_image.size

    for frame_i, (zoom, pan_x_frac, pan_y_frac) in enumerate(_camera_path(num_frames, zoom_amplitude, pan_amplitude, pan_phase_deg, zoom_pattern, pan_bias_angle_deg)):
        if frame_i < start_frame:
            continue
        box = _crop_box(zoom, pan_x_frac, pan_y_frac, width, height, padded_w, padded_h)
        resampled = padded_image.resize((width, height), Image.BILINEAR, box=box)
        yield np.asarray(resampled, dtype="float32")


def apply_camera_drift_to_depth(depth_array, num_frames, zoom_amplitude=0.05, pan_amplitude=0.045, pan_phase_deg=100.0,
                                 zoom_pattern="breathe", pan_bias_angle_deg=None, start_frame=0):
    """Thin, depth-specifically-named alias for apply_camera_drift_to_array - kept for existing call-site clarity/compatibility."""
    yield from apply_camera_drift_to_array(depth_array, num_frames, zoom_amplitude, pan_amplitude, pan_phase_deg, zoom_pattern, pan_bias_angle_deg, start_frame=start_frame)


def apply_camera_drift_to_region_rg(region_rg_array, num_frames, zoom_amplitude=0.05, pan_amplitude=0.045, pan_phase_deg=100.0,
                                     zoom_pattern="breathe", pan_bias_angle_deg=None, start_frame=0):
    """
    Generator - drifts a (height, width, 2) uint8 "region_rg" array (see
    scene_warp.render_region_warp's docstring - R=visibility/alpha mask,
    G=displacement-amplitude weight) through the SAME per-frame camera
    path as the photo/depth. Each of the 2 channels is resampled independently via
    apply_camera_drift_to_array, then re-stacked - yields one (height,
    width, 2) uint8 array per frame.

    start_frame: see apply_camera_drift_to_array's matching param - passed
    through to both inner channel generators so they stay in lockstep with
    each other and with whatever start_frame-aware effect consumes this.
    """
    alpha_channel = region_rg_array[:, :, 0].astype("float32") / 255.0
    weight_channel = region_rg_array[:, :, 1].astype("float32") / 255.0

    alpha_frames = apply_camera_drift_to_array(alpha_channel, num_frames, zoom_amplitude, pan_amplitude, pan_phase_deg, zoom_pattern, pan_bias_angle_deg, start_frame=start_frame)
    weight_frames = apply_camera_drift_to_array(weight_channel, num_frames, zoom_amplitude, pan_amplitude, pan_phase_deg, zoom_pattern, pan_bias_angle_deg, start_frame=start_frame)

    for alpha_frame, weight_frame in zip(alpha_frames, weight_frames):
        yield np.stack([
            np.clip(alpha_frame * 255.0, 0.0, 255.0).astype("uint8"),
            np.clip(weight_frame * 255.0, 0.0, 255.0).astype("uint8"),
        ], axis=-1)


def apply_camera_drift_to_color_array(color_array, num_frames, zoom_amplitude=0.05, pan_amplitude=0.045, pan_phase_deg=100.0,
                                       zoom_pattern="breathe", pan_bias_angle_deg=None, start_frame=0):
    """
    Generator - the (height, width, 3) float32 counterpart to
    apply_camera_drift_to_array/apply_camera_drift_to_region_rg, for
    god-ray/lens-flare/starburst textures (see
    god_rays.build_ray_texture/build_lens_flare_texture/
    build_starburst_texture) were computed ONCE from the static original
    photo and never resampled again, so the ray shaft / flare ghosts /
    starburst spikes stayed visually "glued" to the light source's
    ORIGINAL screen position while the base photo panned/zoomed
    underneath them during camera drift - the same registration-mismatch
    pattern already fixed for vegetation/cloud/particle sampling
    (apply_camera_drift_to_array) and segmentation masks
    (apply_camera_drift_to_region_rg). `build_lens_flare_texture`'s
    per-pixel `color_array` return is (height, width, 3)
    float32 in 0..1, one value per ghost/halo color channel - not a
    uint8 image and not a 2-channel rg pair, so neither existing helper
    fits directly; each of the 3 channels is resampled independently via
    apply_camera_drift_to_array (kept as float32 throughout, unlike
    apply_camera_drift_to_region_rg's uint8 round-trip, since a lens
    flare's ghost colors are meant to carry full float precision rather
    than a mask's already-8-bit-quantized alpha/weight), then re-stacked
    - yields one (height, width, 3) float32 array per frame.
    """
    channel_frames = [
        apply_camera_drift_to_array(
            np.ascontiguousarray(color_array[:, :, channel], dtype="float32"),
            num_frames, zoom_amplitude, pan_amplitude, pan_phase_deg, zoom_pattern, pan_bias_angle_deg,
            start_frame=start_frame,
        )
        for channel in range(3)
    ]
    for red_frame, green_frame, blue_frame in zip(*channel_frames):
        yield np.stack([red_frame, green_frame, blue_frame], axis=-1)
