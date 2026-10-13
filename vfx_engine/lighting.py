"""
lighting.py

Light flicker (e.g. lamp, campfire, candle). Unlike snow/rain/fog, this
effect doesn't add a new transparent layer on top of the image - it takes
an already-finished frame sequence and brightens pixels in it, frame by
frame, so it looks like a living, flickering light.

The "screen blend" used here (`1 - (1-a)*(1-b)`, the same formula
PIL's ImageChops.screen implements) brightens the image in a way that
doesn't immediately "blow it out" to white - the brighter a pixel already
is, the less extra brightening it gets. It's the same effect graphics
programs use under the "Screen" blend mode name.

Operates on linear-light float32 (height, width, 3) numpy arrays, not gamma-encoded PIL Images - see color.py's module docstring for
why (this module's screen blend is exactly the kind of additive
brightening operation that's mathematically wrong to do directly on
sRGB-encoded values) and compositor.py for where frames enter this linear
representation.
"""

import numpy as np
import torch
from PIL import Image, ImageFilter

import config
from vfx_engine import color as _color


def estimate_ambient_light_color(base_image, illumination_result=None):
    """
    a scene-wide ambient light color term, distinct from
    god_rays.sample_light_color's small local neighborhood sample around
    the single detected dominant light source. A real scene's ambient
    light (the soft, directionless "fill" from skylight, inter-
    reflection, or overcast haze) is a property of the WHOLE photo's
    color cast, not just the few dozen pixels immediately around the
    brightest point - sampling only that small patch misses, e.g., a
    warm sunset's overall orange cast when the sun itself is a small,
    near-white overexposed disc surrounded by a much larger, much more
    saturated sky.

    Baseline heuristic (used whenever illumination_result is None, or
    config.ENABLE_LEARNED_ILLUMINATION is off): the original design's own
    suggested approach - heavily downsample the photo (so individual
    objects/edges disappear and only the coarse overall color cast
    survives) and blur what's left, then take its mean color. Cheap,
    dependency-free, matches the "small thumbnail, cheap heuristic"
    spirit of every other per-photo color estimate already in this
    module/codebase (resolve_ambient_drift_scale, god_rays.
    detect_light_source).

    learned-illumination upgrade: when the caller has already computed
    illumination.py's Marigold intrinsic-image-decomposition result for
    this photo (config.ENABLE_LEARNED_ILLUMINATION - pass it as
    `illumination_result` to avoid re-running that model here), its
    `shading` channel is used instead. `shading` is, by construction,
    this decomposition's estimate of how light actually falls on the
    scene with each object's own base color (albedo) already factored
    OUT - a materially better "what color is the light in this scene"
    estimate than a blurred RGB photo copy, whose color still has every
    object's own albedo baked in (a scene full of green trees would read
    as having greenish ambient light under a blurred-photo estimate even
    under neutral white daylight, because the blur never separates "the
    trees are green" from "the light is green").

    No new model is downloaded/introduced for this upgrade path - it
    reuses illumination.py's already-cached Marigold checkpoint
 via
    whatever result the caller already computed for other purposes; this
    function itself never calls estimate_illumination.

    Returns an (r, g, b) 0-255 int tuple.
    """
    if illumination_result is not None and config.ENABLE_LEARNED_ILLUMINATION:
        shading = illumination_result.get("shading")
        if shading is not None and shading.size > 0:
            mean = shading.reshape(-1, 3).mean(axis=0)
            # shading is a multiplicative lighting term, not meant to be
            # viewed directly as a color swatch - typically fairly dim -
            # so its peak channel is renormalized up to a sensible
            # display brightness before converting to 0-255, otherwise
            # this nearly always comes out a dark, muddy tone that
            # dominates whatever it's blended into far too weakly.
            peak = float(mean.max())
            if peak > 1e-6:
                mean = mean / peak
            return tuple(int(np.clip(channel * 255.0, 0, 255)) for channel in mean)

    thumb = base_image.convert("RGB").resize((16, 16), resample=Image.BOX)
    thumb = thumb.filter(ImageFilter.GaussianBlur(radius=3))
    rgb = np.asarray(thumb, dtype="float32").reshape(-1, 3)
    mean = rgb.mean(axis=0)
    return tuple(int(np.clip(channel, 0, 255)) for channel in mean)


def _resolve_ambient_drift_cycles(num_frames):
    """
    config.AMBIENT_DRIFT_SECONDS_PER_CYCLE is a RATE (seconds
    per warm<->cool swing), not a fixed "exactly one swing over the entire
    render" - converts it into a whole cycle count (whole numbers only, so
    apply_ambient_drift's cos term still closes the loop exactly) sized to
    THIS render's actual duration (num_frames/config.FPS), so a long render
    gets proportionally more, naturally-paced swings instead of one lone
    swing stretched across the whole thing. max(1, ...) keeps a very short
    render from collapsing to 0 whole cycles.
    """
    duration_seconds = max(num_frames, 1) / config.FPS
    return max(1, round(duration_seconds / config.AMBIENT_DRIFT_SECONDS_PER_CYCLE))


def build_light_position_mask(width, height, light_pos_uv, radius_frac=0.55):
    """
    builds a soft, radial 0.0-1.0 falloff mask
    (height, width) centered on `light_pos_uv` (normalized 0.0-1.0 image
    coordinates, y=0 at the top - see main.py's resolve_light_pos_uv,
    which wraps god_rays.detect_light_source), for use as apply_flicker's
    `mask` parameter - so flicker can automatically scope itself to a
    photo's own detected light source instead of needing a generic,
    light-unaware mask (config.FLICKER_USE_MASK's shared_mask) manually
    pointed at it.

    Deliberately mirrors vfx_engine/shaders/blur.glsl's `light_glow` term
    exactly - the same aspect-corrected Gaussian falloff
    (`exp(-(dist^2) / (2 * radius^2))`, distance measured in
    aspect-corrected UV units, x-axis scaled by width/height before
    measuring distance so the glow reads as a circle rather than an
    ellipse on a non-square frame) already used everywhere else in this
    codebase a light position needs to fall off softly with distance -
    this is "the identical mechanism" the original design explicitly calls out
    reusing, just evaluated once in numpy instead of once per fragment.

    radius_frac matches blur.glsl's own default light_radius (0.55, in
    those same aspect-corrected units) unless overridden.

    Returns None if `light_pos_uv` is None (no light detected/available -
    matches resolve_light_pos_uv's own None-when-no-image contract), so a
    caller can fall through to its own no-mask default.
    """
    if light_pos_uv is None:
        return None

    light_x_uv, light_y_uv = light_pos_uv
    aspect = width / height

    y_coords, x_coords = np.mgrid[0:height, 0:width].astype("float32")
    x_uv = (x_coords + 0.5) / width
    y_uv = (y_coords + 0.5) / height

    dx = (x_uv - light_x_uv) * aspect
    dy = y_uv - light_y_uv
    dist_to_light = np.sqrt(dx * dx + dy * dy)

    return np.exp(-(dist_to_light ** 2) / (2.0 * radius_frac ** 2)).astype("float32")


def apply_flicker(frames, num_frames, intensity=0.15, speed=0.3, color=(255, 220, 150), mask=None, seed=None, start_frame=0):
    """
    Generator - applies a flickering light to an already-finished frame
    sequence (linear float32 (height, width, 3) arrays - e.g. a photo with
    snow, already composited into one frame) and yields new frames one at
    a time.

    num_frames: number of frames in `frames` (needed up front to compute
                the flicker curve before the frame stream is consumed)
    intensity: 0.0-1.0, maximum brightening strength
    speed: how fast it flickers - higher value = faster changes
    color: (r, g, b) 0-255 (sRGB, the usual way colors are specified in
        this codebase) - glow color (e.g. warm lamp light), decoded to
        linear once at setup.
    mask: optional mask (0..1 array from vfx_engine/mask.py) limiting the
          flicker to a chosen area of the image (e.g. around a lamp);
          None = flickers over the whole image
    seed: RNG-determinism audit (2026-09-17) - feeds the noise-flutter RNG
        in _build_flicker_curve. None (default) preserves the old fresh-
        OS-entropy behavior; callers with a source photo should pass
        vfx_engine.lens_dirt.content_seed(image, salt="flicker") so the
        same photo always produces the same flicker flutter.
    start_frame: render-checkpoint/resume support (see render_checkpoint.py).
        The full-length flicker_curve is still built from num_frames (a
        pure function of frame count/speed/seed, not of start_frame), but
        `frames` itself is assumed to be an ALREADY start_frame-truncated
        stream (i.e. its first item is frame start_frame, not frame 0) -
        exactly like scene_warp.render_region_warp's animated-source
        contract - so the curve is sliced to [start_frame:] before zipping
        against it, keeping curve values aligned to the same real frame
        indices `frames` is actually yielding.
    """
    flicker_curve = _build_flicker_curve(num_frames, speed, seed=seed)[start_frame:]
    linear_color = _decode_color(color)

    for frame, curve_value in zip(frames, flicker_curve):
        strength = curve_value * intensity
        yield brighten_frame(frame, strength, linear_color, mask)


def _decode_color(srgb_color):
    """Decodes an (r, g, b) 0-255 sRGB color tuple to a linear float32 (3,) numpy array - see color.py."""
    return _color.srgb_to_linear(np.asarray(srgb_color, dtype="float32") / 255.0)


def _build_flicker_curve(num_frames, speed, seed=None):
    """
    Builds a sequence of 0.0-1.0 values (one per frame) describing the
    flicker strength over time - a combination of a gentle sine wave (slow
    "breathing" of the light) and smoothed random noise (small, uneven
    flutter, like a real flame).

    The slow wave's frequency is quantized to a whole number of cycles
    over num_frames (the same trick particles.py's wind/wobble math uses)
    so it closes the loop exactly - a raw `sin(t * speed)` has no reason
    to land back on its frame-0 value at frame num_frames, and for a
    live-wallpaper loop that's a visible brightness jump at the seam. The
    noise flutter is small and high-frequency enough that any residual
    mismatch at the seam is imperceptible next to it - not worth the same
    treatment.

    seed: see apply_flicker's own `seed` doc.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(num_frames)

    # Convert the requested `speed` into a whole number of cycles across
    # the loop, then rebuild the per-frame angular speed from that -
    # closest achievable match to the requested speed that still closes
    # the loop exactly.
    approx_cycles = max(1, round(speed * num_frames / (2.0 * np.pi)))
    quantized_speed = approx_cycles * 2.0 * np.pi / num_frames
    slow_wave = 0.5 + 0.5 * np.sin(t * quantized_speed)

    noise = rng.normal(0, 0.2, num_frames)
    smoothing_window = np.ones(3) / 3
    smooth_noise = np.convolve(noise, smoothing_window, mode="same")

    return np.clip(slow_wave + smooth_noise, 0.0, 1.0)


def resolve_ambient_drift_scale(base_image, min_scale=0.5, max_scale=1.6):
    """
    returns a scale factor for apply_ambient_drift's
    `strength`, derived cheaply from the photo's own overall brightness
    and warmth - a coarser stand-in for a full color-temperature estimate,
    in the same "small thumbnail, cheap heuristic" spirit as
    god_rays.detect_light_source (illumination.py's learned relighting
    estimate is a dormant, unintegrated exploratory spike - not something to build a hard dependency on here).

    A dim, warm-toned photo (more likely dusk/dawn/interior, where a
    sinking or artificial light source's color temperature can visibly
    shift even over a short clip) gets a LARGER scale, since more
    oscillation reads as natural there; a bright, neutral/cool-toned photo
    (more likely midday, where the sun's color temperature is essentially
    constant over a short loop) gets a SMALLER scale - otherwise the same
    fixed strength would show an unrealistically large swing on a bright
    neutral scene while under-shooting an already-warm dusk one (see this
    function's call sites and the original design's own writeup for the full
    reasoning).

    Brightness and warmth are each folded into their own 0.0-1.0 "more
    drift is natural" signal, then averaged - a photo needs to be BOTH dim
    AND warm to earn the full max_scale; a bright-but-warm (overexposed
    golden hour) or dim-but-neutral (overcast dusk) photo lands somewhere
    in between rather than snapping to an extreme from either signal
    alone.
    """
    thumb = base_image.convert("RGB").resize((32, 32))
    rgb = np.asarray(thumb, dtype="float32") / 255.0
    mean_rgb = rgb.reshape(-1, 3).mean(axis=0)
    brightness = float(mean_rgb.mean())

    # r-b channel difference is a simple, cheap warmth proxy - normalized
    # so it saturates at a modest, realistic difference rather than
    # needing a literally pure-red image to reach 1.0
    warmth = float(np.clip((mean_rgb[0] - mean_rgb[2]) / 0.25, 0.0, 1.0))
    dimness_signal = float(np.clip(1.0 - brightness, 0.0, 1.0))

    combined = 0.5 * (dimness_signal + warmth)
    return min_scale + combined * (max_scale - min_scale)


def apply_ambient_drift(frames, num_frames, strength=0.06, warm_color=(255, 225, 190), cool_color=(190, 205, 230), start_frame=0):
    """
    Generator - applies a slow, cyclic ambient color drift to an already-
    finished frame sequence (linear float32 (height, width, 3) arrays),
    swinging gently between `warm_color` and `cool_color` over the loop
    (real ambient light isn't perfectly constant even over a short span -
    drifting clouds, atmosphere - a perfectly static color grade for
    20-30 seconds reads as flat/synthetic next to genuinely moving effects
    like rain/snow/camera drift).

    Purely a function of loop phase (a cosine, at a whole-cycle-count
    frequency - see _resolve_ambient_drift_cycles), so it closes the loop
    exactly like every other cyclic effect in this pipeline - frame 0 and
    frame num_frames land on the same tint.

    start_frame: render-checkpoint/resume support (see render_checkpoint.py).
        `frames` is assumed to already be an ALREADY start_frame-truncated
        stream (its first item is frame start_frame, not frame 0) - so the
        frame_i used for the loop-phase `theta` below is offset by
        start_frame (via enumerate's own `start=`), keeping the cosine's
        phase aligned to the true absolute frame index instead of
        restarting the drift cycle from frame 0 on every resume.
    """
    linear_warm = _decode_color(warm_color)
    linear_cool = _decode_color(cool_color)
    drift_cycles = _resolve_ambient_drift_cycles(num_frames)

    # Part 21: lazily converted to device tensors the first time a
    # tensor-typed frame is seen, then cached across every remaining
    # iteration of this generator - avoids a per-frame numpy->tensor
    # conversion for these two small, constant (3,) colors.
    tensor_warm = None
    tensor_cool = None

    for frame_i, frame in enumerate(frames, start=start_frame):
        theta = 2.0 * np.pi * drift_cycles * (frame_i / max(num_frames, 1))
        warmth = np.cos(theta)   # +1 = fully warm, -1 = fully cool, smoothly cyclic

        if abs(warmth) < 0.001:
            yield frame
            continue

        tint_strength = abs(warmth) * strength

        if isinstance(frame, torch.Tensor):
            if tensor_warm is None:
                tensor_warm = torch.as_tensor(linear_warm, dtype=frame.dtype, device=frame.device)
                tensor_cool = torch.as_tensor(linear_cool, dtype=frame.dtype, device=frame.device)
            target_color = tensor_warm if warmth > 0 else tensor_cool
            yield frame * (1.0 - tint_strength) + target_color * tint_strength
            continue

        target_color = linear_warm if warmth > 0 else linear_cool
        yield frame * (1.0 - tint_strength) + target_color * tint_strength


def brighten_frame(frame, strength, linear_color, mask):
    """
    Brightens a single linear-light frame towards `linear_color` (a linear
    float32 (3,) array - see _decode_color/color.py), with strength
    `strength` (0..1), via a screen blend (`1 - (1-frame)*(1-tint)`) - see
    this module's docstring for why this now happens in linear light
    rather than on gamma-encoded 8-bit values.
    """
    if strength <= 0.001:
        return frame

    if isinstance(frame, torch.Tensor):
        tensor_color = torch.as_tensor(linear_color, dtype=frame.dtype, device=frame.device)
        tint = tensor_color * strength
        brightened = 1.0 - (1.0 - frame) * (1.0 - tint)

        if mask is None:
            return brightened

        tensor_mask = torch.as_tensor(mask, dtype=frame.dtype, device=frame.device)
        mask_3d = tensor_mask[:, :, None]
        return brightened * mask_3d + frame * (1.0 - mask_3d)

    tint = linear_color * strength
    brightened = 1.0 - (1.0 - frame) * (1.0 - tint)

    if mask is None:
        return brightened

    mask_3d = mask[:, :, None]
    return brightened * mask_3d + frame * (1.0 - mask_3d)
