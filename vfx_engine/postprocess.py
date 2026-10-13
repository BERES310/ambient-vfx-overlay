"""
postprocess.py

A small screen-space post-processing stack applied to the already-finished,
fully composited frames (photo + all effects baked in), right before
saving - the same kind of pass real-time engines and compositing tools
(Nuke, After Effects, game post-process stacks) run at the very end of the
pipeline:

    bloom               - bright areas softly glow/bleed into their surroundings
    chromatic_aberration - a subtle red/blue channel shift near the frame edges,
                            like a real camera lens
    motion_blur         - a directional, velocity-scaled streak along the
                            camera's own actual per-frame drift direction
, so panning/zooming leaves a real
                            directional trail instead of a flat, non-
                            directional whole-frame ghost
    tonemap             - a gentle filmic S-curve, so bright areas (bloom,
                            lightning) roll off softly instead of clipping
                            to flat white
    grain               - a subtle, ANIMATED (frame-varying) fine-noise
                            layer, applied last - real film has sensor-
                            level noise; a mathematically smooth composite
                            with none is one of the most recognizable
                            "rendered, not photographed" tells. Also doubles as dither ahead of the
                            final 8-bit quantization (image_io.
                            encode_frames_to_images) - a small amount of
                            noise before rounding to 8-bit breaks up
                            banding in smooth gradients (sky, fog, bloom)
                            that flat rounding alone would show.

Every stage is optional and independently tunable - see POSTPROCESS_PARAMS
in config.py. Order matters: bloom and tonemap work in that order because
bloom is meant to look like real light bleeding *before* the "camera"
rolls off highlights, motion blur runs first of all because it needs each
frame's un-blurred neighbor, and grain runs LAST (after tonemap) since
real sensor noise is added at the point of capture/output, not baked into
the scene's own lighting.

Operates on linear-light float32 (height, width, 3) arrays,
not gamma-encoded PIL Images - see color.py's module docstring for why.
This specifically matters here more than almost anywhere else in the
pipeline: bloom's bright-pixel threshold and tonemap's
highlight roll-off are both meant to operate on real HDR headroom (values
that can genuinely exceed 1.0 once several bright layers have stacked) -
on the old 8-bit pipeline, every value was already hard-clipped to 0-255
by the time these stages ran, so there was nothing left above "1.0" for
either of them to actually do their real job on. Bloom's own Gaussian
blur is implemented directly here (not PIL's ImageFilter, which only
supports 8-bit-per-channel or single-band 32-bit-INT modes - no
floating-point image mode that can hold values above 1.0 without clipping).

Part 21 (GPU-resident pipeline): every private stage below now has a full
torch-tensor equivalent alongside its original numpy implementation,
selected via an `isinstance(frame, torch.Tensor)` check at the top of each
public-facing helper - none of these private helpers (_box_blur_1d,
_gaussian_blur_linear, _bilinear_sample_channel, and the five
_apply_* functions) have any external callers outside this module (grep-
confirmed), so it's safe for them to fully branch internally rather than
needing a numpy-preserving fallback signature.

`_apply_motion_blur` used to be a flat, non-
directional, non-velocity-aware whole-frame blend with the PREVIOUS
frame (`frame * (1 - strength) + previous_frame * strength`) - this
smeared the ENTIRE frame uniformly regardless of how fast (or whether at
all) the camera was actually moving at that instant, and in the wrong
"blend with a stale frame" direction rather than a true directional
streak. It is now a real velocity-buffer-style directional blur: given
this frame's own camera-motion pixel vector (see camera.py's
compute_global_drift_pixel_vectors), it shift-and-averages several
samples of the CURRENT frame along the reverse of that vector via
_bilinear_sample_channel (already used by chromatic aberration), so the
streak direction and length both track the camera's actual per-frame
speed - including going to exactly zero on frames where the camera is
momentarily at rest (e.g. near the "breathe" envelope's turning points),
which the old fixed-strength blend could never do. This only covers the
camera-driven global-drift case; see compute_global_drift_pixel_vectors'
docstring for why the scene-content (vegetation/cloud) case is out of
scope here, and why fast independently-moving content (rain/snow) is
correctly left to particles.py's own dedicated mechanism instead.

`_apply_grain` used to draw every pixel's noise
from the same flat standard deviation regardless of that pixel's own
brightness - real sensor noise is dominated by photon shot noise, whose
variance scales with the signal itself, plus a constant read-noise
floor, so shadows read visibly grainier than highlights on real film/
sensors. `_apply_grain` now scales its noise per-pixel by
sqrt(_GRAIN_SHOT_NOISE_WEIGHT * luminance + _GRAIN_READ_NOISE_FLOOR),
normalized so a pixel at luminance 1.0 still gets exactly the configured
`strength` as its std-dev (no change to existing calibration at that
reference point).
"""

import numpy as np
import torch


def apply_postprocess(frames, bloom=None, chromatic_aberration=None, motion_blur=None, tonemap=None, grain=None, seed=None, start_frame=0):
    """
    Generator - applies the configured post-processing stages to each
    linear-light frame in `frames`, in order, and yields the result. Any
    stage left as None (the default) is skipped entirely - see the dict
    shapes below.

    bloom: {"threshold": 0.7, "strength": 0.5} or None
    chromatic_aberration: {"strength": 2.0} (pixel shift at the frame edge) or None
    motion_blur: {"strength": 0.35, "vectors": [(dx, dy), ...]} (strength
        0.0-1.0, how pronounced the directional streak is; vectors is one
        (dx, dy) pixel-displacement tuple per frame, covering the FULL
        num_frames render (see camera.py's compute_global_drift_pixel_
        vectors) - indexed below by this frame's ABSOLUTE position
        (start_frame + this generator's own local position), not its local
        position in a possibly start_frame-truncated `frames` stream) or None
    tonemap: {"strength": 0.5} (0.0-1.0, how strongly the filmic curve is applied) or None
    grain: {"strength": 0.02} (0.0-1.0+, std-dev of the added noise at
        luminance 1.0, in the same linear-light units as the frame itself -
        darker pixels get progressively less absolute noise, toward a
        nonzero floor - see _apply_grain) or None
    seed: RNG-determinism audit (2026-09-17) - feeds grain's per-frame RNG
        (see the `frame_rng =` line below). None (default) preserves the
        old fresh-OS-entropy behavior; callers with a source photo should
        pass vfx_engine.lens_dirt.content_seed(image, salt="grain") so the
        same photo always produces the same grain pattern.
    start_frame: render-checkpoint/resume support (see render_checkpoint.py).
        `frames` may already be truncated to start at absolute frame
        start_frame (its first item IS frame start_frame, not frame 0) -
        the same contract used throughout this codebase (see god_rays.py/
        scene_warp.py/compositor.py's docstrings). Without this, grain's
        per-frame reseed and motion_blur's vector lookup below would both
        index by this generator's own LOCAL position (0, 1, 2, ...)
        instead of the frame's true absolute position in the render,
        producing a visibly different (wrong) grain pattern/blur vector
        for every frame after a resume than an uninterrupted render would
        have produced at that same absolute frame - defeating the whole
        point of a frame-level resume being indistinguishable from an
        uninterrupted render.
    """
    for local_index, frame in enumerate(frames):
        frame_index = start_frame + local_index
        if motion_blur is not None:
            motion_vector = motion_blur["vectors"][frame_index]
            frame = _apply_motion_blur(frame, motion_vector, motion_blur["strength"])

        if bloom is not None:
            frame = _apply_bloom(frame, bloom["threshold"], bloom["strength"])

        if chromatic_aberration is not None:
            frame = _apply_chromatic_aberration(frame, chromatic_aberration["strength"])

        if tonemap is not None:
            frame = _apply_tonemap(frame, tonemap["strength"])

        if grain is not None:
            # Render-checkpoint/resume (closing the gap flagged in this
            # module's RNG-determinism audit entry): the old code held ONE
            # `grain_rng = np.random.default_rng(seed)` Generator before
            # this loop and drew from it every frame, so its internal
            # state after frame N depended on having sequentially drawn
            # every frame from 0 through N-1 first - not index-addressable.
            # A resumed render that starts a fresh process at frame_index
            # (skipping the frames before it, see vfx_engine/
            # render_checkpoint.py) would create a brand-new Generator
            # that's only drawn ZERO times, producing frame_index's OLD
            # noise pattern (the one frame 0 would have drawn) instead of
            # the noise a continuous single-process render would have
            # reached by that point - a visible grain discontinuity right
            # at the resume point. Reseeding a fresh Generator from
            # (seed, frame_index) every frame instead makes each frame's
            # grain a pure function of its own index, matching the same
            # RNG-determinism pattern already used for spawn/jitter seeds
            # elsewhere (see lens_dirt.content_seed, particles.py's `seed`
            # param) - identical output whether this frame is reached by
            # rendering from frame 0 or by resuming straight into it. When
            # seed is None (old non-deterministic default), each frame
            # still just draws fresh OS entropy independently rather than
            # continuing one sequential stream - already non-deterministic
            # either way, so nothing observable changes for that case.
            frame_rng = np.random.default_rng(seed if seed is None else (seed, frame_index))
            frame = _apply_grain(frame, grain["strength"], frame_rng)

        yield frame


def _box_blur_1d(array, radius, axis):
    """
    Box blur along a single axis via a cumulative-sum sliding window -
    O(n) per pixel regardless of radius, pure numpy, no scipy/PIL. Edge-
    padded (not wrapped) so the blur doesn't bleed content from the
    opposite edge of the frame.
    """
    if isinstance(array, torch.Tensor):
        return _box_blur_1d_torch(array, radius, axis)
    kernel_size = 2 * radius + 1
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (radius + 1, radius)
    padded = np.pad(array, pad_width, mode="edge")
    cumsum = np.cumsum(padded, axis=axis)
    length = cumsum.shape[axis]
    front = np.take(cumsum, range(kernel_size, length), axis=axis)
    back = np.take(cumsum, range(0, length - kernel_size), axis=axis)
    return (front - back) / kernel_size


def _box_blur_1d_torch(tensor, radius, axis):
    """
    Torch-tensor equivalent of _box_blur_1d, computed directly on
    `tensor`'s own device - identical edge-padded cumulative-sum sliding
    window, using torch.narrow/torch.cat in place of numpy's np.pad/
    np.take (torch has no direct "pad by edge-replication" primitive that
    accepts an arbitrary axis the way np.pad does, so the edge padding is
    built explicitly here by repeating the first/last slice).
    """
    kernel_size = 2 * radius + 1
    first_slice = tensor.narrow(axis, 0, 1)
    last_slice = tensor.narrow(axis, tensor.shape[axis] - 1, 1)
    front_pad = torch.cat([first_slice] * (radius + 1), dim=axis)
    back_pad = torch.cat([last_slice] * radius, dim=axis)
    padded = torch.cat([front_pad, tensor, back_pad], dim=axis)
    cumsum = torch.cumsum(padded, dim=axis)
    length = cumsum.shape[axis]
    front = torch.narrow(cumsum, axis, kernel_size, length - kernel_size)
    back = torch.narrow(cumsum, axis, 0, length - kernel_size)
    return (front - back) / kernel_size


def _gaussian_blur_linear(array, radius):
    """
    Approximates a Gaussian blur via three passes of a box blur of the
    same radius - repeated box convolution converges toward a Gaussian
    shape (Central Limit Theorem), the same well-known cheap-blur
    approximation used in several real-time engines. Needed instead of
    PIL's ImageFilter.GaussianBlur specifically because this operates on
    unclamped linear-light float32 values (see this module's docstring) -
    PIL has no floating-point image mode that survives values above 1.0.
    """
    radius = max(1, int(round(radius)))
    result = array
    for _ in range(3):
        result = _box_blur_1d(result, radius, axis=0)
        result = _box_blur_1d(result, radius, axis=1)
    return result


def _apply_bloom(frame, threshold, strength):
    """Extracts pixels brighter than `threshold` (linear-light, can meaningfully exceed the old 0-1 gamma-space range), blurs just those at multiple scales, and adds the glow back additively."""
    # Standard luma weights (roughly 0.299R + 0.587G + 0.114B), not a flat
    # channel average - human perceived brightness weights green far more
    # heavily than red or blue. This is the same formula particles.glsl's
    # backlight calculation already uses (`dot(background, vec3(0.299,
    # 0.587, 0.114))`) - reusing it here fixes a real inconsistency
    #: a flat average lets a saturated blue pixel register
    # as "bright" for bloom purposes even though it isn't, perceptually.
    height, width = frame.shape[0], frame.shape[1]
    base_radius = max(2, int(min(width, height) * 0.01))

    # Multi-scale bloom: sum a tight, bright, small-radius
    # core glow with progressively wider, fainter halos rather than one
    # fixed-radius blur - real bloom scatters light at several scales at
    # once in a lens/eye (a crisp glow immediately around the source AND
    # a much broader, softer halo reaching further across the frame), and
    # a single blur pass can only ever produce one of those two. Radii
    # are geometric multiples of the base radius (roughly matching the
    # standard downsample-blur-upsample mip-chain trick's effective
    # scale progression); weights bias toward the tight core so the
    # source itself doesn't wash out into a uniform haze.
    scales = (
        (base_radius, 0.55),
        (base_radius * 3, 0.30),
        (base_radius * 8, 0.15),
    )

    if isinstance(frame, torch.Tensor):
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=frame.dtype, device=frame.device)
        brightness = torch.tensordot(frame, weights, dims=([2], [0]))[..., None]
        bright_mask = torch.clamp((brightness - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
        bright_pixels = frame * bright_mask

        glow_pixels = torch.zeros_like(frame)
        for radius, weight in scales:
            glow_pixels = glow_pixels + _gaussian_blur_linear(bright_pixels, radius=radius) * weight

        return frame + glow_pixels * strength

    brightness = np.tensordot(frame, np.array([0.299, 0.587, 0.114], dtype="float32"), axes=([2], [0]))[..., None]
    bright_mask = np.clip((brightness - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    bright_pixels = frame * bright_mask

    glow_pixels = np.zeros_like(frame)
    for radius, weight in scales:
        glow_pixels += _gaussian_blur_linear(bright_pixels, radius=radius) * weight

    # No clipping here (unlike the old 8-bit version) - headroom above 1.0
    # is exactly what the tonemap stage further down the chain is meant to
    # roll off gracefully; clipping now would just reintroduce the same
    # "nothing left for tonemap to do" problem this linear pipeline fixes.
    return frame + glow_pixels * strength


def _bilinear_sample_channel(channel_plane, x_coords, y_coords):
    """
    Samples `channel_plane` (a single (height, width) array) at fractional
    (x_coords, y_coords) positions via bilinear interpolation - the old chromatic-aberration shift truncated to whole-pixel
    `int32` offsets and indexed directly (nearest-neighbor by
    construction), which reads as slightly stepped/aliased at the small
    pixel-shift scale this effect operates at. Interpolating between the
    4 surrounding integer pixels instead gives the smooth, continuous
    color fringing real lens dispersion produces.
    """
    if isinstance(channel_plane, torch.Tensor):
        height, width = channel_plane.shape
        x_coords = torch.clamp(x_coords, 0.0, width - 1.0)
        y_coords = torch.clamp(y_coords, 0.0, height - 1.0)

        x0 = torch.floor(x_coords).long()
        y0 = torch.floor(y_coords).long()
        x1 = torch.clamp(x0 + 1, max=width - 1)
        y1 = torch.clamp(y0 + 1, max=height - 1)

        fx = x_coords - x0.to(x_coords.dtype)
        fy = y_coords - y0.to(y_coords.dtype)

        top = channel_plane[y0, x0] * (1.0 - fx) + channel_plane[y0, x1] * fx
        bottom = channel_plane[y1, x0] * (1.0 - fx) + channel_plane[y1, x1] * fx
        return top * (1.0 - fy) + bottom * fy

    height, width = channel_plane.shape
    x_coords = np.clip(x_coords, 0.0, width - 1.0)
    y_coords = np.clip(y_coords, 0.0, height - 1.0)

    x0 = np.floor(x_coords).astype("int32")
    y0 = np.floor(y_coords).astype("int32")
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)

    fx = (x_coords - x0)
    fy = (y_coords - y0)

    top = channel_plane[y0, x0] * (1.0 - fx) + channel_plane[y0, x1] * fx
    bottom = channel_plane[y1, x0] * (1.0 - fx) + channel_plane[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


def _apply_chromatic_aberration(frame, strength):
    """Shifts the red channel outward and the blue channel inward, more so towards the frame edges, like lens dispersion."""
    height, width = frame.shape[0], frame.shape[1]

    # Cubic (not linear) radial falloff: real lens CA stays
    # nearly invisible through most of the center of frame and ramps up
    # disproportionately fast only in the outer 20-30% - `radial_distance
    # ** 3` concentrates the shift at the extreme edges instead of
    # spreading it evenly across the whole frame the way a linear falloff
    # does, matching how real lenses (and VFX chromatic-aberration tools
    # that deliberately match them) actually look.

    if isinstance(frame, torch.Tensor):
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, dtype=frame.dtype, device=frame.device),
            torch.arange(width, dtype=frame.dtype, device=frame.device),
            indexing="ij",
        )
        center_x, center_y = width / 2.0, height / 2.0
        dx, dy = (x_coords - center_x) / center_x, (y_coords - center_y) / center_y
        radial_distance = torch.sqrt(dx * dx + dy * dy)

        shift = (radial_distance ** 3) * strength
        shift_x = dx * shift
        shift_y = dy * shift

        def shifted_channel_torch(channel_index, sign):
            src_x = x_coords + sign * shift_x
            src_y = y_coords + sign * shift_y
            return _bilinear_sample_channel(frame[:, :, channel_index], src_x, src_y)

        result = frame.clone()
        result[:, :, 0] = shifted_channel_torch(0, -1)   # red pushed outward
        result[:, :, 2] = shifted_channel_torch(2, 1)    # blue pulled inward
        return result

    y_coords, x_coords = np.mgrid[0:height, 0:width].astype("float32")
    center_x, center_y = width / 2.0, height / 2.0
    dx, dy = (x_coords - center_x) / center_x, (y_coords - center_y) / center_y
    radial_distance = np.sqrt(dx * dx + dy * dy)

    shift = (radial_distance ** 3) * strength
    # Kept as float32 (no int32 truncation) - fractional shift_x/shift_y
    # feed directly into the bilinear sampler below.
    shift_x = dx * shift
    shift_y = dy * shift

    def shifted_channel(channel_index, sign):
        src_x = x_coords + sign * shift_x
        src_y = y_coords + sign * shift_y
        return _bilinear_sample_channel(frame[:, :, channel_index], src_x, src_y)

    result = frame.copy()
    result[:, :, 0] = shifted_channel(0, -1)   # red pushed outward
    result[:, :, 2] = shifted_channel(2, 1)    # blue pulled inward
    return result


_MOTION_BLUR_SAMPLES = 6  # shift-and-average sample count for the directional streak


def _apply_motion_blur(frame, motion_vector, strength):
    """
    Streaks `frame` along the reverse of `motion_vector` (this frame's own
    camera-motion pixel displacement - see camera.py's
    compute_global_drift_pixel_vectors) via a standard real-time
    "velocity-buffer" shift-and-average: samples the CURRENT frame at
    _MOTION_BLUR_SAMPLES evenly-spaced points between "no shift" and "the
    full motion vector scaled by strength", then averages them - replacing
    the old flat `frame * (1 - strength) + previous_frame * strength`
    whole-frame blend (see this module's docstring for why). t=0 (the
    unshifted frame itself) is always included, so strength only ever
    ADDS streaking on top of the sharp frame, never removes detail from it
    the way the old blend's `frame * (1 - strength)` term did.

    Returns `frame` unchanged whenever there is nothing to blur: strength
    is zero/negative, no motion_vector was supplied, or the vector is
    exactly (0, 0) - e.g. a frame where the camera's own drift is
    momentarily at rest, near the "breathe" envelope's turning points.
    """
    if strength <= 0.0 or motion_vector is None:
        return frame
    dx, dy = motion_vector
    if dx == 0.0 and dy == 0.0:
        return frame

    height, width = frame.shape[0], frame.shape[1]
    is_tensor = isinstance(frame, torch.Tensor)

    if is_tensor:
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, dtype=frame.dtype, device=frame.device),
            torch.arange(width, dtype=frame.dtype, device=frame.device),
            indexing="ij",
        )
        accumulator = torch.zeros_like(frame)
    else:
        y_coords, x_coords = np.mgrid[0:height, 0:width].astype("float32")
        accumulator = np.zeros_like(frame)

    for sample_i in range(_MOTION_BLUR_SAMPLES):
        t = (sample_i / (_MOTION_BLUR_SAMPLES - 1)) * strength
        if t == 0.0:
            accumulator = accumulator + frame
            continue
        src_x = x_coords - t * dx
        src_y = y_coords - t * dy
        if is_tensor:
            sampled = torch.stack(
                [_bilinear_sample_channel(frame[:, :, c], src_x, src_y) for c in range(frame.shape[2])],
                dim=-1,
            )
        else:
            sampled = np.stack(
                [_bilinear_sample_channel(frame[:, :, c], src_x, src_y) for c in range(frame.shape[2])],
                axis=-1,
            )
        accumulator = accumulator + sampled

    return accumulator / _MOTION_BLUR_SAMPLES


def _apply_tonemap(frame, strength):
    """A gentle filmic S-curve (Reinhard-style) so highlights (bloom, lightning) compress smoothly instead of clipping to flat white."""
    # Luminance-preserving: apply the Reinhard curve to
    # overall luminance only, then rescale the original RGB by the ratio
    # of tonemapped-to-original luminance, instead of running the curve
    # on each of R/G/B independently. Per-channel Reinhard compresses
    # each channel at a different rate depending on its own value, which
    # shifts hue/saturation on bright, saturated highlights (a cool
    # blue-white lightning flash, warm god rays, a tinted lens flare -
    # exactly the saturated-at-high-brightness content this pipeline
    # produces plenty of); rescaling by the luminance ratio compresses
    # brightness uniformly while keeping hue and saturation intact.
    if isinstance(frame, torch.Tensor):
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=frame.dtype, device=frame.device)
        luminance = torch.tensordot(frame, weights, dims=([2], [0]))
        mapped_luminance = luminance / (1.0 + luminance)
        safe_luminance = torch.where(luminance > 1e-6, luminance, torch.ones_like(luminance))
        ratio = torch.where(luminance > 1e-6, mapped_luminance / safe_luminance, torch.ones_like(luminance))
        mapped = frame * ratio[..., None]
        return frame * (1.0 - strength) + mapped * strength

    luminance = np.tensordot(frame, np.array([0.299, 0.587, 0.114], dtype="float32"), axes=([2], [0]))
    mapped_luminance = luminance / (1.0 + luminance)
    # Guard against divide-by-zero on nearly-black pixels, where the
    # ratio is meaningless anyway (there's no brightness to preserve);
    # falling back to a ratio of 1.0 leaves those pixels unchanged by
    # this rescale, which is correct since mapped ~= luminance there.
    safe_luminance = np.where(luminance > 1e-6, luminance, 1.0)
    ratio = np.where(luminance > 1e-6, mapped_luminance / safe_luminance, 1.0)
    mapped = frame * ratio[..., None]
    return frame * (1.0 - strength) + mapped * strength


_GRAIN_SHOT_NOISE_WEIGHT = 0.85  # "a" in sigma(L) = sqrt(a*L + b)
_GRAIN_READ_NOISE_FLOOR = 0.15   # "b" in sigma(L) = sqrt(a*L + b); a+b=1.0 so sigma(1.0) == strength exactly


def _apply_grain(frame, strength, rng):
    """
    Adds a fresh (per-frame, ANIMATED - not a static overlay, which would
    read as a dirty lens rather than film) fine-noise layer - fixes
    every layer in this pipeline is mathematically
    smooth/noise-free at the pixel level, which human vision reads as
    "rendered" rather than "photographed."

    Mostly luminance noise (real film grain is largely a brightness
    phenomenon, not a color one) with a much smaller independent
    per-channel component for a little natural chroma variation, rather
    than fully independent per-channel noise (which would look more like
    sensor color noise than film grain).

    real sensor/film noise is dominated by photon SHOT
    noise, whose variance scales with the signal itself, plus a small
    constant READ-noise floor - not one flat standard deviation regardless
    of brightness. Scales the noise per-pixel by
    sigma_field = sqrt(_GRAIN_SHOT_NOISE_WEIGHT * luminance +
    _GRAIN_READ_NOISE_FLOOR), with the two weights summing to 1.0 so a
    pixel at luminance 1.0 still gets exactly `strength` as its std-dev -
    identical to the old flat behavior at that one reference point, with
    darker pixels falling toward a nonzero floor (still visibly grainy,
    never zero) and brighter pixels getting more ABSOLUTE noise but far
    less relative to their own signal - the shadows-grainy/highlights-
    clean look real film has, which the old flat noise couldn't produce.
    """
    height, width = frame.shape[0], frame.shape[1]
    luminance_weights = np.array([0.299, 0.587, 0.114], dtype="float32")

    if isinstance(frame, torch.Tensor):
        weights_t = torch.tensor([0.299, 0.587, 0.114], dtype=frame.dtype, device=frame.device)
        luminance = torch.clamp(torch.tensordot(frame, weights_t, dims=([2], [0])), min=0.0)
        sigma_field = torch.sqrt(_GRAIN_SHOT_NOISE_WEIGHT * luminance + _GRAIN_READ_NOISE_FLOOR)[..., None]
        sigma_field = sigma_field.detach().cpu().numpy().astype("float32")
    else:
        luminance = np.clip(np.tensordot(frame, luminance_weights, axes=([2], [0])), 0.0, None)
        sigma_field = np.sqrt(_GRAIN_SHOT_NOISE_WEIGHT * luminance + _GRAIN_READ_NOISE_FLOOR)[..., None]

    unit_luminance_noise = rng.normal(0.0, strength, (height, width, 1)).astype("float32")
    unit_chroma_noise = rng.normal(0.0, strength * 0.25, (height, width, 3)).astype("float32")
    luminance_noise = unit_luminance_noise * sigma_field
    chroma_noise = unit_chroma_noise * sigma_field

    if isinstance(frame, torch.Tensor):
        # The random values themselves are still drawn from the same numpy
        # `rng` (bit-for-bit identical noise regardless of device) - only
        # the addition happens on-device, via a single host->device copy of
        # the (small, per-frame) noise arrays rather than a torch-native RNG
        # that would produce different numbers than the numpy path.
        luminance_noise = torch.from_numpy(luminance_noise).to(frame.device)
        chroma_noise = torch.from_numpy(chroma_noise).to(frame.device)
        return frame + luminance_noise + chroma_noise

    return frame + luminance_noise + chroma_noise
