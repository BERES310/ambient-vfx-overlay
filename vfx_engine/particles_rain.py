"""
particles_rain.py

Ready-to-use rain effect - a thin settings "wrapper" around the shared
particle engine (particles.py), rendered with its own dedicated shader
(shaders/particles_rain.glsl - refraction, streak/bokeh blending,
dual-tone lighting) since rain and snow need different-enough rendering
that sharing one fragment shader stopped being worth it. See
particles_snow.py for the snow equivalent.
"""

import numpy as np
from PIL import Image, ImageFilter

import config
from vfx_engine import lens_dirt
from vfx_engine.particles import compute_circle_of_confusion, render_particles, sample_ambient_tint

_INTENSITY_PRESETS = {
    "light": {"count": 300, "speed_range": (8.0, 11.0), "size_range": (2, 4), "drift": 0.1},
    "medium": {"count": 1400, "speed_range": (10.0, 14.0), "size_range": (1.5, 3.5), "drift": 0.1},
    # 2026-09-22: count was 4500 - dense enough (especially once doubled up
    # by the near+far layered-rain pass, see config.RAIN_NEAR_LAYER/
    # RAIN_FAR_LAYER's count_scale, which together multiply this base count
    # by ~1.6x more streaks on screen) to fully occlude the photo behind it
    # rather than reading as heavy rain over a visible scene. A first pass
    # at count=2200 was verified by render and still looked like a near-total
    # whiteout, because the faster speed_range above (also a 2026-09-22 fix)
    # stretches each drop's motion-blur streak longer, which claws back
    # most of the screen-coverage reduction a raw count cut buys. Cut much
    # harder instead; speed_range stays raised so storm still reads as more
    # intense than medium via fall speed, not via opacity/coverage.
    "storm": {"count": 850, "speed_range": (30.0, 48.0), "size_range": (1.5, 4.5), "drift": 0.3},
}


def _gaussian_blur_rgba_premultiplied(frame, radius):
    """
    blurs a transparent RGBA frame with a real Gaussian
    kernel, using premultiplied alpha so the blur doesn't pull the
    (meaningless) RGB color sitting under a fully-transparent pixel into
    a blurred edge and darken/fringe it - a plain per-channel
    Image.filter(GaussianBlur) on STRAIGHT (non-premultiplied) RGBA does
    exactly that, since a transparent pixel's RGB is arbitrary but still
    gets averaged in at full weight.

    The premultiply/un-premultiply math itself runs in float32 (plain
    numpy, no PIL involved) so it doesn't compound rounding error across
    the divide-back-out step. Only the actual Gaussian convolution goes
    through PIL's accelerated GaussianBlur, which only accepts 8-bit
    modes ("F"/32-bit-float mode is rejected by PIL's C blur path with
    "image has wrong mode") - and since the source frame is itself 8-bit
    RGBA to begin with, quantizing the premultiplied channels to 8-bit
    for the blur step loses no precision beyond what the source already
    has.
    """
    if radius <= 0.05:
        return frame
    arr = np.asarray(frame, dtype=np.float32)
    alpha = arr[..., 3] / 255.0
    blur = ImageFilter.GaussianBlur(radius=radius)

    alpha_u8 = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
    alpha_img = Image.fromarray(alpha_u8, mode="L").filter(blur)
    blurred_alpha = np.asarray(alpha_img, dtype=np.float32)

    blurred_channels = []
    for c in range(3):
        premultiplied_u8 = np.clip(arr[..., c] * alpha, 0.0, 255.0).astype(np.uint8)
        premultiplied_img = Image.fromarray(premultiplied_u8, mode="L").filter(blur)
        blurred_channels.append(np.asarray(premultiplied_img, dtype=np.float32))

    # Un-premultiply: divide the blurred premultiplied color back by the
    # blurred alpha (normalized to 0..1 for the division). Where blurred
    # alpha is ~0 (a patch of frame that was fully transparent even after
    # blurring, e.g. far from any drop), the color is meaningless and
    # never gets shown anyway - substitute 1.0 to avoid a
    # division-by-zero warning, not because the result matters.
    blurred_alpha_norm = blurred_alpha / 255.0
    safe_alpha_norm = np.where(blurred_alpha_norm > 1e-6, blurred_alpha_norm, 1.0)
    rgb = np.stack([channel / safe_alpha_norm for channel in blurred_channels], axis=-1)
    rgb = np.clip(rgb, 0.0, 255.0)
    out_alpha = np.clip(blurred_alpha, 0.0, 255.0)
    out = np.dstack([rgb, out_alpha]).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def _apply_regional_bokeh_blur(frame, depth_array, focus_depth, bokeh_factor, size, num_bands):
    """
    real depth-keyed variable-radius Gaussian blur pass for
    rain's depth-of-field, replacing the old "smaller and more transparent
    = blurry" approximation (still applied upstream via particle
    size/opacity scaling, see particles.py's _setup_particles) with an
    ACTUAL blur kernel on top of it: real out-of-focus points of light
    spread into soft discs with characteristic falloff, they don't just
    shrink and fade.

    Buckets the frame into `num_bands` depth-of-defocus levels (the same
    depth_array/focus_depth driving the upstream size/opacity scaling, so
    both effects agree on which drops are "out of focus"), blurs the
    WHOLE frame once per band at that band's characteristic radius, then
    picks each pixel's result by linearly interpolating between its two
    nearest bands' blurred output based on its own defocus amount - a
    small, fixed number of full-frame blur passes (num_bands, not one per
    particle), with the per-pixel interpolation keeping a low band count
    from reading as a visibly stepped set of rings instead of a smooth
    depth-of-field falloff.
    """
    if bokeh_factor <= 0.0 or focus_depth is None or depth_array is None or num_bands < 2:
        return frame

    depth = np.asarray(depth_array, dtype=np.float32)
    if depth.shape != (size[1], size[0]):
        depth_img = Image.fromarray(np.clip(depth, 0.0, 1.0), mode="F").resize(size, Image.BILINEAR)
        depth = np.asarray(depth_img, dtype=np.float32)

    # Same real thin-lens circle-of-confusion formula as particles.py's
    # _setup_particles - the original design explicitly
    # complements that per-particle size/opacity scaling with this actual
    # blur kernel, not replaces it, so both need to agree on which drops
    # count as "out of focus" and by how much; sharing one function (not
    # two independently-tuned copies of the same formula) is what
    # guarantees that agreement stays exact. 0.0 = perfectly in focus,
    # 1.0 = maximally blurred.
    defocus = compute_circle_of_confusion(depth, focus_depth)

    max_radius_px = bokeh_factor * config.RAIN_BOKEH_BLUR_MAX_RADIUS_PX
    if max_radius_px <= 0.05:
        return frame

    band_centers = np.linspace(0.0, 1.0, num_bands)
    band_frames = [_gaussian_blur_rgba_premultiplied(frame, float(center) * max_radius_px)
                   for center in band_centers]
    band_stack = np.stack([np.asarray(f, dtype=np.float32) for f in band_frames], axis=0)

    idx_f = defocus * (num_bands - 1)
    idx_lo = np.clip(np.floor(idx_f).astype(np.int32), 0, num_bands - 2)
    idx_hi = idx_lo + 1
    frac = (idx_f - idx_lo)[..., None]

    rows, cols = np.indices(defocus.shape)
    lo_vals = band_stack[idx_lo, rows, cols]
    hi_vals = band_stack[idx_hi, rows, cols]
    result = lo_vals * (1.0 - frac) + hi_vals * frac
    return Image.fromarray(np.clip(result, 0.0, 255.0).astype(np.uint8), mode="RGBA")


def _bokeh_blurred_frames(frames, depth_array, focus_depth, bokeh_factor, size, num_bands):
    for frame in frames:
        yield _apply_regional_bokeh_blur(frame, depth_array, focus_depth, bokeh_factor, size, num_bands)


def render_rain(size, num_frames, intensity="medium", angle=0, blur=True,
                 depth_array=None, base_image=None, haze_strength=0.35,
                 focus_depth=None, bokeh_factor=0.0, backlight_strength=0.4,
                 shadow_strength=0.0,
                 refraction_strength=0.35, color=None, light_pos_uv=None,
                 motion_blur_samples=1, shutter_fraction=0.5,
                 base_image_frames=None, ambient_tint_grid=None,
                 velocity_stretch_scale=1.0, depth_bias_power=1.0, speed_scale=1.0,
                 seed=None, start_frame=0):
    """
    Renders a falling-rain animation generator and returns transparent
    frames (RGBA), ready to be applied to a photo.

    intensity: "light" / "medium" / "storm" - how many drops and how fast
        they fall. Can also be a dict with the same keys as
        _INTENSITY_PRESETS above, to fully customize a preset without
        editing this file.
    angle: drop tilt in degrees from vertical (wind-driven rain)
    blur: whether each drop trails a motion streak. The shader (see
        shaders/particles_rain.glsl) blends each drop between a thin
        streak (near the focus plane, moving fast) and a soft round
        bokeh disc (far out of focus) based on focus_depth/bokeh_factor
        below - the geometry only needs a moderate stretch/ghost-trail to
        seed that blend, not to carry the whole streak look by itself.
    depth_array/base_image/haze_strength: see render_particles - ties drop
        size/speed/opacity to the photo's real scene depth (when
        depth_array is given); haze_strength blends far drops toward the
        photo's actual sky color (not the exact pixel behind them).
    focus_depth/bokeh_factor: see render_particles - a real camera focus
        plane (drops away from it blur AND grow into soft bokeh discs,
        rendered by the shader's streak/bokeh blend above) instead of
        "far = blurry, near = sharp" always. None = the old behavior.
    backlight_strength: see render_particles - drops glow brighter where
        there's something bright directly behind them. Defaults to 0.4
        (not 0.0) as of at 0.0 the shader's own backlight-
        glow-against-bright-sky block (see particles_rain.glsl) was
        provably dead code for every caller that didn't explicitly pass
        this, which is exactly why rain against a bright hazy sky read as
        flat dark streaks instead of catching any light.
    shadow_strength: see render_particles - symmetric counterpart to
        backlight_strength: drops glow DIMMER where there's something dark
        (deep shade) directly behind them, fixing the "rain brighter than
        the shadow behind it" bug.
    refraction_strength: 0.0-1.0, how strongly each drop bends the
        background behind it, like real water refracting light - a real
        lens/optical interaction instead of a flat tinted sprite. Needs
        base_image (uses the same texture as haze/backlight sampling).
    color: overrides the automatic ambient-tint color (see
        sample_ambient_tint) - by default, drop color is sampled from the
        photo's own lighting (warmer at sunset, cooler under overcast)
        instead of a fixed blue-grey regardless of the scene.
    light_pos_uv: see render_particles - ties each drop's dual-tone
        highlight to the photo's real detected light direction instead of
        a fixed "light comes from above" assumption.
    motion_blur_samples/shutter_fraction: see render_particles - REAL
        sub-frame motion blur, replacing postprocess.py's
        old whole-frame ghosting blend for fast rain streaks specifically.
        1 (default) = off.
    base_image_frames: see render_particles - an optional
        iterable yielding one per-frame camera-drifted PIL Image (e.g.
        vfx_engine.camera.apply_camera_drift), used INSTEAD of the static
        base_image for atmosphere/backlight/refraction sampling so drops
        stay in registration with a simultaneously camera-drifting photo.
        None (default) = old behavior (samples the static base_image).
    ambient_tint_grid: see render_particles - the original design, an optional
        coarse per-photo color grid (see particles.build_ambient_tint_grid)
        that gives each drop its OWN local color based on where it falls on
        screen, instead of every drop sharing one flat `color`/drop_color.
        None (default) = old behavior (flat color for every drop). Callers
        should compute this ONCE per photo and pass the same grid used for
        splashes (see ground_fx.render_splashes), so both effects agree.
    velocity_stretch_scale: multiplies the internal
        streak-seed velocity_stretch (9.0 when blur=True, 0.0 otherwise),
        so a caller building a separate "near" and "far" rain layer can
        give the near layer a longer motion-blur seed than the far layer
        without touching the blur flag itself. 1.0 (default) = old
        behavior, unscaled.
    depth_bias_power: if depth_array is given and this is
        not 1.0, the array is raised to this power (after clamping to
        [0, 1]) before being handed to render_particles, which skews the
        REAL per-pixel depth used for size/speed/opacity toward the near
        end (power < 1) or the far end (power > 1) while preserving each
        pixel's relative depth ordering. Lets two independently-configured
        rain passes (see main.render_layered_rain) both sample the same
        underlying depth map but read as a distinct "near" layer and a
        distinct "far" layer instead of identical single-layer rain.
        1.0 (default) = old behavior, unbiased.
    speed_scale: user-requested global fall-speed knob (2026-09-22) -
        multiplies preset["speed_range"] (both ends) before it reaches
        render_particles, independent of which intensity preset is chosen
        and independent of the near/far layers' own relative speed_scale
        (see config.RAIN_NEAR_LAYER/RAIN_FAR_LAYER, which scale THIS
        already-scaled value further, not instead of it). 1.0 (default) =
        old behavior, unscaled.

    when focus_depth/bokeh_factor/depth_array are all given,
    the returned frames also get a real post-render depth-keyed Gaussian
    blur pass (see _apply_regional_bokeh_blur) on top of render_particles'
    existing size/opacity-based defocus approximation - an actual blur
    kernel, not just smaller/fainter drops, so out-of-focus rain reads as
    genuinely soft instead of just shrunken. Gated behind
    config.RAIN_BOKEH_BLUR_BANDS (>=2 required) since it's a real added
    per-frame cost (a handful of extra full-frame blurs).

    seed: RNG-determinism audit (2026-09-17) - see render_particles's own
        `seed` doc. None (the default) here additionally falls back to a
        seed derived from base_image's own content (when given) via
        lens_dirt.content_seed, salted "rain" so a snow pass seeded from
        the same photo draws an independent spawn pattern - rather than
        render_particles' own default of fresh OS entropy every call.
        Pass an explicit seed to override, or base_image=None with no
        seed to keep the old non-deterministic behavior.
    start_frame: render-checkpoint/resume support - see render_particles's
        own `start_frame` doc; passed straight through unchanged. 0
        (default) = old behavior. Note the same caller contract applies
        to base_image_frames here as in render_particles: if it's given,
        it must already start at start_frame - this thin wrapper does no
        fast-forwarding of its own.
    """
    if depth_array is not None and depth_bias_power != 1.0:
        depth_array = np.clip(depth_array, 0.0, 1.0) ** depth_bias_power
    preset = intensity if isinstance(intensity, dict) else _INTENSITY_PRESETS.get(intensity, _INTENSITY_PRESETS["medium"])
    speed_lo, speed_hi = preset["speed_range"]
    scaled_speed_range = (speed_lo * speed_scale, speed_hi * speed_scale)
    drop_color = color or sample_ambient_tint(base_image, fallback=(190, 205, 230))
    if seed is None and base_image is not None:
        seed = lens_dirt.content_seed(base_image, salt="rain")

    frames = render_particles(
        size, num_frames,
        count=preset["count"],
        color=drop_color,
        size_range=preset["size_range"],
        speed_range=scaled_speed_range,
        drift=preset["drift"],
        angle_deg=angle,
        shader_file="particles_rain.glsl",
        # A short ghost trail plus a moderate velocity stretch is now just
        # the SEED geometry the shader's streak/bokeh blend works from
        # (see shaders/particles_rain.glsl) - it doesn't have to carry the
        # whole streak look by itself the way it used to, so both are
        # toned down from earlier, heavier values. # streak_count used to double to 2 ghost-copies when blur=True,
        # which on top of velocity_stretch and the shader's own blend read
        # as "too many rigid parallel lines" per drop - a single seed row
        # is enough now that the stretch/blend actually carries the look.
        streak_count=1,
        streak_spacing=3,
        velocity_stretch=(11.0 if blur else 0.0) * velocity_stretch_scale,
        # Real rain isn't a flat spread of drop sizes - most drops are
        # small with an occasional bigger one (Marshall-Palmer-like
        # distribution), which reads as far more natural than a uniform
        # random size within size_range alone.
        size_skew=0.6,
        # Rain doesn't tumble like snow - no rotational wobble.
        wobble_strength=0.0,
        speed_wobble=0.1,
        density_clumping=0.35,
        # A fast streak spreads its light over more of the frame per
        # exposure than a slow one - reads a little fainter in a real
        # photo, not just longer. Rain-only nuance (see render_particles).
        speed_exposure_loss=0.4,
        # ties an additional opacity fade directly
        # to blur_amount (defocus/recession), on top of the mild
        # depth-linked floor base_opacity already has - so a drop genuinely
        # dissolving into background haze/defocus also visibly loses
        # solidity instead of staying a crisp, fully-opaque streak that's
        # merely softer-edged.
        depth_alpha_falloff=0.5,
        depth_array=depth_array,
        base_image=base_image,
        haze_strength=haze_strength,
        focus_depth=focus_depth,
        bokeh_factor=bokeh_factor,
        backlight_strength=backlight_strength,
        shadow_strength=shadow_strength,
        refraction_strength=refraction_strength,
        light_pos_uv=light_pos_uv,
        motion_blur_samples=motion_blur_samples,
        shutter_fraction=shutter_fraction,
        base_image_frames=base_image_frames,
        ambient_tint_grid=ambient_tint_grid,
        seed=seed,
        start_frame=start_frame,
        # Each drop keeps ONE distance for its whole life (drawn from the
        # scene's depth distribution) and the shader hides drops that fall
        # behind a nearer scene surface - real depth layering.
        persistent_depth=depth_array is not None,
        occlusion_strength=getattr(config, "RAIN_DEPTH_OCCLUSION_STRENGTH", 0.75) if depth_array is not None else 0.0,
    )

    if (focus_depth is not None and bokeh_factor > 0.0 and depth_array is not None
            and config.RAIN_BOKEH_BLUR_BANDS >= 2):
        frames = _bokeh_blurred_frames(
            frames, depth_array, focus_depth, bokeh_factor, size, config.RAIN_BOKEH_BLUR_BANDS,
        )
    return frames
