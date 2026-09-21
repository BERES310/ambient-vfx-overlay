"""
particles_snow.py

Ready-to-use snow effect - a thin settings "wrapper" around the shared
particle engine (particles.py), rendered with its own dedicated shader
(shaders/particles_snow.glsl - ring-bokeh defocus, sky-color haze,
directional sparkle bias) since rain and snow need different-enough
rendering that sharing one fragment shader stopped being worth it. See
particles_rain.py for the rain equivalent.
"""

from vfx_engine import lens_dirt
from vfx_engine.particles import render_particles, sample_ambient_tint


def _snow_color(scene_tint):
    """
    Snow is white lit by the scene, not a mid-grey copy of the scene's
    average color: sample_ambient_tint() returns the photo's *average*
    tone (dark for a forest), which drew flakes as dim grey blobs against
    a bright sky. Keeps only the tint's hue (so a sunset still warms the
    flakes a little) and lifts it to a bright near-white.
    """
    tint = [float(channel) for channel in scene_tint]
    peak = max(max(tint), 1.0)
    hue = [channel / peak for channel in tint]                  # brightest channel = 1.0
    return tuple(int(round(250 * (0.55 + 0.45 * channel))) if channel < 1.0 else 250 for channel in hue)


def render_snow(size, num_frames, density=0.6, wind=0.4, depth=True,
                 depth_array=None, base_image=None, haze_strength=0.35,
                 focus_depth=None, bokeh_factor=0.0, backlight_strength=0.0, shadow_strength=0.0,
                 color=None, light_pos_uv=None, motion_blur_samples=1, shutter_fraction=0.5,
                 base_image_frames=None, seed=None, start_frame=0):
    """
    Renders a falling-snow animation generator and returns transparent
    frames (RGBA), ready to be applied to a photo.

    density: 0.0-1.0, how many flakes are on screen at once
    wind: 0.0-1.0, how strongly flakes sway sideways while falling
    depth: vary size/speed/sharpness based on simulated distance from the
           camera (False = every flake looks the same)
    depth_array/base_image/haze_strength: see render_particles - ties
        flake size/speed/opacity to the photo's real scene depth (when
        depth_array is given); haze_strength blends far flakes toward the
        photo's actual sky color (not the exact pixel behind them - see
        shaders/particles_snow.glsl - a flake in front of dark trees
        shouldn't turn dark green).
    focus_depth/bokeh_factor: see render_particles - a real camera focus
        plane (flakes away from it blur AND grow into soft ring-bokeh
        discs, see the shader) instead of "far = blurry, near = sharp"
        always. None = the old behavior.
    backlight_strength: see render_particles - flakes glow brighter where
        there's something bright directly behind them.
    shadow_strength: see render_particles - symmetric counterpart to
        backlight_strength: flakes glow DIMMER where there's something dark
        (deep shade) directly behind them, fixing the "snow brighter than
        the shadow behind it" bug.
    color: overrides the automatic ambient-tint color (see
        sample_ambient_tint) - by default, flake color is sampled from
        the photo's own lighting (snow reflects ambient light, it isn't a
        fixed pure white regardless of the scene) instead of flat white.
    light_pos_uv: normalized (x, y) screen position of the photo's actual
        detected dominant light source - see render_rain's matching
        parameter. particles_snow.glsl now has its own
        dual-tone lighting hook (ported from particles_rain.glsl), so this
        does real work here too - a broader, softer, lower-contrast
        warm/cool tint across each flake's rotated shape than rain's
        tighter specular-like hook, matching how ice/snow scatters light
        far more diffusely than a wet raindrop's near-mirror surface.
        None (default) = the old fixed "light comes from directly above"
        assumption.
    motion_blur_samples/shutter_fraction: see render_particles - REAL
        sub-frame motion blur. 1 (default) = off.
    base_image_frames: see render_particles - an optional
        iterable yielding one per-frame camera-drifted PIL Image (e.g.
        vfx_engine.camera.apply_camera_drift), used INSTEAD of the static
        base_image for atmosphere/backlight sampling so flakes stay in
        registration with a simultaneously camera-drifting photo. None
        (default) = old behavior (samples the static base_image).
    seed: RNG-determinism audit (2026-09-17) - see render_particles's own
        `seed` doc. None (the default) here additionally falls back to a
        seed derived from base_image's own content (when given) via
        lens_dirt.content_seed, salted "snow" so a rain pass seeded from
        the same photo draws an independent spawn pattern - rather than
        render_particles' own default of fresh OS entropy every call.
    start_frame: render-checkpoint/resume support - see render_particles's
        own `start_frame` doc; passed straight through unchanged. 0
        (default) = old behavior. Note the same caller contract applies
        to base_image_frames here as in render_particles: if it's given,
        it must already start at start_frame - this thin wrapper does no
        fast-forwarding of its own.
    """
    count = max(60, int(1100 * density))   # a real snowfall is hundreds-to-thousands of flakes per frame
    # A slightly desaturated cool grey reads more like real photographic
    # snow than pure white, and is a saner fallback if base_image/tint
    # sampling isn't available for some reason.
    flake_color = color or _snow_color(sample_ambient_tint(base_image, fallback=(230, 235, 245), sky_weight=0.4))
    if seed is None and base_image is not None:
        seed = lens_dirt.content_seed(base_image, salt="snow")

    return render_particles(
        size, num_frames,
        count=count,
        color=flake_color,
        size_range=(4, 10),
        speed_range=(0.8, 2.5),
        drift=wind * 1.5,
        use_depth=depth,
        shader_file="particles_snow.glsl",
        # Snowflakes tumble as they fall (unlike rain drops) - a slow
        # rotational wobble on top of the wind-direction tilt. Toned down
        # slightly from earlier for a calmer, more fluid motion.
        wobble_strength=0.45,
        speed_wobble=0.15,
        density_clumping=0.25,
        # Most flakes are small with an occasional bigger foreground one
        # (the same Marshall-Palmer-like logic rain uses for drop size),
        # rather than a flat uniform spread across size_range.
        size_skew=0.4,
        # Sparkle is now direction-biased in the shader (see
        # particles_snow.glsl - stronger near the top/key-light side of
        # the frame), so the same visual "some flakes glint" effect needs
        # a lower base strength than before to avoid over-sparkling.
        sparkle_strength=0.22,
        # Snow has no angle_deg tilt, so without this each particle would
        # only ever sway locally around its random spawn X for the whole
        # video - any unevenness in that initial random spread would read
        # as a permanent, wrong-looking dense patch instead of mixing away.
        horizontal_wander=1.0,
        depth_array=depth_array,
        base_image=base_image,
        haze_strength=haze_strength,
        focus_depth=focus_depth,
        bokeh_factor=bokeh_factor,
        backlight_strength=backlight_strength,
        shadow_strength=shadow_strength,
        light_pos_uv=light_pos_uv,
        motion_blur_samples=motion_blur_samples,
        shutter_fraction=shutter_fraction,
        base_image_frames=base_image_frames,
        seed=seed,
        start_frame=start_frame,
    )
