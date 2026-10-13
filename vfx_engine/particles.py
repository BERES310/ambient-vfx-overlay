"""
particles.py

Shared, GPU-rendered particle system that snow and rain are built on
(snow.py, rain.py) - the same thing could be reused for e.g. sparks or
fireflies if needed.

Particles are drawn as GPU-instanced quads (see shaders/particles.glsl) -
one small rectangle, reused (instanced) once per particle, rotated to
face its motion. Each particle has a "depth" (0 = far, 1 = near) - either
a random value, or, when a real scene depth map is supplied (see
depth_array below), sampled from the actual photo at the particle's own
position - which its size, speed and sharpness depend on: particles closer
to the camera are bigger, faster and sharper, while ones "further away"
are smaller, slower, dimmer and blurrier (a simple depth-of-field effect),
and - when atmospheric perspective is enabled - blend toward whatever's
actually behind them in the photo, the same way real distant objects
desaturate into haze.

Per-frame particle position is computed from a "curl noise" wind field
(spatially-coherent - nearby particles sway together in the same gust,
unlike an independent per-particle sine wobble), plus a periodic speed
wobble (approximating drag/gust response) and, for tumbling particles
(snow), a rotational wobble. All of it is built to close an exact loop
over `num_frames` frames with no drift - see shaders/particles_wind.comp's
docstring for the full explanation. That math is identical on both paths
below, just running in two different places:
    - GPU path (shaders/particles_wind.comp): a compute shader updates
      every particle's position/rotation directly on the GPU, with no
      per-frame round trip to the CPU. Used when the GPU/driver supports
      compute shaders (OpenGL 4.3+, see gpu.supports_compute()).
    - CPU path (_cpu_dynamic_data below): the exact same formulas,
      vectorized with numpy. Used as a fallback on older GPUs/drivers.

The `drift` (wind strength) passed into render_particles is also scaled,
frame by frame, by the shared gust envelope (see vfx_engine/weather.py) -
the same envelope fog, vegetation sway and cloud drift all sample too, so
a gust visibly leans rain/snow harder in step with trees swaying and fog
surging, instead of each effect's wind pulsing on its own unrelated
schedule.
"""

import math

import moderngl
import numpy as np
from PIL import Image

import config
from vfx_engine import gpu, weather
from vfx_engine import color as _color

# The resolution that the size_range/count values in snow.py/rain.py were
# tuned for. Particle size is automatically rescaled relative to this
# resolution (by the image's diagonal, see _auto_size_scale), and particle
# COUNT is automatically rescaled by its pixel area (see
# _auto_density_scale), so particles don't disappear (or look sparse) on
# large photos, or look oversized (or overcrowded) on small ones.
_REFERENCE_RESOLUTION = (1920, 1080)
_REFERENCE_AREA = _REFERENCE_RESOLUTION[0] * _REFERENCE_RESOLUTION[1]

# A single quad, centered on the origin, corners in -0.5..0.5 - instanced
# once per particle in particles.glsl. Ordered for GL_TRIANGLE_STRIP.
_QUAD_CORNERS = np.array([
    [-0.5, -0.5],
    [0.5, -0.5],
    [-0.5, 0.5],
    [0.5, 0.5],
], dtype="f4")

_COMPUTE_LOCAL_SIZE = 256   # must match `local_size_x` in particles_wind.comp


def _auto_size_scale(size):
    """
    Returns a particle-size multiplier based on the photo's resolution
    (relative to _REFERENCE_RESOLUTION), multiplied by
    config.PARTICLE_SIZE_SCALE (default 1.0 = 100%, e.g. 0.75 = 75% of the
    automatically chosen size).
    """
    reference_diagonal = math.hypot(*_REFERENCE_RESOLUTION)
    image_diagonal = math.hypot(*size)
    return (image_diagonal / reference_diagonal) * config.PARTICLE_SIZE_SCALE


def _auto_density_scale(size):
    """
    Returns a particle-COUNT multiplier based on the photo's pixel area
    relative to _REFERENCE_RESOLUTION (see config.AUTO_DENSITY_SCALE),
    capped at config.MAX_DENSITY_SCALE. Density scaling by area (rather
    than by diagonal, like _auto_size_scale) is what keeps
    particles-per-visible-area roughly constant as photos get bigger -
    without it, a wide/large photo ends up with the same raw particle
    count as a small one, which reads as noticeably sparser.
    """
    if not config.AUTO_DENSITY_SCALE:
        return 1.0
    area_scale = (size[0] * size[1]) / _REFERENCE_AREA
    return min(area_scale, config.MAX_DENSITY_SCALE)


def render_particles(size, num_frames, count, color, size_range, speed_range,
                      drift=0.5, upward=False, glow=False, streak_count=1, streak_spacing=3,
                      use_depth=True, angle_deg=0.0, noise_scale=1.0 / 400.0,
                      speed_jitter=0.15, brightness_jitter=0.15, size_skew=0.0, edge_fade=True,
                      wind_mass_scaling=True, speed_wobble=0.08, wobble_strength=0.0,
                      noise_loop_radius=3.0, velocity_stretch=0.0, density_clumping=0.0,
                      depth_array=None, base_image=None, haze_strength=0.35, horizontal_wander=0.0,
                      shape="round", sparkle_strength=0.0, focus_depth=None, bokeh_factor=0.0,
                      speed_exposure_loss=0.0, backlight_strength=0.0, shadow_strength=0.0,
                      shader_file="particles.glsl", refraction_strength=0.0, haze_color=None, light_pos_uv=None,
                      motion_blur_samples=1, shutter_fraction=0.5, base_image_frames=None, ambient_tint_grid=None,
                      depth_alpha_falloff=0.0, seed=None, start_frame=0, persistent_depth=False,
                      occlusion_strength=0.0):
    """
    Generator - for each of the `num_frames` frames of particle animation
    (e.g. snowflakes or raindrops), yields a single transparent PIL frame
    (RGBA), ready to be applied to another image. Frames are rendered on
    the fly, one at a time - so long animations (see
    config.VIDEO_DURATION_SECONDS) don't need to hold every frame in memory
    at once.

    speed_jitter: 0.0-1.0+, extra per-particle random speed variation on
        top of the depth-based speed range, so particles at the same
        simulated depth don't all move at identical speed (nature doesn't
        obey a strict depth->speed function).
    brightness_jitter: 0.0-1.0+, per-particle random brightness variation,
        so particles aren't identically lit stamped copies of each other.
    size_skew: 0.0 = uniform size distribution within size_range (default).
        >0.0 skews the distribution so most particles stay near the small
        end with an occasional larger one (Marshall-Palmer-like rain drop
        size distribution) instead of a flat uniform spread - see rain.py.
    edge_fade: fades particles out near the top/bottom frame edges, hiding
        the "pop" of a particle wrapping back around to the opposite edge
        when it loops - purely position-based (not time-based), so it
        stays correct for any loop length instead of needing to match a
        specific crossfade window.
    wind_mass_scaling: True = smaller/lighter particles get pushed by wind
        more than larger/heavier ones (real air-drag behavior), instead of
        the same wind strength applying to every particle regardless of size.
    speed_wobble: 0.0-1.0+, periodic per-particle fall-speed variation
        (fast/slow pulses) approximating drag/gust response - a pure
        sinusoid of the loop phase, so it varies motion without breaking
        the loop's exact closure (see particles_wind.comp).
    wobble_strength: 0.0-1.0+, extra tumble/rotation wobble on top of the
        wind-direction tilt - meaningful mostly for tumbling particles like
        snow (see snow.py), 0.0 (default) for particles that shouldn't
        tumble (e.g. rain).
    noise_loop_radius: how far the wind noise's time-sampling circle
        reaches (see particles_wind.comp) - larger values give more wind
        variation over one loop; 0.0 makes the wind field static.
    velocity_stretch: 0.0 = round particles (default). >0.0 elongates each
        particle's sprite along its own fall direction, proportional to
        its speed relative to speed_range - a real velocity-streak look
        (see rain.py), can replace most or all of streak_count's
        ghost-copy trail approach.
    density_clumping: 0.0 = perfectly uniform particle density (default).
        >0.0 modulates each particle's opacity by a coherent noise field
        sampled at its spawn position, so some areas of the frame read as
        visibly denser "gusts" than others instead of uniform per-particle
        scatter.
    depth_array: optional numpy float32 array (height, width), 0.0 (far)
        to 1.0 (near), from vfx_engine.depth.estimate_depth() - when given
        (and use_depth=True), each particle's depth (and therefore its
        size/speed/opacity/blur) is sampled from the ACTUAL scene depth at
        its spawn position instead of a random value, so e.g. rain over
        the near foreground is genuinely bigger/faster than rain over a
        distant lake. Falls back to a random depth scalar when omitted.
    base_image: optional PIL Image (the actual photo) - required for
        atmospheric perspective (see haze_strength). When omitted,
        atmospheric perspective is skipped even if haze_strength > 0.
    base_image_frames: optional iterable yielding one per-frame drifted
        PIL Image (e.g. vfx_engine.camera.apply_camera_drift) - when
        given, this (not the static base_image) is what atmosphere/
        backlight/refraction actually sample from, updated once per
        OUTPUT frame.
        base_image itself is still used for the one-time sample_sky_color/
        sample_ambient_tint color derivation regardless.
    haze_strength: 0.0-1.0, how strongly far particles (per depth, see
        depth_array) blend into the photo's own background color behind
        them - real atmospheric perspective instead of just shrinking/
        dimming far particles. No effect without base_image.
    horizontal_wander: 0.0 = particles never travel horizontally except
        local wind sway around their spawn X (default - fine for effects
        with a nonzero angle_deg tilt, like rain, which already drifts
        sideways plenty via that tilt over a loop). >0.0 gives each
        particle its own net sideways drift across roughly that many
        screen-widths over the loop (quantized to a whole number, so it
        stays exactly loop-safe) - needed for effects with no angle tilt
        (e.g. snow, angle_deg=0), otherwise particles never leave the
        neighborhood of their random spawn X, and an uneven-by-chance
        initial spawn distribution reads as a persistent, wrong-looking
        dense patch for the entire length of a long loop instead of
        mixing away like it does vertically. See snow.py.
    shape: "round" (default) = a soft radial glow, good for snow/dust -
        real bright-center falloff, not a flat disk. "streak" = a thin
        tapered line with independent width/length falloff, for rain -
        without this, a stretched round particle just looks like a
        blurry oval, not a real rain streak. See particles.glsl.
    sparkle_strength: 0.0-1.0, "round" shape only - adds an occasional
        cross-shaped glint on brighter particles (some snow/dust visibly
        catches the light), 0.0 = no sparkle.
    focus_depth: None (default) = the old behavior, blur increases
        linearly as depth decreases (far = blurrier, near = sharpest).
        0.0-1.0 = a real camera focus PLANE at that depth - particles
        AWAY from it blur more in either direction (both too-near and
        too-far go soft), matching how actual camera depth-of-field
        works, instead of "far always blurry, near always sharp."
    bokeh_factor: 0.0 = off (default). >0.0, only meaningful together
        with focus_depth - particles far from the focus plane also grow
        physically larger (real out-of-focus bokeh discs are bigger, not
        just softer-edged, than in-focus points), proportional to how far
        out of focus they are.
    speed_exposure_loss: 0.0-1.0+, fades faster particles' opacity down a
        bit (relative to this effect's own speed_range) - a fast streak
        spreads its light across more of the frame per exposure than a
        slow one does, so in a real photo it reads a little fainter, not
        just longer/blurrier. Mainly a rain nuance; leave at 0.0 for snow.
    backlight_strength: 0.0-1.0, how much particles brighten when
        whatever's directly behind them (sampled from base_image, same as
        haze_strength's atmosphere sampling) is itself bright - a rough
        approximation of real forward light scattering (rain/snow glow
        noticeably brighter against a backlit sky or light source than
        against a dark backdrop). Requires base_image, same as haze_strength.
    shadow_strength: 0.0-1.0, symmetric counterpart to backlight_strength -
        DARKENS particles when whatever's directly behind them is itself
        dark (deep shade/shadow), instead of only ever brightening them.
        Fixes the "rain/snow brighter than the shadow behind it" bug: a
        one-directional backlight-only model left particles rendering at a
        fixed brightness regardless of how dark the background actually
        was, which reads as obviously composited over a shadowed area.
        Requires base_image, same as backlight_strength.
    motion_blur_samples: 1 (default) = the old behavior, one render per
        output frame. >1 = REAL sub-frame motion blur:
        renders `motion_blur_samples` sub-frames at fractional positions
        within a shutter window centered on each output frame_i (see
        shutter_fraction), decodes each to linear light, averages them,
        and re-encodes ONE motion-blurred frame - genuine sub-sample
        accumulation, not postprocess.py's old whole-frame ghosting blend
        (which also incorrectly smeared the static, non-moving photo
        behind the particles). Costs `motion_blur_samples` times the
        render/readback cost per frame - only meaningful because this
        pipeline already parametrizes particle position as a continuous
        function of frame_i (see particles_wind.comp's docstring), so a
        fractional frame_i is already exactly as valid an input as an
        integer one - no separate sub-frame architecture needed.
    shutter_fraction: 0.0-1.0, only meaningful when motion_blur_samples > 1
        - how much of one full frame's time span the sub-samples spread
        across (0.5 = a real camera's common "180-degree shutter",
        sub-samples span half a frame's worth of motion either side of
        the frame's own nominal instant).
    light_pos_uv: optional (x, y) normalized 0.0-1.0 screen position of
        the photo's actual dominant light source (see
        vfx_engine/god_rays.py's detect_light_source) - used by
        particles_rain.glsl's dual-tone lighting so each drop's warm
        highlight faces the real light direction instead of a fixed
        "light always comes from directly above" assumption. None (the
        default) falls back to that fixed assumption.
    ambient_tint_grid: optional (grid_h, grid_w, 3) uint8 array from
        sample_ambient_tint.build_ambient_tint_grid - the original design, local
        color sampling instead of one global tint. When given, uploaded as
        a small GL_LINEAR-filtered texture and sampled per-fragment at
        each particle's own screen position (particles_rain.glsl only -
        the plain particles.glsl/particles_snow.glsl shaders don't declare
        the matching uniforms, so this is a no-op for snow/other effects
        sharing this same function, by design: 3.3 scopes local tint to
        rain and splash only). None (the default) keeps the old behavior:
        every particle uses the flat `color` uniform regardless of where
        it lands.
    depth_alpha_falloff: 0.0 (default) = the old behavior, unaffected.
        >0.0 additionally fades a particle's opacity as its blur_amount
        rises (i.e. as it defocuses/recedes) - on top of the mild
        depth-linked floor base_opacity already has, this gives a
        stronger, blur-tied fade so a particle genuinely dissolving into
        background haze/defocus also visibly loses solidity, not just
        gets softer-edged at a constant opacity. Mainly a rain nuance
; leave at 0.0 for
        snow and other effects that shouldn't fade this way.
    seed: RNG-determinism audit (2026-09-17) - integer seed for this
        function's own spawn-position/jitter RNG (see the `rng =` line
        near the top of the setup block below). None (the default) keeps
        the old behavior: a fresh OS-entropy seed every call, meaning
        even re-rendering the SAME photo twice draws a different spawn
        pattern. Callers with a source photo available should pass
        lens_dirt.content_seed(base_image, salt=...) here instead, same
        reasoning/precedent as that function's own docstring - this is a
        pure function of frame_i once seeded, so a deterministic seed is
        also what makes the render-checkpoint/resume feature safe to
        resume mid-render without a visible discontinuity.
    start_frame: render-checkpoint/resume support - 0 (default) renders
        every frame from the start, exactly the old behavior. >0 skips
        yielding frames [0, start_frame) entirely, resuming output at
        frame start_frame, while still running every one-time setup step
        above (seed/static-data generation, shader/texture/VAO/FBO setup,
        compute-program setup) UNCONDITONALLY - those are cheap relative
        to the per-frame GPU render+readback cost this is meant to skip,
        and every frame's particle state is already a pure function of
        (frame_i, num_frames) plus that one-time setup (see the `seed`
        param above), so jumping straight to start_frame reproduces
        bit-identical frames from there onward instead of needing to
        replay frames [0, start_frame) first. IMPORTANT caller contract
        for base_image_frames: when animated_base is True, this function
        does NOT fast-forward base_frames_iter by calling next() on it
        start_frame times - that would waste exactly the per-frame camera-
        drift work start_frame is meant to let callers skip. Instead, the
        caller must construct/advance base_image_frames (e.g. via
        vfx_engine.camera.apply_camera_drift(..., start_frame=start_frame))
        so its FIRST yielded item already corresponds to output frame
        start_frame - this mirrors camera.apply_camera_drift's own
        start_frame contract exactly, and is the same "trust the animated
        iterable already begins at start_frame" convention used by this
        module's siblings (see vfx_engine/god_rays.py, lens_dirt.py,
        scene_warp.py).
    """
    size_scale = _auto_size_scale(size)
    size_range = (size_range[0] * size_scale, size_range[1] * size_scale)
    count = max(1, round(count * _auto_density_scale(size)))
    direction = -1.0 if upward else 1.0
    angle_slant = float(np.tan(np.radians(angle_deg)))

    seeds, static_data = _setup_particles(
        size, count, size_range, speed_range, use_depth, glow, streak_count, streak_spacing, direction,
        speed_jitter, brightness_jitter, size_skew, wind_mass_scaling, velocity_stretch,
        density_clumping, depth_array, horizontal_wander,
        focus_depth, bokeh_factor, speed_exposure_loss, depth_alpha_falloff,
        seed=seed, persistent_depth=persistent_depth,
    )
    total_instances = seeds.shape[0]

    context = gpu.get_context()
    program = gpu.load_shader_program(shader_file)

    def set_uniform(name, value):
        """Sets a uniform only if the compiled shader actually declares it - particles_rain.glsl/particles_snow.glsl each have their own uniform set, not every uniform exists in every shader."""
        try:
            program[name].value = value
        except KeyError:
            pass

    set_uniform("viewport_size", size)
    set_uniform("particle_color", tuple(channel / 255.0 for channel in color))
    set_uniform("edge_fade", 1.0 if edge_fade else 0.0)
    set_uniform("particle_shape", 1.0 if shape == "streak" else 0.0)
    set_uniform("sparkle_strength", sparkle_strength)
    set_uniform("refraction_strength", refraction_strength)
    set_uniform("has_light_pos", 1.0 if light_pos_uv is not None else 0.0)
    set_uniform("light_pos_uv", light_pos_uv if light_pos_uv is not None else (0.5, 0.0))

    use_atmosphere = base_image is not None and haze_strength > 0.0
    use_backlight = base_image is not None and backlight_strength > 0.0
    use_shadow = base_image is not None and shadow_strength > 0.0
    set_uniform("atmosphere_enabled", 1.0 if use_atmosphere else 0.0)
    set_uniform("haze_strength", haze_strength)
    set_uniform("backlight_strength", backlight_strength if use_backlight else 0.0)
    set_uniform("shadow_strength", shadow_strength if use_shadow else 0.0)
    resolved_haze_color = haze_color if haze_color is not None else sample_sky_color(base_image)
    set_uniform("haze_color", tuple(channel / 255.0 for channel in resolved_haze_color))
    # Camera-drift-aware atmosphere/backlight/refraction sampling: base_image_frames, when given, is an iterable yielding one
    # per-frame drifted photo (e.g. vfx_engine.camera.apply_camera_drift) -
    # without this, atmosphere/backlight/refraction always sampled the
    # photo's ORIGINAL, undrifted framing even while camera drift panned/
    # zoomed the photo underneath the particles, a registration mismatch
    # that grows with drift amplitude (this module's own docstring already
    # flagged the trade-off; base_image_frames is what actually fixes it).
    # base_image itself (a single static Image) is still used for the
    # one-time sample_sky_color/sample_ambient_tint calls above - those
    # are a single scalar color sample, not a per-pixel texture, so
    # they're unaffected by camera drift either way.
    animated_base = base_image_frames is not None
    base_frames_iter = iter(base_image_frames) if animated_base else None

    def _prepare_base_rgb(image):
        rgb = image.convert("RGB")
        return rgb.resize(size) if rgb.size != size else rgb

    use_base_texture = (base_image is not None or animated_base) and (use_atmosphere or use_backlight or use_shadow or refraction_strength > 0.0)
    base_texture = None
    if use_base_texture:
        first_base_image = next(base_frames_iter) if animated_base else base_image
        base_rgb = _prepare_base_rgb(first_base_image)
        base_texture = context.texture(size, 3, base_rgb.tobytes())
        base_texture.use(location=0)
        program["base_texture"].value = 0

    # Local ambient tint - a tiny grid uploaded as its own
    # texture and sampled with GL_LINEAR filtering, which performs the
    # "bilinearly interpolate between the 4 nearest grid cells" lookup the
    # the original design describes in hardware, for free, rather than needing
    # manual lerp code in the fragment shader. CLAMP_TO_EDGE keeps edge
    # cells from wrapping into the opposite side of the frame the way
    # REPEAT would.
    use_ambient_tint_grid = ambient_tint_grid is not None
    ambient_tint_texture = None
    if use_ambient_tint_grid:
        grid_h, grid_w = ambient_tint_grid.shape[0], ambient_tint_grid.shape[1]
        ambient_tint_texture = context.texture((grid_w, grid_h), 3, np.ascontiguousarray(ambient_tint_grid).tobytes())
        ambient_tint_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        ambient_tint_texture.repeat_x = False
        ambient_tint_texture.repeat_y = False
        ambient_tint_texture.use(location=1)
        set_uniform("ambient_tint_texture", 1)
    set_uniform("has_ambient_tint_grid", 1.0 if use_ambient_tint_grid else 0.0)

    # Scene-depth occlusion (see particles_rain.glsl): drops whose own
    # distance is farther than the scene surface under them are hidden.
    # Only shaders that declare `in_depth`/`depth_texture` use it.
    shader_uses_depth = False
    try:
        program["in_depth"]
        shader_uses_depth = True
    except KeyError:
        pass
    use_occlusion = shader_uses_depth and depth_array is not None and occlusion_strength > 0.0
    depth_texture = None
    if use_occlusion:
        depth_image = Image.fromarray(np.clip(depth_array, 0.0, 1.0).astype("float32"), mode="F")
        if depth_image.size != size:
            depth_image = depth_image.resize(size, Image.BILINEAR)
        depth_texture = context.texture(size, 1, np.asarray(depth_image, dtype="float32").tobytes(), dtype="f4")
        depth_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        depth_texture.use(location=2)
        set_uniform("depth_texture", 2)
    set_uniform("occlusion_strength", occlusion_strength if use_occlusion else 0.0)

    quad_vbo = context.buffer(_QUAD_CORNERS.tobytes())
    static_vbo = context.buffer(static_data.tobytes())
    dynamic_vbo = context.buffer(reserve=total_instances * 3 * 4)   # x, y, rotation - f4 each

    vao = context.vertex_array(program, [
        (quad_vbo, "2f", "in_corner"),
        (dynamic_vbo, "2f 1f/i", "in_position", "in_rotation"),
        # The 7th static float (depth) is skipped over as 4 bytes of padding for shaders that don't declare in_depth.
        (static_vbo, "1f 1f 1f 1f 1f 1f " + ("1f" if shader_uses_depth else "4x") + "/i",
         "in_size", "in_alpha", "in_blur", "in_brightness", "in_stretch", "in_shape_seed",
         *(("in_depth",) if shader_uses_depth else ())),
    ])

    fbo = context.framebuffer(color_attachments=[context.texture(size, 4)])
    context.enable(moderngl.BLEND)
    # Separate alpha blend: RGB accumulates premultiplied (un-premultiplied on
    # readback below) while alpha accumulates as a + dst_a*(1-a). A plain
    # SRC_ALPHA on the alpha channel too gave alpha squared, and the
    # compositor treats overlays as straight alpha - so every particle came
    # out darker and thinner than the shader intended.
    context.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA,
                          moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)

    use_compute = gpu.supports_compute()
    compute_program = seed_ssbo = group_count = None
    if use_compute:
        compute_program = gpu.load_compute_program("particles_wind.comp")
        compute_program["viewport_size"].value = size
        compute_program["num_frames"].value = float(max(num_frames, 1))
        compute_program["direction"].value = direction
        compute_program["angle_slant"].value = angle_slant
        compute_program["noise_scale"].value = noise_scale
        compute_program["noise_loop_radius"].value = noise_loop_radius
        compute_program["speed_wobble"].value = speed_wobble
        compute_program["wobble_strength"].value = wobble_strength
        seed_ssbo = context.buffer(seeds.tobytes())
        group_count = -(-total_instances // _COMPUTE_LOCAL_SIZE)   # ceil division

    try:
        for frame_i in range(start_frame, num_frames):
            # Rebind every frame, not just once before the loop: moderngl's
            # `.use(location=N)` binds a texture to a GL texture unit that's
            # GLOBAL to the shared context (see gpu.get_context()) - other
            # effect generators (fog.py, scene_warp.py) bind their OWN
            # textures to the same unit numbers (0/1/2) during their own
            # one-time setup. Since compositor.layer_effects interleaves
            # every effect generator via zip() (one next() call per effect
            # per output frame), another effect's setup can run in between
            # two of THIS generator's frames and silently steal unit 0/1
            # away from base_texture/ambient_tint_texture - every frame
            # after that then samples whatever texture the other effect
            # left bound there instead (confirmed root cause of the
            # rain+fog bug report: rain samples fog's 1-channel depth/luma
            # textures through an RGB sampler, which reads back as (r,0,0),
            # a strong red cast). Rebinding here every frame costs one
            # cheap glBindTexture call and makes this generator immune to
            # whatever any other effect's textures are doing.
            if use_base_texture:
                base_texture.use(location=0)
            if use_ambient_tint_grid:
                ambient_tint_texture.use(location=1)
            if depth_texture is not None:
                depth_texture.use(location=2)

            if animated_base and use_base_texture and frame_i > start_frame:
                # Advanced once per OUTPUT frame (not per motion-blur
                # sub-sample) - background registration doesn't need the
                # same sub-frame precision the particles' own positions do.
                base_texture.write(_prepare_base_rgb(next(base_frames_iter)).tobytes())

            if motion_blur_samples <= 1:
                sub_frame_times = (float(frame_i),)
            else:
                # Centered shutter window: sub-samples spread symmetrically
                # around this output frame's own nominal instant, spanning
                # shutter_fraction of one frame's worth of motion - a real
                # camera's shutter integrates continuously over an exposure
                # window, not at one instantaneous moment.
                sub_frame_times = tuple(
                    frame_i + shutter_fraction * (k / (motion_blur_samples - 1) - 0.5)
                    for k in range(motion_blur_samples)
                )

            linear_accum = None
            for sub_frame_i in sub_frame_times:
                # Shared weather gust coupling (see vfx_engine/weather.py) -
                # this frame's wind DRIFT strength is scaled by the same
                # gust envelope every other wind-driven effect (fog,
                # vegetation sway, cloud drift) samples for the same
                # (frame_i, num_frames), so a gust visibly moves rain/snow
                # in step with trees swaying and fog drifting, not on its
                # own independent schedule. config.WEATHER_GUST_STRENGTH =
                # 0.0 makes this an exact no-op (gust_envelope returns 1.0
                # for every frame). `drift` itself now stays
                # RAW (no CPU pre-multiplication) - each particle computes
                # its OWN gust value from its own current position, in the
                # compute shader (particles_wind.comp's gust_at()) or the
                # CPU fallback (weather.gust_multiplier_at, inside
                # _cpu_dynamic_data below), so a gust reaches particles on
                # the downwind side of a wide frame later than ones on the
                # upwind side, instead of every particle in the frame
                # scaling by one identical frame-global value.
                gust_uniforms = weather.gust_shader_uniforms(sub_frame_i, num_frames, elastic=False)

                if use_compute:
                    compute_program["frame_i"].value = gust_uniforms["frame_i"]
                    compute_program["drift"].value = drift
                    compute_program["gust_phase_offset"].value = gust_uniforms["gust_phase_offset"]
                    compute_program["gust_cycles_a"].value = gust_uniforms["gust_cycles_a"]
                    compute_program["gust_cycles_b"].value = gust_uniforms["gust_cycles_b"]
                    compute_program["gust_strength"].value = gust_uniforms["gust_strength"]
                    compute_program["gust_spatial_delay"].value = gust_uniforms["gust_spatial_delay"]
                    compute_program["gust_wind_bias"].value = gust_uniforms["gust_wind_bias"]
                    seed_ssbo.bind_to_storage_buffer(0)
                    dynamic_vbo.bind_to_storage_buffer(1)
                    compute_program.run(group_count)
                    context.memory_barrier()
                else:
                    dynamic_data = _cpu_dynamic_data(
                        seeds, sub_frame_i, size, direction, angle_slant, drift, noise_scale,
                        num_frames, noise_loop_radius, speed_wobble, wobble_strength,
                    )
                    dynamic_vbo.write(dynamic_data.tobytes())

                fbo.use()
                context.clear(0.0, 0.0, 0.0, 0.0)
                vao.render(moderngl.TRIANGLE_STRIP, instances=total_instances)

                raw_pixels = fbo.read(components=4, dtype="f1")
                sub_frame = Image.frombytes("RGBA", size, raw_pixels).transpose(Image.FLIP_TOP_BOTTOM)
                sub_frame = _unpremultiply(sub_frame)

                if motion_blur_samples <= 1:
                    yield sub_frame   # old behavior, bit-for-bit: no decode/encode round trip when not sub-sampling
                else:
                    # Accumulate PREMULTIPLIED linear colour: a plain RGBA average
                    # halves a flake's colour wherever only one sub-sample covers
                    # it (its other sample has rgb 0), turning the disc dark grey.
                    sub_linear = _color.decode_image(sub_frame)
                    sub_linear = _color.torch.cat([sub_linear[:, :, 0:3] * sub_linear[:, :, 3:4], sub_linear[:, :, 3:4]], dim=2)
                    linear_accum = sub_linear if linear_accum is None else linear_accum + sub_linear

            if motion_blur_samples > 1:
                averaged_alpha = linear_accum[:, :, 3:4] / len(sub_frame_times)
                averaged_rgb = linear_accum[:, :, 0:3] / _color.torch.clamp(linear_accum[:, :, 3:4], min=1e-6)
                averaged_linear = _color.torch.cat([averaged_rgb, averaged_alpha], dim=2)
                yield _color.encode_image(averaged_linear, mode="RGBA")
    finally:
        vao.release()
        quad_vbo.release()
        static_vbo.release()
        dynamic_vbo.release()
        if seed_ssbo is not None:
            seed_ssbo.release()
        if base_texture is not None:
            base_texture.release()
        if ambient_tint_texture is not None:
            ambient_tint_texture.release()
        if depth_texture is not None:
            depth_texture.release()
        fbo.release()
        program.release()
        if compute_program is not None:
            compute_program.release()


def _unpremultiply(image):
    """Premultiplied-RGB RGBA PIL image (see the blend_func note in the
    render loop) -> straight-alpha RGBA, the form compositor/color expect."""
    pixels = np.asarray(image, dtype=np.float32)
    alpha = pixels[..., 3:4]
    rgb = np.where(alpha > 0.0, pixels[..., 0:3] * (255.0 / np.maximum(alpha, 1.0)), 0.0)
    out = np.concatenate([np.clip(rgb + 0.5, 0.0, 255.0), alpha], axis=2).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def sample_ambient_tint(base_image, fallback=(200, 210, 225), sky_weight=0.6):
    """
    Derives a particle tint from the ACTUAL photo's ambient lighting,
    instead of a fixed color - real rain/snow picks up the color of the
    light around it (warm at sunset, cool under an overcast sky, etc.),
    it isn't a fixed shade of white/blue regardless of the scene.

    Blends the average color of the top ~20% of the frame (a cheap proxy
    for "the sky"/whatever's lighting the scene from above) with the
    overall image average, weighted by sky_weight - grounded in the
    actual photo's pixels, not a guess. Returns `fallback` unchanged when
    no base_image is given.
    """
    if base_image is None:
        return fallback

    # A tiny thumbnail is plenty for an average color and is nearly free -
    # no need to touch the full-resolution image for this.
    thumb = base_image.convert("RGB").resize((64, 64))
    pixels = np.asarray(thumb, dtype="float32")

    sky_region = pixels[: max(1, pixels.shape[0] // 5)]   # top ~20% of rows
    sky_color = sky_region.reshape(-1, 3).mean(axis=0)
    overall_color = pixels.reshape(-1, 3).mean(axis=0)

    tint = sky_weight * sky_color + (1.0 - sky_weight) * overall_color
    # Pull partway back toward the fallback so the tint stays a plausible
    # particle color (a very warm/dark photo shouldn't turn snow orange or
    # near-black - it should just lean that direction) rather than tracking
    # the scene's average color exactly, which can look wrong for particles.
    tint = 0.55 * tint + 0.45 * np.array(fallback, dtype="float32")
    return tuple(int(np.clip(channel, 0, 255)) for channel in tint)


def build_ambient_tint_grid(base_image, grid_size=(12, 8), fallback=(200, 210, 225), blend=0.55):
    """
    "splash & rain: local color sampling instead of one
    global tint". sample_ambient_tint() above (and this function's own
    per-cell math mirrors its blend-to-fallback design) reduces the whole
    photo to ONE scalar color; a splash landing in a shadowed corner and
    one landing in a sunlit patch of the same photo have no way to differ
    under that model, which is visibly wrong wherever a scene has any
    lighting variation at all - most real photos.

    Returns a small (grid_size[1], grid_size[0], 3) uint8 numpy array: the
    photo's own average color in each of a coarse grid_size[0] x
    grid_size[1] screen-space cell grid. `PIL.Image.resize(..., Image.BOX)`
    does exactly the area-average this needs in one call - each output
    pixel is the mean of the input pixels that map to it, not a resampled/
    blurred point sample - so this is a real regional average, not
    pixel-scale detail (deliberately coarse: a splash/drop's local light
    should track "this general area of the photo", not the exact pixel
    behind it - sampling the exact pixel is the same "distant particle in
    front of dark trees turns dark green" silhouette bug sample_sky_color's
    docstring already flags for haze, just relocated to per-drop tint).

    Callers (particles.py's render_particles, ground_fx.py's
    render_splashes) then bilinearly interpolate this grid at each
    particle/splash's own spawn UV - see sample_tint_grid_bilinear below
    for the CPU-side lookup (splash) and render_particles' own
    ambient_tint_grid handling for the GPU-texture-sample equivalent
    (rain) - a tiny texture sampled with GL_LINEAR filtering performs
    exactly the same 4-nearest-cell bilinear blend in hardware, no manual
    interpolation code needed on that path.

    Blended 55%/45% toward `fallback` per cell, same reasoning and same
    ratio as sample_ambient_tint: an individual cell that happens to be a
    small saturated/near-black patch (a bright red awning, a deep shadow
    pocket) shouldn't turn every drop over it solid red or literally
    black - it should just lean that direction, the same way the single
    global sample already does.
    """
    if base_image is None:
        return None

    grid_w, grid_h = grid_size
    cells = np.asarray(base_image.convert("RGB").resize((grid_w, grid_h), Image.BOX), dtype="float32")
    blended = blend * cells + (1.0 - blend) * np.array(fallback, dtype="float32")
    return np.clip(blended, 0, 255).astype("uint8")


def sample_tint_grid_bilinear(grid, u, v):
    """
    CPU-side bilinear lookup into a build_ambient_tint_grid() grid at
    normalized screen position (u, v) (0.0-1.0 each, scalars or same-shape
    numpy arrays) - the splash-side equivalent of render_particles'
    GPU-texture-sample lookup for rain (see build_ambient_tint_grid's
    docstring). Returns an (..., 3) float32 array in the grid's own 0-255
    range, matching sample_ambient_tint's return convention closely enough
    to drop straight into the same "fill_color"-shaped call sites once
    rounded/cast by the caller.
    """
    grid_h, grid_w = grid.shape[0], grid.shape[1]
    grid_float = grid.astype("float32")

    # Cell-center sampling (+0.5/-0.5) so a spawn point dead in the middle
    # of the frame lands on an exact cell center rather than a grid corner
    # - matches how GL_LINEAR sampling of the same grid, uploaded as a
    # texture, addresses cell centers too.
    fx = np.clip(u * grid_w - 0.5, 0.0, grid_w - 1.0)
    fy = np.clip(v * grid_h - 0.5, 0.0, grid_h - 1.0)
    x0 = np.floor(fx).astype("int32")
    y0 = np.floor(fy).astype("int32")
    x1 = np.clip(x0 + 1, 0, grid_w - 1)
    y1 = np.clip(y0 + 1, 0, grid_h - 1)
    tx = (fx - x0)[..., None]
    ty = (fy - y0)[..., None]

    top = grid_float[y0, x0] * (1.0 - tx) + grid_float[y0, x1] * tx
    bottom = grid_float[y1, x0] * (1.0 - tx) + grid_float[y1, x1] * tx
    return top * (1.0 - ty) + bottom * ty


def sample_sky_color(base_image, fallback=(210, 218, 230)):
    """
    Returns the average color of the top ~12% of base_image (a cheap
    proxy for "the sky"), for use as a haze/fog blend TARGET - distinct
    from sample_ambient_tint(), which blends toward a fallback and is
    meant for particle base color, not this. Blending distant particles
    toward the sampled pixel directly behind them (the old approach)
    makes them pick up whatever's actually there - a distant particle in
    front of dark trees turns dark green, which reads as a silhouette,
    not haze. Real atmospheric haze is a roughly uniform sky/horizon
    color regardless of what's behind it at any given point.
    """
    if base_image is None:
        return fallback

    thumb = base_image.convert("RGB").resize((64, 64))
    pixels = np.asarray(thumb, dtype="float32")
    sky_region = pixels[: max(1, int(pixels.shape[0] * 0.12))]
    sky_color = sky_region.reshape(-1, 3).mean(axis=0)
    return tuple(int(np.clip(channel, 0, 255)) for channel in sky_color)


def _sample_depth_at(depth_array, xs, ys):
    """Nearest-neighbor lookup of depth_array (height, width) at pixel coords xs, ys (numpy arrays)."""
    height, width = depth_array.shape
    col = np.clip(xs.astype("int32"), 0, width - 1)
    row = np.clip(ys.astype("int32"), 0, height - 1)
    return depth_array[row, col]


def compute_circle_of_confusion(depth, focus_depth):
    """
    real thin-lens circle-of-confusion (CoC) defocus curve,
    replacing the old ad hoc `abs(depth - focus_depth) * 2.2` linear
    approximation that used to live duplicated here (_setup_particles) and
    in particles_rain.py's _apply_regional_bokeh_blur. Standard optics
    formula for the diameter of the blur circle a point at distance S2
    projects when the lens is focused at distance S1 (see e.g. Kingslake,
    "Optics in Photography", or any camera-optics reference on
    depth-of-field/circle-of-confusion derivation from thin-lens geometry):

        CoC = (A * f * |S2 - S1|) / (S2 * (S1 - f))

    where A is the aperture diameter, f is focal length, and S1/S2 are
    real-world distances from the lens. This pipeline's own depth map is a
    normalized 0.0 (far) - 1.0 (near) value with no physical unit, so
    config.DOF_NEAR_METERS/DOF_FAR_METERS linearly remap it onto a
    simulated real-world distance range first (near_m at depth=1.0, far_m
    at depth=0.0, matching this codebase's 0=far/1=near depth convention -
    see this module's own header docstring), and config.DOF_FOCAL_LENGTH_MM/
    DOF_F_NUMBER simulate the lens itself (aperture diameter A = f /
    f_number).

    Args:
        depth: numpy array (any shape), 0.0 (far) - 1.0 (near).
        focus_depth: python float, 0.0-1.0, same convention as `depth` -
            the depth value the simulated lens is focused on.

    Returns:
        float32 array, same shape as `depth`, 0.0 (perfectly in focus) to
        1.0 (maximally defocused) - normalized against the WORST-CASE CoC
        across the full 0.0-1.0 depth range for this exact focus_depth/lens
        combination (a handful of scalar ops, computed fresh each call so
        it stays correct if the config values above are changed), so the
        output keeps spanning close to the full 0..1 range regardless of
        which physical aperture/focal-length/near/far values are
        configured - the same 0..1 defocus contract the old linear formula
        had, so callers (particle size/opacity scaling, the regional bokeh
        blur pass) don't need to change at all, only the curve SHAPE
        feeding them is now physically derived instead of a flat
        multiply-and-clip.
    """
    near_m = config.DOF_NEAR_METERS
    far_m = config.DOF_FAR_METERS
    focal_m = config.DOF_FOCAL_LENGTH_MM / 1000.0
    f_number = max(config.DOF_F_NUMBER, 0.1)
    aperture_m = focal_m / f_number

    depth_clamped = np.clip(depth, 0.0, 1.0)
    # depth=1.0 (near) -> near_m; depth=0.0 (far) -> far_m.
    distance_m = far_m + depth_clamped * (near_m - far_m)
    focus_m = far_m + float(np.clip(focus_depth, 0.0, 1.0)) * (near_m - far_m)
    # S1 - f: how far the focus plane sits beyond the lens's own focal
    # length - the thin-lens formula's denominator term. Guarded away from
    # zero (a focus plane sitting exactly at the focal length is a
    # degenerate/unphysical lens configuration, not something any of this
    # module's config defaults can actually produce, but the guard keeps
    # a pathological custom config from raising instead of just saturating).
    denom = max(abs(focus_m - focal_m), 1e-6)

    coc = aperture_m * focal_m * np.abs(distance_m - focus_m) / (np.maximum(distance_m, 1e-6) * denom)

    # Worst case is whichever end of the configured near/far range sits
    # furthest (in real-world CoC terms) from the current focus plane -
    # used purely to rescale the output back into a comparable 0..1 range,
    # not as a physical quantity anyone else reads.
    worst_candidates_m = (near_m, far_m)
    worst_case_m = max(worst_candidates_m, key=lambda d: abs(d - focus_m))
    worst_case_coc = aperture_m * focal_m * abs(worst_case_m - focus_m) / (max(worst_case_m, 1e-6) * denom)
    normalization = max(worst_case_coc, 1e-6)

    return np.clip(coc / normalization, 0.0, 1.0).astype(np.float32)


def _setup_particles(size, count, size_range, speed_range, use_depth, glow, streak_count, streak_spacing, direction,
                      speed_jitter=0.15, brightness_jitter=0.15, size_skew=0.0, wind_mass_scaling=True,
                      velocity_stretch=0.0, density_clumping=0.0, depth_array=None, horizontal_wander=0.0,
                      focus_depth=None, bokeh_factor=0.0, speed_exposure_loss=0.0, depth_alpha_falloff=0.0,
                      seed=None, persistent_depth=False):
    """
    Randomizes each particle's initial properties and returns two numpy
    arrays ready to upload to the GPU:
        seeds       (start_x, start_y, speed, noise_phase, wind_scale,
                    wander_loops) - read by the per-frame position update
                    (GPU compute shader or CPU fallback), constant for the
                    whole animation.
        static_data (size, alpha, blur, brightness, stretch, shape_seed, depth) -
                    read directly by particles.glsl, also constant for the
                    whole animation.

    When streak_count > 1, each particle is expanded into streak_count
    "seed" rows trailing behind it (same idea as a comet's tail) - one row
    per trailing point, each with a fixed offset baked into its start_y
    and a fainter alpha the further back it trails. This used to be
    recomputed every frame in Python; baking it into the seeds once at
    setup means the per-frame cost (GPU or CPU) only has to move
    `count * streak_count` points, not rebuild them.

    speed_jitter/brightness_jitter/size_skew/wind_mass_scaling/
    velocity_stretch/density_clumping/depth_array: see render_particles's
    docstring - all per-particle variance, applied once here at setup so
    it stays constant for a given particle across its whole trailing
    streak (a comet's tail shouldn't have a different brightness per point).

    seed: see render_particles's own `seed` doc - feeds this function's rng.
    """
    rng = np.random.default_rng(seed)

    start_x = rng.uniform(0, size[0], count)
    start_y = rng.uniform(0, size[1], count)

    # use_depth=False -> every particle has the same size/speed/sharpness
    # (simpler, more uniform look, e.g. for light rain seen up close).
    # When a real scene depth map is available, sample the ACTUAL depth at
    # each particle's spawn position instead of a random scalar - ties
    # particle size/speed/blur to the photo's real geometry (e.g. rain
    # over the near foreground reads as genuinely closer than rain over a
    # distant lake), not just a plausible-looking random spread.
    if not use_depth:
        depth = np.full(count, 0.7)
    elif depth_array is not None and persistent_depth:
        # Each drop keeps ONE distance for its whole life, drawn from the
        # scene's own depth distribution at random pixels - NOT from the
        # depth under its random spawn point. Drops wrap and fall through
        # every screen row, so a depth tied to spawn position made a near
        # (big, sharp) drop cross distant sky at full size; with a
        # position-independent distance plus shader-side occlusion by the
        # scene depth (see particles_rain.glsl), near drops read as near
        # everywhere and far drops vanish behind foreground objects.
        depth = _sample_depth_at(depth_array, rng.uniform(0, size[0], count), rng.uniform(0, size[1], count))
    elif depth_array is not None:
        depth = _sample_depth_at(depth_array, start_x, start_y)
    else:
        depth = rng.uniform(0.0, 1.0, count)

    size_min, size_max = size_range
    particle_sizes = size_min + (size_max - size_min) * depth
    if glow:
        particle_sizes = particle_sizes * 2.5   # a bigger, softer blob looks like a glow

    if size_skew > 0.0:
        # Right-skewed multiplier (few notably larger particles, most near
        # 1x) instead of the flat uniform spread size_range alone gives -
        # a Marshall-Palmer-like rain drop size distribution, mean ~1.0.
        skew = rng.standard_exponential(count) * size_skew
        particle_sizes = particle_sizes * np.clip(1.0 + skew - size_skew, 0.5, 2.5)

    speed_min, speed_max = speed_range
    speeds = speed_min + (speed_max - speed_min) * depth
    if speed_jitter > 0.0:
        speeds = speeds * rng.uniform(1.0 - speed_jitter, 1.0 + speed_jitter, count)

    base_opacity = 0.55 + 0.45 * depth   # particles "further away" are dimmer

    if focus_depth is not None:
        # Real camera depth-of-field: blur increases with DISTANCE FROM
        # THE FOCUS PLANE in either direction, not just "far = blurry" -
        # a particle in the near foreground, well in front of where the
        # camera is focused, should go soft too, exactly like a real lens.
        defocus = compute_circle_of_confusion(depth, focus_depth)
        blur_amount = np.full(count, 1.0) if glow else defocus
        if bokeh_factor > 0.0:
            # Real out-of-focus points don't just get blurrier, they get
            # physically BIGGER (bokeh discs) - proportional to how far
            # out of focus they are.
            particle_sizes = particle_sizes * (1.0 + bokeh_factor * defocus)
    else:
        blur_amount = np.full(count, 1.0) if glow else (1.0 - depth)

    if depth_alpha_falloff > 0.0:
        # the depth-linked base_opacity floor above
        # is deliberately mild (0.55-1.0 range) - it alone doesn't make a
        # defocused/receding particle read as genuinely dissolving into
        # haze. This ties an ADDITIONAL fade directly to blur_amount (not
        # depth) so it also applies correctly under focus_depth's real
        # depth-of-field model, where blur is a function of distance from
        # the focus plane rather than raw depth.
        base_opacity = base_opacity * (1.0 - depth_alpha_falloff * blur_amount)

    brightness = rng.uniform(1.0 - brightness_jitter, 1.0 + brightness_jitter, count) if brightness_jitter > 0.0 else np.ones(count)

    if density_clumping > 0.0:
        # Coherent (not per-particle-independent) noise sampled at each
        # particle's own spawn position, so nearby particles share a
        # similar density multiplier - visible denser "gusts" in some
        # areas of the frame instead of uniform per-particle scatter.
        # Static per particle (not re-sampled per frame): an approximation
        # of clumping, not literally moving cluster fronts - see
        # the original design's Physical motion simulation section.
        #
        # _value_noise() is a pure deterministic hash of its input
        # coordinates - with no offset, EVERY render of EVERY photo at
        # this resolution would get the identical clump pattern in the
        # identical screen location, which is exactly what happened
        # during testing (a strong, obviously-wrong-looking cluster
        # appeared reliably in the same spot). A random offset per render
        # (sampled fresh from this call's own rng) decorrelates it - a
        # different, still-coherent clump pattern every time instead of a
        # fixed watermark.
        clump_offset = rng.uniform(0.0, 5000.0, 2)
        clump_noise = _value_noise(np.stack([start_x, start_y], axis=1) * 0.0045 + clump_offset)
        clump_factor = 1.0 - density_clumping * 0.35 + density_clumping * 0.7 * clump_noise
        base_opacity = base_opacity * np.clip(clump_factor, 0.4, 1.3)

    # Wind sensitivity: real air drag pushes lighter/smaller particles
    # sideways more than bigger/heavier ones - scale each particle's wind
    # response inversely to its own size instead of one global strength
    # for every particle regardless of size.
    if wind_mass_scaling and size_max > size_min:
        relative_size = (particle_sizes - particle_sizes.min()) / max(1e-6, particle_sizes.max() - particle_sizes.min())
        wind_scale = 1.6 - 1.1 * relative_size   # smaller (relative_size~0) -> ~1.6x wind, bigger -> ~0.5x
    else:
        wind_scale = np.ones(count)

    # Velocity-streak elongation: faster particles (relative to this
    # effect's own speed_range) stretch more, giving a real motion-streak
    # instead of the old fixed ghost-copy trail approach.
    relative_speed = np.clip((speeds - speed_min) / max(1e-6, speed_max - speed_min), 0.0, 1.0) if speed_max > speed_min else np.zeros(count)
    stretch = velocity_stretch * (0.3 + 0.7 * relative_speed) if velocity_stretch > 0.0 else np.zeros(count)

    if speed_exposure_loss > 0.0:
        # A fast-moving drop spreads its light over more sensor pixels per
        # exposure than a slow one - in a real photo it reads as fainter,
        # not just longer. Purely a rain-relevant nuance (speed_range
        # variation for snow is small enough this barely matters there).
        base_opacity = base_opacity * (1.0 - speed_exposure_loss * relative_speed * 0.35)

    noise_phase = rng.uniform(0, 1000, count)   # per-particle offset into the wind-noise field

    # Shape variance: a single per-particle random value,
    # 0.0-1.0, that each fragment shader hashes into 2-4+ distinct
    # silhouette variants (a slightly different core-position offset, a
    # slightly different lobe count/aspect, a small per-particle asymmetry
    # noise term) - see particles.glsl/particles_snow.glsl/
    # particles_rain.glsl. Without this, every particle of a given effect
    # was one fixed silhouette function just scaled/tinted, which is what
    # reads as "stamped" even with every other statistical variance
    # (size/speed/brightness) already in place.
    shape_seed = rng.uniform(0.0, 1.0, count)

    if horizontal_wander > 0.0:
        # A whole number of screen-widths (direction randomized per
        # particle) so this closes the loop exactly - see
        # render_particles's docstring for why this needs to exist at all
        # (snow, which has no angle_slant tilt, would otherwise never
        # leave the neighborhood of its random spawn X).
        wander_direction = rng.choice([-1.0, 1.0], count)
        # Must stay a whole number of screen-widths (see _cpu_dynamic_data's
        # wander_speed) for the loop to close exactly - but NOT forced up to
        # a minimum of 1.0, since that turns any small horizontal_wander
        # (e.g. 0.1, meant to give subtle drift) into a full screen-width
        # traversal for nearly every particle. Rounding to 0 for a small
        # horizontal_wander is the correct, intended behavior: those
        # particles simply don't wander this loop.
        wander_magnitude = np.round(rng.uniform(0.6, 1.4, count) * horizontal_wander)
        wander_loops = wander_direction * wander_magnitude
    else:
        wander_loops = np.zeros(count)

    seed_rows = []
    static_rows = []
    for step in range(streak_count):
        offset = step * streak_spacing
        fade = 1.0 - (step / max(streak_count, 1))
        shifted_start_y = (start_y - direction * offset) % size[1]

        seed_rows.append(np.stack([start_x, shifted_start_y, speeds, noise_phase, wind_scale, wander_loops], axis=1))
        static_rows.append(np.stack([particle_sizes, base_opacity * fade, blur_amount, brightness, stretch, shape_seed, depth], axis=1))

    seeds = np.concatenate(seed_rows, axis=0).astype("f4")
    static_data = np.concatenate(static_rows, axis=0).astype("f4")
    return seeds, static_data


def _hash(p):
    """Vectorized numpy equivalent of the `hash()` function in particles_wind.comp."""
    return np.mod(np.sin(p[..., 0] * 127.1 + p[..., 1] * 311.7) * 43758.5453123, 1.0)


def _value_noise(p):
    """Vectorized numpy equivalent of `value_noise()` in particles_wind.comp."""
    cell = np.floor(p)
    f = p - cell

    a = _hash(cell)
    b = _hash(cell + np.array([1.0, 0.0]))
    c = _hash(cell + np.array([0.0, 1.0]))
    d = _hash(cell + np.array([1.0, 1.0]))

    u = f * f * (3.0 - 2.0 * f)
    ux, uy = u[..., 0], u[..., 1]
    return a + (b - a) * ux + (c - a) * uy * (1.0 - ux) + (d - b) * ux * uy


def _curl_noise(p, eps=0.05):
    """Vectorized numpy equivalent of `curl()` in particles_wind.comp - see that file for the explanation."""
    n_up = _value_noise(p + np.array([0.0, eps]))
    n_down = _value_noise(p - np.array([0.0, eps]))
    n_right = _value_noise(p + np.array([eps, 0.0]))
    n_left = _value_noise(p - np.array([eps, 0.0]))

    dy = (n_up - n_down) / (2.0 * eps)
    dx = (n_right - n_left) / (2.0 * eps)
    return np.stack([dy, -dx], axis=-1)


def _cpu_dynamic_data(seeds, frame_i, size, direction, angle_slant, drift, noise_scale,
                       num_frames, noise_loop_radius, speed_wobble, wobble_strength):
    """CPU fallback for particles_wind.comp - see that file's docstring for what this computes and why (must stay in sync with it)."""
    start_x, start_y, speed, noise_phase, wind_scale, wander_loops = (
        seeds[:, 0], seeds[:, 1], seeds[:, 2], seeds[:, 3], seeds[:, 4], seeds[:, 5]
    )
    theta = 2.0 * np.pi * (frame_i / max(num_frames, 1))

    # Quantize each particle's fall speed to a whole number of screen-
    # heights per loop, so its position at frame `num_frames` is exactly
    # its position at frame 0 - see particles_wind.comp's docstring.
    raw_loops = np.maximum(speed * num_frames / size[1], 1.0)
    loops = np.floor(raw_loops + 0.5)
    effective_speed = loops * size[1] / max(num_frames, 1)

    # Additive (not multiplicative) periodic offset - see
    # particles_wind.comp's docstring for why this has to be additive to
    # keep the loop's exact closure intact.
    wobble_amplitude = speed_wobble * size[1] * 0.03
    fallen = direction * effective_speed * frame_i + wobble_amplitude * np.sin(2.0 * theta + noise_phase)

    # Net sideways wander - see ParticleSeed.wander_loops in
    # particles_wind.comp for why this exists (without it, e.g. snow's
    # angle_slant=0 particles never actually travel horizontally).
    wander_speed = wander_loops * size[0] / max(num_frames, 1)
    base_x = np.mod(start_x + fallen * angle_slant + wander_speed * frame_i, size[0])
    base_y = np.mod(start_y + fallen, size[1])

    noise_time = noise_loop_radius * np.array([np.cos(theta), np.sin(theta)])
    noise_pos = np.stack([base_x, base_y], axis=1) * noise_scale
    noise_pos += np.stack([noise_phase, np.zeros_like(noise_phase)], axis=1)
    noise_pos += noise_time

    # each particle's OWN gust value, from its OWN current
    # position (base_x/base_y, before wind displacement) - see
    # particles_wind.comp's gust_at() for the identical GPU-side formula
    # this must stay in sync with. `drift` arrives here RAW (no longer
    # CPU-pre-multiplied by a single frame-global gust_envelope() value).
    gust_multiplier = weather.gust_multiplier_at(base_x / size[0], base_y / size[1], frame_i, num_frames)
    wind = _curl_noise(noise_pos) * drift * gust_multiplier[:, None] * 40.0 * wind_scale[:, None]

    x = np.mod(base_x + wind[:, 0], size[0])
    y = np.mod(base_y + wind[:, 1], size[1])

    # Real net velocity, added as vector components before taking the
    # angle - fix for the confirmed rotation bug, must stay
    # in sync with particles_wind.comp's identical fix. See that file's
    # comment for the full derivation of the sign/argument-order
    # convention (atan2(-horizontal, vertical), not atan2(vertical,
    # horizontal)).
    net_vertical = direction * effective_speed + wind[:, 1]
    net_horizontal = angle_slant * effective_speed + wind[:, 0]
    wind_rotation = np.arctan2(-net_horizontal, net_vertical)

    # Tumble scaled by local wind strength - see particles_wind.comp's
    # matching comment for why (must stay in sync with it).
    local_wind_mag = np.linalg.norm(wind, axis=1) / max(drift * 40.0 * 0.5, 0.001)
    tumble_coherence = np.clip(local_wind_mag, 0.3, 1.8)
    tumble = wobble_strength * tumble_coherence * np.sin(3.0 * theta + noise_phase * 7.0)
    rotation = wind_rotation + tumble

    return np.stack([x, y, rotation], axis=1).astype("f4")
