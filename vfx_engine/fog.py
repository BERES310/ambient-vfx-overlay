"""
fog.py

A GPU-rendered, drifting, translucent haze - used for fog, smoke, steam, or
heat haze (same density/color/movement parameters, tuned differently).
Unlike snow/rain, this isn't individual particles, just a single
full-screen shader (shaders/blur.glsl) that generates a noise pattern
directly on the GPU and shifts it over time - the drift path is a closed
circle in noise-space (see blur.glsl's docstring), so it closes the loop
exactly, the same way every other cyclic effect in this pipeline does.

The drift is also biased toward the shared prevailing wind direction and
pulses with the shared gust envelope (see vfx_engine/weather.py) - the
same direction/timing rain, snow, vegetation sway and cloud drift are
coupled to, so fog visibly surges in step with the same gusts moving
everything else instead of drifting on its own unrelated schedule.
"""

import moderngl
import numpy as np
from PIL import Image, ImageFilter

from vfx_engine import gpu, weather

# Two triangles covering the whole screen, in normalized coordinates
# (-1..1) - the shader decides what to draw at every pixel on its own, so
# this quad only exists to make the fragment shader run for every pixel on
# screen.
_FULLSCREEN_QUAD = np.array([
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
    -1.0,  1.0,
     1.0, -1.0,
     1.0,  1.0,
], dtype="f4")


def render_fog(size, num_frames, density=0.35, movement=0.3, color=(215, 220, 230),
                depth_array=None, depth_power=1.5, depth_smoothing_px=25,
                ground_fog_strength=0.55, light_pos_uv=None, light_color=None, light_coupling_strength=0.55,
                base_image=None, ground_texture_strength=0.3, ground_texture_blur_px=6, start_frame=0):
    """
    Generator - for each frame of the drifting fog/smoke/steam animation,
    yields a single transparent frame (RGBA), ready to be applied to a
    photo. Rendered on the fly, one frame at a time (see particles.py for
    why).

    density: 0.0-1.0, overall opacity of the fog
    movement: how far the haze wanders over the loop (a closed drift path,
        not a fixed speed - see blur.glsl)
    color: (r, g, b) 0-255, fog color
    depth_array: optional numpy float32 array (height, width), 0.0 (far)
        to 1.0 (near), from vfx_engine.depth.estimate_depth() - when
        given, fog density is scaled by (1 - depth) ** depth_power, so
        it's genuinely thicker in the distance and thinner in the near
        foreground (real atmospheric haze), instead of a flat uniform
        density plastered over the whole frame regardless of what's near
        or far. Falls back to uniform density when omitted. Also used to give near/far fog their own wisp scale and drift
        speed - see blur.glsl's depth-parallax comment - instead of the
        entire noise field panning as one uniform sheet.
    depth_power: >0, how sharply fog concentrates into the distance when
        depth_array is given - 1.0 is a linear falloff, higher values
        (e.g. 2-3) keep the near foreground clearer for longer before fog
        density ramps up.
    depth_smoothing_px: Gaussian blur radius (px) applied to the resulting
        depth-to-density gradient before it's used as fog alpha - fixes a
        confirmed bug: monocular depth models are
        specifically weak on large, low-texture, hazy regions (a distant
        misty forest canopy is exactly that), so a discontinuous jump in
        the raw depth estimate can pass through completely unsmoothed and
        become a hard visible line cutting through the fog layer. Fog is
        meant to read as a soft atmospheric medium anyway, so smoothing
        its density field costs nothing visually. 0 disables the
        smoothing (raw per-pixel gradient, the old buggy behavior).
    ground_fog_strength: 0.0-1.0 - how much thinner fog gets
        toward the TOP of frame vs. the bottom, independent of near/far
        depth. Real ground fog/mist settles low and thins with height; 0.0
        = no vertical stratification (old behavior, fog density depends
        only on depth/noise).
    light_pos_uv/light_color/light_coupling_strength: -
        fog brightens/tints toward light_color specifically AROUND
        light_pos_uv (normalized 0.0-1.0 screen position - see
        god_rays.detect_light_source/sample_light_color), instead of
        being one flat, position-agnostic tint. Real haze is brighter and
        more visibly lit specifically around a light source - that's
        literally the mechanism that makes god rays visible in the first
        place. light_pos_uv=None (default) disables this (old behavior).
    base_image/ground_texture_strength/ground_texture_blur_px: without this, fog density depends only on procedural noise, the
        coarse depth gradient, and vertical screen position, with zero
        relationship to the photo's actual content, so it can look
        "pasted on" rather than sitting in the scene. When base_image (a
        PIL Image, any mode, already at `size`) is given, fog picks up a
        little ambient-bounce brightness over already-bright ground and
        thins along the photo's own strong edges (so it pools around
        rocks/fences/treelines instead of smearing across them) - see
        blur.glsl's ground-anchoring block. ground_texture_strength
        (0.0-1.0) scales how strongly this couples in, and is itself
        scoped to the lower/ground part of frame via the existing
        ground_fog_strength stratification, never the sky. base_image=None
        (default) disables this (old behavior). ground_texture_blur_px is
        a MUCH lighter blur than depth_smoothing_px's - just enough to
        stop raw photo sensor noise from reading as fake fog texture,
        while deliberately preserving real object edges (the whole point
        of the edge term above), unlike depth_smoothing_px's blur which
        exists specifically to erase depth-map inconsistencies.
    start_frame: skips yielding/rendering frames 0..start_frame-1 (see
        render_checkpoint.py) - every one-time setup step above (GPU
        program/uniform setup, depth/luma texture upload, the
        depth_gradient precompute) still runs unconditionally regardless
        of this value, and each frame's GPU output is a pure function of
        its own frame_i/num_frames plus that fixed setup, so resuming the
        loop below at start_frame produces bit-identical frames to an
        uninterrupted run from frame 0, just without paying the GPU cost
        of the frames already rendered. 0 (default) = old behavior.
    """
    context = gpu.get_context()
    program = gpu.load_shader_program("blur.glsl")

    program["resolution"].value = size
    program["fog_color"].value = tuple(channel / 255.0 for channel in color)
    program["density"].value = density
    program["movement"].value = movement
    program["num_frames"].value = float(max(num_frames, 1))
    program["ground_fog_strength"].value = ground_fog_strength
    # Shared weather wind direction (see vfx_engine/weather.py) - the
    # per-frame gust envelope is set inside the render loop below.
    program["wind_bias"].value = weather.wind_direction_vector()

    program["has_light_pos"].value = 1.0 if light_pos_uv is not None else 0.0
    program["light_pos_uv"].value = light_pos_uv if light_pos_uv is not None else (0.5, 1.0)
    resolved_light_color = light_color if light_color is not None else color
    program["light_tint_color"].value = tuple(channel / 255.0 for channel in resolved_light_color)
    program["light_coupling_strength"].value = light_coupling_strength

    vbo = context.buffer(_FULLSCREEN_QUAD.tobytes())
    vao = context.vertex_array(program, [(vbo, "2f", "in_position")])

    fbo = context.framebuffer(color_attachments=[context.texture(size, 4)])
    context.enable(moderngl.BLEND)
    context.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    # Depth-parallax texture - the RAW depth array (not the
    # power-curved/blurred depth_gradient below, which is a different,
    # already-reshaped quantity meant for the alpha post-multiply), so the
    # shader sees the scene's actual depth to bend its noise-sampling
    # coordinate by. A texture is bound either way (a 1x1 dummy when no
    # real depth is available) so the sampler always has something valid;
    # the shader's own has_depth flag decides whether it's actually used.
    if depth_array is not None:
        depth_uint8 = np.clip(depth_array * 255.0, 0.0, 255.0).astype("uint8")
        depth_texture = context.texture(size, 1, np.ascontiguousarray(depth_uint8).tobytes())
        program["has_depth"].value = 1.0
    else:
        depth_texture = context.texture((1, 1), 1, bytes([128]))
        program["has_depth"].value = 0.0
    depth_texture.use(location=0)
    program["depth_texture"].value = 0

    # Ground-anchoring texture - a lightly-blurred grayscale
    # of the base photo, see blur.glsl's ground-anchoring block and this
    # function's own docstring. Bound either way (a 1x1 dummy when no
    # base_image is given), same pattern as depth_texture above.
    if base_image is not None:
        luma_image = base_image.convert("L").resize(size)
        if ground_texture_blur_px > 0:
            luma_image = luma_image.filter(ImageFilter.GaussianBlur(ground_texture_blur_px))
        luma_texture = context.texture(size, 1, np.ascontiguousarray(np.asarray(luma_image, dtype="uint8")).tobytes())
        program["has_luma"].value = 1.0
        program["ground_texture_strength"].value = ground_texture_strength
    else:
        luma_texture = context.texture((1, 1), 1, bytes([128]))
        program["has_luma"].value = 0.0
        program["ground_texture_strength"].value = 0.0
    luma_texture.use(location=1)
    program["luma_texture"].value = 1

    depth_gradient = None
    if depth_array is not None:
        depth_gradient = np.clip(1.0 - depth_array, 0.0, 1.0) ** depth_power
        if depth_smoothing_px > 0:
            # Deliberately a plain Gaussian blur, not depth.py's guided
            # filter - the guided filter exists to PRESERVE real edges
            # (snapping depth boundaries back to the photo's own luminance
            # edges), the opposite of what's needed here: a smooth,
            # low-texture hazy region has no local contrast for the guided
            # filter to grab onto in the first place, so a depth
            # inconsistency introduced there passes straight through it.
            gradient_image = Image.fromarray((depth_gradient * 255.0).astype("uint8"), mode="L")
            gradient_image = gradient_image.filter(ImageFilter.GaussianBlur(depth_smoothing_px))
            depth_gradient = np.asarray(gradient_image, dtype="float32") / 255.0

    try:
        for frame_i in range(start_frame, num_frames):
            # Rebind every frame: moderngl's `.use(location=N)` binds to a
            # GL texture unit that's GLOBAL to the shared context, not
            # scoped to this generator. compositor.layer_effects()
            # interleaves multiple effect generators (rain, fog,
            # scene_warp) frame-by-frame via zip(), so another generator's
            # one-time setup can run in between two of THIS generator's
            # frames and steal unit 0/1 (see particles.py's render_rain
            # for the confirmed real-world case this caused: rain
            # sampling fog's 1-channel textures through an RGB sampler,
            # producing a red cast). Rebind here so fog is immune too.
            depth_texture.use(location=0)
            luma_texture.use(location=1)

            # fog has no elastic/damped-spring variant, so
            # elastic=False always here - the shader's own gust_at()
            # then behaves exactly like the plain weather.gust_envelope()
            # this used to precompute CPU-side, just evaluated per-
            # fragment now instead of once per frame.
            gust_uniforms = weather.gust_shader_uniforms(frame_i, num_frames, elastic=False)
            program["frame_i"].value = gust_uniforms["frame_i"]
            program["gust_phase_offset"].value = gust_uniforms["gust_phase_offset"]
            program["gust_cycles_a"].value = gust_uniforms["gust_cycles_a"]
            program["gust_cycles_b"].value = gust_uniforms["gust_cycles_b"]
            program["gust_amp_a"].value = gust_uniforms["gust_amp_a"]
            program["gust_lag_a"].value = gust_uniforms["gust_lag_a"]
            program["gust_amp_b"].value = gust_uniforms["gust_amp_b"]
            program["gust_lag_b"].value = gust_uniforms["gust_lag_b"]
            program["gust_strength"].value = gust_uniforms["gust_strength"]
            program["gust_spatial_delay"].value = gust_uniforms["gust_spatial_delay"]

            fbo.use()
            context.clear(0.0, 0.0, 0.0, 0.0)
            vao.render(moderngl.TRIANGLES)

            raw_pixels = fbo.read(components=4, dtype="f1")
            frame = Image.frombytes("RGBA", size, raw_pixels).transpose(Image.FLIP_TOP_BOTTOM)

            if depth_gradient is not None:
                frame_array = np.asarray(frame)
                alpha = (frame_array[:, :, 3].astype("float32") * depth_gradient).astype("uint8")
                frame_array = frame_array.copy()
                frame_array[:, :, 3] = alpha
                frame = Image.fromarray(frame_array, mode="RGBA")

            yield frame
    finally:
        vao.release()
        vbo.release()
        fbo.release()
        depth_texture.release()
        luma_texture.release()
        program.release()
