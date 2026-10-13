"""
scene_warp.py

Shared GPU engine behind vegetation.py's canopy sway and cloud.py's sky
drift: both are "wind moves a masked region of the photo's OWN pixels"
image-space displacement warps (see shaders/region_warp.glsl and
the design notes Vegetation animation / Cloud motion sections), just
with different masks and wind parameters - identical mechanism, so it
lives here once instead of being duplicated in both callers, the same way
particles.py is shared by particles_rain.py/particles_snow.py.
"""

from typing import Literal

import moderngl
import numpy as np
from PIL import Image

from vfx_engine import gpu, weather

# Two triangles covering the whole screen, in normalized coordinates
# (-1..1) - see fog.py's identical fullscreen quad; the fragment shader
# decides what to draw at every pixel on its own.
_FULLSCREEN_QUAD = np.array([
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
    -1.0,  1.0,
     1.0, -1.0,
     1.0,  1.0,
], dtype="f4")


def render_region_warp(size, num_frames, base_image, region_rg_array,
                        drift_strength, noise_scale, wind_bias=(0.0, 0.0),
                        noise_loop_radius=3.0, flutter_strength=0.0,
                        flutter_scale=0.02, flutter_cycles=5.0,
                        wave_strength=0.0, wave_direction=(1.0, 0.0),
                        wave_frequency=0.02, wave_cycles=3.0, gust_kind: Literal["plain", "elastic"] = "plain",
                        gap_flicker_strength=0.0, gap_flicker_scale=0.08,
                        phase_offset_frames=0.0, depth_array=None, start_frame=0):
    """
    Generator - for each of `num_frames` frames, yields a transparent RGBA
    frame: the photo's OWN pixels within `region_rg_array`'s masked area,
    resampled through a wind-driven UV displacement field (see
    shaders/region_warp.glsl), fully transparent everywhere else - ready
    to be composited on top of the (static or camera-drifting) base photo,
    UNDER every particle effect, so rain/snow still falls in FRONT of the
    swaying trees/drifting sky, not behind them.

    base_image: the actual photo (PIL Image) - resampled through the warp,
        not a separate generated texture, so the swayed/drifted result
        stays real photographic content, not synthesized detail. Can
        also be an ITERABLE yielding one PIL Image per frame (e.g.
        vfx_engine.camera.apply_camera_drift) instead of a single static
        Image - fixes without this, vegetation sway/cloud
        drift always sampled the photo's ORIGINAL, undrifted framing even
        while camera drift panned/zoomed the photo underneath them,
        a registration mismatch that grows with drift amplitude. The
        texture is re-uploaded (not recreated) once per frame in that
        case - see the per-frame loop below.
    region_rg_array: a (height, width, 2) uint8 numpy array - R channel is
        the region's visibility/alpha mask (0-255), G channel is the
        per-pixel displacement AMPLITUDE weight (0-255, independent of
        alpha - e.g. vegetation.py's canopy weighting, where trunk pixels
        stay visible but barely move while canopy pixels above them move
        a lot). See vegetation.py/cloud.py for how each builds this. Can
        also be an ITERABLE yielding one (height, width, 2) uint8 array
        per frame - fixes segmentation masks previously had
        NO camera-drift-aware counterpart at all (unlike depth.py), so
        the mask deciding which pixels are even eligible to sway/drift
        stayed locked to the original, undrifted framing even while the
        photo panned/zoomed underneath it. See
        vfx_engine.camera.apply_camera_drift_to_array (the same
        mechanism depth.py already uses) for how callers build this.
    drift_strength: overall displacement magnitude in pixels at weight=1.0.
    noise_scale: spatial scale of the wind field - smaller values give
        broader, slower-varying gusts; larger values give more localized,
        finer variation.
    wind_bias: (x, y) constant directional bias added to the curl-noise
        wind vector before scaling - (0, 0) for effects that should only
        sway/gust in place (vegetation), a real direction for effects that
        should visibly drift one way overall (cloud motion).
    noise_loop_radius: how far the wind noise's time-sampling circle
        reaches (see particles_wind.comp's docstring) - 0.0 makes the wind
        field static.
    flutter_strength/flutter_scale/flutter_cycles: an optional second,
        faster/finer independent noise layer on top of the main wind - see
        shaders/region_warp.glsl's matching uniforms. 0.0 (default) = off.
    wave_strength/wave_direction/wave_frequency/wave_cycles: an optional
        genuine TRAVELING WAVE displacement (a coherent ripple crossing
        the region), distinct from the curl-noise field's in-place
        turbulent gusting - see shaders/region_warp.glsl's matching
        uniforms and vegetation.py's render_grass_sway.
        wave_strength=0.0 (default) = off, unchanged behavior.
    gust_kind: "plain" (default) or "elastic" - an explicit, caller-set tag
        selecting which gust-response model the shader's own gust_at()
        uses. "plain" uses
        the instantaneous envelope (weather.gust_envelope's shape);
        "elastic" uses the damped-spring amplitude/lag values for callers wanting a
        lagged/springy response instead of one that snaps to the driving
        signal instantly. the design notesix: this used to be an
        identity check against the two literal function objects
        weather.gust_envelope/weather.elastic_gust_envelope, which broke
        silently if a caller ever wrapped or partially-applied either
        function (changing its identity while leaving its behavior
        unchanged) and had no way to extend to a third gust-response
        model without editing this dispatch. An explicit tag is a
        documented contract callers opt into, and adding a third
        gust-response model in the future only means adding a new
        Literal value here plus a new branch in gust_shader_uniforms -
        not touching this identity-comparison logic at all.
    gap_flicker_strength/gap_flicker_scale: an optional fast, fine-scale
        alpha flicker approximating light/gaps flickering through foliage
 - see shaders/region_warp.glsl's matching uniforms.
        gap_flicker_strength=0.0 (default) = off, unchanged behavior.
    phase_offset_frames: shifts every theta-driven
        term (the main wind field, flutter, wave, gap-flicker, and the
        gust envelope) by this many frames' worth of phase, without
        affecting which real frame_i drives base_image/region_rg_array
        advancement. 0.0 (default) = unchanged behavior. cloud.py's
        shape-evolving cross-fade calls this generator twice per render -
        once at 0.0, once at num_frames/2.0 (exactly half a loop, i.e.
        theta+pi) - and blends the two outputs, so the same wind field is
        read at two different phases instead of introducing any new noise
        layer.
    depth_array: optional numpy float32 array (height, width), 0.0 (far)
        to 1.0 (near), from vfx_engine.depth.estimate_depth() - when given, the traveling-wave term (wave_strength
        above) gets a per-pixel depth-driven frequency (see
        shaders/region_warp.glsl's matching depth_texture/has_depth
        uniforms and wave block) - far grass reads a tighter, slower wave,
        near grass a wider, faster one - instead of one flat frequency
        across the whole region regardless of how much depth range it
        spans. None (default) = the old flat behavior. Has no effect on
        callers that don't use wave_strength (vegetation.py's canopy sway,
        cloud.py's drift).
    start_frame: skips yielding frames 0..start_frame-1 (see
        render_checkpoint.py) - every one-time GPU setup step above
        (shader program/uniform setup, the first base/region texture
        upload, depth texture upload, VBO/VAO/FBO creation) still runs
        unconditionally regardless of this value, so the GPU-side state
        the loop below depends on is identical whether start_frame is 0
        or not. CONTRACT when base_image/region_rg_array are animated
        iterables (not static Image/np.ndarray): the caller must already
        have built each so its first item corresponds to frame
        `start_frame` (e.g. by passing the same start_frame through to
        camera.apply_camera_drift / apply_camera_drift_to_array) - this
        function's own priming next() calls above (first_base_rgb/
        first_region_array) simply consume that already-correctly-
        positioned first item, and the loop's "frame_i > 0" advance
        checks below become "frame_i > start_frame" so no extra items are
        discarded to "catch up" (which would waste exactly the per-frame
        camera-drift resample work start_frame exists to skip). Each
        frame's shader output is otherwise a pure function of its own
        frame_i/effective_frame_i plus the fixed setup above, so resuming
        at start_frame reproduces bit-identical frames to an
        uninterrupted run from frame 0, just without paying the GPU cost
        of the frames already rendered. 0 (default) = old behavior.
    """
    if gust_kind not in ("plain", "elastic"):
        raise ValueError(f"gust_kind must be 'plain' or 'elastic', got {gust_kind!r}")

    context = gpu.get_context()
    program = gpu.load_shader_program("region_warp.glsl")

    program["resolution"].value = size
    program["num_frames"].value = float(max(num_frames, 1))
    program["noise_scale"].value = noise_scale
    program["noise_loop_radius"].value = noise_loop_radius
    program["wind_bias"].value = tuple(wind_bias)
    program["flutter_scale"].value = flutter_scale
    program["flutter_cycles"].value = flutter_cycles
    program["wave_direction"].value = tuple(wave_direction)
    program["wave_frequency"].value = wave_frequency
    program["wave_cycles"].value = wave_cycles
    program["gap_flicker_strength"].value = gap_flicker_strength
    program["gap_flicker_scale"].value = gap_flicker_scale
    # drift_strength/flutter_strength/wave_strength are set ONCE here, at
    # their RAW (unscaled) values - gust scaling used to be
    # a CPU-precomputed frame-global scalar baked into these three
    # uniforms every frame; the shader now computes its OWN per-fragment
    # gust value (see region_warp.glsl's gust_at()) using the raw
    # ingredients set inside the loop below, so a gust reaches the
    # downwind side of a wide frame later than the upwind side instead of
    # the whole region swaying in perfect lockstep.
    program["drift_strength"].value = drift_strength
    program["flutter_strength"].value = flutter_strength
    program["wave_strength"].value = wave_strength

    animated_base = not isinstance(base_image, Image.Image)
    base_frames_iter = iter(base_image) if animated_base else None

    def _prepare_base_rgb(image):
        rgb = image.convert("RGB")
        return rgb.resize(size) if rgb.size != size else rgb

    first_base_rgb = _prepare_base_rgb(next(base_frames_iter) if animated_base else base_image)
    base_texture = context.texture(size, 3, first_base_rgb.tobytes())
    base_texture.use(location=0)
    program["base_texture"].value = 0

    animated_region = not isinstance(region_rg_array, np.ndarray)
    region_frames_iter = iter(region_rg_array) if animated_region else None
    first_region_array = next(region_frames_iter) if animated_region else region_rg_array

    region_texture = context.texture(size, 2, np.ascontiguousarray(first_region_array, dtype="uint8").tobytes())
    region_texture.use(location=1)
    program["region_texture"].value = 1

    # a texture is bound either way (a 1x1 dummy when
    # no real depth is available), same pattern as fog.py's depth_texture;
    # the shader's own has_depth flag decides whether it's actually used.
    if depth_array is not None:
        depth_resized = depth_array
        if depth_resized.shape[:2][::-1] != size:
            depth_image = Image.fromarray((np.clip(depth_array, 0.0, 1.0) * 255.0).astype("uint8"), mode="L")
            depth_resized = np.asarray(depth_image.resize(size), dtype="float32") / 255.0
        depth_uint8 = np.clip(depth_resized * 255.0, 0.0, 255.0).astype("uint8")
        depth_texture = context.texture(size, 1, np.ascontiguousarray(depth_uint8).tobytes())
        program["has_depth"].value = 1.0
    else:
        depth_texture = context.texture((1, 1), 1, bytes([128]))
        program["has_depth"].value = 0.0
    depth_texture.use(location=2)
    program["depth_texture"].value = 2

    vbo = context.buffer(_FULLSCREEN_QUAD.tobytes())
    vao = context.vertex_array(program, [(vbo, "2f", "in_position")])

    fbo = context.framebuffer(color_attachments=[context.texture(size, 4)])
    context.enable(moderngl.BLEND)
    context.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    # whether THIS caller wants the elastic/damped-spring
    # response or the plain instantaneous
    # envelope (cloud.py/water.py's default) - an explicit tag the caller
    # sets, not an identity check against a specific function object (see
    # this function's gust_kind docstring for why), used to pick which
    # spring-response amplitude/lag values gust_shader_uniforms bakes in
    # for the shader's own gust_at().
    uses_elastic_gust = gust_kind == "elastic"

    try:
        for frame_i in range(start_frame, num_frames):
            # Rebind every frame: moderngl's `.use(location=N)` binds to a
            # GL texture unit that's GLOBAL to the shared context, not
            # scoped to this generator. compositor.layer_effects()
            # interleaves multiple effect generators (rain, fog,
            # scene_warp) frame-by-frame via zip(), so another generator's
            # one-time setup can run in between two of THIS generator's
            # frames and steal units 0/1/2 (see particles.py's render_rain
            # for the confirmed real-world case this caused elsewhere: a
            # texture-unit collision producing a strong red color cast).
            # Rebind here so scene_warp is immune too.
            base_texture.use(location=0)
            region_texture.use(location=1)
            depth_texture.use(location=2)

            if animated_base and frame_i > start_frame:
                base_texture.write(_prepare_base_rgb(next(base_frames_iter)).tobytes())
            if animated_region and frame_i > start_frame:
                region_texture.write(np.ascontiguousarray(next(region_frames_iter), dtype="uint8").tobytes())

            # phase_offset_frames shifts only the theta-driven wind/gust
            # phase, never which real frame's base image/region mask gets
            # sampled above - that stays tied to the loop's own frame_i.
            effective_frame_i = frame_i + phase_offset_frames
            gust_uniforms = weather.gust_shader_uniforms(effective_frame_i, num_frames, elastic=uses_elastic_gust)
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
            program["gust_wind_bias"].value = gust_uniforms["gust_wind_bias"]

            fbo.use()
            context.clear(0.0, 0.0, 0.0, 0.0)
            # Blending is OFF for this draw: one full-screen quad writes the
            # shader's straight (non-premultiplied) RGBA directly. A plain
            # SRC_ALPHA blend onto the transparent buffer would store
            # colour*alpha and alpha*alpha, which the compositor then reads
            # as straight alpha - darkening every feathered edge and every
            # alpha dip (gap flicker showed up as dark blocks over hazy
            # forest). The context state is global and shared with the other
            # interleaved effect generators, so it is restored afterwards.
            context.disable(moderngl.BLEND)
            vao.render(moderngl.TRIANGLES)
            context.enable(moderngl.BLEND)

            raw_pixels = fbo.read(components=4, dtype="f1")
            frame = Image.frombytes("RGBA", size, raw_pixels).transpose(Image.FLIP_TOP_BOTTOM)
            yield frame
    finally:
        vao.release()
        vbo.release()
        base_texture.release()
        region_texture.release()
        depth_texture.release()
        fbo.release()
        program.release()
