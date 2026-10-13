"""
main.py

Main entry point. Takes every photo from input_images/, applies the effects
enabled in config.py (snow, rain, fog, light flicker) and saves the result
in output/.

All settings (which effects are enabled, how strong, which format to save
the result in) are changed in config.py - not in this file.

Running (from inside the vfx_overlay/ folder):

    python main.py
"""

import json
import math

from PIL import Image

import config
from vfx_engine import camera, color, color_grade, compositor, frame_cache, god_rays, ground_fx, image_io, lens_dirt, lens_rain, lighting, mask, particles, presets, provenance, render_checkpoint
from vfx_engine.cloud import render_cloud_drift
from vfx_engine.fog import render_fog
from vfx_engine.lighting import apply_ambient_drift, apply_flicker
from vfx_engine.lightning import apply_flash, build_strike_schedule, render_lightning, resolve_strike_geometry
from vfx_engine.postprocess import apply_postprocess
from vfx_engine.progress import with_progress
from vfx_engine import particles_rain
from vfx_engine.particles_rain import render_rain
from vfx_engine.particles_snow import render_snow
from vfx_engine.vegetation import render_grass_sway, render_palm_sway, render_vegetation_sway
from vfx_engine.water import render_water_ripple
from vfx_engine import weather, weather_grade


def resolve_num_frames(output_mode):
    """A static image only needs one frame; an animation (gif/mp4) needs however many frames follow from the duration (config.VIDEO_DURATION_SECONDS) and frame rate (config.FPS)."""
    return 1 if output_mode == "image" else round(config.VIDEO_DURATION_SECONDS * config.FPS)


def resolve_camera_composition_kwargs(image):
    """
    Returns the zoom_pattern/pan_bias_angle_deg kwargs every camera-drift
    call site in this file should share - one shared
    dict, the same "single detection, reused everywhere" pattern
    resolve_light_pos_uv below already established for the photo's
    detected light source, so the base photo, its depth/mask companions
    (resolve_base_frames/resolve_mask_frames), and vegetation/cloud/rain's
    own drift-sync (_resolve_scene_warp_drift_kwargs/
    resolve_particle_depth_realism) all pick the SAME zoom shape and pan
    bias for one photo, instead of computing it differently in different
    places.

    zoom_pattern comes straight from config.CAMERA_ZOOM_PATTERN
.

    pan_bias_angle_deg is None (old content-agnostic behavior, unchanged)
    when config.CAMERA_COMPOSITION_AWARE_ENABLED is off or no image is
    available - otherwise it's the angle (image pixel convention: 0 =
    due right, 90 = straight down, matching vfx_engine.camera._camera_path)
    from the photo's center toward its own detected dominant light source
    (see resolve_light_pos_uv/god_rays.detect_light_source) - "the pan/zoom path is generic, not composition-aware." Rotating the
    pan ellipse toward the photo's own most visually interesting point
    (the same detection god rays/lens flare/rain's dual-tone lighting
    already reuse) is a real, if modest, step toward the "cinematographer
    chooses pan direction based on composition" ideal in the original design
    describes, without needing full saliency/subject detection to do it.
    """
    zoom_pattern = config.CAMERA_ZOOM_PATTERN
    if not config.CAMERA_COMPOSITION_AWARE_ENABLED or image is None:
        return dict(zoom_pattern=zoom_pattern, pan_bias_angle_deg=None)

    light_x, light_y = god_rays.detect_light_source(image)
    width, height = image.size
    dx = light_x - width / 2.0
    dy = light_y - height / 2.0
    pan_bias_angle_deg = math.degrees(math.atan2(dy, dx))
    return dict(zoom_pattern=zoom_pattern, pan_bias_angle_deg=pan_bias_angle_deg)


def resolve_base_frames(image, num_frames, depth_array=None, depth_confidence=None, start_frame=0):
    """
    Returns either the original static `image` (num_frames == 1, i.e. a
    still image, or config.CAMERA_DRIFT_ENABLED is off), or a generator of
    num_frames slowly zoomed/panned copies of it (see vfx_engine/camera.py)
    for compositor.composite_onto_base to drift the photo itself under the
    effects instead of keeping it perfectly pixel-locked - a static image
    only needs one frame in the first place, so drift would be meaningless
    there.

    depth_array: optional - when given (and config.CAMERA_PARALLAX_ENABLED),
        uses real depth-driven multi-layer parallax instead of the old
        flat single-transform drift, so foreground content visibly slides
        across frame faster than background content during the pan. Falls
        back to the flat drift when no depth data is available.

    depth_confidence: optional - `depth_array`'s matching
        per-pixel confidence field, from depth.py's estimate_depth(...,
        return_confidence=True). When given, the layered-parallax layer
        split (vfx_engine.camera._compute_layer_depth_centers) weights its
        percentiles by confidence, so a large low-confidence region (sky,
        calm water) can't skew where the near/mid/far layer boundaries
        land as easily as a well-textured, high-confidence region can.
        Must be the SAME value (or both None) passed to
        resolve_mask_frames for this same render - see
        apply_layered_camera_drift_to_depth's docstring for why.

    start_frame: render-checkpoint/resume support (see render_checkpoint.py)
        - forwarded into the flat (non-parallax) camera.apply_camera_drift
        path, which is fully start_frame-aware, so the base photo stream
        composite_onto_base zips against an already start_frame-truncated
        overlay stream stays frame-for-frame aligned with it instead of
        starting back over at frame 0. The CAMERA_PARALLAX_ENABLED branch
        (camera.apply_layered_camera_drift) has NO start_frame parameter at
        all - this is the same accepted, documented, deliberately deferred
        gap already called out in resolve_mask_frames's own docstring for
        its matching apply_layered_camera_drift_to_depth call (parallax is
        off by default; wiring start_frame through the full layered-
        parallax path is out of scope for this pass).
    """
    if num_frames <= 1 or not config.CAMERA_DRIFT_ENABLED:
        return image
    composition_kwargs = resolve_camera_composition_kwargs(image)
    if config.CAMERA_PARALLAX_ENABLED and depth_array is not None:
        return camera.apply_layered_camera_drift(
            image, depth_array, num_frames,
            zoom_amplitude=config.CAMERA_ZOOM_AMPLITUDE,
            pan_amplitude=config.CAMERA_PAN_AMPLITUDE,
            pan_phase_deg=config.CAMERA_PAN_PHASE_DEG,
            depth_confidence=depth_confidence,
            **composition_kwargs,
        )
    return camera.apply_camera_drift(
        image, num_frames,
        zoom_amplitude=config.CAMERA_ZOOM_AMPLITUDE,
        pan_amplitude=config.CAMERA_PAN_AMPLITUDE,
        pan_phase_deg=config.CAMERA_PAN_PHASE_DEG,
        start_frame=start_frame,
        **composition_kwargs,
    )


def resolve_postprocess_params(image, num_frames, size):
    """
    Returns config.POSTPROCESS_PARAMS, with motion_blur's "vectors" key
 filled in from THIS render's own actual camera-
    drift path - reusing the exact same config.CAMERA_ZOOM_AMPLITUDE/
    CAMERA_PAN_AMPLITUDE/CAMERA_PAN_PHASE_DEG/resolve_camera_composition_
    kwargs values resolve_base_frames above already used to drift this
    same photo, so the blur direction/magnitude can never silently drift
    out of sync with the camera path actually visible on screen.

    motion_blur is off (None) by default - see config.py's own comment on
    why - so this returns config.POSTPROCESS_PARAMS completely unchanged
    whenever it's off, or whenever there's no camera drift to derive a
    vector from at all (num_frames <= 1, or config.CAMERA_DRIFT_ENABLED
    is off), matching resolve_base_frames' own static-frame fallback.
    """
    params = dict(config.POSTPROCESS_PARAMS)
    # RNG-determinism audit (2026-09-17): content-derived seed for
    # apply_postprocess's grain RNG, so re-rendering the same photo always
    # draws the same grain pattern - see lens_dirt.content_seed's docstring.
    params["seed"] = lens_dirt.content_seed(image, salt="grain") if image is not None else None
    motion_blur = params.get("motion_blur")
    if motion_blur is None or num_frames <= 1 or not config.CAMERA_DRIFT_ENABLED:
        return params

    composition_kwargs = resolve_camera_composition_kwargs(image)
    width, height = size
    vectors = camera.compute_global_drift_pixel_vectors(
        num_frames, width, height,
        zoom_amplitude=config.CAMERA_ZOOM_AMPLITUDE,
        pan_amplitude=config.CAMERA_PAN_AMPLITUDE,
        pan_phase_deg=config.CAMERA_PAN_PHASE_DEG,
        **composition_kwargs,
    )
    params["motion_blur"] = dict(motion_blur, vectors=vectors)
    return params


def resolve_effect_suffix():
    """
    Returns a filename suffix like "_rain", "_snow", "_rain_snow" listing
    which of the enabled effects actually produced this output - so
    re-running with different effects on the same photo never produces an
    identically-named file as a previous, different-looking result, and
    it's obvious from the filename alone what's in it without opening it.
    """
    parts = []
    if config.SNOW_ENABLED:
        parts.append("snow")
    if config.RAIN_ENABLED:
        parts.append("rain")
    if config.FOG_ENABLED:
        parts.append("fog")
    if config.FLICKER_ENABLED:
        parts.append("flicker")
    if config.LIGHTNING_ENABLED:
        parts.append("lightning")
    return ("_" + "_".join(parts)) if parts else ""


_OUTPUT_EXTENSIONS = {"image": "png", "gif": "gif", "mp4": "mp4"}


def predict_output_path(image_path, output_mode):
    """
    Predicts the exact path image_io.save_result() will save this photo's
    output to, WITHOUT rendering anything - mirrors save_result's own
    stem+suffix+extension logic exactly (see vfx_engine/image_io.py) so the
    two can never drift apart silently. Used by resolve_resume_skip below to
    check for an already-finished output before spending any render time.
    """
    stem = image_path.stem + resolve_effect_suffix()
    return config.OUTPUT_DIR / f"{stem}.{_OUTPUT_EXTENSIONS[output_mode]}"


def resolve_resume_skip(image_path):
    """
    Whole-photo-level resume for a multi-photo batch (explicitly requested
    by the user: "in the case the run gets interrupted there would be no
    need to calculate from the beginning"), complementing the per-
    computation caching already built into vfx_engine/cache_store.py
    (depth/segmentation/surface-normals results, keyed by image content +
    relevant config). That per-computation cache alone doesn't help a
    photo whose full render already finished before an interruption -
    process_image would still redo mask-building, every enabled effect,
    and final encoding for it. This check closes that gap: it predicts the
    output path and provenance sidecar (vfx_engine/provenance.py) this
    photo would produce under the CURRENT config, and if a sidecar already
    exists recording that the exact same source photo (by name AND raw
    content hash, so a file re-exported/re-edited under its old filename
    isn't mistaken for the original - see provenance._file_content_hash)
    was rendered to that exact path under the exact same config.py
    settings and manual mask file(s), the work is already done - skip it
    entirely instead of re-rendering.

    Deliberately strict: ANY mismatch (missing output file, missing or
    unreadable sidecar, different source image name OR content, a changed
    manual mask file, different output mode, or even a single differing
    config field) means "render it" rather than risk silently reusing a
    stale or partial result - this is a pure speed optimization on top of
    correctness, never the other way around.

    Returns True if this photo's render is already up to date and should
    be skipped, False if it should be rendered.
    """
    output_mode = config.resolve_output_mode()
    out_path = predict_output_path(image_path, output_mode)
    sidecar_path = out_path.with_name(out_path.name + ".provenance.json")
    if not out_path.exists() or not sidecar_path.exists():
        return False

    try:
        with open(sidecar_path, "r", encoding="utf-8") as sidecar_file:
            saved_provenance = json.load(sidecar_file)
    except (OSError, json.JSONDecodeError):
        return False

    if saved_provenance.get("source_image") != image_path.name:
        return False
    if saved_provenance.get("source_image_hash") != provenance._file_content_hash(image_path):
        return False
    if saved_provenance.get("output_mode") != output_mode:
        return False
    if saved_provenance.get("config") != provenance._snapshot_config():
        return False

    current_mask_hashes = {
        mask_path.name: provenance._file_content_hash(mask_path)
        for mask_path in provenance.resolve_mask_paths(image_path)
    }
    if saved_provenance.get("mask_hashes", {}) != current_mask_hashes:
        return False

    return True


def any_effect_needs_mask():
    """Checks in config.py whether at least one enabled effect should be limited by a mask."""
    return (
        (config.SNOW_ENABLED and config.SNOW_USE_MASK)
        or (config.RAIN_ENABLED and config.RAIN_USE_MASK)
        or (config.FOG_ENABLED and config.FOG_USE_MASK)
        or (config.FLICKER_ENABLED and config.FLICKER_USE_MASK)
    )


def resolve_mask_frames(image, image_path, depth_array, num_frames, shared_mask, segmentation=None, depth_confidence=None, start_frame=0):
    """
    Returns a zero-argument callable that produces a fresh mask each time
    it's called - used for SNOW_USE_MASK/RAIN_USE_MASK/FOG_USE_MASK (see
    render_enabled_effects), each of which needs its OWN independent mask
    "stream" rather than sharing one.

    That independence only actually matters in one case: MASK_MODE =
    "depth" with camera drift active (see config.CAMERA_DRIFT_ENABLED) -
    there, mask.build_mask returns a single-use GENERATOR of per-frame
    masks (see vfx_engine.camera.apply_camera_drift_to_depth), one
    resampled through each frame's own zoom/pan crop so the occlusion
    mask tracks where the "camera" has actually drifted to. If snow AND
    rain both used that same generator, whichever consumed it first would
    exhaust it and starve the other - so this builds a brand new one on
    every call instead.

    In every other case (no camera drift, or a non-depth MASK_MODE) the
    mask is a single, cheap-to-build static numpy array - `shared_mask`
    (already computed once by the caller) is just returned unchanged, no
    need to rebuild it per effect.

    segmentation: optional dict from vfx_engine.segmentation.estimate_segmentation
        - passed straight through to mask.build_mask; ignored when that
        config setting is None.

    depth_confidence: optional - see resolve_base_frames's
        docstring; passed straight through to
        vfx_engine.camera.apply_layered_camera_drift_to_depth and MUST be
        the same value resolve_base_frames was called with for this same
        render, so the depth frames this factory yields stay in the exact
        same per-layer registration as the photo frames they're meant to
        occlude-mask.

    start_frame: render-checkpoint/resume support (see render_checkpoint.py) -
        baked into the returned factory() closure so callers of it (snow/
        rain/fog's SNOW_USE_MASK/RAIN_USE_MASK/FOG_USE_MASK paths in
        render_enabled_effects) don't need to pass anything extra
        themselves. Only wired into the non-parallax
        camera.apply_camera_drift_to_depth branch below (confirmed
        start_frame-ready) - the CAMERA_PARALLAX_ENABLED branch
        (apply_layered_camera_drift_to_depth) is off by default and is a
        documented, deliberately deferred gap, matching resolve_base_frames'
        own parallax branch which has the same limitation.
    """
    needs_per_frame = (
        config.CAMERA_DRIFT_ENABLED and num_frames > 1
        and config.MASK_MODE == "depth" and depth_array is not None
    )
    if not needs_per_frame:
        return lambda: shared_mask

    def factory():
        # Must match resolve_base_frames's own composition kwargs exactly
        # - a mask drifting through a different zoom
        # shape/pan bias than the photo it's meant to occlude would track
        # the wrong framing.
        composition_kwargs = resolve_camera_composition_kwargs(image)
        if config.CAMERA_PARALLAX_ENABLED:
            # Same multi-layer parallax the photo itself now drifts
            # through - see resolve_base_frames - so the
            # occlusion mask tracks the SAME per-layer motion the photo
            # actually shows, not a flat single-transform approximation
            # of it.
            depth_frames = camera.apply_layered_camera_drift_to_depth(
                image, depth_array, num_frames,
                zoom_amplitude=config.CAMERA_ZOOM_AMPLITUDE,
                pan_amplitude=config.CAMERA_PAN_AMPLITUDE,
                pan_phase_deg=config.CAMERA_PAN_PHASE_DEG,
                depth_confidence=depth_confidence,
                **composition_kwargs,
            )
        else:
            depth_frames = camera.apply_camera_drift_to_depth(
                depth_array, num_frames,
                zoom_amplitude=config.CAMERA_ZOOM_AMPLITUDE,
                pan_amplitude=config.CAMERA_PAN_AMPLITUDE,
                pan_phase_deg=config.CAMERA_PAN_PHASE_DEG,
                start_frame=start_frame,
                **composition_kwargs,
            )
        return mask.build_mask(image, image_path, depth_array=depth_array, depth_frames=depth_frames, segmentation=segmentation)

    return factory


def resolve_light_pos_uv(image):
    """
    Returns the photo's detected dominant light source position (see
    vfx_engine/god_rays.py's detect_light_source), normalized to 0.0-1.0
    screen coordinates for use as a shader uniform - or None if `image`
    isn't available. Shared by god rays, lens flare, and
    particles_rain.glsl's dual-tone lighting (see resolve_particle_depth_realism
    below) - one detection, reused everywhere "where's the light coming
    from in this photo" matters.
    """
    if image is None:
        return None
    light_x, light_y = god_rays.detect_light_source(image)
    width, height = image.size
    return (light_x / width, light_y / height)


def resolve_flicker_mask(image, shared_mask):
    """
    returns the mask apply_flicker's `mask` parameter
    should use. When config.FLICKER_AUTO_LIGHT_MASK is on, builds a soft
    falloff mask centered on the photo's own auto-detected dominant light
    position (resolve_light_pos_uv + lighting.build_light_position_mask -
    the same detection every other light-coupled effect in this pipeline
    already reuses), so flicker automatically scopes itself to wherever
    the photo's actual light source is. Falls back to the pre-existing
    FLICKER_USE_MASK/shared_mask behavior (a generic, light-unaware
    depth/segmentation mask) unchanged whenever the new flag is off, or
    whenever no light could be detected (image is None) - so nothing
    about an existing render changes unless FLICKER_AUTO_LIGHT_MASK is
    explicitly turned on.
    """
    if config.FLICKER_AUTO_LIGHT_MASK:
        light_pos_uv = resolve_light_pos_uv(image)
        if light_pos_uv is not None:
            width, height = image.size
            return lighting.build_light_position_mask(width, height, light_pos_uv, radius_frac=config.FLICKER_LIGHT_MASK_RADIUS)
    return shared_mask if config.FLICKER_USE_MASK else None


def resolve_rain_angle_deg(num_frames):
    """
    Returns the droplet tilt (degrees from vertical) render_rain should use
    this render, replacing the old
    RAIN_PARAMS["angle"] fixed independent value.

    Ties the tilt to the SAME shared wind direction every other wind-driven
    effect already agrees on (weather.wind_direction_vector - see
    vfx_engine/weather.py's module docstring) instead of an unrelated,
    independently-tuned number: the wind's horizontal (+x/-x) component
    scales config.RAIN_WIND_ANGLE_MAX_DEG, so a wind blowing due
    horizontal tilts rain at the full max angle, a wind blowing straight
    up/down imparts no sideways tilt at all (real vertically-falling rain
    in a headwind/tailwind doesn't slant sideways), and anything between
    scales smoothly. The vertical wind component is intentionally unused
    here - rain's `angle_deg` in render_particles is a single sideways
    slant-from-vertical value, not a full 2D vector; a wind mostly blowing
    "up" has no sideways component to represent regardless.

    num_frames <= 1 (a single static image, config.OUTPUT_MODE = "png")
    additionally scales the result down by
    config.RAIN_STATIC_IMAGE_ANGLE_SCALE: a strongly wind-tilted rain
    streak with nothing else in the single frame able to show that same
    wind (vegetation sway/cloud drift/water ripple all need num_frames > 1
    to animate at all) reads as an isolated, unmotivated slant rather than
    "this scene has wind in it" - toning it down keeps the implied wind
    subtle enough not to call attention to that mismatch. Animated output
    (gif/mp4/live-wallpaper, num_frames > 1) is unaffected, since real
    scene motion is visible there to back the tilt up.
    """
    wind_x, _wind_y = weather.wind_direction_vector()
    angle = wind_x * config.RAIN_WIND_ANGLE_MAX_DEG
    if num_frames <= 1:
        angle *= config.RAIN_STATIC_IMAGE_ANGLE_SCALE
    return angle


def resolve_wind_motion_advisory(num_frames):
    """
    Returns a one-line advisory string to print for this render, or None if
    there's nothing worth flagging.

    Static PNG output (num_frames <= 1) has no code path at all for
    wind-driven SCENE motion - render_vegetation_effects/
    render_cloud_effects/render_water_effects in this file all early-return
    [] below that threshold, since there's nowhere for a single frame to
    animate sway/drift/ripple across. resolve_rain_angle_deg above already
    softens rain's own tilt for that case (config.RAIN_STATIC_IMAGE_ANGLE_
    SCALE) so it doesn't visually overstate wind the rest of the frame
    can't back up - but if the CONFIGURED wind (config.WEATHER_WIND_
    DIRECTION_DEG, measured here on the UNSCALED tilt so this reflects how
    windy the scene is actually set up to be, not how subtle the softened
    render ends up looking) is strong enough that even the softened tilt
    would read as "this scene is windy," a static image still can't show
    the branches/water/mist agreeing with that. Rather than silently
    leaving that mismatch for the user to notice (or forcing everyone onto
    slower animated output), this surfaces it as a printed suggestion only
    - config.RAIN_STATIC_WIND_ADVISORY_THRESHOLD_DEG controls the
    sensitivity; which OUTPUT_MODE to actually use stays the user's call.
    """
    if num_frames > 1 or not config.RAIN_ENABLED:
        return None
    wind_x, _wind_y = weather.wind_direction_vector()
    unscaled_angle = abs(wind_x * config.RAIN_WIND_ANGLE_MAX_DEG)
    if unscaled_angle < config.RAIN_STATIC_WIND_ADVISORY_THRESHOLD_DEG:
        return None
    return (
        f"  Note: WEATHER_WIND_DIRECTION_DEG={config.WEATHER_WIND_DIRECTION_DEG:g} implies "
        f"fairly strong wind (~{unscaled_angle:.1f} deg rain tilt before static-image "
        "softening), but OUTPUT_MODE=\"image\" can't show wind-driven scene motion "
        "(vegetation sway/cloud drift/water ripple all require an animated OUTPUT_MODE). "
        "Consider OUTPUT_MODE=\"gif\" or \"mp4\" for a scene this windy."
    )


def resolve_particle_depth_realism(image, depth_array, num_frames=1, start_frame=0):
    """
    Returns the kwargs render_snow/render_rain should get for depth-driven
    realism (see config.PARTICLE_DEPTH_REALISM), or an empty dict when it's
    off/unavailable - so render_enabled_effects can just splat this in
    regardless of whether depth data ended up being computed.

    num_frames: also folds in a FRESH per-frame camera-drifted photo
    stream (base_image_frames) via _resolve_scene_warp_drift_kwargs -
    the original design's third named-but-previously-unfixed subsystem (the other
    two, vegetation/cloud, were fixed via that same helper) - so rain/snow
    atmosphere/backlight/refraction sampling stays in registration with a
    simultaneously camera-drifting photo too, not just vegetation sway and
    cloud drift. Call this separately per effect (not once and reused) -
    each needs its OWN independent generator, same reasoning as
    _resolve_scene_warp_drift_kwargs's docstring.

    start_frame: render-checkpoint/resume support (see render_checkpoint.py) -
        forwarded to _resolve_scene_warp_drift_kwargs so the base_image_frames
        stream it may build (if any) already starts at frame start_frame,
        keeping it in registration with render_snow/render_rain's own
        start_frame-truncated frame loop instead of starting back at frame 0.
    """
    if not config.PARTICLE_DEPTH_REALISM or depth_array is None:
        return {}
    kwargs = dict(
        depth_array=depth_array, base_image=image, haze_strength=config.HAZE_STRENGTH,
        focus_depth=config.FOCUS_DEPTH, bokeh_factor=config.BOKEH_FACTOR,
        backlight_strength=config.BACKLIGHT_STRENGTH,
        shadow_strength=config.SHADOW_STRENGTH,
        light_pos_uv=resolve_light_pos_uv(image),
    )
    drift_kwargs = _resolve_scene_warp_drift_kwargs(image, num_frames, start_frame=start_frame)
    if "base_image_frames" in drift_kwargs:
        kwargs["base_image_frames"] = drift_kwargs["base_image_frames"]
    return kwargs


def render_god_ray_effects(size, num_frames, image, start_frame=0):
    """
    Returns a list of god-ray frame streams (see vfx_engine/god_rays.py),
    one per detected light source,
    each computed ONCE (light sources don't move over one loop) with just
    a cyclic intensity pulse varying per frame. Meant to be layered UNDER
    the falling particles (light fills the atmosphere; particles are lit
    by/visible within it) - unlike lens flare (see
    render_lens_flare_effects), which is a camera artifact and belongs on
    top of everything.
    """
    if image is None or not config.GOD_RAYS_ENABLED:
        return []

    # Fog x god-rays coupling - real light shafts are
    # visible BECAUSE fog/haze particulate scatters the light, so the
    # ray texture's visibility/diffuseness is coupled to the scene's
    # actual fog density instead of being two unrelated strength sliders
    # that happen to composite on the same frame.
    fog_density = config.FOG_PARAMS.get("density", 0.0) if config.FOG_ENABLED else 0.0
    # the same fixed ground_fog_strength
    # render_fog itself always uses (fog.py's render_fog signature default,
    # not currently exposed via FOG_PARAMS/config), reused here so the ray
    # texture's vertical falloff matches the ground fog actually being
    # rendered this session, and is 0.0 (no falloff) when fog is off.
    ground_fog_strength = 0.55 if config.FOG_ENABLED else 0.0

    light_sources = god_rays.detect_light_sources(
        image, max_sources=config.MAX_LIGHT_SOURCES,
        min_relative_strength=config.MIN_LIGHT_SOURCE_RELATIVE_STRENGTH,
    )
    # see _resolve_light_effect_drift_kwargs's docstring.
    # Empty dict (falsy) when camera drift is off/unavailable, so the
    # `if drift_kwargs` check below cleanly falls back to the original
    # static-texture behavior with no branching duplication.
    drift_kwargs = _resolve_light_effect_drift_kwargs(image, num_frames)
    overlays = []
    for source in light_sources:
        # Secondary sources are dimmer in reality too, not just detected as
        # such - scaling GOD_RAYS_STRENGTH by the source's own relative
        # brightness keeps a faint streetlight's shaft visibly fainter than
        # the dominant sun/lamp's, instead of every detected source
        # producing an identically-strong ray fan.
        ray_texture = god_rays.build_ray_texture(
            image, source["pos"], threshold=config.GOD_RAYS_THRESHOLD,
            strength=config.GOD_RAYS_STRENGTH * source["strength"], fog_density=fog_density,
            ground_fog_strength=ground_fog_strength,
        )
        if drift_kwargs:
            ray_texture = camera.apply_camera_drift_to_array(ray_texture, num_frames, start_frame=start_frame, **drift_kwargs)
        overlays.append(god_rays.render_light_overlay(
            size, num_frames, ray_texture, color=config.GOD_RAYS_COLOR, pulse_amount=config.GOD_RAYS_PULSE_AMOUNT,
            start_frame=start_frame,
        ))
    return overlays


def render_lens_flare_effects(size, num_frames, image, start_frame=0):
    """
    Returns a list of lens-flare frame streams (see vfx_engine/god_rays.py)
    - a camera-lens artifact, not scene content, so it belongs layered on
    TOP of everything else (particles, ground effects, lightning), unlike
    god rays (see render_god_ray_effects). One pass per detected light
    source,
    each scaled by that source's relative strength so secondary flares
    read as genuinely fainter reflections, not identical copies.

    Uses render_lens_flare_overlay (not the shared render_light_overlay)
    since the original design gave build_lens_flare_texture a per-pixel color
    texture alongside the alpha texture - per-ghost color variance that a
    single flat `color` tint can't express.
    """
    if image is None or not config.LENS_FLARE_ENABLED:
        return []

    light_sources = god_rays.detect_light_sources(
        image, max_sources=config.MAX_LIGHT_SOURCES,
        min_relative_strength=config.MIN_LIGHT_SOURCE_RELATIVE_STRENGTH,
    )
    # see _resolve_light_effect_drift_kwargs's docstring.
    drift_kwargs = _resolve_light_effect_drift_kwargs(image, num_frames)
    overlays = []
    for source in light_sources:
        alpha_texture, color_texture = god_rays.build_lens_flare_texture(
            size, source["pos"], strength=config.LENS_FLARE_STRENGTH * source["strength"],
            ghost_count=config.LENS_FLARE_GHOST_COUNT, color=config.LENS_FLARE_COLOR,
        )
        if drift_kwargs:
            alpha_texture = camera.apply_camera_drift_to_array(alpha_texture, num_frames, start_frame=start_frame, **drift_kwargs)
            color_texture = camera.apply_camera_drift_to_color_array(color_texture, num_frames, start_frame=start_frame, **drift_kwargs)
        overlays.append(god_rays.render_lens_flare_overlay(
            size, num_frames, alpha_texture, color_texture, pulse_amount=0.15,
            start_frame=start_frame,
        ))
    return overlays


def render_starburst_effects(size, num_frames, image, start_frame=0):
    """
    Returns a list of starburst/diffraction-spike frame streams (see
    vfx_engine/god_rays.py's build_starburst_texture) - a camera-lens
    artifact like lens flare, so it belongs layered on TOP of everything
    else alongside it. One pass per detected light source, each scaled
    by that source's relative strength.
    """
    if image is None or not config.STARBURST_ENABLED:
        return []

    light_sources = god_rays.detect_light_sources(
        image, max_sources=config.MAX_LIGHT_SOURCES,
        min_relative_strength=config.MIN_LIGHT_SOURCE_RELATIVE_STRENGTH,
    )
    # see _resolve_light_effect_drift_kwargs's docstring.
    drift_kwargs = _resolve_light_effect_drift_kwargs(image, num_frames)
    overlays = []
    for source in light_sources:
        starburst_texture = god_rays.build_starburst_texture(
            size, source["pos"], num_spikes=config.STARBURST_SPIKE_COUNT,
            spike_length_fraction=config.STARBURST_LENGTH_FRACTION,
            angular_sharpness=config.STARBURST_ANGULAR_SHARPNESS,
            strength=config.STARBURST_STRENGTH * source["strength"],
        )
        if drift_kwargs:
            starburst_texture = camera.apply_camera_drift_to_array(starburst_texture, num_frames, start_frame=start_frame, **drift_kwargs)
        overlays.append(god_rays.render_light_overlay(
            size, num_frames, starburst_texture, color=config.STARBURST_COLOR,
            pulse_amount=config.STARBURST_PULSE_AMOUNT,
            start_frame=start_frame,
        ))
    return overlays


def render_lens_dirt_effects(size, num_frames, image, illumination_result=None, start_frame=0):
    """
    Returns a list (0 or 1 items) of lens-dirt frame streams (see
    vfx_engine/lens_dirt.py) - grime/dust on the lens itself, a camera
    artifact like lens flare, layered on top of everything else. The grime
    PATTERN itself is independent of scene content (it's on the LENS, not
    in the scene) but `image` is now also used for the original design's light-
    source coupling: the same detected light position/color that already
    feeds god rays, lens flare, and ground_fx.render_wet_sheen so grime scatters/glows near the light instead of only
    ever darkening uniformly. `image` is also used for the original design
    brightness-dependent visibility - grime reads more strongly over
    bright regions of the photo (open sky) than dark ones (shadowed
    foliage), instead of applying uniformly regardless of what's behind
    the lens dirt.

    `build_dirt_texture`'s noise seed is derived from
    the photo's own content (lens_dirt.content_seed) instead of the
    previous unseeded (fully OS-entropy-random every call) default, so
    the same photo re-rendered reproduces the same grime layout while
    different photos still get their own distinct one.

    the glow SHAPE (not the static grime pattern
    itself, which must stay frame-invariant) is built once from the
    static, undrifted light position and then, when camera drift is
    active, warped through the exact same
    _resolve_light_effect_drift_kwargs/camera.apply_camera_drift_to_array
    path already used for god rays/lens flare/starburst -
    so lens dirt's veiling-glare glow tracks the light's actual per-frame
    drifted screen position instead of freezing at wherever the light sat
    in the single static frame the position was sampled from, staying
    consistent with every other light-coupled effect during playback.
    """
    if image is None or not config.LENS_DIRT_ENABLED:
        return []

    light_pos_px = god_rays.detect_light_source(image)
    light_pos_uv = (light_pos_px[0] / size[0], light_pos_px[1] / size[1])
    light_color = god_rays.sample_light_color(image, light_pos_px, illumination_result=illumination_result)

    glow_shape = lens_dirt.build_light_glow_field(size, light_pos_uv, config.LENS_DIRT_GLOW_RADIUS)
    drift_kwargs = _resolve_light_effect_drift_kwargs(image, num_frames)
    if drift_kwargs:
        glow_shape = camera.apply_camera_drift_to_array(glow_shape, num_frames, start_frame=start_frame, **drift_kwargs)

    dirt_texture = lens_dirt.build_dirt_texture(
        size, seed=lens_dirt.content_seed(image),
        speck_scale=config.LENS_DIRT_SPECK_SCALE, speck_threshold=config.LENS_DIRT_SPECK_THRESHOLD,
        speck_edge_power=config.LENS_DIRT_SPECK_EDGE_POWER, speck_weight=config.LENS_DIRT_SPECK_WEIGHT,
    )
    return [lens_dirt.render_lens_dirt(
        size, num_frames, dirt_texture, strength=config.LENS_DIRT_STRENGTH,
        light_pos_uv=light_pos_uv, light_color=light_color,
        glow_strength=config.LENS_DIRT_GLOW_STRENGTH, glow_radius=config.LENS_DIRT_GLOW_RADIUS,
        glow_shape=glow_shape,
        image=image, brightness_min_visibility=config.LENS_DIRT_BRIGHTNESS_MIN_VISIBILITY,
        brightness_max_visibility=config.LENS_DIRT_BRIGHTNESS_MAX_VISIBILITY,
        brightness_blur_fraction=config.LENS_DIRT_BRIGHTNESS_BLUR_FRACTION,
        start_frame=start_frame,
    )]


def render_lens_rain_effects(size, num_frames, image, illumination_result=None, start_frame=0):
    """
    Returns a list (0 or 1 items) of lens-rain frame streams (see
    vfx_engine/lens_rain.py) - soft, out-of-focus water droplets sitting
    directly on the camera lens itself, a camera artifact layered on top
    of everything else (same category as render_lens_dirt_effects above,
    which this mirrors structurally), distinct from lens_dirt's static
    dust/grime and from particles_rain's sharp in-scene falling drops.

    the original design: gated to heavy-rain intensity
    specifically (config.LENS_RAIN_REQUIRED_INTENSITY, "storm" by default)
    - light/medium rain doesn't throw enough water at an exposed lens for
    this to be a realistic thing to see, so this only ever activates
    alongside RAIN_ENABLED at that intensity.

    Camera-drift fix: each droplet's own screen position must stay fixed
    (it's stuck to the lens, not the scene), but what it optically
    refracts of the scene behind it has to keep tracking the actual
    panned/zoomed framing whenever CAMERA_DRIFT_ENABLED is on, or it reads
    as a dark, misregistered "hole" that desyncs from the real background
    as the pan progresses. `_resolve_scene_warp_drift_kwargs` already
    builds exactly the fresh per-frame drifted-photo generator this needs
    (the same one vegetation/grass/palm sway consume) - lens_rain only
    needs its `base_image_frames` key, not `camera_drift_params` (which
    exists for drifting a full-frame mask/texture, not raw photo content).
    """
    if (image is None or not config.LENS_RAIN_ENABLED or not config.RAIN_ENABLED
            or config.RAIN_PARAMS.get("intensity") != config.LENS_RAIN_REQUIRED_INTENSITY):
        return []

    light_pos_px = god_rays.detect_light_source(image)
    light_pos_uv = (light_pos_px[0] / size[0], light_pos_px[1] / size[1])
    light_color = god_rays.sample_light_color(image, light_pos_px, illumination_result=illumination_result)

    drift_kwargs = _resolve_scene_warp_drift_kwargs(image, num_frames, start_frame=start_frame)

    return [lens_rain.render_lens_rain(
        size, num_frames, image,
        droplet_count=config.LENS_RAIN_DROPLET_COUNT,
        radius_range=config.LENS_RAIN_RADIUS_RANGE,
        elongation_range=config.LENS_RAIN_ELONGATION_RANGE,
        magnification=config.LENS_RAIN_MAGNIFICATION,
        blur_radius_frac=config.LENS_RAIN_BLUR_RADIUS_FRACTION,
        strength=config.LENS_RAIN_STRENGTH,
        rim_darken_strength=config.LENS_RAIN_RIM_DARKEN_STRENGTH,
        wobble_px=config.LENS_RAIN_WOBBLE_PX,
        pulse_amount=config.LENS_RAIN_PULSE_AMOUNT,
        pulse_speed=config.LENS_RAIN_PULSE_SPEED,
        highlight_strength=config.LENS_RAIN_HIGHLIGHT_STRENGTH,
        light_pos_uv=light_pos_uv, light_color=light_color,
        start_frame=start_frame,
        base_image_frames=drift_kwargs.get("base_image_frames"),
    )]


def resolve_segmentation(image, depth_array, num_frames):
    """
    Returns a dict from vfx_engine.segmentation.estimate_segmentation, or
    None when nothing currently enabled would actually use it - this
    project's segmentation consumers today are render_ground_effects
    (needs a depth map too, to build the ground heuristic segmentation
    then gates - see ground_fx.build_ground_mask), render_vegetation_effects,
    render_cloud_effects, render_water_effects, and mask.build_mask's
    optional MASK_SEGMENTATION_CATEGORY combination layer. Skipping the estimate entirely when
    none of these apply avoids loading/running a whole extra model for
    nothing.
    """
    needs_for_ground = config.SEGMENTATION_ENABLED and config.GROUND_FX_ENABLED and depth_array is not None
    needs_for_vegetation = config.SEGMENTATION_ENABLED and config.VEGETATION_SWAY_ENABLED and num_frames > 1
    needs_for_cloud = config.SEGMENTATION_ENABLED and config.CLOUD_DRIFT_ENABLED and num_frames > 1
    needs_for_water = config.SEGMENTATION_ENABLED and config.WATER_RIPPLE_ENABLED and num_frames > 1
    needs_for_mask = config.SEGMENTATION_ENABLED and config.MASK_SEGMENTATION_CATEGORY is not None and any_effect_needs_mask()
    if not (needs_for_ground or needs_for_vegetation or needs_for_cloud or needs_for_water or needs_for_mask):
        return None

    from vfx_engine import segmentation as segmentation_module
    return segmentation_module.estimate_segmentation(image)


def _resolve_scene_warp_drift_kwargs(image, num_frames, start_frame=0):
    """
    Returns the kwargs vegetation.py/cloud.py's render_*_sway/render_cloud_drift
    need to stay in registration with camera drift - a
    fresh per-frame drifted-photo generator (base_image_frames) plus the
    camera-drift parameters needed to resample their own masks through
    the identical path (camera_drift_params) - or an empty dict when
    camera drift is off, num_frames <= 1, or no image is available
    (matching every other camera-drift guard in this file).

    A FRESH generator every call (not a shared one) - vegetation/grass/
    cloud each need their OWN independent per-frame photo stream, since a
    generator consumed by one would be exhausted for the others. Uses the
    same FLAT apply_camera_drift path 8.3/9.6's own write-ups reference
    (not the layered parallax) - both findings were
    written against, and explicitly ask for, that original mechanism.

    Also folds in resolve_camera_composition_kwargs so
    vegetation sway/cloud drift/rain-snow atmosphere sampling all track
    the SAME zoom shape/pan bias resolve_base_frames picked for the photo
    itself, not the old always-"breathe"/never-biased defaults.

    start_frame: render-checkpoint/resume support (see render_checkpoint.py) -
        forwarded straight into camera.apply_camera_drift so base_image_frames'
        first yielded item already corresponds to frame start_frame, matching
        every other start_frame-aware generator's own truncated frame loop
        rather than restarting the drifted-photo stream back at frame 0.
    """
    if not config.CAMERA_DRIFT_ENABLED or image is None or num_frames <= 1:
        return {}
    camera_drift_params = dict(
        zoom_amplitude=config.CAMERA_ZOOM_AMPLITUDE,
        pan_amplitude=config.CAMERA_PAN_AMPLITUDE,
        pan_phase_deg=config.CAMERA_PAN_PHASE_DEG,
        **resolve_camera_composition_kwargs(image),
    )
    return dict(
        base_image_frames=camera.apply_camera_drift(image, num_frames, start_frame=start_frame, **camera_drift_params),
        camera_drift_params=camera_drift_params,
    )


def _resolve_light_effect_drift_kwargs(image, num_frames):
    """
    Returns the camera_drift_params dict render_god_ray_effects/
    render_lens_flare_effects/render_starburst_effects need to keep their
    ray-shaft/ghost-chain/starburst-spike textures registered to the
    light source's ACTUAL drifted screen position each frame. Empty dict when camera drift is off,
    num_frames <= 1, or no image is available - callers check for that
    and fall back to the original static-texture behavior untouched.

    Deliberately returns only the drift PARAMETERS, not a pre-built
    generator like _resolve_scene_warp_drift_kwargs does for the photo
    itself: unlike the base photo (one shared drift path for everyone)
    or a mask (one per consumer but built once per call), each of these
    three effects builds its OWN texture from a light source's position
    on the STATIC original image first (build_ray_texture/
    build_lens_flare_texture/build_starburst_texture all need the
    undrifted image/position to sample from), and only THEN needs to
    drift that resulting texture - so the actual
    camera.apply_camera_drift_to_array/apply_camera_drift_to_color_array
    generator has to be constructed fresh, per light source, at the
    point each texture is built, using these same params every time so
    every source's texture (and the base photo itself) drifts through
    the identical camera path.
    """
    if not config.CAMERA_DRIFT_ENABLED or image is None or num_frames <= 1:
        return {}
    return dict(
        zoom_amplitude=config.CAMERA_ZOOM_AMPLITUDE,
        pan_amplitude=config.CAMERA_PAN_AMPLITUDE,
        pan_phase_deg=config.CAMERA_PAN_PHASE_DEG,
        **resolve_camera_composition_kwargs(image),
    )


def render_vegetation_effects(size, num_frames, image, segmentation, depth_array=None, start_frame=0):
    """
    Returns a list (0-4 items) of vegetation-sway frame streams (see
    vfx_engine/vegetation.py) - real image-space wind displacement of the
    photo's own tree/plant/palm/flower/grass pixels, from AI segmentation.
    Empty when disabled, no image/segmentation is available, or there's
    only a single static frame to begin with (nothing to animate).

    vegetation is no longer one undifferentiated mask/motion
    model - segmentation.py now splits it into three per-species
    categories (vegetation_tree, vegetation_palm, vegetation_flower), each
    routed through the motion model that actually matches how that kind
    of plant moves in wind: tree/plant keeps the original canopy-height-
    weighted sway (render_vegetation_sway); palm gets its own reduced-
    drift/rapid-flutter frond whip (render_palm_sway); flower is routed
    through the SAME low-stiffness traveling-wave motion grass uses
.

    Grass/field (segmentation's "ground_soft" category) still gets its
    OWN traveling-wave sway (render_grass_sway), separate from tree/
    plant's canopy-height-weighted sway - grass has none of
    a tree's rigid trunk-to-canopy structure, and the two were producing
    genuinely wrong-looking motion when run through the same model (see
    render_grass_sway's docstring for why grass previously got no sway at
    all, not just an under-weighted one).

    depth_array: optional - passed straight through to
        both render_grass_sway calls below (flower and grass/field) so
        their traveling wave reads depth-appropriate spacing/speed across
        a mask that spans a real distance range. Not used by tree/palm
        sway, which have no traveling-wave term to modulate. None
        (default) = old flat, depth-independent wave.
    """
    if not config.VEGETATION_SWAY_ENABLED or image is None or segmentation is None or num_frames <= 1:
        return []

    # A FRESH generator per consumer below (not one shared dict) - see
    # _resolve_scene_warp_drift_kwargs's docstring for why.
    streams = []
    tree_mask = segmentation.get("vegetation_tree")
    if tree_mask is not None:
        streams.append(render_vegetation_sway(
            size, num_frames, image, tree_mask,
            wind_strength=config.VEGETATION_SWAY_STRENGTH, gust_scale=config.VEGETATION_GUST_SCALE,
            start_frame=start_frame,
            **_resolve_scene_warp_drift_kwargs(image, num_frames, start_frame=start_frame),
        ))

    palm_mask = segmentation.get("vegetation_palm")
    if palm_mask is not None:
        streams.append(render_palm_sway(
            size, num_frames, image, palm_mask,
            wind_strength=config.VEGETATION_SWAY_STRENGTH, gust_scale=config.VEGETATION_GUST_SCALE,
            start_frame=start_frame,
            **_resolve_scene_warp_drift_kwargs(image, num_frames, start_frame=start_frame),
        ))

    flower_mask = segmentation.get("vegetation_flower")
    if flower_mask is not None:
        streams.append(render_grass_sway(
            size, num_frames, image, flower_mask,
            wind_strength=config.VEGETATION_SWAY_STRENGTH * config.GRASS_SWAY_STRENGTH_MULTIPLIER,
            gust_scale=config.VEGETATION_GUST_SCALE,
            depth_array=depth_array,
            start_frame=start_frame,
            **_resolve_scene_warp_drift_kwargs(image, num_frames, start_frame=start_frame),
        ))

    grass_mask = segmentation.get("ground_soft")
    if grass_mask is not None:
        streams.append(render_grass_sway(
            size, num_frames, image, grass_mask,
            wind_strength=config.VEGETATION_SWAY_STRENGTH * config.GRASS_SWAY_STRENGTH_MULTIPLIER,
            gust_scale=config.VEGETATION_GUST_SCALE,
            depth_array=depth_array,
            start_frame=start_frame,
            **_resolve_scene_warp_drift_kwargs(image, num_frames, start_frame=start_frame),
        ))

    return streams


def render_cloud_effects(size, num_frames, image, segmentation, start_frame=0):
    """
    Returns a list (0 or 1 items) of cloud-drift frame streams (see
    vfx_engine/cloud.py) - real image-space wind displacement of the
    photo's own sky pixels, from AI segmentation. Same guards as
    render_vegetation_effects above.
    """
    if not config.CLOUD_DRIFT_ENABLED or image is None or segmentation is None or num_frames <= 1:
        return []
    sky_mask = segmentation.get("sky")
    if sky_mask is None:
        return []
    return [render_cloud_drift(
        size, num_frames, image, sky_mask,
        drift_strength=config.CLOUD_DRIFT_STRENGTH,
        start_frame=start_frame,
        **_resolve_scene_warp_drift_kwargs(image, num_frames, start_frame=start_frame),
    )]


def render_water_effects(size, num_frames, image, segmentation, start_frame=0):
    """
    Returns a list (0 or 1 items) of water-ripple frame streams (see
    vfx_engine/water.py, the original design) - real image-space wind
    displacement of the photo's own water pixels, from AI segmentation.
    Same guards as render_vegetation_effects/render_cloud_effects above.
    """
    if not config.WATER_RIPPLE_ENABLED or image is None or segmentation is None or num_frames <= 1:
        return []
    water_mask = segmentation.get("water")
    if water_mask is None:
        return []
    return [render_water_ripple(
        size, num_frames, image, water_mask,
        ripple_strength=config.WATER_RIPPLE_STRENGTH,
        start_frame=start_frame,
        **_resolve_scene_warp_drift_kwargs(image, num_frames, start_frame=start_frame),
    )]


def render_ground_effects(size, num_frames, depth_array, segmentation=None, image=None, ambient_tint_grid=None, illumination_result=None, start_frame=0):
    """
    Returns a list of ground-interaction frame streams (see
    vfx_engine/ground_fx.py) for whichever of rain/snow are enabled -
    wet-sheen + splashes for rain, a gather-and-settle build-up for snow.
    Returns an empty list when config.GROUND_FX_ENABLED is off or no depth
    map is available (ground_fx needs one to guess where "ground" is).

    segmentation: optional dict from vfx_engine.segmentation.estimate_segmentation
        (see config.SEGMENTATION_ENABLED) - sharpens ground_fx's ground
        mask beyond the depth/position heuristic alone (see
        ground_fx.build_ground_mask's docstring). Rain and snow each build
        their OWN ground mask below (not a shared one): rain wants only
        "hard" ground (paved/bare - a real wet-sheen surface), snow also
        wants "soft" ground (grass/field - snow settles there too, but it
        wouldn't get a wet-sheen highlight from rain) - see
        build_ground_mask's include_soft_ground param.
    image: the original photo - passed through to render_splashes so splash
        color can be sampled from the scene's own ambient light instead of a fixed constant.
    ambient_tint_grid: the SAME coarse per-photo color grid
        (see vfx_engine.particles.build_ambient_tint_grid) passed to
        render_rain in render_enabled_effects, computed once per photo by
        the caller and threaded through to render_splashes here so rain
        and splashes agree on local color instead of each inventing its
        own. None keeps splashes on the old whole-photo light_color/
        sample_ambient_tint behavior.
    """
    if not config.GROUND_FX_ENABLED or depth_array is None:
        return []

    frame_iterables = []

    # One shared roughness field per photo - reused
    # across wet sheen/darkening/snow accumulation below, so all three
    # read as variation in the SAME physical surface rather than each
    # inventing its own unrelated pattern. One detected-light-color sample
    # similarly shared between wet sheen and splashes.
    roughness = ground_fx.build_roughness_field(size)
    light_color = None
    ground_light_pos_uv = None
    if image is not None:
        light_pos = god_rays.detect_light_source(image)
        light_color = god_rays.sample_light_color(image, light_pos, illumination_result=illumination_result)
        ground_light_pos_uv = (light_pos[0] / size[0], light_pos[1] / size[1])

    # normals (surface_normals.py) + a pseudo light direction
    # derived from the same ground_light_pos_uv above, plus (the original design
    # 9.9(a)) illumination.py's per-pixel albedo when available - feeds
    # render_wet_sheen's materials.shade_surface BRDF path below. Any of
    # these being unavailable (ENABLE_SURFACE_NORMALS off, no image, no
    # detected light) falls through to render_wet_sheen's old flat-
    # highlight_color behavior automatically (see that function's
    # docstring) - never a hard failure.
    wet_sheen_normals = None
    wet_sheen_light_dir = None
    wet_sheen_albedo = None
    if image is not None and config.ENABLE_SURFACE_NORMALS:
        try:
            from vfx_engine import surface_normals
            wet_sheen_normals = surface_normals.estimate_surface_normals(image)
        except RuntimeError as exc:
            print(f"  (surface normals unavailable, wet sheen falling back to flat highlight_color - {exc})")
        if ground_light_pos_uv is not None:
            from vfx_engine import materials
            wet_sheen_light_dir = materials.resolve_pseudo_light_direction(ground_light_pos_uv)
        if illumination_result is not None:
            wet_sheen_albedo = illumination_result.get("albedo")

    if config.RAIN_ENABLED:
        ground_mask_hard = ground_fx.build_ground_mask(depth_array, segmentation=segmentation, include_soft_ground=False, image=image)
        frame_iterables.append(ground_fx.render_wet_darkening(
            size, num_frames, ground_mask_hard, strength=config.WET_DARKENING_STRENGTH, roughness=roughness,
            start_frame=start_frame,
        ))
        frame_iterables.append(ground_fx.render_wet_sheen(
            size, num_frames, ground_mask_hard, strength=config.WET_SHEEN_STRENGTH,
            roughness=roughness, highlight_color=light_color or (255, 255, 255),
            normals=wet_sheen_normals, light_dir=wet_sheen_light_dir,
            material_category="ground_hard", illumination_albedo=wet_sheen_albedo,
            start_frame=start_frame,
        ))
        frame_iterables.append(ground_fx.render_splashes(
            size, num_frames, ground_mask_hard, count=config.SPLASH_COUNT, base_image=image, light_color=light_color,
            light_pos_uv=ground_light_pos_uv, ambient_tint_grid=ambient_tint_grid,
            start_frame=start_frame,
        ))

    if config.SNOW_ENABLED:
        ground_mask_all = ground_fx.build_ground_mask(depth_array, segmentation=segmentation, include_soft_ground=True, image=image)
        frame_iterables.append(
            ground_fx.render_snow_accumulation(
                size, num_frames, ground_mask_all, max_opacity=config.SNOW_ACCUMULATION_OPACITY, roughness=roughness,
                start_frame=start_frame,
            )
        )

    return frame_iterables


def _build_layered_rain_intensity(base_intensity, count_scale, size_scale, speed_scale):
    """
    derives a near/far rain preset by scaling whatever
    preset config.RAIN_PARAMS["intensity"] already resolves to (a name
    like "storm" or a fully custom dict), rather than hardcoding a second,
    separate set of presets that would silently drift out of sync with
    whatever intensity the user actually configured.
    """
    base = base_intensity if isinstance(base_intensity, dict) else particles_rain._INTENSITY_PRESETS.get(
        base_intensity, particles_rain._INTENSITY_PRESETS["medium"])
    size_lo, size_hi = base["size_range"]
    speed_lo, speed_hi = base["speed_range"]
    return dict(
        count=max(1, int(round(base["count"] * count_scale))),
        size_range=(max(1.0, size_lo * size_scale), max(1.0, size_hi * size_scale)),
        speed_range=(speed_lo * speed_scale, speed_hi * speed_scale),
        drift=base["drift"],
    )


def render_layered_rain(size, num_frames, rain_params, angle, ambient_tint_grid, depth_kwargs_factory, image=None, start_frame=0):
    """
    renders rain as two independently-tuned depth layers (a
    small/sharp/numerous "far" layer and a big/blurry/sparse "near" layer)
    instead of one continuous depth gradient. A single depth-driven
    gradient can already size/speed/blur particles continuously by real
    scene depth (see resolve_particle_depth_realism), but it can't give
    the near and far ends of that gradient their own streak length, drop
    size, or wind angle the way two visually distinct layers of real rain
    at different distances actually look.

    rain_params: config.RAIN_PARAMS (or an equivalent dict) - "intensity"
        is scaled per-layer (see _build_layered_rain_intensity), "blur"/
        "motion_blur_samples"/"shutter_fraction" are shared unchanged by
        both layers.
    depth_kwargs_factory: a zero-argument callable returning a FRESH
        resolve_particle_depth_realism(...) dict each call (mirroring
        mask_frames_factory's pattern elsewhere in this file) - each layer
        needs its OWN independent base_image_frames generator, not one
        shared object two render_rain calls would race to exhaust.

    Returns [far_frames, near_frames] in that order, meant to be appended
    to effect_frame_iterables in that same order so the near layer's
    bigger/closer drops draw on top of the far layer's smaller ones,
    matching how compositor.layer_effects/composite_onto_base draws
    effect_frame_iterables in list order - no manual alpha-compositing
    code needed here.
    """
    base_intensity = rain_params.get("intensity", "medium")
    shared = {k: v for k, v in rain_params.items() if k not in ("intensity", "blur")}
    blur = rain_params.get("blur", True)

    # RNG-determinism audit (2026-09-17): each layer needs its OWN
    # distinct-but-still-deterministic seed (same salted-content-hash
    # reasoning as render_snow's call site above) - a shared seed between
    # the two layers would make the near/far spawn patterns visibly
    # correlated instead of looking like two independent rain passes.
    far_seed = lens_dirt.content_seed(image, salt="rain_far") if image is not None else None
    near_seed = lens_dirt.content_seed(image, salt="rain_near") if image is not None else None

    far_cfg = config.RAIN_FAR_LAYER
    far_frames = render_rain(
        size, num_frames,
        intensity=_build_layered_rain_intensity(
            base_intensity, far_cfg["count_scale"], far_cfg["size_scale"], far_cfg["speed_scale"]),
        angle=angle + far_cfg["angle_offset_deg"],
        blur=blur,
        velocity_stretch_scale=far_cfg["velocity_stretch_scale"],
        depth_bias_power=far_cfg["depth_bias_power"],
        ambient_tint_grid=ambient_tint_grid,
        seed=far_seed,
        start_frame=start_frame,
        **shared, **depth_kwargs_factory(),
    )

    near_cfg = config.RAIN_NEAR_LAYER
    near_frames = render_rain(
        size, num_frames,
        intensity=_build_layered_rain_intensity(
            base_intensity, near_cfg["count_scale"], near_cfg["size_scale"], near_cfg["speed_scale"]),
        angle=angle + near_cfg["angle_offset_deg"],
        blur=blur,
        velocity_stretch_scale=near_cfg["velocity_stretch_scale"],
        depth_bias_power=near_cfg["depth_bias_power"],
        ambient_tint_grid=ambient_tint_grid,
        seed=near_seed,
        start_frame=start_frame,
        **shared, **depth_kwargs_factory(),
    )
    return [far_frames, near_frames]


def render_enabled_effects(size, num_frames, mask_frames_factory, lightning_schedule=None, lightning_geometry=None, image=None, depth_array=None, segmentation=None, illumination_result=None, start_frame=0):
    """
    Renders every effect enabled in config.py and returns a list of frame
    streams (generators) - one stream per effect.

    mask_frames_factory: see main.resolve_mask_frames - a zero-argument
        callable returning a fresh mask (static array or per-frame
        generator) each call, so SNOW_USE_MASK/RAIN_USE_MASK/FOG_USE_MASK
        each get their own independent mask instead of racing to exhaust
        one shared one. None when any_effect_needs_mask() was False. Its
        own start_frame alignment (if any) is baked in by the caller when
        it built the factory (see resolve_mask_frames) - not this
        function's concern.
    segmentation: optional dict from vfx_engine.segmentation.estimate_segmentation
        (see config.SEGMENTATION_ENABLED) - passed through to
        render_ground_effects, render_vegetation_effects and
        render_cloud_effects, see their docstrings.
    start_frame: render-checkpoint/resume support (see render_checkpoint.py)
        - forwarded into every effect call below so each effect's frame
        stream starts at frame start_frame instead of frame 0, matching
        the num_frames-start_frame length that layer_effects'/
        composite_onto_base's zip() further downstream will expect once
        the base photo/camera-drift stream is truncated the same way.
    """
    effect_frame_iterables = []

    # one coarse local-color grid computed ONCE per photo and
    # shared by rain and splashes below - each drop/splash looks up its OWN
    # region's color instead of every particle in the scene sharing a
    # single whole-photo average. Deliberately NOT threaded into
    # resolve_particle_depth_realism (shared by render_snow too): snow's
    # signature has no matching parameter, and 3.3 explicitly scopes local
    # tint to rain + splash only.
    ambient_tint_grid = particles.build_ambient_tint_grid(image) if image is not None else None

    if config.SNOW_ENABLED:
        # RNG-determinism audit (2026-09-17): pass an explicit content-
        # derived seed regardless of whether PARTICLE_DEPTH_REALISM's own
        # base_image kwarg happens to be active below, so re-rendering the
        # same photo twice always draws the same flake spawn pattern (a
        # prerequisite for the planned frame-level render-checkpoint/
        # resume feature - see lens_dirt.content_seed's docstring).
        snow_seed = lens_dirt.content_seed(image, salt="snow") if image is not None else None
        frames = render_snow(size, num_frames, **config.SNOW_PARAMS, seed=snow_seed, start_frame=start_frame,
                              **resolve_particle_depth_realism(image, depth_array, num_frames, start_frame=start_frame))
        if config.SNOW_USE_MASK:
            frames = mask.apply_mask_to_frames(frames, mask_frames_factory())
        effect_frame_iterables.append(frames)

    if config.RAIN_ENABLED:
        rain_angle = resolve_rain_angle_deg(num_frames)
        # only meaningful with real depth data to split into
        # near/far bands - falls back to the old single continuous-depth
        # rain pass otherwise, same gating as PARTICLE_DEPTH_REALISM
        # elsewhere in this pipeline.
        use_rain_layers = config.RAIN_DEPTH_LAYERS_ENABLED and config.PARTICLE_DEPTH_REALISM and depth_array is not None
        if use_rain_layers:
            far_frames, near_frames = render_layered_rain(
                size, num_frames, config.RAIN_PARAMS, rain_angle, ambient_tint_grid,
                lambda: resolve_particle_depth_realism(image, depth_array, num_frames, start_frame=start_frame),
                image=image, start_frame=start_frame,
            )
            if config.RAIN_USE_MASK:
                far_frames = mask.apply_mask_to_frames(far_frames, mask_frames_factory())
                near_frames = mask.apply_mask_to_frames(near_frames, mask_frames_factory())
            effect_frame_iterables.append(far_frames)
            effect_frame_iterables.append(near_frames)
        else:
            # RNG-determinism audit (2026-09-17): explicit seed, independent
            # of whether PARTICLE_DEPTH_REALISM's own base_image kwarg is
            # active below - see the SNOW_ENABLED block above for the same
            # reasoning.
            rain_seed = lens_dirt.content_seed(image, salt="rain") if image is not None else None
            frames = render_rain(size, num_frames, **config.RAIN_PARAMS,
                                  angle=rain_angle,
                                  ambient_tint_grid=ambient_tint_grid,
                                  seed=rain_seed,
                                  start_frame=start_frame,
                                  **resolve_particle_depth_realism(image, depth_array, num_frames, start_frame=start_frame))
            if config.RAIN_USE_MASK:
                frames = mask.apply_mask_to_frames(frames, mask_frames_factory())
            effect_frame_iterables.append(frames)

    # Ground effects, god rays, vegetation sway, cloud drift, and water
    # ripple are all layered UNDER the falling particles above (each
    # prepended to the list, in reverse of their intended bottom-to-top
    # order) so splashes/wet-sheen/accumulation sit on the scene surface,
    # light shafts fill the atmosphere, and swaying trees/drifting sky/
    # rippling water are part of the SCENE itself - with the particles
    # still falling in front of all of it. Vegetation/cloud/water go
    # absolute bottom-most (prepended last): they replace/warp the static
    # scene's own pixels, so everything else (god rays' light, ground
    # effects' surface detail) should sit visually on TOP of them, not
    # the other way around.
    effect_frame_iterables[0:0] = render_ground_effects(size, num_frames, depth_array, segmentation=segmentation, image=image, ambient_tint_grid=ambient_tint_grid, illumination_result=illumination_result, start_frame=start_frame)
    effect_frame_iterables[0:0] = render_god_ray_effects(size, num_frames, image, start_frame=start_frame)
    effect_frame_iterables[0:0] = render_vegetation_effects(size, num_frames, image, segmentation, depth_array=depth_array, start_frame=start_frame)
    effect_frame_iterables[0:0] = render_cloud_effects(size, num_frames, image, segmentation, start_frame=start_frame)
    effect_frame_iterables[0:0] = render_water_effects(size, num_frames, image, segmentation, start_frame=start_frame)

    if config.FOG_ENABLED:
        fog_depth_array = depth_array if config.FOG_DEPTH_AWARE else None
        # Fog x light coupling - fog brightens/tints
        # toward the photo's actual detected light color specifically
        # AROUND the light position, instead of being one flat, position-
        # agnostic tint - see fog.py/blur.glsl.
        fog_light_pos_uv = None
        fog_light_color = None
        if image is not None:
            fog_light_pos_px = god_rays.detect_light_source(image)
            fog_light_pos_uv = (fog_light_pos_px[0] / size[0], fog_light_pos_px[1] / size[1])
            fog_light_color = god_rays.sample_light_color(image, fog_light_pos_px, illumination_result=illumination_result)
        frames = render_fog(
            size, num_frames, **config.FOG_PARAMS, depth_array=fog_depth_array, depth_power=config.FOG_DEPTH_POWER,
            depth_smoothing_px=config.FOG_DEPTH_SMOOTHING_PX, light_pos_uv=fog_light_pos_uv, light_color=fog_light_color,
            base_image=image if config.FOG_GROUND_TEXTURE_ENABLED else None,
            ground_texture_strength=config.FOG_GROUND_TEXTURE_STRENGTH,
            start_frame=start_frame,
        )
        if config.FOG_USE_MASK:
            frames = mask.apply_mask_to_frames(frames, mask_frames_factory())
        effect_frame_iterables.append(frames)

    if config.LIGHTNING_ENABLED:
        params = config.LIGHTNING_PARAMS
        effect_frame_iterables.append(render_lightning(
            size, lightning_schedule, lightning_geometry, color=params["color"],
            branch_chance=params["branch_chance"], strike_frames=params["strike_frames"],
            start_frame=start_frame,
        ))

    # Lens flare, starburst spikes, lens dirt, and lens rain are camera
    # artifacts, not scene content - they belong on top of literally
    # everything else (unlike god rays, see above, which are scene light
    # and sit under the particles). Lens rain is added last so a droplet
    # sitting on top of existing grime/flare reads correctly (more visible
    # than either alone), matching real footage where a wet lens shows
    # everything else through/around the water.
    effect_frame_iterables.extend(render_lens_flare_effects(size, num_frames, image, start_frame=start_frame))
    effect_frame_iterables.extend(render_starburst_effects(size, num_frames, image, start_frame=start_frame))
    effect_frame_iterables.extend(render_lens_dirt_effects(size, num_frames, image, illumination_result=illumination_result, start_frame=start_frame))
    effect_frame_iterables.extend(render_lens_rain_effects(size, num_frames, image, illumination_result=illumination_result, start_frame=start_frame))

    return effect_frame_iterables


def process_image(image_path):
    """Processes one photo: applies every enabled effect and saves the result in output/."""
    print(f"Processing: {image_path.name}")

    output_mode = config.resolve_output_mode()

    image = image_io.load_base_image(image_path)
    output_size = image.size
    num_frames = resolve_num_frames(output_mode)

    wind_motion_advisory = resolve_wind_motion_advisory(num_frames)
    if wind_motion_advisory:
        print(wind_motion_advisory)

    # Spatial supersampling: render the ENTIRE pipeline at
    # config.SUPERSAMPLE_FACTOR times the final output resolution, then
    # box-filter-downsample once at the very end (see the matching
    # downsample step further below) - rain streaks and small snowflakes
    # otherwise only have the shaders' own smoothstep edge-softening for
    # anti-aliasing, which visibly aliases/shimmers frame to frame at
    # native resolution. Upscaling the base photo itself here, before
    # anything else runs, means every downstream consumer (depth,
    # segmentation, every effect module) automatically operates at the
    # higher working resolution with no changes needed anywhere else -
    # `size` below IS the supersampled size for the rest of this function.
    ssaa_factor = max(1, int(config.SUPERSAMPLE_FACTOR))
    if ssaa_factor > 1:
        render_size = (output_size[0] * ssaa_factor, output_size[1] * ssaa_factor)
        image = image.resize(render_size, Image.LANCZOS)
    size = image.size

    # Computed once and reused for both masking (MASK_MODE = "depth") and
    # particle depth-realism (config.PARTICLE_DEPTH_REALISM) - estimating
    # depth twice would be wasted work since both want the same map.
    depth_array = None
    depth_confidence = None
    if config.MASK_MODE == "depth" or config.PARTICLE_DEPTH_REALISM:
        from vfx_engine import depth as depth_module
        depth_array, depth_confidence = depth_module.estimate_depth(image, return_confidence=True)

    # AI scene segmentation (see vfx_engine/segmentation.py) - see
    # resolve_segmentation's docstring for exactly which effects consume it.
    segmentation = resolve_segmentation(image, depth_array, num_frames)

    # learned-illumination: illumination.py's Marigold intrinsic-image
    # decomposition (albedo/shading), computed ONCE per photo here (like
    # depth_array/segmentation above) and threaded through to whichever
    # effects want it - lighting.estimate_ambient_light_color (via
    # god_rays.sample_light_color) prefers its `shading` channel over a
    # blurred-photo heuristic, and ground_fx.render_wet_sheen's materials.
    # shade_surface BRDF prefers its `albedo` channel over a flat per-
    # category tint (see resolve_ground_effects/render_enabled_effects).
    # This was previously a "dormant, unintegrated exploratory spike" (see
    # lighting.py's own prior docstring wording) - this is its first real
    # integration. Wrapped in try/except: the Marigold checkpoint is
    # already cached locally, but estimate_illumination
    # still raises a plain RuntimeError on any other model-loading failure
    # (see that module's docstring) - a cosmetic lighting refinement
    # should never take down an entire render over that.
    illumination_result = None
    if config.ENABLE_LEARNED_ILLUMINATION:
        try:
            from vfx_engine import illumination as illumination_module
            illumination_result = illumination_module.estimate_illumination(image)
        except RuntimeError as exc:
            print(f"  (learned illumination unavailable, falling back to heuristics - {exc})")

    # Base-photo color-grade unification pass. Runs here, after segmentation
    # (so it can use the real AI sky mask when one was already computed for
    # another effect) but before shared_mask/render_enabled_effects/
    # resolve_base_frames touch `image` at all - every downstream consumer
    # of `image` (masking, every effect's own color sampling, the visible
    # base frames themselves) automatically inherits the corrected, unified
    # grade with no changes needed anywhere else.
    if config.COLOR_GRADE_UNIFY_ENABLED:
        image = color_grade.unify_color_grade(
            image, segmentation=segmentation, strength=config.COLOR_GRADE_UNIFY_STRENGTH,
        )
    if config.WEATHER_GRADE_STYLE:
        image = weather_grade.apply_weather_grade(image, config.WEATHER_GRADE_STYLE, config.WEATHER_GRADE_STRENGTH)

    # Frame-level render checkpoint/resume - distinct
    # from resolve_resume_skip's WHOLE-PHOTO batch resume above: this
    # tracks progress WITHIN one photo's render, so an interruption partway
    # through a long animated render doesn't force redoing every frame
    # already written. Reuses the exact same fingerprint fields
    # resolve_resume_skip already validates a completed render against
    # (source image name+content hash, output mode, full config snapshot,
    # manual mask file hashes) plus num_frames itself - see
    # render_checkpoint.build_fingerprint. Only meaningful for actual
    # animations (num_frames > 1); a single still image has nothing
    # mid-render to resume from, so this is skipped entirely for it, same
    # gating as config.AMBIENT_DRIFT_ENABLED and with_progress below.
    start_frame = 0
    fingerprint = None
    if num_frames > 1:
        source_image_hash = provenance._file_content_hash(image_path)
        config_snapshot = provenance._snapshot_config()
        mask_hashes = {
            mask_path.name: provenance._file_content_hash(mask_path)
            for mask_path in provenance.resolve_mask_paths(image_path)
        }
        fingerprint = render_checkpoint.build_fingerprint(
            image_path, source_image_hash, output_mode, config_snapshot, mask_hashes, num_frames,
        )
        start_frame = render_checkpoint.load_checkpoint(fingerprint)
        if start_frame > 0:
            print(f"  Resuming render from frame {start_frame}/{num_frames} (checkpoint found)")

    shared_mask = mask.build_mask(image, image_path, depth_array=depth_array, segmentation=segmentation) if any_effect_needs_mask() else None
    mask_frames_factory = resolve_mask_frames(image, image_path, depth_array, num_frames, shared_mask, segmentation=segmentation, depth_confidence=depth_confidence, start_frame=start_frame)

    # Built once so render_lightning() (the bolt overlay) and apply_flash()
    # (the scene-wide brightening, applied further below) strike on exactly
    # the same frames, with the same bolt endpoint - see lightning.py.
    lightning_schedule = None
    lightning_geometry = None
    if config.LIGHTNING_ENABLED:
        lightning_params = config.LIGHTNING_PARAMS
        lightning_schedule = build_strike_schedule(
            num_frames, fps=config.FPS,
            frequency=lightning_params["frequency"], strike_frames=lightning_params["strike_frames"],
        )
        lightning_geometry = resolve_strike_geometry(
            lightning_schedule, size, depth_array=depth_array, strike_frames=lightning_params["strike_frames"],
        )

    effect_frame_iterables = render_enabled_effects(
        size, num_frames, mask_frames_factory, lightning_schedule, lightning_geometry,
        image=image, depth_array=depth_array, segmentation=segmentation, illumination_result=illumination_result,
        start_frame=start_frame,
    )
    overlay_frames = compositor.layer_effects(size, num_frames, effect_frame_iterables, start_frame=start_frame)
    base_frames = resolve_base_frames(image, num_frames, depth_array=depth_array, depth_confidence=depth_confidence, start_frame=start_frame)
    final_frames = compositor.composite_onto_base(base_frames, overlay_frames)

    if config.FLICKER_ENABLED:
        flicker_mask = resolve_flicker_mask(image, shared_mask)
        # RNG-determinism audit (2026-09-17): see the SNOW_ENABLED block
        # above for why this is passed explicitly rather than relying on
        # config.FLICKER_PARAMS to carry it.
        flicker_seed = lens_dirt.content_seed(image, salt="flicker") if image is not None else None
        final_frames = apply_flicker(final_frames, num_frames, mask=flicker_mask, seed=flicker_seed, start_frame=start_frame, **config.FLICKER_PARAMS)

    if config.LIGHTNING_ENABLED:
        lightning_params = config.LIGHTNING_PARAMS
        final_frames = apply_flash(
            final_frames, lightning_schedule, lightning_geometry, intensity=lightning_params["intensity"],
            color=lightning_params["color"], strike_frames=lightning_params["strike_frames"],
            falloff_radius_frac=lightning_params["flash_falloff_radius_frac"],
            min_intensity_frac=lightning_params["flash_min_intensity_frac"],
            start_frame=start_frame,
        )

    if config.AMBIENT_DRIFT_ENABLED and num_frames > 1:
        drift_strength = config.AMBIENT_DRIFT_STRENGTH * lighting.resolve_ambient_drift_scale(image)
        final_frames = apply_ambient_drift(final_frames, num_frames, strength=drift_strength, start_frame=start_frame)

    final_frames = apply_postprocess(final_frames, **resolve_postprocess_params(image, num_frames, size), start_frame=start_frame)

    # Supersampling downsample - box-filter back down from
    # the supersampled working resolution to the real output resolution,
    # in LINEAR light (color.downsample_linear), before anything gets
    # quantized to 8-bit. A no-op (ssaa_factor == 1) when supersampling is
    # disabled.
    if ssaa_factor > 1:
        final_frames = (color.downsample_linear(frame, ssaa_factor) for frame in final_frames)

    # End of the linear-light compositing/post-processing chain - encode back to gamma-encoded 8-bit PIL Images exactly ONCE,
    # here, right before saving.
    final_frames = image_io.encode_frames_to_images(final_frames)

    # Frame-level render checkpoint/resume, continued from the fingerprint/
    # start_frame block above: wraps the (num_frames - start_frame) newly-
    # encoded frames with frame_cache.resume_or_checkpoint_frames, which
    # prepends the start_frame already-finished frames read back from disk
    # and caches+checkpoints each new one as it's produced - see that
    # function's docstring. Placed HERE, right after encode_frames_to_images
    # and before with_progress, rather than inside image_io.save_result
    # itself: save_result's job is writing a specific container format
    # (mp4/gif/png) and knows nothing about resume state, while this
    # wrapper only needs a plain "one already-encoded PIL Image in, one
    # PIL Image out" generator - inserting it here keeps save_mp4/_save_gif
    # completely untouched and keeps with_progress's `total=num_frames`
    # math correct unchanged, since the combined stream is always exactly
    # num_frames items long regardless of start_frame.
    if fingerprint is not None:
        final_frames = frame_cache.resume_or_checkpoint_frames(final_frames, fingerprint, start_frame, num_frames)

    if num_frames > 1:
        final_frames = with_progress(final_frames, num_frames)

    out_path = image_io.save_result(final_frames, image_path, output_mode, suffix=resolve_effect_suffix())
    print(f"  Saved: {out_path.name}")

    # Now that save_result has fully consumed every frame without raising,
    # this render is genuinely complete - permanently delete its checkpoint
    # and cached frames. Left as plain
    # sequential statements (not inside a try/finally) so that an exception
    # anywhere above propagates out of process_image before reaching here,
    # leaving the checkpoint/frame cache intact for the next run to resume
    # from - deleting them on a failed/interrupted render would defeat the
    # entire point of this feature.
    if fingerprint is not None:
        render_checkpoint.delete_checkpoint(fingerprint)
        frame_cache.clear_frame_cache(fingerprint)

    # Render provenance - every rendered output, not just
    # an "approved" one, gets a small JSON sidecar recording the exact
    # config.py parameter set that produced it.
    provenance_path = provenance.save_provenance(out_path, image_path, num_frames, output_mode)
    print(f"  Saved: {provenance_path.name}")


def main():
    presets.apply_preset(config.ACTIVE_PRESET)
    if config.ACTIVE_PRESET is not None:
        print(f"Using preset: {config.ACTIVE_PRESET}")

    config.make_sure_folders_exist()

    images = image_io.find_input_images(config.INPUT_DIR)
    if not images:
        print(f"No photos found in: {config.INPUT_DIR}")
        print("Drop .png/.jpg/.jpeg/.bmp/.webp files there and run the program again.")
        return

    rendered_count = 0
    skipped_count = 0
    for image_path in images:
        if resolve_resume_skip(image_path):
            print(f"Skipping (already rendered under current settings): {image_path.name}")
            skipped_count += 1
            continue
        process_image(image_path)
        rendered_count += 1

    # Automatic end-of-run finalization (explicitly requested by the user
    # alongside the resumable cache itself: "automatically after the whole
    # program finishes working"). The actual cache writes already happen
    # incrementally, immediately after each expensive computation finishes
    # (see vfx_engine/cache_store.py's module docstring) - that's what
    # makes an interrupted run resumable in the first place, so there is
    # nothing left to flush here. What belongs here instead is a clear,
    # real (non-placeholder) summary of what this run actually resumed vs.
    # freshly computed, so the resume behavior is visible, not silent.
    print(f"\nDone! {len(images)} photo(s) total -> {config.OUTPUT_DIR}")
    print(f"  Rendered: {rendered_count}")
    print(f"  Skipped (already up to date from a previous run): {skipped_count}")
    cache_root = config.CACHE_DIR
    if cache_root.exists():
        cached_entry_count = sum(1 for _ in cache_root.rglob("*.npz"))
        print(f"  Resumable cache: {cached_entry_count} entrie(s) on disk in {cache_root}")


if __name__ == "__main__":
    main()
