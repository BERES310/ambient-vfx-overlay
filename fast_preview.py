"""
fast_preview.py

Renders a short, low-resolution DRAFT of the current config.* effect
settings entirely in memory (no video file, no two-pass ffmpeg encode) -
built for quick visual feedback while choosing parameters, not final
quality. See live_wallpaper.process_image for the real, full-quality
render this is a cheap stand-in for.

Two things make this fast instead of taking minutes like the real render:
    - The image is downscaled to a small max dimension, only a couple of
      seconds of frames are generated instead of the full
      LIVE_WALLPAPER_DURATION_SECONDS, and those frames are generated (and
      played back) at a dedicated low PREVIEW_FPS rather than the real
      render's config.FPS - fewer, cheaper frames to compute, and a
      calmer, slower loop to watch.
    - Depth estimation and AI segmentation - by far the slowest steps, and
      the only ones that don't depend on which effects/intensities are
      enabled - are cached per (photo, preview size), so tweaking a rain
      slider and re-previewing the same photo doesn't re-run those models.
      Learned illumination and the color-grade-unify pass are skipped
      outright (real-time-not-realism is the point here); everything else
      (particles, fog, lightning, flicker, postprocess, seamless loop) is
      the exact same code path the real render uses, just at draft size.

Callers apply their config.* overrides with setattr (same restore-after
pattern vfx_gui.py's _render_worker and live_wallpaper.py's on_rerender
already use) *before* calling render_fast_preview(), same as the real
render.
"""

from PIL import Image

import config
from main import (
    any_effect_needs_mask, render_enabled_effects, resolve_base_frames,
    resolve_flicker_mask, resolve_mask_frames, resolve_postprocess_params,
    resolve_segmentation,
)
from vfx_engine import compositor, image_io, lighting, mask
from vfx_engine.lighting import apply_ambient_drift, apply_flicker
from vfx_engine.lightning import apply_flash, build_strike_schedule, resolve_strike_geometry
from vfx_engine.loop import make_seamless_loop
from vfx_engine.postprocess import apply_postprocess
from vfx_engine import weather_grade

PREVIEW_MAX_DIMENSION = 360
PREVIEW_DURATION_SECONDS = 2.5
# The real render's config.FPS (usually 30) makes for a short, rapid-fire
# loop here (1.5s @ 30fps = 45 frames blitted 33ms apart) - a dedicated,
# lower preview-only frame rate means fewer frames to compute (faster
# render) *and* a slower, easier-to-watch loop (fixed user complaint: "it
# shouldnt be that fast"). Applied via the same config.* setattr/restore
# override pattern every other render entry point uses, so every
# FPS-derived calculation below (num_frames, crossfade, lightning
# scheduling, the returned fps) automatically picks it up.
PREVIEW_FPS = 12

# A fixed, "many frames" stand-in used only to decide whether segmentation
# is worth computing at all (see main.resolve_segmentation) - every one of
# its num_frames>1 checks behaves identically for any value above 1, so this
# doesn't need to match the real preview frame count exactly.
_SEGMENTATION_GATE_FRAMES = 30

# Keyed by (str(image_path), max_dimension) -> the slow, effect-independent
# per-photo analysis (resized image, depth, segmentation). Cleared only by
# process restart - draft previews are a bounded, short-lived GUI session,
# and each entry is small (a downscaled image + a couple of small arrays).
_analysis_cache = {}


def _get_cached_analysis(image_path, max_dimension):
    key = (str(image_path), max_dimension)
    cached = _analysis_cache.get(key)
    if cached is not None:
        return cached

    image = image_io.load_base_image(image_path)
    image.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

    depth_array = depth_confidence = None
    if config.MASK_MODE == "depth" or config.PARTICLE_DEPTH_REALISM:
        from vfx_engine import depth as depth_module
        depth_array, depth_confidence = depth_module.estimate_depth(image, return_confidence=True)

    segmentation = resolve_segmentation(image, depth_array, _SEGMENTATION_GATE_FRAMES)

    cached = dict(image=image, depth_array=depth_array, depth_confidence=depth_confidence, segmentation=segmentation)
    _analysis_cache[key] = cached
    return cached


def render_fast_preview(image_path, max_dimension=PREVIEW_MAX_DIMENSION, duration_seconds=PREVIEW_DURATION_SECONDS):
    """
    Returns (frames, width, height, fps) - frames is a list of gamma-
    encoded PIL Images ready to blit directly onto a canvas, sized to fit
    within max_dimension while preserving the photo's aspect ratio.
    """
    original_fps = config.FPS
    try:
        config.FPS = PREVIEW_FPS
        analysis = _get_cached_analysis(image_path, max_dimension)
        image = analysis["image"]
        if config.WEATHER_GRADE_STYLE:
            image = weather_grade.apply_weather_grade(image, config.WEATHER_GRADE_STYLE, config.WEATHER_GRADE_STRENGTH)
        depth_array = analysis["depth_array"]
        depth_confidence = analysis["depth_confidence"]
        segmentation = analysis["segmentation"]
        size = image.size

        num_frames = max(1, round(duration_seconds * config.FPS))
        crossfade_frames = min(num_frames // 4, round(config.LIVE_WALLPAPER_CROSSFADE_SECONDS * config.FPS))

        shared_mask = mask.build_mask(image, image_path, depth_array=depth_array, segmentation=segmentation) if any_effect_needs_mask() else None
        mask_frames_factory = resolve_mask_frames(
            image, image_path, depth_array, num_frames, shared_mask,
            segmentation=segmentation, depth_confidence=depth_confidence,
        )

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
            image=image, depth_array=depth_array, segmentation=segmentation, illumination_result=None,
        )
        overlay_frames = compositor.layer_effects(size, num_frames, effect_frame_iterables)
        base_frames = resolve_base_frames(image, num_frames, depth_array=depth_array, depth_confidence=depth_confidence)
        final_frames = compositor.composite_onto_base(base_frames, overlay_frames)

        if config.FLICKER_ENABLED:
            flicker_mask = resolve_flicker_mask(image, shared_mask)
            final_frames = apply_flicker(final_frames, num_frames, mask=flicker_mask, **config.FLICKER_PARAMS)

        if config.LIGHTNING_ENABLED:
            lightning_params = config.LIGHTNING_PARAMS
            final_frames = apply_flash(
                final_frames, lightning_schedule, lightning_geometry, intensity=lightning_params["intensity"],
                color=lightning_params["color"], strike_frames=lightning_params["strike_frames"],
                falloff_radius_frac=lightning_params["flash_falloff_radius_frac"],
                min_intensity_frac=lightning_params["flash_min_intensity_frac"],
            )

        if config.AMBIENT_DRIFT_ENABLED and num_frames > 1:
            drift_strength = config.AMBIENT_DRIFT_STRENGTH * lighting.resolve_ambient_drift_scale(image)
            final_frames = apply_ambient_drift(final_frames, num_frames, strength=drift_strength)

        final_frames = apply_postprocess(final_frames, **resolve_postprocess_params(image, num_frames, size))
        if crossfade_frames > 0:
            final_frames = make_seamless_loop(final_frames, num_frames, crossfade_frames)

        final_frames = list(image_io.encode_frames_to_images(final_frames))
        return final_frames, size[0], size[1], config.FPS
    finally:
        config.FPS = original_fps
