"""
water.py

Wind-driven water ripple - image-space displacement warping of the
photo's OWN water pixels (a lake, river, sea, or pool - see
vfx_engine.segmentation.estimate_segmentation's "water" category),
driven by the same curl-noise wind field family the particle system,
vegetation sway, and cloud drift already use - see
the design notes Part 20.1 and vfx_engine/scene_warp.py for the
shared GPU engine this, vegetation.py's canopy sway, and cloud.py's sky
drift are all built on.

Confirmed by grep before this module existed: `segmentation.py` already
classified "water" as one of its coarse categories on every single
render, and nothing anywhere in the codebase ever read it - a body of
water sat perfectly, permanently static for the whole loop while rain
fell, the camera drifted, vegetation swayed, and clouds drifted directly
above it. At least as visually obvious a "static photo with effects on
it" tell as the vegetation/cloud gaps the original the design notes already
flagged, and arguably more so given how much of a typical landscape
composition a body of water can occupy.

The bigger payoff, named in this project's very first the design notes document as
"the actual payoff beyond per-element motion" and built here for the
first time: water ripple timing is tied to the SAME wind conditions
driving cloud drift directly above it, not an independently-tuned
effect. render_water_ripple deliberately reuses cloud.py's exact
noise_scale/noise_loop_radius (so both sample the identical curl-noise
field's time evolution) and, by default, the identical wind_bias
direction (config.WEATHER_WIND_DIRECTION_DEG via
weather.wind_direction_vector(), the same shared source cloud.py already
defaults to) - and both already sample the SAME global
weather.gust_envelope() per frame (render_region_warp's own default
gust_kind="plain"), the identical mechanism rain/fog/vegetation/cloud
already share. The result: a lake's surface visibly ripples in step with the
same gusts that are simultaneously drifting the sky's clouds and leaning
the rain, instead of two unrelated regions (one static, one animated)
sharing a frame - real cross-element consistency, not just "animate one
more thing."
"""

import numpy as np
from PIL import Image, ImageFilter

from vfx_engine import camera, weather
from vfx_engine.scene_warp import render_region_warp

# Same-sized edge feather as cloud.py's sky mask - a water body's
# shoreline is a similarly soft, natural boundary (not a hard vegetation
# silhouette), and this needs to fade out fully by the water mask's own
# edge for the same reason cloud drift's does.
_MASK_FEATHER = 6


def render_water_ripple(size, num_frames, base_image, water_mask, ripple_strength=0.5, direction_deg=None,
                         base_image_frames=None, camera_drift_params=None, start_frame=0):
    """
    Generator - yields transparent RGBA frames of the photo's own water,
    resampled through a slow, mostly-horizontal wind-driven ripple field
    (see vfx_engine.scene_warp.render_region_warp) - composite this OVER
    the base image and UNDER every particle effect (see main.py's
    render_enabled_effects): water ripple is part of the static SCENE,
    like vegetation sway and cloud drift, not something falling in front
    of it - rain/snow still needs to fall in FRONT of a rippling lake,
    not behind it.

    water_mask: (height, width) float32 array, 0.0-1.0 - see
        vfx_engine.segmentation.estimate_segmentation's "water" category.
        None, or an all-zero/near-zero mask (e.g. a photo with no visible
        water at all) yields an empty (fully transparent) stream instead
        of warping nothing.
    ripple_strength: 0.0-1.0+, overall ripple amplitude - deliberately
        small relative to cloud drift's own drift_strength scale (real
        wind-driven water ripples are a subtle surface texture, not a
        broad structural displacement the way cloud LAYERS sliding across
        frame are).
    direction_deg: degrees from horizontal (0 = due right, 90 = straight
        up) the ripple's dominant traveling-wave direction. None (the
        default) uses the shared prevailing wind direction
        (config.WEATHER_WIND_DIRECTION_DEG via
        weather.wind_direction_vector()) - the SAME source cloud.py's own
        direction_deg=None default resolves to, so a lake's ripple
        travels the same direction the sky above it drifts in by
        default, rather than an independently-set value that could
        disagree with it.
    base_image_frames: optional iterable yielding one per-frame drifted
        PIL Image (e.g. vfx_engine.camera.apply_camera_drift) - the design notes mechanism: when given, THIS (not the static base_image) is
        what the rippling water actually resamples from, so it stays in
        registration with a simultaneously camera-drifting photo instead
        of always sampling the original, undrifted framing.
    camera_drift_params: optional dict {"zoom_amplitude", "pan_amplitude",
        "pan_phase_deg"} - the design notes mechanism: when given, the water
        MASK itself (not just the base texture above) is also resampled
        per-frame through the same camera-drift path (see
        vfx_engine.camera.apply_camera_drift_to_region_rg), so which
        pixels are even eligible to ripple stays registered with the
        drifting photo too. None (default) = the old behavior, one
        static mask reused for every frame.
    start_frame: render-checkpoint/resume support (see render_checkpoint.py) -
        skips yielding frames 0..start_frame-1; identical mechanism to
        vegetation.render_vegetation_sway's matching param - forwarded to
        render_region_warp and, when an animated camera-drift-synced mask
        is in play, to camera.apply_camera_drift_to_region_rg so that
        source's first yielded item lines up with frame start_frame too.
    """
    if water_mask is None or water_mask.max() < 0.05:
        for _ in range(start_frame, num_frames):
            yield Image.new("RGBA", size, (0, 0, 0, 0))
        return

    feathered = water_mask
    if _MASK_FEATHER > 0:
        feathered_image = Image.fromarray((water_mask * 255).astype("uint8"), mode="L")
        feathered_image = feathered_image.filter(ImageFilter.GaussianBlur(_MASK_FEATHER))
        feathered = np.asarray(feathered_image, dtype="float32") / 255.0

    # Uniform amplitude weight = the mask itself, same as cloud.py's sky -
    # a water surface has no rigid "root" the way a tree trunk does, so
    # there's no height-graded weighting to apply here either.
    region_rg = np.stack([
        (feathered * 255.0).astype("uint8"),
        (feathered * 255.0).astype("uint8"),
    ], axis=-1)

    if direction_deg is None:
        wind_bias = weather.wind_direction_vector()
    else:
        direction = np.radians(direction_deg)
        wind_bias = (float(np.cos(direction)), float(np.sin(direction)))

    region_source = region_rg
    if camera_drift_params is not None:
        region_source = camera.apply_camera_drift_to_region_rg(region_rg, num_frames, start_frame=start_frame, **camera_drift_params)

    yield from render_region_warp(
        size, num_frames, base_image_frames if base_image_frames is not None else base_image, region_source,
        # Deliberately much smaller than cloud.py's 26.0x - real water
        # ripple is a small-amplitude surface texture, not a broad
        # structural slide.
        drift_strength=ripple_strength * 3.0,
        # Same noise_scale/noise_loop_radius as cloud.py's own call
        # (0.0009 / 2.0) - this is what actually ties ripple TIMING to
        # cloud drift's: both sample the identical curl-noise field at
        # the identical points in its time evolution, so a gust that
        # visibly speeds up cloud drift at frame N visibly ripples the
        # water harder at that same frame N too, not on an independent
        # schedule.
        noise_scale=0.0009,
        wind_bias=wind_bias,
        noise_loop_radius=2.0,
        # A finer, faster secondary ripple layer on top of the main
        # wind-correlated displacement - real water shows fine surface
        # texture riding on top of the larger wind-driven pattern, the
        # same "two noise layers" idea vegetation.py's flutter adds on
        # top of its own broader gusting.
        flutter_strength=ripple_strength * 1.4,
        flutter_scale=0.045,
        flutter_cycles=8.0,
        # A genuine traveling wave crossing the water in the wind
        # direction (region_warp.glsl's wave_strength/wave_direction,
        # the same mechanism vegetation.render_grass_sway uses for its
        # "wind visibly crossing a lawn" look) - "mostly-horizontal
        # ripple displacement" reads as coherent directional ripples
        # crossing the surface, not just in-place turbulent jitter.
        wave_strength=ripple_strength * 2.5,
        wave_direction=wind_bias,
        wave_frequency=0.03,
        wave_cycles=6.0,
        start_frame=start_frame,
    )
