"""
cloud.py

Drifting cloud motion. A still photo's sky has no second frame to compute
real optical flow from -
this takes the "mirror/warp the existing sky texture" approach rather
than synthesizing new cloud content: the photo's OWN sky/cloud pixels are
resampled through a slow, mostly-horizontal wind-driven displacement
field (see vfx_engine/scene_warp.py, the same engine vegetation.py's
canopy sway is built on, just with different wind parameters and mask).
This keeps the real, recognizable photographed sky instead of replacing
it with anything synthetic, while giving it real, visible drift - a
photo with an otherwise-moving foreground (rain, vegetation sway, camera
drift) next to a perfectly frozen sky reads as visibly wrong once you
notice it, which is why this exists.

The sky region comes from real AI scene segmentation (see
vfx_engine/segmentation.py's "sky" category), not a brightness/position
heuristic.
"""

import itertools
import math

import numpy as np
from PIL import Image, ImageFilter

from vfx_engine import camera, weather
from vfx_engine.scene_warp import render_region_warp

# A wider edge feather than vegetation.py's - sky boundaries (treeline,
# rooftops, distant mountains) are usually softer/further from the camera
# than a vegetation silhouette, and this is a slower, LARGER-amplitude
# drift than vegetation's localized gusts, so it needs more falloff room
# at the mask edge to avoid a visible seam where the sky mask ends.
_MASK_FEATHER = 6

# fraction of frame height, near the TOP and BOTTOM
# edges, over which the displacement AMPLITUDE (not visibility - the sky
# stays just as visible) eases down to 0. At cloud.py's drift_strength *
# 26.0 multiplier - the largest amplitude scalar of any effect sharing
# region_warp.glsl - a pixel very close to the frame's own top/bottom edge
# can, at large enough gust values, sample a displaced UV that wraps and
# reflects (mirror_coord's whole-FRAME fold, not a mask-local one) from
# the opposite edge, pulling a mirrored sliver of foreground/ground
# content into what should read as uniform sky. Keeping displacement near
# zero right at the edge keeps those pixels from ever reaching far enough
# to wrap.
_EDGE_TAPER_FRAC = 0.12


def render_cloud_drift(size, num_frames, base_image, sky_mask, drift_strength=0.5, direction_deg=None,
                        base_image_frames=None, camera_drift_params=None, start_frame=0):
    """
    Generator - yields transparent RGBA frames of the photo's own sky,
    resampled through a slow, mostly-horizontal wind-driven displacement
    field (see vfx_engine.scene_warp.render_region_warp) - composite this
    OVER the base image and UNDER every particle effect/vegetation sway
    (see main.py's render_enabled_effects): sky drift is part of the
    static SCENE, like vegetation sway, not something falling in front of
    it.

    sky_mask: (height, width) float32 array, 0.0-1.0 - see
        vfx_engine.segmentation.estimate_segmentation's "sky" category.
        None, or an all-zero/near-zero mask (e.g. a photo with no visible
        sky at all) yields an empty (fully transparent) stream instead of
        warping nothing.
    drift_strength: 0.0-1.0+, overall drift amplitude.
    direction_deg: degrees from horizontal (0 = due right, 90 = straight
        up) the constant wind bias points - real cloud LAYERS drift in
        roughly one dominant direction at a time (unlike vegetation's
        more localized per-tree gusts), on top of which the curl noise
        still adds organic variation rather than a perfectly straight
        mechanical slide. None (the default) uses the shared prevailing
        wind direction (config.WEATHER_WIND_DIRECTION_DEG - see
        vfx_engine/weather.py), the same direction rain/fog/vegetation
        are already blowing in, rather than an independently-set value
        that could disagree with them; pass an explicit value only to
        deliberately make this sky layer drift differently.
    base_image_frames: optional iterable yielding one per-frame drifted
        PIL Image (e.g. vfx_engine.camera.apply_camera_drift) - when given, THIS (not the static base_image) is what the
        sky drift actually resamples from, so it stays in registration
        with a simultaneously camera-drifting photo instead of always
        sampling the original, undrifted framing.
    camera_drift_params: optional dict {"zoom_amplitude", "pan_amplitude",
        "pan_phase_deg"} - when given, the sky MASK itself
        (not just the base texture above) is also resampled per-frame
        through the same camera-drift path (see
        vfx_engine.camera.apply_camera_drift_to_region_rg), so which
        pixels are even eligible to drift stays registered with the
        drifting photo too. None (default) = the old behavior.
    start_frame: render-checkpoint/resume support (see render_checkpoint.py) -
        skips yielding frames 0..start_frame-1; identical mechanism to
        vegetation.render_vegetation_sway's matching param - forwarded to
        BOTH render_region_warp calls below (so each phase-offset pass
        skips ahead independently in lockstep) and, when an animated
        camera-drift-synced mask is in play, to
        camera.apply_camera_drift_to_region_rg. The crossfade `mix` phase
        below is computed from the true absolute frame index (offset by
        start_frame), not from 0, so resuming mid-render doesn't discontinue
        the Gap A crossfade's sine phase at the resume boundary.

    the design notes (clouds only ever translate, never evolve shape):
    a pure displacement warp can only push the photo's OWN pixels sideways
    along the wind field - it can never billow/stretch/dissipate a cloud
    puff, so at this effect's unusually large 26x drift multiplier a
    recognizable puff visibly slides as a rigid, unchanging blob ("a
    sticker being dragged across the sky"). Fixed the same way
    vegetation's `flutter_strength` already layers a second, differently-
    phased noise sample onto its main one: render_region_warp is called
    TWICE, once at phase 0 and once exactly half a loop out of phase
    (`phase_offset_frames=num_frames/2.0`, i.e. theta+pi on the identical
    underlying wind field), then the two results are cross-faded with a
    mix weight that itself completes exactly one full sine cycle per loop
    (loop-safe). Since both samples read the same noise field at different
    phases, they displace the same cloud puff differently - blending
    between them reads as parts of the puff fading in/out (genuine shape
    change) rather than a second translation layered on the first.
    """
    if sky_mask is None or sky_mask.max() < 0.05:
        for _ in range(start_frame, num_frames):
            yield Image.new("RGBA", size, (0, 0, 0, 0))
        return

    feathered = sky_mask
    if _MASK_FEATHER > 0:
        feathered_image = Image.fromarray((sky_mask * 255).astype("uint8"), mode="L")
        feathered_image = feathered_image.filter(ImageFilter.GaussianBlur(_MASK_FEATHER))
        feathered = np.asarray(feathered_image, dtype="float32") / 255.0

    # Amplitude weight = the mask itself (no canopy-style height biasing -
    # a cloud layer has no "root" that should stay still) - fading the
    # displacement magnitude toward 0 at the same edge the visibility mask
    # fades out at avoids any mismatch between "how visible" and "how much
    # this pixel moves" right at the mask boundary.
    #
    # on top of that, the AMPLITUDE weight (not the
    # visibility mask - sky pixels near the frame's top/bottom edge stay
    # just as visible) additionally eases down to 0 within
    # _EDGE_TAPER_FRAC of the frame's own top and bottom edge, so those
    # pixels never displace far enough for mirror_coord's whole-frame fold
    # to wrap in a mismatched reflection from the opposite edge.
    frame_height = feathered.shape[0]
    row_indices = np.arange(frame_height, dtype=np.float32)
    edge_margin = max(1.0, frame_height * _EDGE_TAPER_FRAC)
    top_ease = np.clip(row_indices / edge_margin, 0.0, 1.0)
    bottom_ease = np.clip((frame_height - 1 - row_indices) / edge_margin, 0.0, 1.0)
    edge_taper = np.minimum(top_ease, bottom_ease)[:, np.newaxis]   # (height, 1), broadcasts over width
    amplitude_weight = feathered * edge_taper

    region_rg = np.stack([
        (feathered * 255.0).astype("uint8"),
        (amplitude_weight * 255.0).astype("uint8"),
    ], axis=-1)

    if direction_deg is None:
        wind_bias = weather.wind_direction_vector()
    else:
        direction = np.radians(direction_deg)
        wind_bias = (float(np.cos(direction)), float(np.sin(direction)))

    region_source = region_rg
    if camera_drift_params is not None:
        region_source = camera.apply_camera_drift_to_region_rg(region_rg, num_frames, start_frame=start_frame, **camera_drift_params)

    # render_region_warp is called TWICE - once at the
    # real phase, once a half-loop out of phase - and the two streams are
    # cross-faded below. Both calls consume the SAME base_image/region_source
    # sequence, so if either is a live, single-use iterable (e.g.
    # camera.apply_camera_drift's generator) it needs to be split into two
    # independent, replayable iterators - a generator can only be walked
    # once, and the second render_region_warp call needs its own pass over
    # every frame.
    #
    # This used to be done by _materialize_if_iterable(source) calling
    # list(source) ONCE and sharing that list between both passes - correct,
    # but it fully materializes EVERY frame of the entire clip, at full
    # render resolution, into memory at once (a 720-frame render at ~3840px
    # wide is ~24GB of RGBA frames) before either pass even starts. Confirmed
    # as the actual root cause of multi-hour render stalls: system RAM
    # exhaustion (observed down to ~1GB free of 32GB) forces the OS into
    # heavy paging, which is consistent with the escalating per-frame ETA
    # seen in real runs (a render that gets progressively slower frame over
    # frame as this list grows, rather than one one-time stall).
    #
    # itertools.tee(source, 2) gives two independent iterators over the SAME
    # underlying generator instead, buffering only the frames one consumer
    # is ahead of the other. The two consumers here (frames_a/frames_b
    # below) are driven by the SAME outer `zip(frames_a, frames_b)` loop, so
    # they always advance in lockstep - tee's internal buffer never holds
    # more than one frame, instead of the whole clip.
    base_source = base_image_frames if base_image_frames is not None else base_image
    base_source_a, base_source_b = _tee_if_iterable(base_source)
    region_source_a, region_source_b = _tee_if_iterable(region_source)

    common_kwargs = dict(
        drift_strength=drift_strength * 26.0,
        noise_scale=0.0009,
        wind_bias=wind_bias,
        noise_loop_radius=2.0,
        flutter_strength=0.0,
    )
    frames_a = render_region_warp(size, num_frames, base_source_a, region_source_a,
                                   phase_offset_frames=0.0, start_frame=start_frame, **common_kwargs)
    frames_b = render_region_warp(size, num_frames, base_source_b, region_source_b,
                                   phase_offset_frames=num_frames / 2.0, start_frame=start_frame, **common_kwargs)

    # frame_i must reflect the TRUE absolute frame index (offset by
    # start_frame), not always start at 0 - otherwise resuming mid-render
    # would discontinue the crossfade's sine phase right at the resume
    # boundary (see start_frame's docstring paragraph above).
    for frame_i, (frame_a, frame_b) in enumerate(zip(frames_a, frames_b), start=start_frame):
        mix = 0.5 + 0.5 * math.sin(2.0 * math.pi * frame_i / max(num_frames, 1))
        yield _crossfade_rgba(frame_a, frame_b, mix)


def _tee_if_iterable(source):
    """
    A single PIL Image or a single (height, width, 2) numpy array is
    already reusable as-is across multiple render_region_warp calls -
    only an ITERABLE of per-frame values (a live, single-use generator,
    e.g. camera.apply_camera_drift's) needs to be split so cloud.py's two
    parallel render_region_warp calls each get their
    own independent, replayable sequence of frames instead of racing to
    exhaust the same generator.

    Returns a (source_a, source_b) pair either way (both the same object
    for the non-iterable case - a PIL Image/numpy array is safely reusable
    without copying). For the iterable case, uses itertools.tee instead of
    list(source): tee gives two independent iterators over the SAME
    underlying generator, buffering only the frames one consumer is ahead
    of the other, instead of eagerly pulling and holding the ENTIRE clip's
    frames in memory at once before either consumer starts (see the call
    site's comment for the ~24GB real-world memory blowup this caused).
    Safe here specifically because both returned iterators are driven by
    the SAME outer zip() loop back in render_cloud_drift, so they always
    advance in lockstep and tee's buffer never grows past ~1 frame.
    """
    if isinstance(source, (Image.Image, np.ndarray)):
        return source, source
    return itertools.tee(source, 2)


def _crossfade_rgba(frame_a, frame_b, mix):
    """
    Linear-blend two transparent RGBA PIL frames (straight, not
    premultiplied, alpha - matches render_region_warp's own output) by
    `mix` (0.0 = all frame_a, 1.0 = all frame_b) - both color and alpha
    channels blend the same way, so a pixel opaque in one frame and
    transparent in the other fades smoothly rather than popping.
    """
    if mix <= 0.001:
        return frame_a
    if mix >= 0.999:
        return frame_b
    array_a = np.asarray(frame_a, dtype="float32")
    array_b = np.asarray(frame_b, dtype="float32")
    blended = array_a * (1.0 - mix) + array_b * mix
    return Image.fromarray(blended.astype("uint8"), mode="RGBA")
