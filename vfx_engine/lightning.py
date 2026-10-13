"""
lightning.py

Procedural lightning bolts + the accompanying full-frame flash. Unlike the
particle effects (snow/rain), a lightning bolt is generated once as a 2D
branching shape (via midpoint displacement - the standard technique for
procedural lightning/cracks/rivers) and reused for however many frames it
stays on screen (real bolts last only a couple of frames).

This is CPU/PIL work rather than a GPU shader: a bolt is a handful of line
segments drawn once per strike, not thousands of particles redrawn every
frame, so there's no performance reason to involve the GPU here, and PIL's
line drawing + Gaussian blur (for the glow) are simple and robust.

The bolt (render_lightning, a transparent overlay layer) and the
full-frame flash (apply_flash, brightens the already-composited frames)
are two separate stages of the same pipeline and need to strike at exactly
the same frames - build_strike_schedule() computes that timing once, up
front, and both functions are given the same schedule, instead of each
rolling its own independent dice (which could easily drift out of sync
depending on the order the two generators happen to be pulled from).
"""

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from vfx_engine.lighting import _decode_color, brighten_frame


def build_strike_schedule(num_frames, frequency=0.15, strike_frames=3, fps=24, seed=None):
    """
    Precomputes, for every frame, how many frames are left in a strike that
    covers it (0 = no strike). Pass the same schedule to both
    render_lightning() and apply_flash() so the bolt shape and the
    scene-wide flash always land on the same frames.

    frequency: roughly how many strikes per second on average
    strike_frames: how many frames one strike lasts (a bright flash frame,
                   then a couple of dimmer afterglow frames)
    """
    rng = random.Random(seed)
    strike_probability_per_frame = frequency / fps

    schedule = []
    frames_left = 0
    for _ in range(num_frames):
        if frames_left <= 0 and rng.random() < strike_probability_per_frame:
            frames_left = strike_frames
        schedule.append(frames_left)
        if frames_left > 0:
            frames_left -= 1
    return schedule


def resolve_strike_geometry(schedule, size, depth_array=None, strike_frames=3, seed=None, sky_depth_threshold=0.35):
    """
    Precomputes the start/end point (and a resolved distance_factor, see
    below) of every strike in `schedule` - one dict, keyed by the frame
    index each strike starts on (schedule[i] == strike_frames), so both
    render_lightning() (which draws the bolt) and apply_flash() (which
    needs the same endpoint to center its radial falloff mask, see 4.3)
    agree on exactly where each strike hits, the same reason
    build_strike_schedule() exists as a single shared source of truth
    rather than each function rolling its own dice. Each dict value is
    now a 3-tuple `(start, end, distance_factor)`, not a 2-tuple - see
    _sample_distance_factor's docstring for the third element.

    Fixes the confirmed "bolt doesn't connect to anything" gap: the old code picked the endpoint as `height *
    rng.uniform(0.35, 0.7)` - a blind fraction of frame height with zero
    relationship to the actual photo, so the bolt just stopped in mid-air.
    When `depth_array` is given (see vfx_engine.depth.estimate_depth,
    0.0=far/1.0=near, the same convention fog.py/ground_fx.py use), the
    endpoint instead terminates at the first "near-ish" pixel scanning
    down the bolt's own column - i.e. the actual skyline/horizon silhouette
    for that column, whether that's the true horizon, a ridgeline, or a
    genuinely tall foreground object (a tree, a building) that happens to
    occlude the sky first. That single scan naturally covers both cases
    the design notes calls out, without needing to special-case "is this a
    horizon or an object" - whatever the photo's actual geometry puts
    there first IS the answer either way.

    Fixes the design notes ("bolt always starts at literal row 0"): the
    start point now goes through the SAME column-scan helper
    (_column_depth_profile) as the endpoint, via _resolve_strike_start,
    instead of being hardcoded to `(start_x, 0.0)` - see that function's
    docstring for why this only nudges the start down in tight
    compositions and otherwise stays visually identical to the old
    behavior.
    """
    rng = random.Random(seed)
    width, height = size
    geometry = {}

    for frame_index, frames_left in enumerate(schedule):
        if frames_left != strike_frames:
            continue
        start_x = rng.uniform(width * 0.1, width * 0.9)
        end_x = start_x + rng.uniform(-width * 0.15, width * 0.15)
        start_y = _resolve_strike_start(start_x, height, depth_array, sky_depth_threshold)
        end_y = _resolve_strike_height(end_x, height, depth_array, rng, sky_depth_threshold)
        distance_factor = _sample_distance_factor(end_x, end_y, depth_array)
        geometry[frame_index] = ((start_x, start_y), (end_x, end_y), distance_factor)

    return geometry


def _column_depth_profile(x, depth_array):
    """
    Returns the (height,) windowed-average depth profile for a small
    horizontal band of columns centered on `x` - shared by both
    _resolve_strike_height (the bolt's endpoint) and _resolve_strike_start
, so both scan the photo's
    real geometry through the identical column-averaging logic instead of
    two independently-written threshold walks that could quietly drift out
    of sync with each other over time.

    Averaging a small horizontal window of columns (not just the single
    nearest column) matters because monocular depth maps are noisy
    pixel-to-pixel, and a single column's first threshold-crossing row can
    jitter wildly between two visually-identical strikes a few pixels
    apart.
    """
    depth_height, depth_width = depth_array.shape
    center_col = int(np.clip(round(x), 0, depth_width - 1))
    half_window = max(1, int(depth_width * 0.01))
    col_start = max(0, center_col - half_window)
    col_end = min(depth_width, center_col + half_window + 1)
    return depth_array[:, col_start:col_end].mean(axis=1)


def _resolve_strike_height(x, height, depth_array, rng, sky_depth_threshold=0.35):
    """
    Returns the y coordinate a bolt at horizontal position `x` should
    strike down to - see resolve_strike_geometry's docstring. Falls back
    to the old random-fraction behavior when no depth data is available.
    """
    if depth_array is None:
        return height * rng.uniform(0.35, 0.7)

    column_profile = _column_depth_profile(x, depth_array)

    # First row (scanning down from the top, i.e. from the sky) where
    # depth rises past the "sky" threshold - the actual skyline/horizon/
    # tall-object silhouette for this column.
    hits = np.nonzero(column_profile > sky_depth_threshold)[0]
    if len(hits) == 0:
        # No near-ish pixel anywhere in this column (e.g. it's all open
        # sky) - strike near the bottom of frame as a last resort, rather
        # than mid-air.
        return float(depth_array.shape[0]) * 0.92
    return float(hits[0])


# how much of the frame's own height the column's real
# "sky band above the first near pixel" has to be THINNER than before the
# bolt's start point gets nudged down from row 0 at all. A generous sky
# band (the ordinary landscape-photo case: hits[0] is far down the column,
# well past this cap) means row 0 genuinely IS inside real photographed
# sky, so the start point is left exactly where it always was. Only a
# tight composition - the sky-classified band above the first near pixel
# occupying less than this fraction of frame height - triggers the nudge,
# since that's the specific case where starting at literal row 0 would
# read as originating from inside solid foreground content instead of a
# visible cloud base.
_START_SKY_CAP_FRAC = 0.25


def _resolve_strike_start(x, height, depth_array, sky_depth_threshold=0.35):
    """
    Returns the y coordinate a bolt at horizontal position `x` should
    START from - see resolve_strike_geometry's and _START_SKY_CAP_FRAC's
    docstrings. Falls back to the old hardcoded row-0 behavior when no
    depth data is available, exactly like _resolve_strike_height does for
    the endpoint.
    """
    if depth_array is None:
        return 0.0

    column_profile = _column_depth_profile(x, depth_array)
    hits = np.nonzero(column_profile > sky_depth_threshold)[0]
    if len(hits) == 0:
        return 0.0   # this column is open sky all the way down - row 0 is correct as-is

    boundary = float(hits[0])
    return boundary if boundary < height * _START_SKY_CAP_FRAC else 0.0


def _sample_distance_factor(x, y, depth_array):
    """
    samples the ACTUAL scene depth (0.0=far, 1.0=near,
    same convention as every other depth-aware effect in this codebase) at
    a strike's own resolved endpoint, so _generate_bolt can render a bolt
    striking a close ridgeline thicker/brighter/warmer than one striking a
    hazy, distant horizon - the depth information needed for this was
    already sitting computed and unused in resolve_strike_geometry.
    Returns 1.0 (near - the old, unaffected rendering) when no depth_array
    is available, so callers that don't pass one keep the current
    behavior exactly.
    """
    if depth_array is None:
        return 1.0
    depth_height, depth_width = depth_array.shape
    row = int(np.clip(round(y), 0, depth_height - 1))
    col = int(np.clip(round(x), 0, depth_width - 1))
    return float(depth_array[row, col])


def _flicker_intensity(elapsed, strike_frames):
    """
    Returns 0.0-1.0 brightness for a strike `elapsed` frames after it
    started (0 = the very first, brightest frame). Fixes the confirmed
    "one decay curve, no flicker" gap: the old version was
    `frames_left / strike_frames` - a single, perfectly monotonic linear
    fade, arguably the single biggest "this is a shape fading out" tell in
    the whole effect. Real lightning is a rapid, UNEVEN multi-pulse strobe
    (a sharp initial return stroke, then one or more dimmer, briefly-
    separated restrikes), not a dimmer switch turning down smoothly.

    A closed-form curve (fast-decaying initial spike + two Gaussian bumps
    for secondary/tertiary restrikes) reads as "electrical" instead of
    "fading" and needs no extra timing budget - it's evaluated at whatever
    `strike_frames` already is, from a single shared frame position, so
    render_lightning's bolt alpha and apply_flash's screen brightness stay
    exactly in sync the same way they always have (same call, same inputs,
    same output) without needing a lookup table sized to a specific frame
    count.
    """
    t = elapsed / max(strike_frames - 1, 1)   # 0.0 (strike start) .. 1.0 (strike end)
    pulse = (
        1.0 * math.exp(-t * 7.0)                        # sharp initial stroke, decays fast
        + 0.55 * math.exp(-((t - 0.45) ** 2) / 0.01)     # brief secondary restrike partway through
        + 0.25 * math.exp(-((t - 0.75) ** 2) / 0.006)    # smaller tertiary flicker near the tail
    )
    return min(1.0, pulse)


def render_lightning(size, schedule, strike_geometry, color=(230, 235, 255), branch_chance=0.35, strike_frames=3,
                     start_frame=0):
    """
    Generator - for each frame in `schedule` (see build_strike_schedule),
    yields a transparent RGBA overlay frame: the bolt itself while a
    strike is active, fully transparent otherwise. A new bolt shape is
    generated at the start of each strike and regenerated at each subsequent pulse boundary, fading out
    (afterglow) as the strike ends.

    strike_geometry: dict from resolve_strike_geometry(), keyed by the
        frame index each strike starts on - the SAME dict passed to
        apply_flash(), so the bolt drawn here and the flash's radial
        falloff mask always agree on where the strike actually hit
. Each value is now a 3-tuple `(start, end,
        distance_factor)` - see resolve_strike_geometry's docstring.

    the design notes ("every restrike reuses the literal same bolt_image"):
    a real multi-pulse strike doesn't hold pixel-perfectly still between its
    return stroke and its restrikes - individual minor branches visibly
    appear/disappear/re-route between pulses while the main channel (the
    ionized trunk path the current already carved through the air) stays
    put. `_PULSE_THRESHOLDS` mirrors _flicker_intensity's own three pulse
    centers (t=0.0 initial stroke, t=0.45 secondary restrike, t=0.75
    tertiary flicker) - crossing each one regenerates the bolt via a fresh
    `branch_rng` (varying branch appearance/shape) while `trunk_seed` stays
    fixed for the whole strike (identical main channel every regeneration),
    via _generate_bolt/_draw_branch's trunk_rng vs branch_rng split.

    start_frame: render-checkpoint/resume support (see render_checkpoint.py)
        - skips YIELDING frames [0, start_frame), but still walks the state
        machine (strike/pulse detection, bolt regeneration) for those
        frames, since `rng`/`trunk_seed`/`fired_pulse_thresholds`/
        `bolt_image` are carried frame-to-frame within a strike and can't be
        reconstructed from frame_index alone the way other effects' pure
        per-frame setups can. NOTE: unlike every other effect in this
        codebase, `rng = random.Random()` here is intentionally unseeded
        (bolt *shape* only - see the comment below), so a resumed render's
        bolts will look different from what an uninterrupted run would have
        drawn even with this replay - only the strike *timing* (from
        `schedule`/`strike_geometry`, both precomputed and passed in
        unchanged) stays identical. This is a pre-existing property of
        lightning's design, not something the checkpoint feature regresses.
    """
    rng = random.Random()   # decides bolt *shape* (trunk seed + branching), doesn't need to match apply_flash's timing
    bolt_image = None
    current_geometry = None
    trunk_seed = None
    fired_pulse_thresholds = set()
    _PULSE_THRESHOLDS = (0.0, 0.45, 0.75)

    for frame_index, frames_left in enumerate(schedule):
        if frames_left == strike_frames:   # a new strike just started
            current_geometry = strike_geometry[frame_index]
            trunk_seed = rng.random()
            fired_pulse_thresholds = set()

        if frames_left > 0:
            elapsed = strike_frames - frames_left
            t = elapsed / max(strike_frames - 1, 1)
            for threshold in _PULSE_THRESHOLDS:
                if threshold not in fired_pulse_thresholds and t >= threshold:
                    start, end, distance_factor = current_geometry
                    bolt_image = _generate_bolt(size, start, end, color, branch_chance, trunk_seed, rng, distance_factor)
                    fired_pulse_thresholds.add(threshold)
            if frame_index < start_frame:
                continue
            fade = _flicker_intensity(elapsed, strike_frames)
            yield _fade_bolt(bolt_image, fade)
        else:
            if frame_index < start_frame:
                continue
            yield Image.new("RGBA", size, (0, 0, 0, 0))


def apply_flash(frames, schedule, strike_geometry, intensity=0.9, color=(230, 235, 255), strike_frames=3,
                falloff_radius_frac=0.85, min_intensity_frac=0.25, start_frame=0):
    """
    Generator - applies a full-frame brightness flash to an already
    composited sequence of frames, at the same frames `schedule` (see
    build_strike_schedule, pass the same one used for render_lightning)
    marks as part of a strike. This is what makes a strike look like it's
    actually lighting up the whole scene, not just drawing a bright
    squiggle on top of it.

    Fixes the confirmed "flash has no spatial relationship to the bolt"
    gap: the old version called brighten_frame(..., mask=None)
    - a perfectly uniform, full-frame brightness increase regardless of
    where the bolt struck. Now a radial mask centered on the strike's own
    endpoint (strike_geometry, the same dict render_lightning() uses to
    draw the bolt) makes the flash genuinely brightest near the strike and
    fall off with distance from it - `min_intensity_frac` keeps a floor
    everywhere else, since a real flash still lights the whole sky, just
    unevenly, not a hard point-light spotlight confined to one corner.

    strike_geometry: see render_lightning's docstring - same dict.
    falloff_radius_frac: distance (as a fraction of the frame diagonal) at
        which the flash reaches `min_intensity_frac` and stays there.
    min_intensity_frac: 0.0-1.0, the flash's minimum strength far from the
        strike, relative to its peak strength at the strike point itself.
    start_frame: render-checkpoint/resume support (see render_checkpoint.py).
        `frames` is assumed to already be start_frame-truncated by its
        caller (the usual contract elsewhere in this codebase - see
        apply_flicker/apply_ambient_drift in lighting.py), but `schedule`
        and `strike_geometry` are the same full-length, absolute-frame-
        indexed structures build_strike_schedule/resolve_strike_geometry
        always produce (unchanged, shared with render_lightning). So
        `schedule` is sliced to `[start_frame:]` to line back up with the
        truncated `frames` stream via zip, and the loop's own frame_index
        is produced via `enumerate(..., start=start_frame)` instead of
        starting at 0, so `strike_geometry[frame_index]` lookups (still
        keyed by the ORIGINAL, absolute frame index) keep resolving to the
        correct strike geometry.
    """
    mask = None
    linear_color = _decode_color(color)

    for frame_index, (frame, frames_left) in enumerate(zip(frames, schedule[start_frame:]), start=start_frame):
        if frames_left == strike_frames:   # a new strike just started
            _, end, _ = strike_geometry[frame_index]
            frame_height, frame_width = frame.shape[0], frame.shape[1]
            mask = _radial_flash_mask((frame_width, frame_height), end, falloff_radius_frac, min_intensity_frac)

        if frames_left > 0:
            fade = _flicker_intensity(strike_frames - frames_left, strike_frames)
            yield brighten_frame(frame, intensity * fade, linear_color, mask=mask)
        else:
            yield frame


def _radial_flash_mask(size, center, falloff_radius_frac, min_intensity_frac):
    """
    Returns a (height, width) float32 array, 0..1: 1.0 at `center` (the
    strike's own endpoint), falling off toward `min_intensity_frac` by
    `falloff_radius_frac` of the frame diagonal away, and staying at that
    floor beyond it - see apply_flash's docstring for why a floor (not a
    fade to 0) is correct here.
    """
    width, height = size
    y_coords, x_coords = np.mgrid[0:height, 0:width].astype("float32")
    distance = np.sqrt((x_coords - center[0]) ** 2 + (y_coords - center[1]) ** 2)

    diagonal = (width ** 2 + height ** 2) ** 0.5
    radius = max(1.0, diagonal * falloff_radius_frac)
    falloff = np.clip(1.0 - distance / radius, 0.0, 1.0)
    return min_intensity_frac + (1.0 - min_intensity_frac) * falloff


def _generate_bolt(size, start, end, color, branch_chance, trunk_seed, branch_rng, distance_factor=1.0):
    """
    Builds one lightning bolt as a transparent RGBA image: a bright,
    jagged core line from `start` to `end` (see resolve_strike_geometry),
    plus random side branches, with a soft glow (Gaussian blur of the same
    shape) behind the sharp core.

    Hot core / cold glow color split: the old version drew
    both the sharp core and the blurred glow in the exact same flat
    `color` - real overexposed electrical discharge clips its core toward
    near-white, while the surrounding air glow keeps the actual tinted
    color. The branching shape is drawn only ONCE (into a plain-white
    `shape_alpha` mask) so the core and glow are guaranteed to share
    identical geometry - drawing the branches twice with the same rng
    would advance its state differently each time and produce two
    DIFFERENT jagged shapes, misaligning the core and its own glow.

    the design notes (trunk/branch rng split): `trunk_seed` reseeds a
    fresh `trunk_rng` every call, so the MAIN channel's own midpoint-
    displacement shape is bit-identical across repeated calls sharing the
    same seed (render_lightning calls this multiple times per strike, once
    per pulse boundary, always with the same trunk_seed) - only the
    independently-varying `branch_rng` controls whether/where minor side
    branches spawn and their own shape, so restrikes show the same main
    channel with different branch detail, matching how real multi-pulse
    strikes behave.

    the design notes (distance-based tapering): `distance_factor`
    (0.0=far, 1.0=near, from resolve_strike_geometry's depth sample)
    scales line_width and glow_radius and cools/desaturates `color` toward
    a fixed haze tint for far strikes, and dims overall alpha slightly -
    a close strike should read thick, bright, and sharply defined; a
    strike against a distant, hazy horizon should read thinner, dimmer,
    softer-edged and cooler in color, the same kind of atmospheric
    perspective every other depth-aware effect in this codebase already
    applies. Defaults to 1.0 (near - today's unaffected look) so any
    caller that doesn't have depth data keeps the exact old rendering.
    """
    line_width = max(1, int(min(size) * 0.0015))
    line_width = max(1, int(round(line_width * (0.55 + 0.6 * distance_factor))))

    trunk_rng = random.Random(trunk_seed)
    shape_layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shape_layer)
    _draw_branch(draw, start, end, trunk_rng, branch_rng, branch_chance, depth=0,
                 color=(255, 255, 255), line_width=line_width, is_branch=False)
    shape_alpha = shape_layer.getchannel("A")

    alpha_scale = 0.6 + 0.4 * distance_factor
    if alpha_scale < 0.999:
        alpha_arr = np.asarray(shape_alpha, dtype="float32") * alpha_scale
        shape_alpha = Image.fromarray(alpha_arr.astype("uint8"), mode="L")

    # far strikes cool/desaturate toward a fixed hazy blue-grey, same
    # atmospheric-perspective idea fog.py/particles_rain.py already apply
    haze_tint = (150, 160, 180)
    cool_amount = (1.0 - distance_factor) * 0.6
    tinted_color = tuple(int(c + (h - c) * cool_amount) for c, h in zip(color, haze_tint))

    core_color = tuple(min(255, int(channel + (255 - channel) * 0.7)) for channel in tinted_color)   # clipped near-white
    core = Image.new("RGBA", size, (*core_color, 0))
    core.putalpha(shape_alpha)

    glow_radius = max(4, int(min(size) * 0.01))
    glow_radius = max(2, int(round(glow_radius * (0.7 + 0.6 * (1.0 - distance_factor)))))
    glow_source = Image.new("RGBA", size, (*tinted_color, 0))
    glow_source.putalpha(shape_alpha)
    glow = glow_source.filter(ImageFilter.GaussianBlur(radius=glow_radius))

    return Image.alpha_composite(glow, core)


def _draw_branch(draw, start, end, trunk_rng, branch_rng, branch_chance, depth, color, line_width, max_depth=6, is_branch=False):
    """
    Recursive midpoint displacement: draws a jagged line from `start` to
    `end` by repeatedly splitting segments in half and randomly offsetting
    the midpoint perpendicular to the segment - the standard technique for
    procedural lightning/cracks/coastlines. Occasionally spawns a shorter
    side branch, fading with recursion depth.

    `is_branch` tracks whether the current recursive
    call is still tracing the bolt's own MAIN channel (False) or has
    descended into a side branch (True, set for every call inside a
    branch and all its own sub-branches). Main-channel midpoint
    displacement draws from `trunk_rng` (identical sequence every call
    sharing the same seed -> identical trunk shape across pulses); once
    inside a branch, everything - the branch's own shape, and any further
    nested branch_chance rolls within it - draws from `branch_rng`. The
    branch_chance roll that decides whether a NEW branch spawns off the
    main channel always uses `branch_rng` regardless of `is_branch`, since
    that's precisely the "does a branch appear here or not" decision that
    should vary between pulses.
    """
    if depth >= max_depth or (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2 < 16:
        draw.line([start, end], fill=(*color, 255), width=max(1, line_width - depth // 2))
        return

    shape_rng = branch_rng if is_branch else trunk_rng
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    segment_length = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
    displacement = shape_rng.uniform(-1, 1) * segment_length * 0.25
    # perpendicular direction to the segment
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    perp_x, perp_y = -dy / length, dx / length
    mid = (mid_x + perp_x * displacement, mid_y + perp_y * displacement)

    _draw_branch(draw, start, mid, trunk_rng, branch_rng, branch_chance, depth + 1, color, line_width, max_depth, is_branch)
    _draw_branch(draw, mid, end, trunk_rng, branch_rng, branch_chance, depth + 1, color, line_width, max_depth, is_branch)

    if branch_rng.random() < branch_chance and depth < max_depth - 1:
        branch_end = (mid[0] + branch_rng.uniform(-1, 1) * segment_length * 0.6, mid[1] + segment_length * branch_rng.uniform(0.3, 0.7))
        _draw_branch(draw, mid, branch_end, trunk_rng, branch_rng, branch_chance * 0.6, depth + 2, color,
                     max(1, line_width - 1), max_depth, is_branch=True)


def _fade_bolt(bolt_image, fade):
    """Scales a bolt frame's alpha channel by `fade` (0..1), for the flash-then-afterglow decay."""
    if fade >= 0.999:
        return bolt_image
    alpha = np.asarray(bolt_image.getchannel("A"), dtype="float32") * fade
    result = bolt_image.copy()
    result.putalpha(Image.fromarray(alpha.astype("uint8"), mode="L"))
    return result
