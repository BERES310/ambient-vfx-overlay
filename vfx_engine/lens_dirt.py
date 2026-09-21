"""
lens_dirt.py

A static camera-lens grime/dust overlay - a soft, cloud-like film
concentrated more toward the frame edges/corners (real lens dirt
accumulates there; the center gets cleaned/wiped more often) - a subtle
"shot through an actual, slightly imperfect lens" touch. This is a camera
artifact, not scene content, the same category as lens flare (see
god_rays.py) - it belongs layered on top of everything else, not mixed in
with the scene's own particles/lighting.

Built from coherent (fbm) noise, NOT discrete blob/ellipse shapes - an
earlier version used clusters of overlapping ellipses, which looked fine
on a busy, high-contrast photo (the blobs blended into the existing
texture) but read as clearly artificial, distinct dark circles on a
low-contrast, fairly uniform photo (confirmed visually: a drone shot of
solid green forest made every blotch obvious). Coherent noise has no
"shape" to read as artificial in the first place - it's just an uneven
density field, the same way real grime/haze actually looks.

Computed once (a physical smudge on a lens doesn't move or change over a
20-30s loop) and applied as a low, mostly-constant multiplicative
darkening, with only the faintest cyclic pulse so it isn't a perfectly
dead-static overlay for the whole video - see render_lens_dirt.
"""

import hashlib
import math

import numpy as np
from PIL import Image, ImageFilter


def _hash(p):
    """Vectorized pseudo-random value for 2D points - see particles.py's identical helper."""
    return np.mod(np.sin(p[..., 0] * 127.1 + p[..., 1] * 311.7) * 43758.5453123, 1.0)


def _value_noise(p):
    """Smoothly interpolated ("value") noise - see particles.py's identical helper."""
    cell = np.floor(p)
    f = p - cell

    a = _hash(cell)
    b = _hash(cell + np.array([1.0, 0.0]))
    c = _hash(cell + np.array([0.0, 1.0]))
    d = _hash(cell + np.array([1.0, 1.0]))

    u = f * f * (3.0 - 2.0 * f)
    ux, uy = u[..., 0], u[..., 1]
    return a + (b - a) * ux + (c - a) * uy * (1.0 - ux) + (d - b) * ux * uy


def _fbm(p, octaves=4):
    """Fractal Brownian motion - several octaves of noise summed together, for an organic, cloud-like density field (see shaders/blur.glsl's identical technique for fog)."""
    value = np.zeros(p.shape[:-1], dtype="float32")
    amplitude = 0.5
    for _ in range(octaves):
        value += amplitude * _value_noise(p)
        p = p * 2.0
        amplitude *= 0.5
    return value


def _build_grime_layer(size, rng, scale, edge_power, threshold):
    """
    Returns a (height, width) float32 array, 0.0-1.0, of one coherent-noise
    grime layer - see build_dirt_texture, which combines two calls to this
    at different frequencies/thresholds. Shared by both the
    haze-film and dust-speck layers so they only differ in their
    scale/edge_power/threshold tuning, not in the underlying noise method.
    """
    width, height = size
    offset = rng.uniform(0.0, 1000.0, 2)

    y_coords, x_coords = np.mgrid[0:height, 0:width].astype("float32")
    aspect = height / width
    noise_x = (x_coords / width) * scale + offset[0]
    noise_y = (y_coords / height) * scale * aspect + offset[1]
    noise = _fbm(np.stack([noise_x, noise_y], axis=-1))
    noise = (noise - noise.min()) / max(1e-6, noise.max() - noise.min())

    center_x, center_y = width / 2.0, height / 2.0
    dx = (x_coords - center_x) / center_x
    dy = (y_coords - center_y) / center_y
    radial = np.clip(np.sqrt(dx * dx + dy * dy) / 1.2, 0.0, 1.0) ** edge_power

    weighted = noise * radial
    grime = np.clip((weighted - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    return grime.astype("float32")


def build_dirt_texture(size, seed=None, scale=3.0, edge_power=2.4, threshold=0.6,
                        speck_scale=14.0, speck_threshold=0.78, speck_edge_power=0.6, speck_weight=0.7):
    """
    Returns a (height, width) float32 array, 0.0-1.0, where higher values
    mean MORE grime (used as a darkening strength, not a color) - a soft,
    uneven noise field, weighted toward the frame edges/corners via
    edge_power (real lens dirt accumulates there, not dead-center - the
    center of a lens gets wiped clean more often).

    Fixes real lens imperfections are usually a mix of two
    visually distinct things at once - a broad, soft overall haze/grime
    film, and a separate handful of small, sharp, sparse dust specks or
    hairs. A single noise field at one frequency/threshold can only
    convincingly read as one of the two (the broad film, which is why the
    original single-layer version was tuned that way - see this module's
    docstring on why discrete blob shapes were rejected). This now combines
    TWO independently-seeded `_build_grime_layer` calls: the original
    broad, edge-weighted "haze film" (scale/edge_power/threshold, as
    before) plus a second, higher-frequency, higher-threshold, much less
    edge-biased "dust speck" layer (speck_scale/speck_threshold/
    speck_edge_power - real dust motes and hairs can land anywhere on a
    lens, not just the edges, unlike the slowly-accumulating haze film).
    The two layers are added, not maxed or averaged - a speck sitting on
    top of haze should read as MORE grime there, not just whichever layer
    was already stronger.

    scale: spatial frequency of the haze-film layer's underlying noise -
        lower values give broader, cloudier patches; higher values give
        finer, more speckled grime.
    edge_power: >0, how sharply the haze film concentrates toward the
        edges - 1.0 is a gentle radial falloff, higher values (2-3) keep
        the center noticeably cleaner before grime density ramps up.
    threshold: 0.0-1.0, only haze-film noise values above this (after edge
        weighting) show as visible grime at all - keeps most of the frame
        completely clean, with just a few soft patches standing out.
    speck_scale/speck_threshold/speck_edge_power: the same three
        parameters, tuned instead for the fine dust-speck layer - see
        config.py's LENS_DIRT_SPECK_* defaults for the actual tuned
        values (much higher frequency and threshold, much gentler edge
        bias, than the haze film).
    speck_weight: >=0, how strongly the speck layer contributes once
        combined with the haze film - lower than 1.0 keeps specks reading
        as a subtle accent rather than doubling the overall grime density.
    """
    rng = np.random.default_rng(seed)
    haze = _build_grime_layer(size, rng, scale, edge_power, threshold)
    specks = _build_grime_layer(size, rng, speck_scale, speck_edge_power, speck_threshold)
    return np.clip(haze + specks * speck_weight, 0.0, 1.0).astype("float32")


def content_seed(image, salt=""):
    """
    Gap B fix - returns a deterministic integer seed derived
    from a cheap downsampled thumbnail of `image`'s own pixel content,
    for use as `build_dirt_texture`'s `seed` argument. Without this, the
    grime layout either comes from a fixed/config-level seed (identical
    grime on every photo this pipeline ever renders, which a viewer
    comparing several outputs side by side would quickly notice and
    correctly perceive as a stock overlay rather than an emergent
    property of any particular photo) or - the actual previous behavior
    here, since no seed was ever passed at the `build_dirt_texture` call
    site - a fresh OS-entropy seed every single call, meaning even
    re-rendering the SAME photo twice produced two different grime
    layouts, which breaks reproducibility for regression testing and for
    a user re-running the tool on the same input expecting the same
    result. Hashing the photo's own content gives each distinct input
    photo its own distinct, but still fully deterministic, grime layout.

    Downsampled to a tiny 32x32 thumbnail before hashing (not the full-
    resolution pixel buffer) so this stays cheap regardless of the
    source photo's actual resolution - full-photo-vs-thumbnail content
    hashes still differ between any two visually distinct photos, which
    is all a seed needs to do here.

    salt: generalization (RNG-determinism audit, 2026-09-17) for reuse
        beyond lens-dirt grime - every other unseeded RNG call site found
        in the codebase (particle spawn positions, flicker curve, film
        grain) also needs a deterministic-per-photo seed, but several of
        them derive from the SAME `image`. Without a salt, rain and snow
        seeded from one photo would draw the exact same spawn-position
        sequence (visually correlated, wrong-looking particle patterns).
        Folding a short caller-chosen string into the hash keeps each use
        site's seed distinct while still being 100% reproducible for a
        given (image, salt) pair. Default "" preserves this function's
        original output exactly, so the one pre-existing call site
        (build_dirt_texture's seed, main.py) is unaffected.
    """
    thumbnail = image.convert("RGB").resize((32, 32), Image.BILINEAR)
    digest = hashlib.sha256(thumbnail.tobytes() + salt.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def build_light_glow_field(size, light_pos_uv, radius):
    """
    Returns a (height, width) float32 array, 0.0-1.0, of a Gaussian falloff
    centered on `light_pos_uv` (normalized 0.0-1.0 IMAGE-space coordinates,
    y=0 at the top - the same convention main.py's resolve_light_pos_uv/
    god_rays.detect_light_source use) - how close each pixel is to the
    photo's own detected light source, in aspect-corrected units so the
    glow reads as a circle rather than an ellipse on non-square photos.

    Mirrors shaders/blur.glsl's identical `light_glow` computation for
    fog's own light-position coupling - same falloff
    shape, same aspect-correction trick, just evaluated on the CPU here
    since lens dirt is a plain numpy overlay, not a GPU shader.

    Public (not `_`-prefixed) since Gap A has main.py build
    this once from the STATIC, undrifted light position and then
    optionally warp the result through camera.apply_camera_drift_to_array
    itself, exactly like god_rays.build_ray_texture/build_lens_flare_
    texture/build_starburst_texture's own textures do -
    see render_lens_dirt's `glow_shape` parameter.
    """
    width, height = size
    y_coords, x_coords = np.mgrid[0:height, 0:width].astype("float32")
    aspect = width / height

    x_norm = (x_coords / width) * aspect
    y_norm = y_coords / height
    light_x, light_y = light_pos_uv
    dx = x_norm - light_x * aspect
    dy = y_norm - light_y

    dist_sq = dx * dx + dy * dy
    return np.exp(-dist_sq / (2.0 * radius * radius)).astype("float32")


def _build_brightness_visibility_field(image, size, min_visibility, max_visibility, blur_fraction):
    """
    Returns a (height, width) float32 array - an alpha MULTIPLIER derived
    from how bright the photo itself is at each pixel, ranging from
    min_visibility (over the darkest regions) to max_visibility (over the
    brightest). Fixes real dust/grime is far more visually
    apparent against a bright, even background (it stands out clearly
    against open sky) than a dark, low-contrast one (barely perceptible
    against shadowed foliage) - the old code applied grime alpha uniformly
    regardless of what was actually behind it.

    Uses the same perceptual luma weights (0.299R + 0.587G + 0.114B) as
    postprocess.py's bloom brightness extraction rather
    than a flat channel average, for the same reason: human-perceived
    brightness weights green far more heavily than red or blue.

    Blurred (blur_fraction * width, as a Gaussian radius) before use so
    this reads as a smooth "local background" brightness rather than
    being sensitive to per-pixel noise/texture in the source photo - a
    single stray bright pixel in a dark region shouldn't make grime
    right on top of it suddenly pop.
    """
    width, height = size
    resized = image.resize((width, height), Image.BILINEAR).convert("RGB")
    blur_radius = max(1.0, blur_fraction * width)
    resized = resized.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    array = np.asarray(resized, dtype="float32") / 255.0
    luma = np.tensordot(array, np.array([0.299, 0.587, 0.114], dtype="float32"), axes=([2], [0]))
    return (min_visibility + (max_visibility - min_visibility) * luma).astype("float32")


def render_lens_dirt(size, num_frames, texture, strength=0.35, pulse_amount=0.08, pulse_speed=1,
                      light_pos_uv=None, light_color=None, glow_strength=0.9, glow_radius=0.55,
                      glow_shape=None,
                      image=None, brightness_min_visibility=0.35, brightness_max_visibility=1.35,
                      brightness_blur_fraction=0.03, start_frame=0):
    """
    Generator - yields num_frames RGBA overlay frames of the lens grime
    (see build_dirt_texture), composited over the frame with a color that
    varies per pixel between near-black (plain darkening, the old
    behavior) and the photo's detected light color (a bright, localized
    glow) depending on how much each pixel's grime density overlaps with
    where the light source actually sits.

    Fixes real lens grime/dust doesn't just darken evenly -
    it SCATTERS light, and that scattering becomes strongly, specifically
    visible wherever a bright light source sits near/behind the dirt in
    frame (this is literally why "dirty lens" veiling-glare is its own
    distinct cinematography look, not just a darkened image). The third
    module in this codebase with this exact gap, after fog.py (1.6/5.4)
    and ground_fx.render_wet_sheen (3.5) - all three now draw on the same
    per-photo light-position/color detection main.py already computes
    once (resolve_light_pos_uv/god_rays.sample_light_color).

    light_pos_uv: normalized (0.0-1.0) IMAGE-space light position - see
        main.py's resolve_light_pos_uv. None (default) disables the glow
        entirely - pure darkening, old behavior, unchanged.
    light_color: (r, g, b) 0-255 - the photo's detected light color (see
        god_rays.sample_light_color). Falls back to a neutral warm white
        when light_pos_uv is given but light_color isn't.
    glow_strength: 0.0-1.0+, how strongly the glow can overpower the plain
        darkening at its brightest point (dead-center on the light,
        directly under a dense patch of grime).
    glow_radius: >0, in the same aspect-corrected units as
        build_light_glow_field - how far the glow reaches from the light
        position before fading out. 0.55 matches fog's own light_radius
        (shaders/blur.glsl) so both effects glow over a visually
        consistent area around the same detected light source. Ignored
        when glow_shape is given explicitly.
    glow_shape: optional pre-built raw light-glow field (see
        build_light_glow_field) - either a single static (height, width)
        array (equivalent to the default, computed-once-from-light_pos_uv
        behavior below) or Gap A, a per-frame iterable (see
        camera.apply_camera_drift_to_array) so the glow tracks the
        light's ACTUAL drifted screen position each frame instead of
        staying glued to the original photo's framing while camera drift
        pans/zooms the base image underneath it - the same registration-
        mismatch class already fixed for god_rays.render_light_overlay/
        render_lens_flare_overlay. The grime `texture`
        itself is NEVER drift-warped (real lens grime is physically fixed
        to the lens, not the scene) - only this glow SHAPE, which tracks
        where the light appears on screen, is per-frame-capable; each
        frame's actual glow field is still `texture * current_glow_shape`,
        so the grime pattern stays static while just the glow sweeps
        across it. None (default) falls back to computing this once,
        internally, from light_pos_uv/glow_radius exactly as before -
        unchanged behavior whenever camera drift is disabled or the
        caller has no per-frame position to track. Same
        isinstance(..., np.ndarray) static-or-iterable convention as
        god_rays.render_light_overlay.

    Also fixes `image` (the same source photo everything
    else in this pipeline already uses), when given, scales the effective
    grime texture by _build_brightness_visibility_field before anything
    else below is computed - so BOTH the plain darkening and the design notes-
    15.1 glow read more strongly over bright regions of the photo and
    more faintly over dark ones, instead of applying uniformly regardless
    of what's actually behind the lens dirt. None (default) skips this
    entirely - texture is used exactly as given, old behavior unchanged.
    brightness_min_visibility/brightness_max_visibility: the alpha
        multiplier range mapped from the darkest to the brightest regions
        of the photo - see config.py's LENS_DIRT_BRIGHTNESS_* defaults.
    brightness_blur_fraction: >0, Gaussian blur radius for the brightness
        reference, as a fraction of image width - smooths pixel-level
        noise into a "local background" read.

    Only the faintest cyclic pulse in strength - real lens grime doesn't
    visibly animate; a tiny bit of variation just keeps a 20-30s loop from
    having one perfectly dead-flat layer sitting over the whole thing.
    pulse_speed MUST be a whole number for the loop to close exactly (see
    god_rays.render_light_overlay's matching note).

    start_frame: skips yielding frames 0..start_frame-1 (see
        render_checkpoint.py). CONTRACT when `glow_shape` is an animated
        iterable (not a static np.ndarray): the caller must already have
        built it so it begins at `start_frame` (e.g. by passing the same
        start_frame through to camera.apply_camera_drift_to_array) -
        this function does NOT (and must not) discard `start_frame` items
        via next() itself, since that would waste exactly the per-frame
        work start_frame exists to skip; it simply trusts the iterable's
        first item corresponds to frame `start_frame`, the same contract
        god_rays.render_light_overlay documents for its own animated
        texture parameter. A static glow_shape (or no glow at all) is
        unaffected either way. 0 (default) = old behavior.
    """
    if image is not None:
        visibility_field = _build_brightness_visibility_field(
            image, size, brightness_min_visibility, brightness_max_visibility, brightness_blur_fraction)
        texture = np.clip(texture * visibility_field, 0.0, 1.0).astype("float32")

    height, width = texture.shape
    black_rgb = np.zeros((height, width, 3), dtype="float32")

    glow_field = None
    animated_glow_shape = False
    glow_shape_iter = None
    if light_pos_uv is not None or glow_shape is not None:
        if glow_shape is None:
            glow_shape = build_light_glow_field(size, light_pos_uv, glow_radius)
        animated_glow_shape = not isinstance(glow_shape, np.ndarray)
        glow_shape_iter = iter(glow_shape) if animated_glow_shape else None
        current_glow_shape = next(glow_shape_iter) if animated_glow_shape else glow_shape
        glow_field = texture * current_glow_shape
        resolved_light_color = light_color if light_color is not None else (255, 244, 214)
        light_rgb = np.empty((height, width, 3), dtype="float32")
        for channel in range(3):
            light_rgb[:, :, channel] = resolved_light_color[channel]

    for frame_i in range(start_frame, num_frames):
        if animated_glow_shape and frame_i > start_frame:
            current_glow_shape = next(glow_shape_iter)
            glow_field = texture * current_glow_shape

        theta = 2.0 * math.pi * pulse_speed * (frame_i / max(num_frames, 1))
        pulse = 1.0 + pulse_amount * math.sin(theta)
        dark_alpha = texture * strength * pulse

        if glow_field is None:
            total_alpha = np.clip(dark_alpha, 0.0, 1.0)
            rgba = np.zeros((height, width, 4), dtype="uint8")
            rgba[:, :, 3] = (total_alpha * 255).astype("uint8")
            yield Image.fromarray(rgba, mode="RGBA")
            continue

        glow_alpha = glow_field * glow_strength * pulse
        total_alpha = np.clip(dark_alpha + glow_alpha, 0.0, 1.0)

        # Per-pixel color blend, NOT a separate layer: away from the light
        # (glow_alpha ~ 0) this collapses to pure black - the old, unchanged
        # darkening. Near the light, where grime and light position overlap,
        # the blended color shifts toward the actual light color and the
        # combined alpha rises above what plain darkening alone would give -
        # composited with normal "over" alpha blending, that reads as a
        # localized brightening/tint (glow) instead of more darkening.
        glow_weight = (glow_alpha / np.maximum(total_alpha, 1e-6))[:, :, None]
        rgb = black_rgb * (1.0 - glow_weight) + light_rgb * glow_weight

        rgba = np.empty((height, width, 4), dtype="uint8")
        rgba[:, :, 0:3] = np.clip(rgb, 0.0, 255.0).astype("uint8")
        rgba[:, :, 3] = (total_alpha * 255).astype("uint8")
        yield Image.fromarray(rgba, mode="RGBA")
