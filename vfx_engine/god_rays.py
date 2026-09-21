"""
god_rays.py

Volumetric light shafts ("god rays" / crepuscular rays) radiating from the
dominant light source detected in the photo, plus a simple camera lens
flare - the design notes' Lighting section. Real light rarely stays
perfectly diffuse when there's haze, fog, or falling rain/snow in the air;
it visibly streaks toward the camera, and a bright enough source produces
a lens flare (halo + a few faint "ghost" circles) as an artifact of the
camera itself, not the scene.

Both the ray shaft texture and the flare texture are computed ONCE from
the base photo - the light source's position doesn't move over a 20-30s
loop, so recomputing an expensive radial blur every single frame would be
pure waste. What varies per frame is a slow, cyclic intensity pulse
(loop-safe, the same phase-based approach as everything else in this
pipeline - see particles_wind.comp's docstring for why "additive/
multiplicative sinusoid of the loop phase" is the standard trick here),
layered on top of the static texture.
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import config


def detect_light_source(base_image, top_bias=0.7):
    """
    Returns (x, y) pixel coordinates of the likely dominant light source in
    the photo - the brightest region, weighted toward the upper part of the
    frame (sky/sun/lamps are usually up there, not on the ground) so a
    stray bright reflection low in the frame doesn't win over the actual
    sky. Cheap: works on a small thumbnail, not the full-resolution image.
    Shared by god rays, lens flare, and particles_rain.glsl's dual-tone
    lighting (see main.py's resolve_light_direction) - one detection, reused
    everywhere that needs "where's the light coming from" in this photo.
    """
    thumb_size = 96
    thumb = base_image.convert("L").resize((thumb_size, thumb_size))
    luminance = np.asarray(thumb, dtype="float32") / 255.0

    rows = np.arange(thumb_size, dtype="float32").reshape(-1, 1) / max(thumb_size - 1, 1)
    vertical_weight = 1.0 - top_bias * rows   # rows near the top (small row index) keep more of their weight
    weighted = luminance * vertical_weight

    flat_index = int(np.argmax(weighted))
    row, col = divmod(flat_index, thumb_size)
    width, height = base_image.size
    return (col / thumb_size) * width, (row / thumb_size) * height


def detect_light_sources(base_image, top_bias=0.7, max_sources=3, min_relative_strength=0.35):
    """
    Returns a list of up to `max_sources` dicts `{"pos": (x, y), "strength":
    0.0-1.0}`, one per distinct bright region in the photo, sorted brightest
    first (`strength=1.0` for the dominant one, everyone else expressed as a
    fraction of it) - fixes `detect_light_source` (singular)
    always picked exactly one point via a single `argmax`, so a photo with
    several comparably bright sources (streetlights down a road, multiple
    windows, headlights) only ever got flare/ray treatment from whichever
    one happened to be brightest, with every other real light source in
    frame silently ignored - even though each would, in reality, cast its
    own (fainter, since it isn't dominant) flare pattern.

    Works the same way `detect_light_source` does (small thumbnail,
    top-biased luminance so sky/lamps win over stray ground reflections),
    but instead of stopping at one `argmax`, repeatedly takes the current
    brightest remaining point, records it, then suppresses it via
    connected-component labeling: every pixel connected to that peak whose
    value is at least `component_threshold_frac` of the peak's own value
    gets zeroed out in one pass (`_flood_fill_mask` below - a small
    dependency-free flood fill, since this project keeps everything to
    numpy/PIL/torch rather than pulling in scipy just for this one
    lookup), not just a small fixed-radius box around the peak. A large
    diffuse light source (a
    bright overcast sky patch) and a tiny pinpoint one (a distant
    streetlamp) both get their true full extent suppressed as a single
    piece this way, regardless of how big that extent actually is - a
    fixed-radius box either under-suppresses a large/plateaued source
    (its own unsuppressed edges then get "rediscovered" as spurious extra
    sources on the next pass) or over-suppresses a small one's
    surroundings for no reason. Stops once either `max_sources` is
    reached or the next-brightest remaining point is dimmer than
    `min_relative_strength` of the dominant one (a photo with one
    obviously dominant sun and otherwise unremarkable brightness
    shouldn't have `max_sources` worth of near-noise treated as real
    secondary lights).

    `component_threshold_frac` (the flood-fill
    suppression bar, "connected pixels at least this fraction of the
    peak's own value belong to the same source") used to be one fixed
    constant (0.3) regardless of how this particular photo's brightness
    is actually distributed. A high-contrast scene (one sharp, isolated
    light against an otherwise dark frame) can afford a LOWER fraction -
    doing so just sweeps up that same source's own soft graded edges in
    one pass instead of truncating it early and having stray unsuppressed
    edge pixels get "rediscovered" as a spurious extra source next
    iteration, and a genuinely separate dim secondary source stays far
    enough below even a low bar (and is not 4-connected to the peak
    anyway) to survive untouched. A low-contrast/overcast scene (the
    brightest sky patch only marginally brighter than the rest of the
    sky) needs a HIGHER fraction, or the flood fill would spread across
    most of the merely-bright sky before hitting a genuinely darker
    pixel, gobbling up a good chunk of the frame as "one light source"
    and starving every real secondary source of its own contrast budget
    against `min_relative_strength`. Fixed here by deriving the fraction,
    per detected source, from the gap between that source's own peak
    luminance and the thumbnail's overall median/background luminance
    (already-available data from the same 96x96 detection thumbnail, no
    new sampling) - a bigger peak-to-background gap (more contrast) maps
    to a lower fraction, a smaller gap (less contrast) maps to a higher
    one. This only touches this function's own internal threshold
    computation; no change to `_flood_fill_mask`, `detect_light_source`,
    or any caller's interface (no caller ever passed the old fixed
    parameter explicitly).
    """
    thumb_size = 96
    # Box resampling (pure local averaging, no overshoot) instead of the
    # default resize filter - a 5x+ downsample of a sharp bright disc with
    # a ringing-prone filter (bicubic/lanczos) can produce a faint
    # secondary "halo" ring of elevated brightness away from the true
    # peak, which non-max suppression below would then misread as a
    # second, distinct light source rather than resampling artifact of
    # the SAME source.
    thumb = base_image.convert("L").resize((thumb_size, thumb_size), resample=Image.BOX)
    luminance = np.asarray(thumb, dtype="float32") / 255.0

    rows = np.arange(thumb_size, dtype="float32").reshape(-1, 1) / max(thumb_size - 1, 1)
    vertical_weight = 1.0 - top_bias * rows
    working = luminance * vertical_weight

    width, height = base_image.size

    # the photo's own median/background luminance
    # (unweighted by top_bias, which only exists to bias WHERE the peak
    # search looks, not to redefine what "background brightness" means),
    # computed once up front from the same thumbnail already in hand.
    background_luminance = float(np.median(luminance))

    sources = []
    dominant_value = None
    for _ in range(max_sources):
        flat_index = int(np.argmax(working))
        value = float(working.flat[flat_index])
        if value <= 0.0:
            break

        if dominant_value is None:
            dominant_value = value
        relative_strength = value / dominant_value
        if relative_strength < min_relative_strength:
            break

        row, col = divmod(flat_index, thumb_size)
        pos = ((col / thumb_size) * width, (row / thumb_size) * height)
        sources.append({"pos": pos, "strength": relative_strength})

        # contrast_ratio: 0.0 (this source's peak is no brighter than the
        # scene's median/background) to 1.0 (peak is vastly brighter than
        # background, the classic isolated-point-light case). Mapped onto
        # [0.15, 0.55] - high contrast -> low fraction (sweep up the full
        # graded source in one pass), low contrast -> high fraction (don't
        # let the fill spread across a broad, merely-bright sky).
        peak_luminance = float(luminance.flat[flat_index])
        contrast_ratio = float(np.clip(
            (peak_luminance - background_luminance) / max(peak_luminance, 1e-6), 0.0, 1.0))
        component_threshold_frac = 0.55 - 0.40 * contrast_ratio

        component_mask = _flood_fill_mask(working >= value * component_threshold_frac, row, col)
        working[component_mask] = 0.0

    return sources


def _flood_fill_mask(binary_mask, start_row, start_col):
    """
    Returns a boolean array the same shape as `binary_mask`, marking the
    4-connected region of True pixels reachable from `(start_row,
    start_col)` - a minimal, dependency-free connected-component fill
    (this project avoids adding scipy as a dependency purely for
    `ndimage.label`; the thumbnail this runs on is tiny, 96x96, so a plain
    stack-based flood fill is more than fast enough) used by
    `detect_light_sources` to suppress a discovered light source's true
    full extent rather than an arbitrary fixed-radius box around its peak.
    """
    visited = np.zeros_like(binary_mask, dtype=bool)
    if not binary_mask[start_row, start_col]:
        visited[start_row, start_col] = True
        return visited

    height, width = binary_mask.shape
    stack = [(start_row, start_col)]
    visited[start_row, start_col] = True
    while stack:
        row, col = stack.pop()
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row, next_col = row + d_row, col + d_col
            if 0 <= next_row < height and 0 <= next_col < width:
                if binary_mask[next_row, next_col] and not visited[next_row, next_col]:
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
    return visited


def sample_light_color(base_image, light_pos, radius_frac=0.05, fallback=(255, 255, 255),
                        ambient_blend=None, illumination_result=None):
    """
    Returns the average RGB color of a small neighborhood around
    `light_pos` (pixel coordinates - see detect_light_source) in the
    actual photo - the closest available local proxy for "what color is
    the dominant light in this scene" without a dedicated illumination-
    color model (a genuinely harder, less standardized task than depth/
    segmentation - see, an exploratory spike, not something
    to build speculatively here).

    Used to tint wet-surface reflections, fog, and lens artifacts toward the photo's real light color
    instead of a fixed white/neutral tone - warm sodium streetlamps
    should produce a warm orange sheen, not a neutral one.

    this local point sample is now blended with
    lighting.estimate_ambient_light_color's scene-WIDE ambient term
    (config.AMBIENT_LIGHT_BLEND_STRENGTH, or `ambient_blend` to override
    it) before being returned - a small neighborhood right around the
    single brightest point can miss the photo's broader color cast (see
    that function's own docstring for the sunset/sun-disc example), so
    every one of this function's callers (fog, wet sheen/splashes, lens
    dirt/rain) gets the benefit automatically without each needing its
    own separate blending logic. `illumination_result`, when given
    (config.ENABLE_LEARNED_ILLUMINATION), lets the ambient term prefer
    illumination.py's learned `shading` channel over a blurred-photo
    heuristic - see estimate_ambient_light_color's.9(a)
    section. Imported lazily (not at module top) purely to keep this
    module's existing top-of-file import list (math, numpy, PIL)
    unchanged for any caller that doesn't need this - lighting.py has no
    reason to import god_rays back, so there is no cycle either way.
    """
    width, height = base_image.size
    radius = max(4, int(min(width, height) * radius_frac))
    x, y = light_pos
    left, right = max(0, int(x - radius)), min(width, int(x + radius))
    top, bottom = max(0, int(y - radius)), min(height, int(y + radius))
    if right <= left or bottom <= top:
        point_color = fallback
    else:
        patch = base_image.convert("RGB").crop((left, top, right, bottom))
        pixels = np.asarray(patch, dtype="float32").reshape(-1, 3)
        point_color = tuple(int(np.clip(channel, 0, 255)) for channel in pixels.mean(axis=0))

    from vfx_engine import lighting
    blend = config.AMBIENT_LIGHT_BLEND_STRENGTH if ambient_blend is None else ambient_blend
    if blend <= 0.0:
        return point_color

    ambient_color = lighting.estimate_ambient_light_color(base_image, illumination_result=illumination_result)
    return tuple(
        int(np.clip((1.0 - blend) * point_channel + blend * ambient_channel, 0, 255))
        for point_channel, ambient_channel in zip(point_color, ambient_color)
    )


def _zoom_blur_toward(image, center, num_samples=72, max_scale=2.4):
    """
    Accumulates `num_samples` progressively-more-zoomed-in copies of
    `image` (each sampled closer to `center`), EQUALLY weighted, and
    averages them - the standard cheap technique for a radial/zoom blur:
    repeated small zooms toward a point, blended together, read as
    streaks radiating away from that point once accumulated. `image` must
    be a single-channel ("L" mode) PIL Image.

    Equal weighting matters: an earlier version of this weighted samples
    by a decaying factor (favoring the least-zoomed, i=0 sample, which is
    essentially the unblurred original image) - the normalized average
    ended up completely dominated by that near-identity sample, and the
    far-reaching, more-stretched samples that actually carry the visible
    "ray" content got diluted to well under 1% of the total. The result
    was confirmed empirically to produce no visible streaking at all.
    Equal weighting gives every sample, near and far, the same say.
    """
    width, height = image.size
    cx, cy = center

    accum = np.zeros((height, width), dtype="float32")
    for i in range(num_samples):
        t = i / max(num_samples - 1, 1)
        scale = 1.0 + (max_scale - 1.0) * t
        inv_scale = 1.0 / scale
        # PIL's AFFINE transform maps OUTPUT pixel (x, y) to an INPUT
        # sample coordinate of (a*x + b*y + c, d*x + e*y + f) - this is
        # the inverse of "scale the image up by `scale` about (cx, cy)",
        # i.e. sampling closer and closer to the center as scale grows.
        a, b, c = inv_scale, 0.0, cx * (1.0 - inv_scale)
        d, e, f = 0.0, inv_scale, cy * (1.0 - inv_scale)
        sampled = image.transform((width, height), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR)
        accum += np.asarray(sampled, dtype="float32")

    return accum / num_samples


def build_ray_texture(base_image, light_pos, threshold=0.72, strength=0.6, fog_density=0.0,
                       ground_fog_strength=0.0):
    """
    Returns a single (height, width) float32 array, 0.0-1.0, of the god-ray
    intensity at every pixel - built once from base_image, reused (with a
    per-frame intensity pulse, see render_light_overlay) for the whole loop.

    threshold: 0.0-1.0, luminance above which the (already radially-
        blurred) result counts as a visible ray, vs. falling back into
        the darker gaps between rays.
    strength: 0.0-1.0+, overall ray brightness multiplier.
    fog_density: 0.0-1.0, the scene's current fog density (config.
        FOG_PARAMS["density"] when fog is enabled, 0.0 otherwise) - fixes
        light shafts are visible BECAUSE OF fog/haze
        particulate scattering the light, not two coincidentally-
        overlapping phenomena, but the old code had zero reference
        between the two. More fog makes real light shafts both MORE
        VISIBLE (there's more particulate to scatter light toward the
        camera) and MORE DIFFUSE (softer-edged, broader shafts rather
        than crisp defined beams - scattering blurs the alternating
        bright/dark structure that makes a shaft read as a shaft at all).
        Coupled here as a single global scalar (the same level of
        sophistication config.py's shared wind/gust state already uses
        elsewhere in this pipeline - a true per-pixel spatial fog-density
        field would need fog's own GPU noise field to be queryable from
        Python, a much larger architecture change out of scope here).
        0.0 (default) = old behavior, completely unchanged.
    ground_fog_strength: 0.0-1.0 - the
        SAME vertical-stratification curve fog.py/blur.glsl already use
        for ground fog (`height_factor` at blur.glsl:262, a closed-form
        function of vertical screen position alone: `mix(1.0 -
        ground_fog_strength, 1.0, pow(1.0 - v_uv.y, 1.6))`, v_uv.y=0 at
        the BOTTOM of frame, 1.0 at the top), reused here as a cheap
        post-multiply on the ray texture's own vertical axis so rays
        thin out toward the top of frame exactly where fog.py's own
        ground fog is thinnest, and stay full-strength low in frame where
        ground fog sits thickest - matching where god rays' underlying
        physical cause (light scattering off fog/haze particulate) is
        actually most concentrated. This is a DELIBERATELY partial fix,
        not a true per-pixel fog field: it only captures the vertical
        ground-fog gradient, not a horizontal fog bank or patchy fog
        shape (`fog.py`'s own bank_noise term, say) - a full per-pixel
        coupling would need fog's own GPU noise field queryable from
        Python, the same larger architecture change `fog_density` above
        already declines to make. 0.0 (default) = old behavior, no
        vertical falloff at all.

    IMPORTANT: the radial blur has to run on the FULL luminance image
    (light AND dark), not a pre-thresholded "bright-only" mask. Real god
    rays come from ALTERNATING bright/dark structure (sun between tree
    branches, gaps in a rooftop) being stretched radially outward from
    the light - a mask that already discarded every dark occluding pixel
    has nothing left to alternate with, and zoom-blurring a plain bright
    blob just produces another plain blob with no visible streaks (this
    was tried and confirmed empirically: a bright disc with no threshold-
    surviving structure around it produced zero visible rays). Threshold
    AFTER blurring, not before.
    """
    grayscale = base_image.convert("L")
    blurred_luminance = _zoom_blur_toward(grayscale, light_pos) / 255.0

    if fog_density > 0.0:
        # Extra softening on top of the zoom blur - denser fog scatters
        # light more broadly, smearing out the alternating bright/dark
        # structure that makes a shaft read as a crisp beam.
        diffuse_radius = max(1, int(min(base_image.size) * 0.01 * fog_density))
        diffuse_image = Image.fromarray((blurred_luminance * 255.0).astype("uint8"), mode="L")
        diffuse_image = diffuse_image.filter(ImageFilter.GaussianBlur(diffuse_radius))
        blurred_luminance = np.asarray(diffuse_image, dtype="float32") / 255.0

    # More fog -> lower effective threshold (more of the frame counts as
    # a visible ray, since scattered light fills in what would otherwise
    # be a dark gap) and higher effective strength (the shaft itself
    # scatters more light toward the camera).
    effective_threshold = threshold * (1.0 - 0.3 * fog_density)
    effective_strength = strength * (1.0 + 0.5 * fog_density)

    # Smoothstep the threshold ramp (a linear ramp leaves a visible hard
    # edge where it starts) and feather the result, so the shaft fades out
    # softly instead of ending in a hard-edged wedge.
    ramp = np.clip((blurred_luminance - effective_threshold) / max(1e-6, 1.0 - effective_threshold), 0.0, 1.0)
    rays = ramp * ramp * (3.0 - 2.0 * ramp)
    feather_radius = max(2, int(min(base_image.size) * 0.012))
    feather_image = Image.fromarray((rays * 255.0).astype("uint8"), mode="L").filter(ImageFilter.GaussianBlur(feather_radius))
    rays = np.asarray(feather_image, dtype="float32") / 255.0
    rays = np.clip(rays * effective_strength, 0.0, 1.0)

    if ground_fog_strength > 0.0:
        # see docstring. rays is a plain
        # numpy array in image row order (row 0 = frame TOP), so v_uv.y
        # (GLSL convention, 1.0 = top, 0.0 = bottom) maps to
        # `1.0 - row / (height - 1)`.
        height = rays.shape[0]
        row_index = np.arange(height, dtype="float32").reshape(-1, 1)
        v_uv_y = 1.0 - row_index / max(height - 1, 1)
        height_factor = (1.0 - ground_fog_strength) + ground_fog_strength * np.power(1.0 - v_uv_y, 1.6)
        rays = rays * height_factor

    return np.clip(rays, 0.0, 1.0)


def _jitter_ghost_color(base_color, rng, warmth_range=0.35, tint_range=0.2):
    """
    Returns an (r, g, b) 0-255 tuple - `base_color` nudged along two
    independent axes to fake chromatic dispersion: real
    lens flare ghosts pick up different color casts from each other
    because light splits slightly differently on each internal reflection
    path, so one ghost often reads distinctly warmer/cooler or more
    tinted than its neighbors even though they all originate from the
    same light source/base color.

    warmth axis: push red up and blue down (or vice versa) - the classic
    warm/cool shift. tint axis: push green up/down against red+blue - a
    second, independent axis so ghosts don't all just fall somewhere on
    one straight warm-cool line, which would still look too uniform/
    patterned across a whole ghost chain.
    """
    r, g, b = (c / 255.0 for c in base_color)
    warmth = rng.uniform(-warmth_range, warmth_range)
    r = np.clip(r + warmth, 0.0, 1.0)
    b = np.clip(b - warmth, 0.0, 1.0)
    tint = rng.uniform(-tint_range, tint_range)
    g = np.clip(g + tint, 0.0, 1.0)
    r = np.clip(r - tint * 0.5, 0.0, 1.0)
    b = np.clip(b - tint * 0.5, 0.0, 1.0)
    return tuple(int(round(c * 255)) for c in (r, g, b))


def _regular_polygon_points(center, radius, sides, rotation_degrees):
    """
    Returns a list of (x, y) vertices for a regular polygon - used to draw
    ghosts as hexagons/heptagons rather than perfect circles.
    """
    center_x, center_y = center
    points = []
    for k in range(sides):
        angle = math.radians(rotation_degrees) + (2.0 * math.pi * k / sides)
        points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))
    return points


def build_lens_flare_texture(size, light_pos, strength=0.5, ghost_count=3, color=(255, 244, 214), seed=None):
    """
    Returns (alpha_texture, color_texture): alpha_texture is a (height,
    width) float32 array, 0.0-1.0, exactly like before; color_texture is
    a NEW (height, width, 3) float32 array, 0.0-1.0, of the per-pixel RGB
    tint to use at that pixel (see render_lens_flare_overlay) - a simple
    camera lens flare: a soft halo centered on the light source, plus a
    few fainter "ghost" shapes positioned along the line from the light
    through the frame center and out the other side (the classic real-lens
    flare pattern - internal reflections between lens elements land ghosts
    on the OPPOSITE side of the frame center from the light itself, at
    varying distances, shrinking and dimming with distance from the
    source).

    Fixes: previously every ghost (and the halo) shared
    one identical flat `color` and was always a perfect circle, and ghost
    spacing/size followed one fixed, perfectly regular arithmetic
    progression (`t = 1.0 + i * 0.55`, `radius = halo_radius * (0.5 -
    i*0.1)`) - identical relative pattern shape for every photo, just
    repositioned/rescaled to that photo's light. Real lens ghost chains
    vary in color (chromatic dispersion), shape (aperture-blade polygons,
    not circles), and aren't perfectly evenly spaced/monotonically
    shrinking. `seed` (None by default, like lens_dirt.build_dirt_texture
    - this pattern is a property of the LENS, not the scene, so run-to-run
    variation is correct/expected, not a bug) drives all of the above via
    one `np.random.default_rng`.

    Fixes the halo used to be a flat, fully-opaque disc
    (`draw.ellipse(fill=255)`) that only became soft because the *entire*
    canvas got one post-hoc Gaussian blur afterward - same shortcut already
    flagged for masks (10.1) and segmentation (9.1): a hard-edged shape
    softened after the fact reads differently than a quantity that was
    actually continuous from the start. A blurred disc's brightness is
    flat across its whole interior and only starts falling off within one
    blur-radius of the (former) edge; a real halo/glow falls off smoothly
    from its brightest point at the center all the way out, with no flat
    plateau. Fixed by computing the halo directly as a continuous radial
    gradient over a coordinate grid (`(1 - r/halo_radius) ** gamma`, the
    same technique `build_starburst_texture` uses for its core blob)
    instead of drawing a disc at all, then blending it with the ghost
    chain's own (still draw+blur, per 16.1/16.3 above - ghosts are meant
    to look like distinct reflected shapes, not a smooth gradient) alpha
    and color arrays via a proper alpha-weighted combine so there's no
    visible seam where the two techniques meet.
    """
    rng = np.random.default_rng(seed)
    width, height = size
    center_x, center_y = width / 2.0, height / 2.0
    light_x, light_y = light_pos

    alpha_canvas = Image.new("L", size, 0)
    alpha_draw = ImageDraw.Draw(alpha_canvas)
    color_canvas = Image.new("RGB", size, (0, 0, 0))
    color_draw = ImageDraw.Draw(color_canvas)

    halo_radius = min(width, height) * 0.09

    # Ghosts: mirror the light's offset from center, at increasing
    # multiples (now jittered - see docstring), shrinking and dimming
    # further from the source.
    dx, dy = center_x - light_x, center_y - light_y
    for i in range(1, ghost_count + 1):
        t = 1.0 + i * 0.55 + rng.uniform(-0.18, 0.18)
        raw_ghost_x = center_x + dx * t
        raw_ghost_y = center_y + dy * t
        size_jitter = rng.uniform(0.72, 1.32)
        ghost_radius = halo_radius * max(0.15, 0.5 - i * 0.1) * size_jitter

        # Clamp into the visible frame (with margin for the ghost's own
        # radius) instead of dropping it outright when the mirrored
        # position lands outside the frame. A light source detected very
        # close to a frame edge (confirmed: a real photo's detected light
        # landing at (x, 0), the very top row) mirrors most or all ghosts
        # far outside the frame - the old bounds check just skipped those,
        # meaning lens flare quietly did nothing for such photos, with no
        # visual acknowledgment that it happened. A real lens flare ghost
        # that would technically fall off-frame still shows a sliver/glow
        # hugging the edge in an actual photo; it doesn't just vanish.
        ghost_x = float(np.clip(raw_ghost_x, ghost_radius * 0.4, width - ghost_radius * 0.4))
        ghost_y = float(np.clip(raw_ghost_y, ghost_radius * 0.4, height - ghost_radius * 0.4))

        # How far the raw (unclamped) position had to be pulled back in -
        # a ghost only slightly outside frame stays essentially full
        # strength, one mirrored WAY off-frame (e.g. an edge-of-frame
        # light source) fades out toward the edge instead of sitting
        # there at full, oddly-static brightness regardless of how far
        # its "true" mirrored position actually is.
        clamp_distance = math.hypot(raw_ghost_x - ghost_x, raw_ghost_y - ghost_y)
        fade = 1.0 / (1.0 + clamp_distance / (halo_radius * 3.0))

        alpha_jitter = rng.uniform(0.65, 1.3)
        ghost_alpha = int(np.clip(160 * (0.6 ** i) * fade * alpha_jitter, 0, 255))
        if ghost_alpha <= 1:
            continue

        ghost_color = _jitter_ghost_color(color, rng)

        # Roughly 40% stay circular, the rest pick up a hexagonal or
        # heptagonal aperture-blade shape - a plausible,
        # not-too-regular mix rather than either "always circles" (the
        # old bug) or "always polygons" (an equally uniform new pattern).
        sides = int(rng.choice([0, 6, 7], p=[0.4, 0.3, 0.3]))
        if sides == 0:
            bbox = [ghost_x - ghost_radius, ghost_y - ghost_radius, ghost_x + ghost_radius, ghost_y + ghost_radius]
            alpha_draw.ellipse(bbox, fill=ghost_alpha)
            color_draw.ellipse(bbox, fill=ghost_color)
        else:
            rotation = float(rng.uniform(0.0, 360.0))
            points = _regular_polygon_points((ghost_x, ghost_y), ghost_radius, sides, rotation)
            alpha_draw.polygon(points, fill=ghost_alpha)
            color_draw.polygon(points, fill=ghost_color)

    blur_radius = max(3, int(halo_radius * 0.35))
    alpha_blurred = alpha_canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    color_blurred = color_canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    alpha_fraction = np.asarray(alpha_blurred, dtype="float32") / 255.0
    color_raw = np.asarray(color_blurred, dtype="float32") / 255.0

    # color_canvas was painted on a BLACK background, so blurring it mixes
    # black into every soft/anti-aliased edge exactly like premultiplied
    # alpha - without unpremultiplying here, every ghost's blurred rim
    # would read as darkening toward black (an obviously wrong color
    # fringe) instead of smoothly fading the correct tint color as alpha
    # drops. Dividing back out by the (unstrengthened) alpha fraction
    # recovers the true per-pixel tint; a small floor on the denominator
    # avoids blowing up fully-transparent pixels, whose color doesn't
    # matter anyway since alpha there is ~0.
    color_array = np.clip(color_raw / np.clip(alpha_fraction, 0.02, None)[:, :, None], 0.0, 1.0)

    # True radial-gradient halo: a continuous falloff
    # computed directly over the pixel grid, brightest at light_pos and
    # smoothly decreasing outward, rather than a disc shape softened only
    # by the shared post-hoc blur above (which is still correct and left
    # untouched for the ghost chain itself).
    ys, xs = np.mgrid[0:height, 0:width]
    halo_dist = np.sqrt((xs - light_x) ** 2 + (ys - light_y) ** 2).astype("float32")
    halo_gamma = 1.7
    halo_gradient = np.clip(1.0 - halo_dist / (halo_radius * 2.2), 0.0, 1.0) ** halo_gamma
    halo_color_norm = np.asarray([np.clip(c, 0, 255) / 255.0 for c in color], dtype="float32")

    # Combine ghost-chain alpha/color with the halo gradient via a proper
    # alpha-weighted blend (a "union" combine for alpha, a
    # coverage-weighted average for color) instead of np.maximum on both -
    # np.maximum on color would hard-cut to whichever layer happens to be
    # brighter at each pixel, producing a visible seam right where the
    # ghost chain's blurred alpha crosses the halo gradient's alpha; the
    # weighted blend below fades between the two continuously instead.
    combined_alpha_unweighted = np.clip(alpha_fraction + halo_gradient - alpha_fraction * halo_gradient, 0.0, 1.0)
    weight_sum = np.clip(alpha_fraction + halo_gradient, 1e-4, None)
    combined_color = (
        color_array * alpha_fraction[:, :, None] + halo_color_norm[None, None, :] * halo_gradient[:, :, None]
    ) / weight_sum[:, :, None]
    color_array = np.clip(combined_color, 0.0, 1.0)

    alpha_array = combined_alpha_unweighted * strength
    return alpha_array, color_array


def build_starburst_texture(size, light_pos, num_spikes=6, spike_length_fraction=0.4, angular_sharpness=48.0,
                             core_radius_fraction=0.012, strength=0.5, seed=None):
    """
    Returns a (height, width) float32 array, 0.0-1.0, of star-shaped
    diffraction spikes radiating from `light_pos` - fixes a
    very recognizable component of "a bright point of light shot through a
    real lens" (star-shaped diffraction spikes, especially prominent on
    streetlights/headlights/sun through branches, caused by light
    diffracting around the aperture blades/sensor structure) was
    previously entirely absent - there was a halo and ghost chain
    (build_lens_flare_texture) but nothing rendering the spike/star
    pattern at the light source's own position. This is a NEW capability,
    not a fix to existing spike code.

    Computed with pure numpy (angular Gaussian x radial falloff) rather
    than PIL line-drawing, since a convincingly tapered, anti-aliased
    spike needs continuous brightness falloff along its whole length, not
    a fixed-width drawn line. `seed` (None by default, matching
    build_lens_flare_texture/build_dirt_texture - this pattern is a
    property of the LENS's aperture, not the scene) jitters each spike's
    angle slightly and the whole pattern's base rotation, so the star
    isn't a perfectly even, identical-every-photo asterisk.
    """
    rng = np.random.default_rng(seed)
    width, height = size
    light_x, light_y = light_pos

    ys, xs = np.mgrid[0:height, 0:width]
    dx = (xs - light_x).astype("float32")
    dy = (ys - light_y).astype("float32")
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    max_len = spike_length_fraction * min(width, height)
    core_radius = max(1.0, core_radius_fraction * min(width, height))

    # Spikes come in opposite pairs through the light source (a real
    # diffraction star's arms are symmetric across the center), evenly
    # spaced by default but each pair gets its own small angle jitter and
    # the whole star gets one random base rotation, per.3's
    # established "not perfectly regular" precedent applied here to a
    # different effect.
    base_rotation = rng.uniform(0.0, 2.0 * math.pi)
    accumulator = np.zeros((height, width), dtype="float32")
    for k in range(num_spikes):
        angle = base_rotation + (math.pi * k / num_spikes) + rng.uniform(-0.035, 0.035)
        length_jitter = rng.uniform(0.75, 1.15)
        spike_max_len = max_len * length_jitter

        # Angular distance from this spike's axis (mod pi, since a spike
        # extends both directions through the center - angle and angle+pi
        # are the same arm).
        angular_diff = np.mod(theta - angle + math.pi / 2.0, math.pi) - math.pi / 2.0
        angular_falloff = np.exp(-(angular_diff ** 2) * angular_sharpness)

        radial_falloff = np.clip(1.0 - r / spike_max_len, 0.0, 1.0) ** 1.6
        accumulator += angular_falloff * radial_falloff

    accumulator = np.clip(accumulator, 0.0, 1.0)

    # A small solid core at the light source itself so the spikes visibly
    # emanate from a bright point rather than converging on empty space.
    core = np.clip(1.0 - r / core_radius, 0.0, 1.0) ** 1.2
    combined = np.clip(np.maximum(accumulator, core), 0.0, 1.0)

    blur_radius = max(1.0, core_radius * 0.3)
    blurred = Image.fromarray((combined * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return (np.asarray(blurred, dtype="float32") / 255.0) * strength


def render_light_overlay(size, num_frames, texture, color=(255, 244, 214), pulse_amount=0.2, pulse_speed=1, start_frame=0):
    """
    Generator - yields num_frames RGBA overlay frames of a precomputed
    texture (see build_ray_texture / build_lens_flare_texture / design notes
    16.2's build_starburst_texture), tinted `color`, with a slow, cyclic
    intensity pulse (real light shafts through moving haze/particles
    aren't perfectly constant - they waver slightly, and a static,
    unmoving overlay for 30 seconds would look flat/fake). The pulse is
    a pure sinusoid of the loop phase, so it closes the loop exactly
    like every other cyclic effect in this pipeline - pulse_speed MUST
    be a whole number for that to hold (a non-integer number of pulse
    cycles would not return to the same value at frame `num_frames` as
    frame 0).

    `texture` is either a single static (height, width) float32 array
    (the original, still-default behavior - a still image, or camera
    drift disabled), OR - an iterable yielding one such
    array per frame (see camera.apply_camera_drift_to_array), so a
    caller can keep the ray-shaft / starburst-spike texture registered
    to the light source's ACTUAL drifted screen position each frame
    instead of it staying glued to the original photo's framing while
    camera drift pans/zooms the base image underneath it - the same
    registration-mismatch class the design notes already fixed for
    vegetation/cloud/particle sampling and segmentation masks (see
    camera.apply_camera_drift_to_array/apply_camera_drift_to_region_rg's
    docstrings). Mirrors scene_warp.render_region_warp's existing
    static-array-or-iterable convention exactly, via the same
    isinstance(..., np.ndarray) check.

    start_frame: skips yielding frames 0..start_frame-1 (see
        render_checkpoint.py). CONTRACT when `texture` is an animated
        iterable (not a static np.ndarray): the caller must already have
        built that iterable so it begins at `start_frame` (e.g. by
        passing the same start_frame through to
        camera.apply_camera_drift_to_array, which - per its own
        start_frame docstring - yields only num_frames-start_frame items,
        already fast-forwarded). This function does NOT (and must not)
        discard `start_frame` items via next() itself - that would waste
        exactly the per-frame GPU/resample work start_frame exists to
        skip - it simply trusts the iterable's first item corresponds to
        frame `start_frame`. A static array is unaffected either way,
        since every frame already reads the same one value. 0 (default)
        = old behavior, identical for both the static and animated case.
    """
    animated_texture = not isinstance(texture, np.ndarray)
    texture_frames_iter = iter(texture) if animated_texture else None
    current_texture = next(texture_frames_iter) if animated_texture else texture

    height, width = current_texture.shape
    rgb = np.empty((height, width, 3), dtype="float32")
    for channel in range(3):
        rgb[:, :, channel] = color[channel]
    rgb_uint8 = rgb.astype("uint8")

    dither_rng = np.random.default_rng(1234)
    for frame_i in range(start_frame, num_frames):
        if animated_texture and frame_i > start_frame:
            current_texture = next(texture_frames_iter)

        theta = 2.0 * math.pi * pulse_speed * (frame_i / max(num_frames, 1))
        pulse = 1.0 + pulse_amount * math.sin(theta)
        alpha = np.clip(current_texture * pulse, 0.0, 1.0)

        rgba = np.empty((height, width, 4), dtype="uint8")
        rgba[:, :, 0:3] = rgb_uint8
        # Dithered quantization - a smooth low-alpha gradient rounded to 8
        # bits otherwise shows up as visible terrace banding.
        dithered = alpha * 255.0 + dither_rng.random((height, width), dtype="float32") - 0.5
        rgba[:, :, 3] = np.clip(dithered + 0.5, 0, 255).astype("uint8")
        yield Image.fromarray(rgba, mode="RGBA")


def render_lens_flare_overlay(size, num_frames, alpha_texture, color_texture, pulse_amount=0.2, pulse_speed=1, start_frame=0):
    """
    Generator - yields num_frames RGBA overlay frames for lens flare
    specifically, using a per-pixel `color_texture` (see
    build_lens_flare_texture's design notes-16.1 color_array return) instead of
    render_light_overlay's single flat tint. render_light_overlay itself
    is left untouched and keeps serving god rays (build_ray_texture),
    which are a single-color light shaft with no per-ghost color variance
    to preserve - only lens flare's ghost chain needed per-pixel color.

    Same sinusoidal-pulse-of-loop-phase convention as render_light_overlay
    (pulse_speed MUST be a whole number so the loop closes exactly).

    `alpha_texture`/`color_texture` each independently accept either a
    single static array (unchanged default behavior) or - a per-frame iterable (alpha_texture: camera.
    apply_camera_drift_to_array; color_texture: camera.
    apply_camera_drift_to_color_array), for the same camera-drift
    registration reason documented on render_light_overlay above - a
    static lens-flare ghost chain would otherwise stay glued to the
    photo's original framing while camera drift moves the base image
    underneath it. Same isinstance(..., np.ndarray) static-or-iterable
    convention as render_light_overlay/scene_warp.render_region_warp.

    start_frame: skips yielding frames 0..start_frame-1 (see
        render_checkpoint.py) - same contract as render_light_overlay's
        own start_frame: when alpha_texture/color_texture are animated
        iterables, the caller must already have built each so its first
        item corresponds to frame `start_frame` (e.g. by passing the same
        start_frame through to camera.apply_camera_drift_to_array /
        apply_camera_drift_to_color_array). This function never discards
        items via next() to "catch up" - it trusts the iterables are
        already positioned correctly, so no GPU/resample work already
        skipped upstream gets wastefully redone or reconsumed here.
    """
    animated_alpha = not isinstance(alpha_texture, np.ndarray)
    alpha_frames_iter = iter(alpha_texture) if animated_alpha else None
    current_alpha = next(alpha_frames_iter) if animated_alpha else alpha_texture

    animated_color = not isinstance(color_texture, np.ndarray)
    color_frames_iter = iter(color_texture) if animated_color else None
    current_color = next(color_frames_iter) if animated_color else color_texture

    height, width = current_alpha.shape

    for frame_i in range(start_frame, num_frames):
        if animated_alpha and frame_i > start_frame:
            current_alpha = next(alpha_frames_iter)
        if animated_color and frame_i > start_frame:
            current_color = next(color_frames_iter)

        rgb = np.clip(current_color, 0.0, 1.0) * 255.0

        theta = 2.0 * math.pi * pulse_speed * (frame_i / max(num_frames, 1))
        pulse = 1.0 + pulse_amount * math.sin(theta)
        alpha = np.clip(current_alpha * pulse, 0.0, 1.0)

        rgba = np.empty((height, width, 4), dtype="uint8")
        rgba[:, :, 0:3] = rgb.astype("uint8")
        rgba[:, :, 3] = (alpha * 255).astype("uint8")
        yield Image.fromarray(rgba, mode="RGBA")
