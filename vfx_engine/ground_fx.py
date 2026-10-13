"""
ground_fx.py

Ground/scene-interaction effects that make rain/snow feel like they're
actually landing on the scene, instead of just falling in front of it:
    - wet darkening + sheen: rain-soaked ground reads darker overall
      (real wet stone/asphalt darkens, it doesn't just get a highlight),
      plus a slow animated specular highlight sweep on top
    - splashes: short-lived expanding/fading filled discs where drops
      "land" (rain)
    - snow accumulation: a soft white build-up on ground-like surfaces
      (snow)

"Ground" starts as a heuristic from the depth map: near + in the lower
part of the frame - good enough to be clearly better than nothing, but
imprecise on its own (a lake sitting at "ground-like" depth/position in
the lower frame, or a splash landing on a raised bench instead of true
flat ground - see build_ground_mask's docstring). When real AI scene
segmentation is available (see vfx_engine/segmentation.py), that
heuristic is additionally GATED by the model's own "is this pixel
actually ground" classification, instead of depth/position guesswork
alone.

Every effect here is built to close an exact loop over `num_frames`
frames, the same way particles.py is - phase (frame_i/num_frames) drives
everything, so nothing needs loop.py's crossfade to hide a seam (see
the design notes Loop generation section). Snow "accumulation" in
particular can't be truly one-directional in a seamless loop (that would
require a jump cut back to empty) - it's a slow build-up-and-settle
breathing cycle instead, see render_snow_accumulation's docstring.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import config
from vfx_engine.depth import _box_filter
from vfx_engine.particles import _value_noise, sample_ambient_tint, sample_tint_grid_bilinear


def _resolve_snow_accumulation_cycles(num_frames):
    """
    config.SNOW_ACCUMULATION_SECONDS_PER_CYCLE is a RATE
    (seconds per build-up-and-settle breath), not a fixed "exactly one
    cycle over the entire render" - converts it into a whole cycle count
    (whole numbers only, so render_snow_accumulation's cosine term still
    closes the loop exactly) sized to THIS render's actual duration
    (num_frames/config.FPS), so a long render gets proportionally more,
    naturally-paced breaths instead of one lone breath stretched across
    the whole thing. max(1, ...) keeps a very short render from
    collapsing to 0 whole cycles.
    """
    duration_seconds = max(num_frames, 1) / config.FPS
    return max(1, round(duration_seconds / config.SNOW_ACCUMULATION_SECONDS_PER_CYCLE))


def build_roughness_field(size, scale=0.02, seed=None):
    """
    Returns a (height, width) float32 array, 0.0-1.0 - a static per-pixel
    surface roughness/puddling proxy, computed once per photo at setup
    (reuses particles.py's value-noise building block rather than
    duplicating that hash-noise formula a fourth time in this codebase).

    Shared by render_wet_sheen/render_wet_darkening/render_snow_accumulation
 so all three read as variation in the SAME
    physical surface instead of each inventing its own unrelated pattern -
    real specular response, wetness darkening, and snow settling all
    depend on the same underlying surface micro-geometry (puddles,
    texture, unevenness), not three independent phenomena.
    """
    width, height = size
    rng = np.random.default_rng(seed)
    # A random offset per call (see particles.py's density_clumping for
    # the identical reasoning) - without it, every photo at a given
    # resolution would get the exact same noise pattern in the exact same
    # screen location.
    offset = rng.uniform(0.0, 5000.0, 2)
    y_coords, x_coords = np.mgrid[0:height, 0:width].astype("float32")
    coords = np.stack([x_coords, y_coords], axis=-1) * scale + offset
    # Two octaves (not one) for a less "single blobby pattern" look, still
    # cheap and still no new textures/dependencies.
    noise = 0.65 * _value_noise(coords) + 0.35 * _value_noise(coords * 2.3 + 91.0)
    return np.clip(noise, 0.0, 1.0)


def build_normal_coherence_mask(image, radius=8):
    """
    returns a (height, width) float32 array, 0.0-1.0 - how
    LOCALLY CONSISTENT the scene's surface normals are around each pixel,
    via vfx_engine.surface_normals' dormant Marigold-normals estimate
    (config.ENABLE_SURFACE_NORMALS - see that module's own evaluation-
    gated framing, mirroring illumination.py's - the design notes). A
    genuinely flat surface (a calm puddle, a level patio, a still lake)
    has normals that agree closely with their neighbors across a small
    window; a merely dark/low-albedo but textured surface (grass, gravel,
    foliage in shadow - today's build_ground_mask false-positive case)
    has normals that disagree from pixel to pixel even though
    depth/position alone can't tell the two apart.

    Computed as the MAGNITUDE of the locally-averaged normal vector (each
    of the three normal components box-filtered independently over a
    `radius`-pixel window, then recombined into a vector and its length
    taken): unit vectors that all point the same direction average back
    to a vector of length ~1.0, while unit vectors pointing in
    inconsistent directions partially cancel and average to something
    shorter - close to 1.0 where the neighborhood agrees/aligns, closer
    to 0 where it's locally inconsistent/noisy. Reuses depth.py's own
    _box_filter (the exact fast integral-image box-mean technique that
    module's guided-filter refinement already implements for a similar
    windowed-average need) rather than reimplementing an equivalent
    routine a second time in this codebase.

    Returns None when config.ENABLE_SURFACE_NORMALS is off, or `image`
    is None - callers should fall back to their plain, unmultiplied
    heuristic in that case, the same dormant-feature gating pattern used
    everywhere else in this codebase.
    """
    if not config.ENABLE_SURFACE_NORMALS or image is None:
        return None

    from vfx_engine import surface_normals
    normals = surface_normals.estimate_surface_normals(image)

    avg_x = _box_filter(normals[:, :, 0], radius)
    avg_y = _box_filter(normals[:, :, 1], radius)
    avg_z = _box_filter(normals[:, :, 2], radius)
    coherence = np.sqrt(avg_x ** 2 + avg_y ** 2 + avg_z ** 2)
    return np.clip(coherence, 0.0, 1.0).astype("float32")


def build_ground_mask(depth_array, segmentation=None, include_soft_ground=False, feather=25, near_percentile=93, image=None):
    """
    Returns a (height, width) float32 array, 0.0-1.0, marking pixels likely
    to be a ground/floor surface: near (high depth_array value) AND in the
    lower part of the frame. Not real surface-normal detection on its own -
    a cheap heuristic that's right often enough to be worth using by
    itself, and meaningfully sharpened when `segmentation` is available
    (see below), and further when `image` is given and surface normals are
    available (see below).

    near_percentile is deliberately strict (default 93rd percentile of
    depth*vertical_weight): a looser threshold catches mid-distance
    surfaces too (e.g. a lake sitting in the lower half of the frame,
    which is genuinely FAR despite its position) and washes out much more
    of the frame than just the actual near foreground floor.

    segmentation: optional dict from vfx_engine.segmentation.estimate_segmentation
        - when given, the depth/position heuristic above is additionally
        GATED (multiplied) by the model's own real "is this pixel actually
        ground" classification (segmentation["ground_hard"] - paved/bare
        surfaces: floor, road, sidewalk, path, earth, sand), instead of
        relying on depth/position guesswork alone. This is what actually
        fixes the two failure cases named above: a lake wrongly scoring
        "ground-like" by depth/position gets excluded because the model
        also knows it's classified as water, not ground; a raised bench
        sitting in the "near + lower frame" zone gets excluded because the
        model doesn't classify it as ground either. Falls back to the pure
        heuristic (unchanged) when segmentation is None (model unavailable/
        disabled - see config.SEGMENTATION_ENABLED) or the model found NO
        ground-like pixels at all in the photo (gating against an all-zero
        mask would silently zero out every ground effect in the whole
        photo, a worse result than the imperfect heuristic alone).
    include_soft_ground: when segmentation is given, also count
        segmentation["ground_soft"] (grass, field) as valid ground, not
        just segmentation["ground_hard"] - grass/field is real settle-able
        terrain (snow accumulation, see render_snow_accumulation, wants
        this), but isn't a hard reflective surface a "wet sheen" highlight
        makes sense on (rain's render_wet_darkening/render_wet_sheen/
        render_splashes want this False - see main.py's render_ground_effects,
        which calls this function once per surface type it needs).
    image: optional original photo (PIL Image) - the design notes. When given
        AND config.ENABLE_SURFACE_NORMALS is on, the depth/position (and
        optional segmentation) heuristic above is additionally multiplied
        by build_normal_coherence_mask's per-pixel normal-coherence score,
        so a merely low-albedo/dark but locally INCONSISTENT surface
        (grass, gravel, foliage in shadow) gets excluded even where
        segmentation is unavailable or too coarse to catch it, while a
        genuinely flat dark surface (a puddle in shade, whose normals
        stay locally coherent despite being dark) still passes. None
        (default), or ENABLE_SURFACE_NORMALS off, keeps the old behavior
        unchanged.
    """
    height, width = depth_array.shape
    rows = np.arange(height, dtype="float32").reshape(-1, 1) / max(height - 1, 1)
    vertical_weight = np.clip((rows - 0.35) / 0.65, 0.0, 1.0)   # only the lower ~65% of the frame counts at all
    raw = depth_array * vertical_weight

    threshold = np.percentile(raw, near_percentile)
    spread = max(1e-6, raw.max() - threshold)
    ground = np.clip((raw - threshold) / spread, 0.0, 1.0)
    ground = ground ** 1.5   # push mid-strength values down further, keep only the clearest hits strong

    if segmentation is not None:
        surface_mask = segmentation.get("ground_hard")
        if surface_mask is not None and include_soft_ground and "ground_soft" in segmentation:
            surface_mask = np.clip(surface_mask + segmentation["ground_soft"], 0.0, 1.0)
        if surface_mask is not None and surface_mask.max() > 0.05:
            ground = ground * surface_mask

    coherence = build_normal_coherence_mask(image)
    if coherence is not None:
        ground = ground * coherence

    if feather > 0:
        ground_image = Image.fromarray((ground * 255).astype("uint8"), mode="L")
        ground_image = ground_image.filter(ImageFilter.GaussianBlur(feather))
        ground = np.asarray(ground_image, dtype="float32") / 255.0

    return ground


def render_wet_darkening(size, num_frames, ground_mask, strength=0.4, roughness=None, start_frame=0):
    """
    Generator - darkens the ground mask's area, like rain-soaked stone or
    asphalt. This is the effect that actually reads as "the ground is
    wet" at a glance - a moving specular highlight alone (render_wet_sheen)
    is a nice detail on TOP of this, but by itself is too subtle to notice
    next to falling rain; a real darkening is what a viewer's eye catches
    first. Nearly static (a very slight breathing pulse) rather than fully
    static, so it doesn't look like a flat baked-in mask.

    roughness: optional (height, width) float32 array, 0.0-1.0 (see
        build_roughness_field) - modulates darkening strength so puddling
        reads as uneven (darker in low points, lighter on raised/textured
        areas - the design notes) instead of one flat gray wash applied
        identically everywhere. None = the old uniform behavior.
    start_frame: skips yielding frames 0..start_frame-1 (see
        render_checkpoint.py) - both setup arrays above (dark_rgba,
        roughness_factor) are cheap, content-derived, and computed
        unconditionally regardless of this value, and each frame's alpha
        below is a pure function of frame_i/num_frames plus those fixed
        arrays, so resuming at start_frame reproduces the exact same
        frames an uninterrupted run from 0 would have produced from that
        point on. 0 (default) = old behavior.
    """
    width, height = size
    dark_rgba = np.zeros((height, width, 4), dtype="uint8")   # black - alpha-compositing this darkens, unlike a white overlay
    roughness_factor = (0.6 + 0.8 * roughness) if roughness is not None else 1.0

    for frame_i in range(start_frame, num_frames):
        theta = 2.0 * np.pi * (frame_i / max(num_frames, 1))
        pulse = 0.92 + 0.08 * np.sin(theta * 2.0)   # subtle breathing, not perfectly flat
        alpha = np.clip(ground_mask * strength * pulse * roughness_factor, 0.0, 1.0)

        rgba = dark_rgba.copy()
        rgba[:, :, 3] = (alpha * 255).astype("uint8")
        yield Image.fromarray(rgba, mode="RGBA")


def render_wet_sheen(size, num_frames, ground_mask, strength=0.4, band_count=3.0, angle_deg=25.0,
                      roughness=None, highlight_color=(255, 255, 255),
                      normals=None, light_dir=None, material_category="ground_hard", illumination_albedo=None,
                      start_frame=0):
    """
    Generator - a slow, animated diagonal specular highlight sweep over
    the ground mask, like light catching a wet surface. Purely a function
    of position and loop phase (theta), so it closes the loop exactly:
    the traveling-wave term is `sin(k . pos - band_count*theta)`, and
    theta already runs exactly 0..2pi over the loop, so frame `num_frames`
    reproduces frame 0's wave pattern bit-for-bit - but only if
    `band_count` is a whole number (same requirement as god_rays.
    render_light_overlay's `pulse_speed`); a non-integer value would not
    return to the same phase at frame `num_frames` as frame 0, so it is
    rounded to the nearest whole number below.

    roughness: optional (height, width) float32 array, 0.0-1.0 (see
        build_roughness_field) - modulates the traveling wave's amplitude
        so the highlight only "catches" in patches (puddles, texture)
        instead of one clean band sweeping the whole mask uniformly
. A contrast curve is applied on top so roughness
        reads as distinct catch/no-catch patches, not just a dimmer
        version of the same uniform sweep. None = the old uniform
        behavior.
    highlight_color: (r, g, b) 0-255 - defaults to flat white. Pass the
        scene's actual detected light color (see god_rays.sample_light_color,
        itself now the design notes ambient-blended) to tint the sheen toward
        it - a real wet-surface reflection picks up the
        color of whatever light it's actually reflecting (warm sodium
        streetlamps produce an orange/yellow sheen, not a neutral white
        one). Also doubles as materials.shade_surface's `light_color`
        when the BRDF path below is active.
    normals, light_dir, material_category, illumination_albedo: when `normals` (see surface_normals.estimate_surface_normals)
        and `light_dir` (see materials.resolve_pseudo_light_direction) are
        both given, this function's flat `highlight_color` fill is
        replaced by vfx_engine.materials.shade_surface's real per-pixel
        Fresnel+roughness BRDF output (materials.MATERIAL_TABLE[
        material_category], with illumination_albedo optionally
        overriding its flat albedo tint per-pixel - the design notes(a)) -
        the shared shading function's first real consumer, replacing the
        old hardcoded uniform tint with a spatially-varying, physically-
        motivated one (surfaces facing the light shine more; the color
        itself now has real per-pixel albedo/Fresnel variation instead of
        being one flat RGB triple stamped everywhere the traveling wave
        happens to be bright). The traveling wave computed below still
        supplies 100% of the TEMPORAL animation (the BRDF term is static
        per photo, like the normals/light position it's derived from);
        only the highlight's spatial COLOR distribution changes.
        Any of the three being None keeps the old flat-highlight_color
        behavior exactly.
    start_frame: skips yielding frames 0..start_frame-1 (see
        render_checkpoint.py) - every setup array above (projected, k,
        patch_factor, per_pixel_color/the one-time shade_surface call)
        still runs unconditionally regardless of this value, and each
        frame's traveling-wave alpha below is a pure function of its own
        frame_i/num_frames plus that fixed setup, so resuming at
        start_frame reproduces the exact same wave phase (and therefore
        bit-identical frames) an uninterrupted run from frame 0 would
        have produced from that point on. 0 (default) = old behavior.
    """
    width, height = size
    rounded_band_count = round(band_count)
    if rounded_band_count != band_count:
        print(
            f"  (ground_fx.render_wet_sheen: band_count={band_count} is not a "
            f"whole number; rounding to {rounded_band_count} so the seamless "
            f"loop closes exactly)"
        )
    band_count = float(rounded_band_count)
    y_coords, x_coords = np.mgrid[0:height, 0:width].astype("float32")
    angle = np.radians(angle_deg)
    projected = x_coords * np.cos(angle) + y_coords * np.sin(angle)
    k = 2.0 * np.pi * band_count / max(width, height)

    if roughness is not None:
        patch = np.clip((roughness - 0.35) / 0.4, 0.0, 1.0) ** 1.5
        patch_factor = 0.15 + 0.85 * patch
    else:
        patch_factor = 1.0

    # static per-photo per-pixel highlight color, computed
    # ONCE up front (like `roughness`/`patch_factor` above) rather than
    # per-frame - normals/light_dir don't change frame to frame, so
    # re-shading every frame would be pure waste, same reasoning
    # god_rays.py's module docstring gives for its own static textures.
    per_pixel_color = None
    if normals is not None and light_dir is not None:
        from vfx_engine import materials
        shaded = materials.shade_surface(
            normals, light_dir, material_category, highlight_color,
            illumination_albedo=illumination_albedo,
        )
        peak = float(np.percentile(shaded, 99.5))
        if peak > 1e-6:
            per_pixel_color = np.clip(shaded / peak, 0.0, 1.0)

    for frame_i in range(start_frame, num_frames):
        theta = 2.0 * np.pi * (frame_i / max(num_frames, 1))
        wave = 0.5 + 0.5 * np.sin(k * projected - band_count * theta)
        wave = wave ** 3   # narrower, more highlight-like peaks instead of a broad soft sine
        wave = wave * patch_factor

        alpha = np.clip(wave * ground_mask * strength, 0.0, 1.0)
        rgba = np.zeros((height, width, 4), dtype="uint8")
        if per_pixel_color is not None:
            rgba[:, :, 0:3] = (per_pixel_color * 255).astype("uint8")
        else:
            rgba[:, :, 0:3] = highlight_color
        rgba[:, :, 3] = (alpha * 255).astype("uint8")
        yield Image.fromarray(rgba, mode="RGBA")


_MAX_CROWN_SPIKES = 14   # upper bound on separated spikes/jets per splash's crown - see render_splashes


def _apply_splash_refraction(canvas, base_array, cx, cy, radius, fade, light_pos_uv, refraction_strength, width, height, fill_color, outer_scale=1.05, peak_alpha=235.0):
    """
    Warps a small patch of `base_array` (the actual photo, HxWx3 float32)
    outward from (cx, cy) like a lens and composites it into `canvas`
    (a shared HxWx4 float32 RGBA accumulator, modified in place) - the
    splash's own crown body, instead of a flat-filled polygon.

    `fill_color` blends in as a small baseline brightness (see
    `baseline_mix` below) so the crown body stays visibly "wet" even when
    the warped photo patch underneath is itself near-black (a splash over
    deep shadow/dark ground) - without it, the refracted body can
    disappear almost entirely, leaving only the bright rim outline
    visible and making the splash read as a hollow wireframe ring instead
    of a filled, glinting droplet crown.

    Directly reuses particles_rain.glsl's raindrop-as-lens idea: a real
    water surface bends whatever's behind/under it, a solid-color shape
    drawn on top never can, no matter how well the timing/silhouette is
    tuned. `(lx, ly)` here is the same role as the shader's `disc` - the
    splash's local -1..1 coordinate, standing in for a surface normal.

    Also applies the same dual-tone (warm toward the light, cool away
    from it) rim tint the rain shader uses, driven by the same
    `light_pos_uv` the rest of the render already resolves once per photo
    (see main.py's resolve_light_pos_uv) - so a splash's highlight faces
    the scene's actual detected light source instead of every splash
    being lit identically regardless of direction.

    Nearest-neighbor sampling (not bilinear) - a few-pixel warp on a
    ~50px patch doesn't need sub-pixel interpolation to read as "water",
    and staying nearest-neighbor keeps this cheap enough to run per
    splash per frame in plain numpy.

    `outer_scale` sets how far past `radius` the lit water disc extends
    (as a multiple of `radius`) - the caller passes the current frame's
    actual bumpy crown-polygon extent so the rim's outermost bumps always sit on top of lit water,
    never past its edge onto bare, unlit background.

    `peak_alpha` (0-255) is the fully-opaque-center alpha this disc tops
    out at - the design notes reuses this same function at a smaller radius
    and lower peak_alpha for the small "wet base" pooling disc at a
    splash's impact point (sitting under the separated spikes), not just
    the old single whole-crown disc, so it needed to stop being hardcoded.

    `base_array=None` (no photo to warp) falls back to a flat wash of
    `fill_color` with the same soft radial edge falloff, instead of
    requiring a caller-side branch - the design notes calls this
    unconditionally for the wet base disc regardless of whether
    base_image was given, same as the per-spike patches.
    """
    if radius < 1.0 or fade <= 0.0:
        return
    r = int(np.ceil(radius * outer_scale)) + 2
    x0, x1 = max(0, int(cx) - r), min(width, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(height, int(cy) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    lx = (xx - cx) / radius
    ly = (yy - cy) / radius
    dist = np.sqrt(lx * lx + ly * ly)
    mask = dist <= outer_scale
    if not np.any(mask):
        return

    fill_arr = np.array(fill_color, dtype=np.float32)
    if base_array is not None:
        disp = refraction_strength * radius * 0.9
        src_x = np.clip(xx + lx * disp, 0, width - 1).astype(np.int32)
        src_y = np.clip(yy + ly * disp, 0, height - 1).astype(np.int32)
        sampled = base_array[src_y, src_x]

        light_dx = light_pos_uv[0] * width - cx
        light_dy = light_pos_uv[1] * height - cy
        light_len = max(1e-4, float(np.hypot(light_dx, light_dy)))
        to_light_x, to_light_y = light_dx / light_len, light_dy / light_len
        light_facing = np.clip((lx * to_light_x + ly * to_light_y) * 1.4, -1.0, 1.0)
        warm = np.array([1.10, 1.04, 0.92], dtype=np.float32)
        cool = np.array([0.88, 0.93, 1.08], dtype=np.float32)
        tint = cool[None, None, :] + (warm - cool)[None, None, :] * ((light_facing[..., None] + 1.0) * 0.5)
        warped = sampled * tint
        # Additive, not a blend toward fill_color - a splash over near-black
        # ground/shadow needs actual brightening (a real water surface is a
        # specular highlight catching whatever light is available, brighter
        # than the dark ground under it), not a weighted average that's still
        # mostly the same near-black value. Scaled by `fade` so it dies down
        # in lockstep with the rest of the splash instead of leaving a fixed
        # glow after everything else has faded.
        baseline_boost = 0.4 * fade
        shaded = np.clip(warped + fill_arr[None, None, :] * baseline_boost, 0.0, 255.0)
    else:
        shaded = np.broadcast_to(fill_arr, xx.shape + (3,)).copy()

    edge_alpha = np.clip((outer_scale - dist) * 3.0, 0.0, 1.0) * fade * peak_alpha
    new_alpha = np.where(mask, edge_alpha, 0.0)[..., None]

    region = canvas[y0:y1, x0:x1]
    existing_alpha = region[..., 3:4]
    out_alpha = new_alpha + existing_alpha * (1.0 - new_alpha / 255.0)
    safe_out_alpha = np.maximum(out_alpha, 1e-4)
    out_rgb = (shaded * new_alpha + region[..., :3] * existing_alpha * (1.0 - new_alpha / 255.0)) / safe_out_alpha
    region[..., :3] = out_rgb
    region[..., 3:4] = out_alpha


def _apply_spike_patch(canvas, base_array, cx, cy, angle, base_r, tip_r, half_width, fade,
                        light_pos_uv, refraction_strength, width, height, fill_color, rim_color,
                        peak_alpha=200.0, rim_boost=1.6):
    """
    draws ONE spike/jet of a splash's crown as an
    independent, feathered, tapering shape radiating out from (cx, cy)
    at `angle`, from `base_r` to `tip_r` with `half_width` at its base
    narrowing toward a point at its tip - composited into `canvas` (the
    same shared float32 RGBA accumulator `_apply_splash_refraction`
    uses). Calling this once PER SPIKE, each with its own angle/height/
    width, instead of filling one polygon through every crown point, is
    what actually produces visible GAPS of background between spikes -
    a real splash coronet is a ring of separated jets, not one smooth
    filled silhouette no matter how bumpy its outline is perturbed.

    Uses the exact same warp-the-photo-like-a-lens technique as
    _apply_splash_refraction (nearest-neighbor sample displaced outward
    along the local surface, same warm/cool light_pos_uv rim tint) when
    `base_array` is given, so a spike's body is real refracted photo
    like the rest of the crown - falls back to a flat `fill_color` wash
    when there's no photo to warp (base_image=None), but even the
    flat-fill fallback keeps the same analytic feathered taper instead
    of a hard-edged polygon fill, per the item's explicit "feather the
    crown/spike edges" note.

    All feathering here is analytic (a numpy alpha field computed
    directly from each pixel's position along/across the spike's own
    local axis), not a separate blur pass - cheaper, and it lets a soft
    highlight be folded straight into the shaded color near the spike's
    tip (where a real raised jet of water actually catches the most
    light) instead of the old single continuous stroked-outline rim,
    which read as a hard geometric ring no matter how much it was
    blurred afterward.
    """
    if tip_r <= base_r + 0.5 or half_width < 0.3 or fade <= 0.0:
        return
    dirx, diry = float(np.cos(angle)), float(np.sin(angle))
    perpx, perpy = -diry, dirx

    # Tight rotated-rectangle bounding box (base_r..tip_r along the
    # spike axis, +/- half_width across it) instead of a full square of
    # side 2*tip_r - with up to ~14 spikes per splash this keeps the
    # per-spike cost close to what one whole-crown disc used to cost.
    margin = 2.0
    along_lo, along_hi = base_r - margin, tip_r + margin
    across_lo, across_hi = -half_width - margin, half_width + margin
    xs_corners = [cx + a * dirx + b * perpx for a in (along_lo, along_hi) for b in (across_lo, across_hi)]
    ys_corners = [cy + a * diry + b * perpy for a in (along_lo, along_hi) for b in (across_lo, across_hi)]
    x0, x1 = int(max(0, min(xs_corners))), int(min(width, max(xs_corners) + 1))
    y0, y1 = int(max(0, min(ys_corners))), int(min(height, max(ys_corners) + 1))
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    rx, ry = xx - cx, yy - cy
    along = rx * dirx + ry * diry
    across = rx * perpx + ry * perpy

    span = max(tip_r - base_r, 1e-3)
    t_len = (along - base_r) / span
    # Wedge/spike taper: full half_width right at the base, narrowing to
    # a point at the tip - with a small mid-length "belly" bulge so each
    # spike reads as a teardrop/thorn rather than a perfectly straight
    # wedge, matching a real jet's slightly bulged profile.
    width_profile = (
        half_width
        * np.clip(1.0 - t_len, 0.0, 1.0) ** 1.15
        * (1.0 + 0.15 * np.sin(np.clip(t_len, 0.0, 1.0) * np.pi))
    )
    width_ratio = np.abs(across) / np.maximum(width_profile, 1e-3)
    cross_falloff = np.clip(1.0 - width_ratio, 0.0, 1.0)
    # Fades in fast from the base (blends into the wet base disc) and
    # fades out toward the tip (a real point is infinitely thin, so
    # taper its opacity too rather than leaving a hard-alpha point).
    length_falloff = np.clip(np.minimum(t_len * 5.0, (1.0 - t_len) * 2.2), 0.0, 1.0)
    alpha_field = (cross_falloff ** 1.3) * length_falloff
    mask = (t_len > -0.05) & (t_len < 1.05) & (width_ratio < 1.15)
    if not np.any(mask):
        return

    fill_arr = np.array(fill_color, dtype=np.float32)
    lx, ly = rx / max(tip_r, 1e-3), ry / max(tip_r, 1e-3)
    if base_array is not None and refraction_strength > 0.0:
        disp = refraction_strength * tip_r * 0.9
        src_x = np.clip(xx + lx * disp, 0, width - 1).astype(np.int32)
        src_y = np.clip(yy + ly * disp, 0, height - 1).astype(np.int32)
        sampled = base_array[src_y, src_x]

        light_dx = light_pos_uv[0] * width - cx
        light_dy = light_pos_uv[1] * height - cy
        light_len = max(1e-4, float(np.hypot(light_dx, light_dy)))
        to_light_x, to_light_y = light_dx / light_len, light_dy / light_len
        light_facing = np.clip((lx * to_light_x + ly * to_light_y) * 1.4, -1.0, 1.0)
        warm = np.array([1.10, 1.04, 0.92], dtype=np.float32)
        cool = np.array([0.88, 0.93, 1.08], dtype=np.float32)
        tint = cool[None, None, :] + (warm - cool)[None, None, :] * ((light_facing[..., None] + 1.0) * 0.5)
        shaded = np.clip(sampled * tint + fill_arr[None, None, :] * 0.4 * fade, 0.0, 255.0)
    else:
        shaded = np.broadcast_to(fill_arr, xx.shape + (3,)).copy()

    # A raised jet catches the most light right near its tip, on the
    # side facing outward/up - a bright highlight baked directly into
    # the shaded color instead of a separate stroked outline drawn on
    # top, so it can never read as a hard geometric ring.
    tip_highlight = np.clip((t_len - 0.45) / 0.5, 0.0, 1.0) * (cross_falloff ** 2.0)
    rim_arr = np.array(rim_color, dtype=np.float32)
    shaded = np.clip(shaded + rim_arr[None, None, :] * (tip_highlight[..., None] * rim_boost * fade), 0.0, 255.0)

    new_alpha = np.where(mask, alpha_field * fade * peak_alpha, 0.0)[..., None]
    region = canvas[y0:y1, x0:x1]
    existing_alpha = region[..., 3:4]
    out_alpha = new_alpha + existing_alpha * (1.0 - new_alpha / 255.0)
    safe_out_alpha = np.maximum(out_alpha, 1e-4)
    out_rgb = (shaded * new_alpha + region[..., :3] * existing_alpha * (1.0 - new_alpha / 255.0)) / safe_out_alpha
    region[..., :3] = out_rgb
    region[..., 3:4] = out_alpha


def _draw_soft_droplet(frame, cx, cy, size, color, alpha, light_pos_uv, width, height):
    """
    the design notes point 2b - a small radial-gradient blob (a bright
    off-center specular highlight fading to a soft-edged base color, no
    hard outline) instead of one flat-filled circle - the design notes calls
    this out as "the cheapest place to test" a soft radial-gradient +
    rim-highlight fill before applying the same idea to the much larger,
    costlier crown spikes themselves (see _apply_spike_patch).

    The highlight is offset toward the scene's detected light direction
    (`light_pos_uv`) rather than sitting dead-center, so a droplet's
    brightest point is where it's actually catching the light, not its
    geometric middle - same convention _apply_splash_refraction and
    _apply_spike_patch already use for the crown's own rim tint.
    """
    if size < 0.4 or alpha <= 0:
        return
    r = int(np.ceil(size * 1.3)) + 1
    x0, x1 = int(max(0, cx - r)), int(min(width, cx + r + 1))
    y0, y1 = int(max(0, cy - r)), int(min(height, cy + r + 1))
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    lx, ly = (xx - cx) / size, (yy - cy) / size
    dist = np.sqrt(lx * lx + ly * ly)
    base_falloff = np.clip((1.15 - dist) * 2.2, 0.0, 1.0) ** 1.4

    light_dx = light_pos_uv[0] * width - cx
    light_dy = light_pos_uv[1] * height - cy
    light_len = max(1e-4, float(np.hypot(light_dx, light_dy)))
    hx, hy = (light_dx / light_len) * 0.35, (light_dy / light_len) * 0.35
    hl_dist = np.sqrt((lx - hx) ** 2 + (ly - hy) ** 2)
    highlight = np.clip(1.0 - hl_dist / 0.55, 0.0, 1.0) ** 1.8

    color_arr = np.array(color, dtype=np.float32)
    white = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    rgb = np.clip(color_arr[None, None, :] + (white - color_arr)[None, None, :] * (highlight[..., None] * 0.75), 0.0, 255.0)

    out_alpha = np.clip(base_falloff * (alpha / 255.0) * 255.0, 0.0, 255.0)
    patch = np.dstack([rgb, out_alpha]).astype(np.uint8)
    frame.alpha_composite(Image.fromarray(patch, mode="RGBA"), (x0, y0))


def _sample_spawn_points_with_spacing(rng, flat_probs, mask_shape, count, min_spacing_px, retry_cap=20):
    """
    rejection-sampled spawn points instead of plain
    `rng.choice(..., p=flat_probs)`. Plain weighted sampling draws WITH
    replacement from every ground pixel independently, so nothing stops
    two (or several) splashes landing on top of or right next to each
    other - a real wet surface gets impacts scattered roughly evenly, not
    clumped into one unnaturally bright/dense spot while the rest of the
    ground mask sits empty.

    For each of `count` points, draws a weighted candidate pixel and
    rejects it (redrawing) if it's closer than `min_spacing_px` to any
    already-accepted point THIS batch, up to `retry_cap` attempts - after
    which it accepts the last candidate anyway, so a dense ground mask
    with too little room for full spacing can't loop forever. Cheap:
    splash counts are tens, not thousands (a few thousand rng.choice calls
    worst case), per the design notes item's own note.
    """
    xs = np.empty(count, dtype=np.float64)
    ys = np.empty(count, dtype=np.float64)
    min_spacing_sq = min_spacing_px * min_spacing_px

    for i in range(count):
        cx = cy = 0.0
        for _attempt in range(retry_cap):
            idx = rng.choice(flat_probs.size, p=flat_probs)
            cy, cx = np.unravel_index(idx, mask_shape)
            cx, cy = float(cx), float(cy)
            if i == 0:
                break
            dx = xs[:i] - cx
            dy = ys[:i] - cy
            if np.all(dx * dx + dy * dy >= min_spacing_sq):
                break
        xs[i] = cx
        ys[i] = cy

    return xs, ys


def render_splashes(size, num_frames, ground_mask, count=90, max_radius=26, duration_fraction=0.1,
                     seed=None, base_image=None, light_color=None, light_pos_uv=None, refraction_strength=0.4,
                     ambient_tint_grid=None, min_spacing_factor=0.9, start_frame=0):
    """
    Generator - short-lived expanding, fading splashes scattered over
    ground_mask, approximating rain impact splashes. Each splash's timing
    is a fixed phase offset (0..1 of the loop) sampled once at setup, so
    the same set of splashes repeats identically every loop - genuinely
    periodic, not just crossfaded at the seam.

    Addresses the confirmed "amateur" tells in the original version
, all fixed together since they all touch the same
    per-splash setup/per-frame draw loop:
      - ease-out growth (real splash crowns expand fast and decelerate
        under drag) instead of the old constant-speed linear growth,
        one of the most recognizable "CG tell" curves there is.
      - an irregular, multi-point crown outline (a closed loop of
        _CROWN_POINTS points, each at a per-splash-and-per-point random
        radius, generated once at setup) instead of a perfectly circular
        ring - a real splash crown has distinct raised points, not smooth
        radial symmetry.
      - 2-4 tiny secondary droplets per splash that separate from the
        crown, arc briefly, and fall back - real splashes fling a few
        droplets outward and up; the old version had nothing but the
        one ring.
      - one bright, fast-decaying impact flash right at t~0 - the
        instant of impact is genuinely brighter than the crown's own
        subsequent glint.
      - splash/droplet/flash color sampled from the photo's own ambient
        light (sample_ambient_tint, the same helper particles.py already
        uses for rain/snow color) instead of a fixed constant, so
        splashes don't visibly mismatch the scene's actual color grade.
      - the crown's own body is real refracted/tinted photo, not a flat
        fill - see _apply_splash_refraction. A splash
        used to be indistinguishable from a colored disc; a real water
        surface bends what's behind it and catches directional light,
        which a flat fill can't do no matter how well-tuned the timing
        and silhouette are.

    base_image: the original photo (PIL Image) or None - see
        vfx_engine.particles.sample_ambient_tint. Also used as the source
        for the refraction warp (_apply_splash_refraction) - None falls
        back to the old flat-fill polygon (no photo to warp from).
    light_color: (r, g, b) 0-255 or None - the scene's actual detected
        light color (see god_rays.sample_light_color), preferred over
        base_image's whole-photo ambient tint when given:
        a splash highlight reflects the specific light it's catching, not
        the photo's overall average color. Falls back to
        sample_ambient_tint(base_image) when None.
    light_pos_uv: normalized (x, y) 0.0-1.0 screen position of the
        photo's detected dominant light source (see main.py's
        resolve_light_pos_uv / god_rays.detect_light_source) - drives
        the refracted crown's warm/cool rim tint, same convention as
        particles_rain.glsl's light_pos_uv. (0.5, 0.0) (top-center) if
        None.
    refraction_strength: 0.0 disables the water-refraction crown (falls
        back to the flat-fill polygon even when base_image is given).
        >0.0 controls how strongly the crown bends the photo underneath
        it - see _apply_splash_refraction.
    ambient_tint_grid: optional (grid_h, grid_w, 3) uint8
        coarse per-photo color grid (see
        vfx_engine.particles.build_ambient_tint_grid). When given, EACH
        splash samples its own fill/rim/droplet/flash color from its own
        spawn position via sample_tint_grid_bilinear, instead of every
        splash in the scene sharing one flat ambient_tint/light_color -
        a splash landing in a bright puddle and one landing in a shadowed
        patch shouldn't glint identically. Callers should pass the SAME
        grid used for render_rain's ambient_tint_grid, computed once per
        photo, so rain and splashes agree on regional color. None
        (default) keeps the old behavior: light_color, or failing that
        one whole-photo sample_ambient_tint() call, used for every splash.
    min_spacing_factor: minimum distance enforced between
        any two splashes' spawn points, as a multiple of `max_radius`
        (e.g. the default 0.9 means splashes must spawn at least 0.9 *
        max_radius pixels apart). Prevents the old plain-weighted-sampling
        behavior where two splashes could land on top of each other and
        read as one unnaturally bright/dense cluster while the rest of the
        ground mask stayed empty - see _sample_spawn_points_with_spacing.
        0 disables spacing entirely (old behavior).
    start_frame: skips yielding frames 0..start_frame-1 (see
        render_checkpoint.py) - every one-time per-splash setup array
        below (spawn positions, phase_offsets, spike/droplet geometry,
        colors, etc., all drawn from `rng` seeded once from `seed`) still
        runs unconditionally regardless of this value, and each frame's
        drawing below is a pure function of its own frame_i/num_frames
        (via `phase` and the per-splash `local_phase` it derives) plus
        that fixed setup - it never mutates `rng` or any setup array
        inside the per-frame loop - so resuming at start_frame reproduces
        bit-identical frames to an uninterrupted run from frame 0, just
        without paying the compositing cost of the frames already
        rendered. 0 (default) = old behavior, including the empty-ground
        early-return path below.
    """
    width, height = size
    rng = np.random.default_rng(seed)

    flat_weights = ground_mask.flatten().astype("float64")
    total_weight = flat_weights.sum()
    if total_weight <= 0.0:
        # No plausible ground found - render an empty (fully transparent)
        # sequence rather than scattering splashes uniformly everywhere.
        for _ in range(start_frame, num_frames):
            yield Image.new("RGBA", size, (0, 0, 0, 0))
        return

    flat_probs = flat_weights / total_weight
    if min_spacing_factor > 0.0:
        splash_x, splash_y = _sample_spawn_points_with_spacing(
            rng, flat_probs, ground_mask.shape, count, max_radius * min_spacing_factor,
        )
    else:
        chosen = rng.choice(flat_weights.size, size=count, p=flat_probs)
        splash_y, splash_x = np.unravel_index(chosen, ground_mask.shape)

    phase_offsets = rng.uniform(0.0, 1.0, count)
    radius_jitter = rng.uniform(0.7, 1.5, count)

    # crown SHAPE: a ring of separated spikes/jets radiating
    # outward, not one filled polygon through perimeter points (however
    # bumpy its per-point radius, that's still a single continuous
    # convex-ish silhouette with no actual gaps - a wobbly circle, not a
    # spiky crown). Each splash gets its own random spike COUNT
    # (`spike_active_count`, 8-14), each slot with its own angle, height,
    # base half-width and lean, all sampled once here at setup exactly
    # like the old crown_angles/crown_radius_mult arrays were - only the
    # shape drawn per spike (each its own independent feathered patch,
    # see _apply_spike_patch, with visible background between neighbors)
    # is new.
    spike_active_count = rng.integers(8, _MAX_CROWN_SPIKES + 1, count)
    spike_angle_phase = rng.uniform(0.0, 2.0 * np.pi, count)
    spike_angles = (
        spike_angle_phase[:, None]
        + np.linspace(0.0, 2.0 * np.pi, _MAX_CROWN_SPIKES, endpoint=False)[None, :]
        + rng.uniform(-0.18, 0.18, (count, _MAX_CROWN_SPIKES))
    )
    spike_height_mult = rng.uniform(0.85, 1.55, (count, _MAX_CROWN_SPIKES))
    spike_half_width_frac = rng.uniform(0.09, 0.20, (count, _MAX_CROWN_SPIKES))
    spike_lean = rng.uniform(-0.14, 0.14, (count, _MAX_CROWN_SPIKES))
    # Each slot "switches on" at its own random point along the splash's
    # age curve (compared per-frame below against the shared height_age
    # envelope) instead of every spike appearing/disappearing in
    # lockstep - real coronets don't gain and lose every jet at once.
    spike_activation_thresh = rng.uniform(0.0, 1.0, (count, _MAX_CROWN_SPIKES))
    # A stable per-(splash, spike-slot) noise lane, same role the old
    # crown_noise_lane played: gives each spike's lean a small, smoothly
    # evolving wobble over its own lifetime instead of holding perfectly
    # rigid. The y-coordinate scrolls with the splash's own age
    # (`local_phase` below) so the wobble evolves smoothly within one
    # splash's life instead of jumping randomly frame to frame.
    spike_noise_lane = np.arange(count * _MAX_CROWN_SPIKES, dtype=np.float32).reshape(count, _MAX_CROWN_SPIKES)

    # 2-4 secondary droplets per splash, each with its own outward angle,
    # travel distance, arc height and size - padded to a fixed 4 slots
    # with a per-splash active count so the per-frame loop below can stay
    # simple (skip slots >= that splash's own droplet_count).
    max_droplets = 4
    droplet_count = rng.integers(2, max_droplets + 1, count)
    droplet_angle = rng.uniform(0.0, 2.0 * np.pi, (count, max_droplets))
    droplet_dist_frac = rng.uniform(0.7, 1.3, (count, max_droplets))
    droplet_arc_frac = rng.uniform(0.35, 0.65, (count, max_droplets))
    droplet_size_frac = rng.uniform(0.09, 0.16, (count, max_droplets))
    # Droplets separate quickly and fall back well before the crown itself
    # finishes fading - real secondary droplets are a brief early event,
    # not something that lingers as long as the main splash ring.
    droplet_life_frac = 0.55
    impact_flash_frac = 0.07   # fraction of a splash's own lifecycle spent as the bright impact flash

    ambient_tint = light_color if light_color is not None else sample_ambient_tint(base_image, fallback=(225, 232, 240))
    if ambient_tint_grid is not None:
        # each splash's own spawn UV looks up its own local
        # color from the shared per-photo grid, instead of every splash
        # sharing one whole-photo average - vectorized once here across all
        # `count` splashes (sample_tint_grid_bilinear broadcasts over
        # array u/v), then indexed per-splash below like every other
        # per-splash setup array in this function.
        splash_u = (splash_x.astype(np.float32) + 0.5) / width
        splash_v = (splash_y.astype(np.float32) + 0.5) / height
        local_tint = sample_tint_grid_bilinear(ambient_tint_grid, splash_u, splash_v)
    else:
        local_tint = np.broadcast_to(np.array(ambient_tint, dtype=np.float32), (count, 3)).copy()
    fill_colors = local_tint
    rim_colors = np.clip(local_tint * 1.06 + 8, 0, 255)
    droplet_colors = np.clip(local_tint * 1.1 + 15, 0, 255)
    flash_colors = np.clip(local_tint * 1.15 + 60, 0, 255)

    resolved_light_pos_uv = light_pos_uv if light_pos_uv is not None else (0.5, 0.0)
    base_array = None
    if base_image is not None and refraction_strength > 0.0:
        base_array = np.asarray(base_image.convert("RGB"), dtype=np.float32)

    for frame_i in range(start_frame, num_frames):
        phase = (frame_i / max(num_frames, 1))
        # Always allocated, regardless of base_array - the design notes spike
        # patches and wet base disc both rely on this shared float32
        # accumulator's soft alpha-blend compositing (real feathered
        # edges) even in the no-photo flat-fill fallback, not just when
        # refraction is active.
        water_rgba = np.zeros((height, width, 4), dtype=np.float32)
        frame = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(frame)

        # How far into its (looping) lifecycle each splash currently is,
        # 0..1, wrapping via modulo - purely phase-based, so it's the same
        # at frame 0 and frame num_frames by construction.
        local_phase = np.mod(phase - phase_offsets, 1.0) / duration_fraction
        active = local_phase < 1.0

        # Vectorized once per frame across every splash/spike slot (cheap -
        # count * _MAX_CROWN_SPIKES is under 1300 evaluations), then
        # indexed per-splash below - see spike_noise_lane's setup comment
        # for why each coordinate is built this way.
        spike_noise_time = np.broadcast_to((local_phase * 3.7)[:, None], (count, _MAX_CROWN_SPIKES))
        spike_noise_coord = np.stack([spike_noise_lane, spike_noise_time], axis=-1)
        spike_wobble = (_value_noise(spike_noise_coord) - 0.5) * 2.0   # -> roughly [-1, 1]

        for i in np.nonzero(active)[0]:
            t = local_phase[i]
            # Ease-out growth: real splash crowns expand
            # fast and decelerate under drag, not at constant speed.
            eased_t = 1.0 - (1.0 - t) ** 2
            radius = max_radius * radius_jitter[i] * eased_t
            if radius < 0.5:
                continue
            cx, cy = float(splash_x[i]), float(splash_y[i])
            fade = (1.0 - t) ** 1.5
            fill_color = tuple(int(c) for c in fill_colors[i])
            rim_color = tuple(int(c) for c in rim_colors[i])
            droplet_color = tuple(int(c) for c in droplet_colors[i])
            flash_color = tuple(int(c) for c in flash_colors[i])

            # crown SHAPE: a small pooling "wet base" disc at
            # the impact point (dimmer/smaller than the old whole-crown
            # disc - it's just standing water, not the rising jets), then
            # each active spike drawn as its own independent feathered
            # patch on top - the actual source of the visible GAPS between
            # spikes, since each _apply_spike_patch call only ever paints
            # its own thin wedge, never a shape spanning the whole ring.
            #
            # Spike height/count envelope over the splash's own age:
            # spikes rise quickly to a peak around t~0.2-0.4, hold
            # briefly, then shrink back down (and drop out one by one,
            # via spike_activation_thresh below) as the crown collapses
            # after t~0.6 - matching real coronet timing instead of every
            # spike appearing/disappearing in lockstep with the crown's
            # own overall radius.
            if t < 0.35:
                height_age = t / 0.35
            elif t < 0.6:
                height_age = 1.0
            else:
                height_age = max(0.0, 1.0 - (t - 0.6) / 0.4)

            base_r = radius * 0.32
            wet_disc_radius = max(1.0, base_r * 1.15)
            _apply_splash_refraction(
                water_rgba, base_array, cx, cy, wet_disc_radius, fade,
                resolved_light_pos_uv, refraction_strength, width, height,
                fill_color, outer_scale=1.05, peak_alpha=150.0,
            )

            active_spikes = np.nonzero(
                spike_activation_thresh[i, :spike_active_count[i]] <= max(height_age, 0.02)
            )[0]
            for k in active_spikes:
                tip_r_full = base_r + max(0.0, radius * spike_height_mult[i, k] - base_r)
                tip_r = base_r + (tip_r_full - base_r) * height_age
                if tip_r <= base_r + 0.5:
                    continue
                # Each spike's own alpha ramps up smoothly around its
                # activation threshold instead of popping on/off, and
                # its lean gains a small extra wobble as the splash ages
                # (spikes visibly lean/sway more as they near collapse).
                spike_alpha_mult = float(np.clip((height_age - spike_activation_thresh[i, k]) / 0.14 + 0.5, 0.0, 1.0))
                if spike_alpha_mult <= 0.0:
                    continue
                lean_wobble = spike_lean[i, k] + spike_wobble[i, k] * 0.05 * np.clip(t / 0.6, 0.0, 1.0)
                angle = float(spike_angles[i, k] + lean_wobble)
                half_width = radius * spike_half_width_frac[i, k]
                _apply_spike_patch(
                    water_rgba, base_array, cx, cy, angle, base_r, tip_r, half_width,
                    fade * spike_alpha_mult, resolved_light_pos_uv, refraction_strength,
                    width, height, fill_color, rim_color,
                )

            # Secondary droplets: separate outward, arc up, then fall
            # back - only while within this splash's own droplet_life_frac
            # window, mapped back to its own 0..1 range so the arc always
            # completes (rises then returns to 0) regardless of
            # duration_fraction.
            if t < droplet_life_frac:
                t_droplet = t / droplet_life_frac
                droplet_ease = 1.0 - (1.0 - t_droplet) ** 2
                arc = 4.0 * t_droplet * (1.0 - t_droplet)   # 0 at start/end, peaks at t_droplet=0.5
                droplet_fade = (1.0 - t_droplet) ** 1.2
                base_extent = max_radius * radius_jitter[i]

                for d in range(droplet_count[i]):
                    dist = base_extent * droplet_dist_frac[i, d] * droplet_ease
                    rise = base_extent * droplet_arc_frac[i, d] * arc
                    dx = cx + dist * np.cos(droplet_angle[i, d])
                    dy = cy + dist * np.sin(droplet_angle[i, d]) - rise
                    d_size = max(0.6, base_extent * droplet_size_frac[i, d] * (0.5 + 0.5 * droplet_ease))
                    d_alpha = int(np.clip(droplet_fade * 235, 0, 255))
                    if d_alpha > 0:
                        _draw_soft_droplet(
                            frame, dx, dy, d_size, droplet_color, d_alpha,
                            resolved_light_pos_uv, width, height,
                        )

            # Impact flash: one bright, fast-decaying burst right at the
            # moment of impact (t~0) - the instant a drop actually hits is
            # genuinely brighter than the crown's subsequent glint.
            if t < impact_flash_frac:
                flash_t = t / impact_flash_frac
                flash_alpha = int(np.clip((1.0 - flash_t) ** 2 * 255, 0, 255))
                flash_radius = max(1.0, max_radius * radius_jitter[i] * 0.35 * (1.0 - 0.4 * flash_t))
                if flash_alpha > 0:
                    draw.ellipse(
                        [cx - flash_radius, cy - flash_radius, cx + flash_radius, cy + flash_radius],
                        fill=(*flash_color, flash_alpha),
                    )

        water_img = Image.fromarray(np.clip(water_rgba, 0, 255).astype("uint8"), "RGBA")
        frame = Image.alpha_composite(water_img, frame)

        yield frame


def render_snow_accumulation(size, num_frames, ground_mask, max_opacity=0.5, roughness=None, start_frame=0):
    """
    Generator - a soft white build-up on ground_mask that grows and
    recedes in one or more breaths over the loop (see
    _resolve_snow_accumulation_cycles), instead of literally accumulating
    forever.

    A live wallpaper loop has to return to its exact starting frame, so
    snow can't actually pile up indefinitely without a visible jump cut
    back to bare ground at the seam - that's a hard conflict between
    "accumulates over time" and "loops seamlessly". This renders slow
    breathing cycles instead - depth ramps 0 -> max_opacity -> 0 across
    each whole cycle (`(1 - cos(theta)) / 2`) - which reads as gently
    gathering and settling snow rather than a one-way pile-up, and closes
    the loop exactly since cos(0) == cos(2*pi*cycles) for whole `cycles`.

    roughness: optional (height, width) float32 array, 0.0-1.0 (see
        build_roughness_field) - biases local opacity so accumulation
        reads as uneven (more where snow would actually settle/catch,
        less elsewhere) instead of one uniform translucent wash growing
        identically everywhere. Real surface-normal/edge
        data doesn't exist in this pipeline (see 1.11/19.1) - this is
        explicitly the "at minimum, a coarse static noise field" fallback
        the design notes itself calls out, not true surface-flatness modeling.
        None = the old uniform behavior.
    start_frame: skips yielding frames 0..start_frame-1 (see
        render_checkpoint.py) - the setup above (ground_rgb,
        roughness_factor, breath_cycles) still runs unconditionally
        regardless of this value, and each frame's depth_frac/alpha below
        is a pure function of its own frame_i/num_frames plus that fixed
        setup, so resuming at start_frame reproduces the exact same
        breathing-cycle phase (and therefore bit-identical frames) an
        uninterrupted run from frame 0 would have produced from that
        point on. 0 (default) = old behavior.
    """
    width, height = size
    ground_rgb = np.zeros((height, width, 4), dtype="uint8")
    ground_rgb[:, :, 0:3] = 255
    roughness_factor = (0.5 + 0.9 * roughness) if roughness is not None else 1.0
    breath_cycles = _resolve_snow_accumulation_cycles(num_frames)

    for frame_i in range(start_frame, num_frames):
        theta = 2.0 * np.pi * breath_cycles * (frame_i / max(num_frames, 1))
        depth_frac = (1.0 - np.cos(theta)) / 2.0
        alpha = np.clip(ground_mask * max_opacity * depth_frac * roughness_factor, 0.0, 1.0)

        rgba = ground_rgb.copy()
        rgba[:, :, 3] = (alpha * 255).astype("uint8")
        yield Image.fromarray(rgba, mode="RGBA")
