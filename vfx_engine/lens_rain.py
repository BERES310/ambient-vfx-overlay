"""
lens_rain.py

Design note #6: a NEW camera-lens artifact, genuinely
missing from the codebase before this - soft, out-of-focus water droplets
sitting directly ON the camera lens itself, distinct from lens_dirt.py's
static dust/grime film and distinct from particles_rain.py's IN-SCENE
falling drops. Real footage shot in heavy rain (through an unprotected
lens, or a windshield/window) shows both at once: sharp-ish falling drops
receding into the scene's depth, AND a handful of much bigger, badly
out-of-focus blobs of water sitting on the glass right in front of the
camera - two visually different things, not one effect at two scales.

Like lens_dirt.py, this is a camera artifact (not scene content) and is
layered on top of literally everything else, including lens_dirt itself
(a droplet sitting on top of existing grime is more visible than either
alone, not less). Also like lens_dirt.py, real lens water droplets held in
place by surface tension mostly DON'T slide around frame to frame over a
short loop - they sit where they landed, refracting/blurring whatever is
behind them, with only the faintest life (a slow surface-tension jiggle,
a specular glint that shifts slightly) to keep a 20-30s loop from looking
like one dead-static sticker. So exactly like lens_dirt.py's own
docstring reasoning, the expensive optical work (refraction sampling +
heavy out-of-focus blur per droplet) is computed ONCE per render, not
once per frame - only a small per-frame wobble/pulse animates on top.

Each droplet is built from three layered pieces, all soft/blurred (a
droplet this close to the lens is by definition miles outside the
camera's focus plane - there is no such thing as a sharp-edged droplet
here):
  1. A refracted/magnified, heavily blurred sample of the photo BEHIND the
     droplet (real water is transparent and convex - it bends and
     compresses the background into itself like a tiny magnifying lens,
     it doesn't just sit there as a flat gray blob). See
     `_sample_refracted_patch`.
  2. A darker meniscus ring near the droplet's true edge (real water beads
     bend light most sharply right at the edge where the surface curves
     away fastest, reading as a subtle dark rim) plus a soft ellipse alpha
     falloff so the whole blob fades out gradually, never with a hard
     silhouette edge.
  3. A small, bright specular highlight offset toward the photo's own
     detected light position (the same light_pos_uv every other lit
     effect in this pipeline shares - see main.py's resolve_light_pos_uv)
     - the actual glint that makes a droplet read as wet/reflective
     instead of just a blurry smudge.

Gated (see main.py's render_lens_rain_effects) to heavy-rain intensity
specifically ("storm") - light/medium rain doesn't throw enough water at
an exposed lens for this to be a realistic thing to see.
"""

import math

import numpy as np
from PIL import Image


def _gaussian_kernel_1d(sigma):
    radius = max(1, int(math.ceil(sigma * 3.0)))
    x = np.arange(-radius, radius + 1, dtype="float64")
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel.astype("float32")


def _gaussian_blur_float(array, radius):
    """
    Separable Gaussian blur applied directly to a float32 array (H, W) or
    (H, W, C), with edge-replicated padding - a manual numpy implementation
    (no scipy dependency - see module docstring) used instead of routing
    through an 8-bit PIL image. Routing a small droplet patch's steep,
    tightly-curved gradients (alpha feather, meniscus rim) through a
    uint8 PIL blur quantizes them before the highlight/rim shading and
    final compositing ever see them - against a near-black background
    that premature quantization shows up as visible banding/posterization
    rings tracing the gradient's level-set contours. Blurring in float32
    end to end (matching the rest of this pipeline's own linear-light
    float32 compositing chain - see image_io.py's module docstring) avoids
    that entirely; only the final RGBA frame is ever quantized to uint8.
    """
    if radius <= 0:
        return array
    sigma = max(radius / 3.0, 0.6)
    kernel = _gaussian_kernel_1d(sigma)
    pad = len(kernel) // 2

    def convolve_axis(arr, axis):
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (pad, pad)
        padded = np.pad(arr, pad_width, mode="edge")
        result = np.zeros_like(arr)
        for i, weight in enumerate(kernel):
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(i, i + arr.shape[axis])
            result += weight * padded[tuple(sl)]
        return result

    blurred = convolve_axis(array.astype("float32"), axis=0)
    blurred = convolve_axis(blurred, axis=1)
    return blurred


def _bilinear_sample(image_array, xs, ys):
    """
    Samples image_array (H, W, 3) float32 at fractional (xs, ys) pixel
    coordinates (same shape as each other), with edge-clamped bilinear
    interpolation. No scipy dependency (not installed in this project's
    venv - see very_important_note.txt rule 1, nothing gets pulled in from
    outside vfx_overlay/) so this is done by hand: the 4 integer neighbor
    lookups plus a manual weighted blend, exactly what
    scipy.ndimage.map_coordinates(order=1) would do internally.
    """
    height, width = image_array.shape[:2]
    xs = np.clip(xs, 0.0, width - 1.001)
    ys = np.clip(ys, 0.0, height - 1.001)

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = (xs - x0)[..., None]
    fy = (ys - y0)[..., None]

    top = image_array[y0, x0] * (1.0 - fx) + image_array[y0, x1] * fx
    bottom = image_array[y1, x0] * (1.0 - fx) + image_array[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


def _bilinear_shift(array, dx, dy):
    """
    Shifts a (h, w) or (h, w, c) array by a FRACTIONAL-pixel (dx, dy) offset
    via edge-clamped bilinear resampling - the sub-pixel-accurate analog of
    _bilinear_sample above, generalized to work on 2D arrays (alpha, rim)
    as well as 3D ones (rgb), which _bilinear_sample's hardcoded [..., None]
    broadcast can't do without breaking on non-square patches.

    Used to move the ALREADY-BAKED droplet patch content by the wobble
    offset without snapping it to the nearest whole pixel first - a whole-
    pixel snap (the previous approach here) makes sub-pixel wobble either
    invisible (rounds to 0 every frame) or jump in visible 1px steps
    instead of drifting smoothly, on top of silently disagreeing with this
    function's own docstring/comment, which already describes the wobble as
    "a fractional-pixel amount".
    """
    h, w = array.shape[:2]
    py, px = np.mgrid[0:h, 0:w].astype("float32")
    xs = np.clip(px - dx, 0.0, w - 1.001)
    ys = np.clip(py - dy, 0.0, h - 1.001)

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = xs - x0
    fy = ys - y0
    if array.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]

    top = array[y0, x0] * (1.0 - fx) + array[y0, x1] * fx
    bottom = array[y1, x0] * (1.0 - fx) + array[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


def _build_droplet_layout(size, seed, count, radius_range, elongation_range):
    """
    Returns a list of droplet dicts (center/radius/elongation/rotation/
    wobble phase) - random, but weighted toward the LOWER half of the
    frame (gravity pulls water down a lens/windshield faster than it
    accumulates fighting gravity near the top) the same way lens_dirt.py's
    grime is weighted toward the edges rather than uniform-random.
    """
    width, height = size
    rng = np.random.default_rng(seed)
    droplets = []
    for _ in range(count):
        cx = rng.uniform(0.03, 0.97) * width
        # Beta(2, 1.15) skews toward 1.0 (the bottom of frame) more than a
        # flat uniform distribution would, without banning droplets from
        # the upper frame entirely - a few always cling higher up too.
        cy = rng.beta(2.0, 1.15) * height
        radius = rng.uniform(*radius_range)
        elongation = rng.uniform(*elongation_range)
        rotation = rng.uniform(-0.35, 0.35)  # slight tilt off pure-vertical teardrop axis
        wobble_phase = rng.uniform(0.0, 2.0 * math.pi)
        wobble_seed_offset = rng.uniform(0.0, 1000.0)
        droplets.append(dict(
            cx=cx, cy=cy, radius=radius, elongation=elongation, rotation=rotation,
            wobble_phase=wobble_phase, wobble_seed_offset=wobble_seed_offset,
        ))
    return droplets


def _sample_refracted_patch(image_array, cx, cy, rx, ry, rotation, magnification, blur_radius):
    """
    Returns (patch_rgb, alpha, rim_darken) for one droplet, each a
    (patch_h, patch_w[, 3]) float32 array local to a bounding box padded
    around the droplet (padded so the blur below doesn't darken/fade the
    droplet's own edge against a hard patch boundary).

    patch_rgb: a refracted, heavily out-of-focus sample of `image_array`
        as seen "through" this droplet - see the module docstring's #1.
    alpha: 0.0-1.0 soft ellipse falloff (the droplet's own shape/opacity),
        already blurred.
    rim_darken: 0.0-1.0, strongest right at the droplet's true edge (the
        meniscus ring, module docstring's #2), 0 at the center and outside
        the droplet entirely.
    """
    height, width = image_array.shape[:2]
    pad = int(math.ceil(max(rx, ry) * 1.6 + blur_radius * 2.0 + 4))
    x0 = int(math.floor(cx - pad))
    x1 = int(math.ceil(cx + pad))
    y0 = int(math.floor(cy - pad))
    y1 = int(math.ceil(cy + pad))
    patch_x0, patch_y0 = max(0, x0), max(0, y0)
    patch_x1, patch_y1 = min(width, x1), min(height, y1)
    if patch_x1 <= patch_x0 or patch_y1 <= patch_y0:
        return None

    py, px = np.mgrid[patch_y0:patch_y1, patch_x0:patch_x1].astype("float32")
    dx = px - cx
    dy = py - cy
    # Rotate into the droplet's own (slightly tilted) major/minor axes
    # before the ellipse math below, so `elongation` stretches the drop
    # along its own tilt instead of always along the frame's vertical.
    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
    local_x = dx * cos_r + dy * sin_r
    local_y = -dx * sin_r + dy * cos_r

    rad = np.sqrt((local_x / rx) ** 2 + (local_y / ry) ** 2)
    rad_clamped = np.clip(rad, 0.0, 1.0)

    # Real convex water pulls in a WIDER cone of background than the
    # droplet's own small footprint and compresses it inward - strongest
    # right at the center (rad=0, the thickest part of the bead), fading
    # to no distortion at all right at the true edge (rad=1, where the
    # water surface is thinnest). Sampling with a SMALLER effective offset
    # at the center than at the edge is exactly that: it pulls in pixels
    # from farther away at the center (a wider background compressed into
    # a small area = magnified) while the rim samples almost undistorted.
    bulge = 1.0 - magnification * (1.0 - rad_clamped)
    src_x = cx + local_x * cos_r - local_y * sin_r  # placeholder, replaced below
    # (local_x/local_y are axis-aligned in droplet space - rotate the
    # BULGE-SCALED offset back into image space to sample the right pixels)
    scaled_local_x = local_x * bulge
    scaled_local_y = local_y * bulge
    src_x = cx + scaled_local_x * cos_r - scaled_local_y * sin_r
    src_y = cy + scaled_local_x * sin_r + scaled_local_y * cos_r

    sampled = _bilinear_sample(image_array, src_x, src_y)

    # Blur the refracted sample - a droplet this close to the lens is far
    # outside the camera's focus plane, so even the "image" it refracts
    # reads as a soft, indistinct blur, never a sharp tiny upside-down
    # scene (that level of optical accuracy would read as a gimmick, not
    # realism, at the scale a rendered droplet appears on screen). Blurred
    # directly in float32 (see _gaussian_blur_float's docstring) rather
    # than through an 8-bit PIL round trip, which would quantize/
    # posterize this small patch's steep gradients.
    patch_rgb = _gaussian_blur_float(sampled, blur_radius)

    # Soft ellipse alpha - smoothstep from fully opaque (rad <= 0.82) down
    # to fully transparent (rad >= 1.0), then blurred again so even that
    # falloff has no hard edge left after resampling.
    feather_start = 0.82
    t = np.clip((rad - feather_start) / (1.0 - feather_start), 0.0, 1.0)
    alpha = (1.0 - (t * t * (3.0 - 2.0 * t))).astype("float32")
    alpha = _gaussian_blur_float(alpha, max(1.0, blur_radius * 0.5))

    # Meniscus rim: a narrow dark band centered right at rad ~= 0.9,
    # zero at the center and zero past the outer feather - real water
    # beads read distinctly darker in a thin ring near their true edge
    # (the surface curves away fastest there, bending reflected light away
    # from the camera) rather than a flat gradient from bright center to
    # transparent edge.
    rim_center, rim_width = 0.9, 0.16
    rim_darken = np.clip(1.0 - np.abs(rad - rim_center) / rim_width, 0.0, 1.0).astype("float32")
    rim_darken *= alpha  # never extends past the droplet's own visible edge
    rim_darken = _gaussian_blur_float(rim_darken, max(1.0, blur_radius * 0.35))

    return dict(x0=patch_x0, y0=patch_y0, rgb=patch_rgb, alpha=alpha, rim=rim_darken,
                local_x=local_x, local_y=local_y, rx=rx, ry=ry)


def render_lens_rain(size, num_frames, image, droplet_count=14, radius_range=(16, 46),
                      elongation_range=(1.05, 1.55), magnification=0.55, blur_radius_frac=0.012,
                      strength=0.85, rim_darken_strength=0.5, wobble_px=1.5, pulse_amount=0.25,
                      pulse_speed=1, highlight_strength=0.8, light_pos_uv=None, light_color=None,
                      seed=None, start_frame=0):
    """
    Generator - yields num_frames RGBA overlay frames of water droplets
    sitting on the lens itself (see module docstring). A camera artifact,
    not scene content - meant to be composited on top of everything else,
    same layer category as lens_dirt/lens flare/starburst.

    image: the source photo (PIL Image) each droplet optically refracts -
        required (unlike lens_dirt, which can run with image=None and just
        skip its light-coupling/brightness-visibility extras, a lens-rain
        droplet with nothing to refract isn't a meaningful effect at all).
    droplet_count: how many droplets sit on the lens.
    radius_range: (min, max) droplet radius in pixels, along its minor
        axis, before `elongation` stretches the major axis.
    elongation_range: (min, max) how much taller than wide each droplet's
        ellipse is (gravity-stretched teardrop, not a perfect circle).
    magnification: 0.0-1.0+, how strongly each droplet bulges/compresses
        the background behind it (see _sample_refracted_patch) - 0
        disables the lensing distortion entirely (still refracts, just
        with an undistorted 1:1 sample before blurring).
    blur_radius_frac: Gaussian blur radius for the refracted sample, as a
        fraction of image width - keeps droplets softly out-of-focus at
        any resolution instead of a fixed pixel radius that would read
        differently on a 1080p vs 4K render.
    strength: 0.0-1.0+, overall opacity multiplier for every droplet.
    rim_darken_strength: 0.0-1.0+, how strongly the meniscus edge ring
        darkens the droplet under it (see _sample_refracted_patch's rim).
    wobble_px: >=0, amplitude (pixels) of each droplet's own slow,
        CLOSED-loop surface-tension jiggle (a tiny elliptical path in
        image space that returns exactly to its start every loop, the
        same "closed drift path" convention fog.py's own movement
        parameter uses - see fog.py's module docstring - so this loops
        seamlessly like every other cyclic effect in this pipeline). 0 =
        perfectly static droplets (still fine - real pinned droplets
        barely move at all).
    pulse_amount/pulse_speed: the droplet's specular highlight brightens/
        dims slightly over the loop (pulse_speed MUST be a whole number
        for the loop to close exactly - see lens_dirt.render_lens_dirt's
        identical note) - keeps a still frame from looking perfectly
        dead-static over a long loop, exactly like lens_dirt's own pulse.
    highlight_strength: 0.0-1.0+, brightness of each droplet's specular
        glint.
    light_pos_uv/light_color: normalized (0.0-1.0) detected light position/
        (r, g, b) color - see main.py's resolve_light_pos_uv/
        god_rays.sample_light_color. None (default) puts the highlight at
        a fixed upper-left offset with a neutral warm-white color instead.
    seed: RNG seed for droplet placement - None = a new random layout
        every render.
    start_frame: skips yielding frames 0..start_frame-1 (see
        render_checkpoint.py) - the expensive one-time optical bake above
        (`baked`, built from `droplets`/`_sample_refracted_patch`) still
        runs unconditionally regardless of this value, and each frame's
        wobble/pulse/highlight below is a pure function of its own
        frame_i/num_frames plus that fixed bake, so resuming at
        start_frame reproduces bit-identical frames to an uninterrupted
        run from frame 0, just without paying the per-frame compositing
        cost of the frames already rendered. 0 (default) = old behavior.
    """
    width, height = size
    image_array = np.asarray(image.convert("RGB").resize(size, Image.BILINEAR), dtype="float32")
    blur_radius = max(1.0, blur_radius_frac * width)

    droplets = _build_droplet_layout(size, seed, droplet_count, radius_range, elongation_range)

    if light_pos_uv is not None:
        light_x = light_pos_uv[0] * width
        light_y = light_pos_uv[1] * height
    else:
        light_x, light_y = None, None
    resolved_light_color = np.array(light_color if light_color is not None else (255, 248, 230), dtype="float32")

    # The expensive optical work - refraction sampling + heavy blur - is
    # computed ONCE here, not per frame (see module docstring: real pinned
    # droplets don't change their own optics frame to frame). Only the
    # highlight position (closed wobble) and its brightness (pulse) below
    # vary per frame.
    baked = []
    for droplet in droplets:
        patch = _sample_refracted_patch(
            image_array, droplet["cx"], droplet["cy"], droplet["radius"],
            droplet["radius"] * droplet["elongation"], droplet["rotation"],
            magnification, blur_radius,
        )
        if patch is None:
            continue

        # Specular highlight: a small bright ellipse offset from the
        # droplet's center TOWARD the detected light source (real water
        # beads catch a glint on the side facing the light, not
        # dead-center) - falls back to a fixed upper-left offset (a light
        # "above and to the left" assumption, same fallback direction
        # god_rays/lighting.py's other un-detected-light defaults use)
        # when no light position was detected.
        if light_x is not None:
            to_light_x, to_light_y = light_x - droplet["cx"], light_y - droplet["cy"]
            light_dist = math.hypot(to_light_x, to_light_y)
            if light_dist > 1e-3:
                to_light_x, to_light_y = to_light_x / light_dist, to_light_y / light_dist
            else:
                to_light_x, to_light_y = -0.6, -0.8
        else:
            to_light_x, to_light_y = -0.6, -0.8
        highlight_offset = droplet["radius"] * 0.35
        highlight_cx = droplet["cx"] + to_light_x * highlight_offset
        highlight_cy = droplet["cy"] + to_light_y * highlight_offset
        highlight_radius = max(2.0, droplet["radius"] * 0.28)

        py = np.arange(patch["y0"], patch["y0"] + patch["alpha"].shape[0])
        px = np.arange(patch["x0"], patch["x0"] + patch["alpha"].shape[1])
        gx, gy = np.meshgrid(px, py)
        highlight_dist_sq = (gx - highlight_cx) ** 2 + (gy - highlight_cy) ** 2
        highlight_field = np.exp(-highlight_dist_sq / (2.0 * highlight_radius ** 2)).astype("float32")
        highlight_field *= patch["alpha"]

        baked.append(dict(patch=patch, droplet=droplet, highlight_field=highlight_field))

    for frame_i in range(start_frame, num_frames):
        t = frame_i / max(num_frames, 1)
        theta = 2.0 * math.pi * t
        pulse = 1.0 + pulse_amount * math.sin(2.0 * math.pi * pulse_speed * t)

        rgb_accum = np.zeros((height, width, 3), dtype="float32")
        alpha_accum = np.zeros((height, width), dtype="float32")

        for entry in baked:
            patch, droplet = entry["patch"], entry["droplet"]
            ph, pw = patch["alpha"].shape
            y0, x0 = patch["y0"], patch["x0"]

            # Closed-loop surface-tension wobble: a small ellipse in
            # image-space that returns exactly to (0, 0) at t=0 and t=1 -
            # shifts the ALREADY-BAKED patch by a fractional-pixel amount
            # via a cheap re-blend against itself shifted, rather than
            # re-running the expensive refraction sample every frame.
            wobble_dx = wobble_px * math.cos(theta + droplet["wobble_phase"])
            wobble_dy = wobble_px * 0.6 * math.sin(theta + droplet["wobble_phase"]) - wobble_px * 0.6 * math.sin(droplet["wobble_phase"])

            # Fractional-pixel shift via bilinear resampling (see
            # _bilinear_shift's docstring) instead of snapping wobble_dx/dy
            # to the nearest whole pixel - the patch is placed at its
            # unmoved integer (y0, x0) below, with the sub-pixel motion
            # baked into the resampled content itself.
            base_rgb = _bilinear_shift(patch["rgb"], wobble_dx, wobble_dy)
            base_alpha = _bilinear_shift(patch["alpha"], wobble_dx, wobble_dy) * strength * pulse
            rim = _bilinear_shift(patch["rim"], wobble_dx, wobble_dy) * rim_darken_strength
            shaded_rgb = base_rgb * (1.0 - rim[..., None])

            highlight = _bilinear_shift(entry["highlight_field"], wobble_dx, wobble_dy) * highlight_strength * pulse
            shaded_rgb = shaded_rgb * (1.0 - highlight[..., None]) + resolved_light_color[None, None, :] * highlight[..., None]

            dst_y0 = y0
            dst_x0 = x0
            dst_y1, dst_x1 = dst_y0 + ph, dst_x0 + pw

            clip_y0, clip_x0 = max(0, dst_y0), max(0, dst_x0)
            clip_y1, clip_x1 = min(height, dst_y1), min(width, dst_x1)
            if clip_y1 <= clip_y0 or clip_x1 <= clip_x0:
                continue
            src_y0, src_x0 = clip_y0 - dst_y0, clip_x0 - dst_x0
            src_y1, src_x1 = src_y0 + (clip_y1 - clip_y0), src_x0 + (clip_x1 - clip_x0)

            region_alpha = base_alpha[src_y0:src_y1, src_x0:src_x1]
            region_rgb = shaded_rgb[src_y0:src_y1, src_x0:src_x1]

            dst_alpha_slice = alpha_accum[clip_y0:clip_y1, clip_x0:clip_x1]
            dst_rgb_slice = rgb_accum[clip_y0:clip_y1, clip_x0:clip_x1]

            # Straight "over" blend, droplet-by-droplet - droplets rarely
            # overlap given their placement spread, and on the few that do
            # this still composites correctly (later droplets drawn over
            # earlier ones), same alpha-over math as compositor.py uses
            # for every other layer in this pipeline.
            new_alpha = region_alpha + dst_alpha_slice * (1.0 - region_alpha)
            safe_new_alpha = np.maximum(new_alpha, 1e-6)
            blended_rgb = (region_rgb * region_alpha[..., None] +
                           dst_rgb_slice * dst_alpha_slice[..., None] * (1.0 - region_alpha[..., None])) / safe_new_alpha[..., None]
            dst_rgb_slice[:] = blended_rgb
            dst_alpha_slice[:] = new_alpha

        rgba = np.empty((height, width, 4), dtype="uint8")
        rgba[:, :, 0:3] = np.clip(rgb_accum, 0.0, 255.0).astype("uint8")
        rgba[:, :, 3] = np.clip(alpha_accum * 255.0, 0.0, 255.0).astype("uint8")
        yield Image.fromarray(rgba, mode="RGBA")
