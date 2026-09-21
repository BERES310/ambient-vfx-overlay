"""
materials.py

a small, shared, physically-motivated material model: one
per-segmentation-category table of surface material properties (base
albedo tint, roughness/specular-sharpness, Fresnel reflectance at normal
incidence, subsurface-scattering strength) plus ONE shared shading
function (shade_surface) implementing a single-bounce Lambertian-diffuse
+ Fresnel-weighted specular BRDF.

This is NOT a full 3D renderer - there's no real light transport, no
shadows, no multi-bounce global illumination, and view/light directions
are cheap 2.5D approximations (see resolve_pseudo_light_direction), not
an inverse-projected 3D scene. It exists so that every effect in this
pipeline that currently invents its OWN ad hoc "how shiny/what color
should this surface look" logic can instead share one physically-
grounded function and one category-keyed parameter table - so a wet
road and a wet rock face read as genuinely different materials instead
of each hand-tuned effect reinventing its own private, mutually
inconsistent "shininess" number. ground_fx.py's render_wet_sheen is the
first real consumer (see that function's docstring).

Categories match vfx_engine/segmentation.py's 9-category output exactly:
sky, water, vegetation_tree, vegetation_palm, vegetation_flower,
ground_hard, ground_soft, rock_terrain, building.
"""

import numpy as np

# category -> {
#   base_albedo_tint: (r, g, b) 0..1 - the material's own base color,
#       independent of lighting. Overridden per-pixel by
#       illumination.py's learned albedo channel when available (design notes
#       9.9(a) - see resolve_albedo below); this flat tint is the
#       fallback when that model is disabled/unavailable.
#   roughness: 0.0 (mirror-smooth) - 1.0 (fully diffuse/matte). Drives
#       shade_surface's specular lobe width (see that function).
#   fresnel_f0: reflectance at normal incidence, 0..1. Real-world
#       dielectrics (water, wet stone, leaves, plaster) mostly cluster in
#       a narrow ~0.02-0.05 band regardless of color - unlike metals,
#       whose F0 is high and itself colored - so these values stay
#       deliberately close together; what actually differentiates the
#       categories' final look is roughness (highlight sharpness) far
#       more than F0.
#   subsurface_strength: 0.0-1.0 - how much light scatters through the
#       material rather than reflecting purely at the surface. Only
#       meaningful for vegetation currently (a sunlit leaf visibly glows
#       from transmitted light) - stored here for future consumers, not
#       yet consumed by shade_surface (see that function's docstring).
# }
MATERIAL_TABLE = {
    "sky": {
        "base_albedo_tint": (0.75, 0.82, 0.95), "roughness": 1.0,
        "fresnel_f0": 0.02, "subsurface_strength": 0.0,
    },
    "water": {
        "base_albedo_tint": (0.35, 0.50, 0.55), "roughness": 0.08,
        "fresnel_f0": 0.02, "subsurface_strength": 0.0,
    },
    "vegetation_tree": {
        "base_albedo_tint": (0.28, 0.42, 0.18), "roughness": 0.75,
        "fresnel_f0": 0.03, "subsurface_strength": 0.45,
    },
    "vegetation_palm": {
        "base_albedo_tint": (0.30, 0.48, 0.22), "roughness": 0.70,
        "fresnel_f0": 0.03, "subsurface_strength": 0.50,
    },
    "vegetation_flower": {
        "base_albedo_tint": (0.55, 0.35, 0.40), "roughness": 0.60,
        "fresnel_f0": 0.03, "subsurface_strength": 0.55,
    },
    "ground_hard": {
        "base_albedo_tint": (0.45, 0.43, 0.40), "roughness": 0.35,
        "fresnel_f0": 0.045, "subsurface_strength": 0.0,
    },
    "ground_soft": {
        "base_albedo_tint": (0.32, 0.35, 0.20), "roughness": 0.85,
        "fresnel_f0": 0.03, "subsurface_strength": 0.10,
    },
    "rock_terrain": {
        "base_albedo_tint": (0.40, 0.38, 0.36), "roughness": 0.65,
        "fresnel_f0": 0.04, "subsurface_strength": 0.0,
    },
    "building": {
        "base_albedo_tint": (0.55, 0.52, 0.48), "roughness": 0.50,
        "fresnel_f0": 0.04, "subsurface_strength": 0.0,
    },
}

# Neutral mid-gray/mid-roughness fallback for an unrecognized/None category -
# never raises, since a caller passing a category string that doesn't (yet)
# have a table entry shouldn't crash a whole render over a cosmetic lookup.
_DEFAULT_MATERIAL = {
    "base_albedo_tint": (0.5, 0.5, 0.5), "roughness": 0.5,
    "fresnel_f0": 0.03, "subsurface_strength": 0.0,
}


def get_material(category):
    """Returns MATERIAL_TABLE[category], or _DEFAULT_MATERIAL for an unknown/None category."""
    return MATERIAL_TABLE.get(category, _DEFAULT_MATERIAL)


def resolve_albedo(category, illumination_albedo=None):
    """
    (a): returns the surface albedo to shade with - either
    illumination.py's own per-pixel Marigold-estimated albedo channel
    (a (H, W, 3) float32 array, 0..1 - see illumination.estimate_
    illumination's `albedo` key) when the caller supplies one, or
    MATERIAL_TABLE's flat per-category tint otherwise.

    A per-pixel measured albedo is strictly more accurate than one flat
    tint per segmentation category: real ground/vegetation/rock all have
    plenty of pixel-to-pixel albedo variation (dirt patches, worn spots,
    leaf-color variation, moss) that a single category-wide color can't
    represent, so it overrides the category default wherever available -
    exactly the "PER-PIXEL override of the per-category flat albedo
    tint" calls for. `illumination_albedo=None` (the default -
    config.ENABLE_LEARNED_ILLUMINATION off, or the caller didn't compute
    it) keeps the old flat-tint behavior.
    """
    if illumination_albedo is not None:
        return illumination_albedo.astype(np.float32)
    return np.asarray(get_material(category)["base_albedo_tint"], dtype=np.float32)


def schlick_fresnel(cos_theta, f0):
    """
    Schlick's approximation of Fresnel reflectance: f0 + (1-f0)*(1-cos)^5 -
    how much MORE reflective a surface becomes at a glancing viewing
    angle than head-on (every real surface, no matter how matte, reflects
    almost perfectly at a near-90-degree grazing angle - the familiar
    "a lake looks like a mirror near the horizon but transparent close to
    my feet" effect). f0 is the material's reflectance at normal
    incidence (cos_theta=1) - see MATERIAL_TABLE's fresnel_f0.

    cos_theta: numpy array or float, expected already clipped to 0..1.
    f0: float (or an array broadcastable against cos_theta) for a
        colored/per-channel Fresnel term.
    """
    return f0 + (1.0 - f0) * np.power(np.clip(1.0 - cos_theta, 0.0, 1.0), 5)


def resolve_pseudo_light_direction(light_pos_uv, elevation=0.55):
    """
    this pipeline has no real 3D scene or camera - only a 2D
    photo plus a detected 2D light POSITION (god_rays.detect_light_source)
    and, where available, per-pixel surface normals in a camera-facing
    frame (surface_normals.py). A physically-motivated specular highlight
    still needs a light DIRECTION, not just a position, so this builds a
    simple, deliberately approximate pseudo-3D direction: the XY
    components point from screen center toward the light's 2D position (a
    light in the upper-right of frame is treated as "coming from" the
    upper-right), and a fixed positive Z component (`elevation`) stands in
    for "the light sits somewhat in front of/above the scene, not
    directly behind the camera" - without it, a light positioned exactly
    at screen center would collapse to a degenerate all-zero XY direction.
    This is a 2.5D approximation for a stylized effects pipeline, not an
    inverse projection of real scene geometry.

    light_pos_uv: (x, y) normalized 0..1 IMAGE-space coordinates (y=0 at
        the top - matches main.py's resolve_light_pos_uv/god_rays'
        convention), or None (no light detected).

    Returns a normalized (3,) float32 array in the same screen-space
    camera frame surface_normals.py's normals use (X=right, Y=up, Z=
    toward camera) - or None if light_pos_uv is None.
    """
    if light_pos_uv is None:
        return None

    light_x_uv, light_y_uv = light_pos_uv
    dx = (light_x_uv - 0.5) * 2.0
    dy = -(light_y_uv - 0.5) * 2.0   # image-space y=0-top -> camera-frame Y=up
    direction = np.array([dx, dy, elevation], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return direction / norm


def shade_surface(normal, light_dir, category, light_color, illumination_albedo=None, view_dir=(0.0, 0.0, 1.0)):
    """
    shared single-bounce BRDF - the ONE function every
    material-aware shading effect in this pipeline should call, instead
    of each effect inventing its own ad hoc "how bright/what color"
    formula. Combines:

      - a Lambertian diffuse term (albedo * N.L), using either
        illumination.py's per-pixel learned albedo or MATERIAL_TABLE's
        flat per-category tint (see resolve_albedo -.9(a)), and
      - a Blinn-Phong specular lobe ((N.H)^shininess, H = normalize(view
        + light)) whose sharpness is derived from the category's
        roughness (roughness=0 -> tight, mirror-like highlight;
        roughness=1 -> broad, barely-there highlight) - standard, cheap,
        and visually convincing; not a physically exact microfacet model
        (e.g. GGX), which this stylized pipeline doesn't need.

    The two terms are blended via Schlick Fresnel reflectance
    (schlick_fresnel(N.V, fresnel_f0)) so a surface is diffuse-dominant
    head-on and specular-dominant at grazing angles, the same real-world
    behavior schlick_fresnel's own docstring describes - not a fixed,
    unphysical 50/50 mix.

    Args:
        normal: (H, W, 3) float32 unit vectors, camera-space (X=right,
            Y=up, Z=toward camera - see surface_normals.py), or None to
            fall back to a flat forward-facing (0,0,1) normal everywhere
            (still produces a valid, if spatially-flat, result).
        light_dir: (3,) unit vector (see resolve_pseudo_light_direction),
            or None (no detected light source) - returns an all-zero
            result in that case, since there is nothing to shade toward.
        category: a MATERIAL_TABLE key (see get_material) - supplies
            roughness/fresnel_f0 and (when illumination_albedo is None)
            the flat albedo tint.
        light_color: (r, g, b) 0-255 sRGB tuple - see
            god_rays.sample_light_color/lighting.estimate_ambient_light_color.
        illumination_albedo: optional (H, W, 3) float32 array, 0..1 -
           .9(a) per-pixel albedo override (see resolve_albedo).
        view_dir: (3,) unit vector - fixed straight-on camera view by
            default (this pipeline never simulates a moving 3D camera
            relative to the scene; camera drift/dolly effects operate on
            the whole 2D frame at the compositing stage, not per-material).

    Returns:
        (H, W, 3) float32 array - the shaded surface color (NOT yet
        composited onto anything; the caller decides how to blend it in,
        e.g. as a tint/highlight overlay). All-zero if light_dir is None.
    """
    material = get_material(category)
    shape = normal.shape[:2] if normal is not None else (1, 1)

    if light_dir is None:
        return np.zeros(shape + (3,), dtype=np.float32)

    if normal is None:
        normal = np.zeros(shape + (3,), dtype=np.float32)
        normal[..., 2] = 1.0

    view = np.asarray(view_dir, dtype=np.float32)
    light = np.asarray(light_dir, dtype=np.float32)
    half_vec = view + light
    half_norm = float(np.linalg.norm(half_vec))
    half_vec = half_vec / half_norm if half_norm > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

    n_dot_h = np.clip(np.tensordot(normal, half_vec, axes=([2], [0])), 0.0, 1.0)
    n_dot_v = np.clip(np.tensordot(normal, view, axes=([2], [0])), 0.0, 1.0)
    n_dot_l = np.clip(np.tensordot(normal, light, axes=([2], [0])), 0.0, 1.0)

    roughness = material["roughness"]
    # Smoother (lower roughness) -> higher exponent -> tighter, brighter
    # peak; a fully rough (roughness=1) surface still gets a broad, dim
    # lobe rather than literally zero specular response (a real fully
    # matte surface still shows some sheen at a grazing enough angle).
    shininess = 2.0 + (1.0 - roughness) ** 2 * 250.0
    spec_shape = np.power(n_dot_h, shininess)

    fresnel = schlick_fresnel(n_dot_v, material["fresnel_f0"])

    albedo = resolve_albedo(category, illumination_albedo)
    linear_light_color = np.asarray(light_color, dtype=np.float32) / 255.0

    diffuse = albedo * (n_dot_l * (1.0 - fresnel))[..., None] * linear_light_color[None, None, :]
    specular = (spec_shape * fresnel * n_dot_l)[..., None] * linear_light_color[None, None, :]

    return (diffuse + specular).astype(np.float32)
