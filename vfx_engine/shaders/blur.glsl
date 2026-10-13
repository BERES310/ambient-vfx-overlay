// blur.glsl
//
// Renders a soft, drifting, translucent haze (fog/smoke/steam/heat haze)
// as a full-screen effect. Instead of blurring a texture of random noise
// (the old numpy-based approach), this generates the noise procedurally,
// directly in the fragment shader, using "fractal Brownian motion" (fbm) -
// several layers ("octaves") of smooth noise at different scales added
// together, which is the standard technique for natural-looking clouds/fog
// in real-time graphics.

#vertex
#version 330

in vec2 in_position;   // a full-screen quad's corners, already in -1..1 range
out vec2 v_uv;

void main() {
    v_uv = (in_position + 1.0) * 0.5;   // 0..1 texture-style coordinates
    gl_Position = vec4(in_position, 0.0, 1.0);
}

#fragment
#version 330

in vec2 v_uv;
out vec4 frag_color;

uniform float frame_i;       // current frame index - drives the drifting motion over time
uniform float num_frames;    // total frames in the loop - see the loop-safety note below
uniform vec2 resolution;
uniform vec3 fog_color;
uniform float density;       // overall opacity, 0.0-1.0
uniform float movement;      // radius of the drift path - how far the haze wanders over one loop
uniform vec2 wind_bias;      // shared weather wind direction (see vfx_engine/weather.py), unit vector -
                              // also used as gust_at()'s spatial-propagation direction below,
                              // since fog.py sets this directly to weather.wind_direction_vector() with no
                              // further per-caller scaling, unlike region_warp.glsl's own wind_bias uniform.
// the design notes (spatially-propagating gust): this used to be one
// CPU-precomputed frame-global `uniform float gust` value (1.0 = no
// coupling) - now this fragment computes its OWN gust value via
// gust_at() below, using this same set of raw ingredients
// region_warp.glsl's identical function does (see
// vfx_engine/weather.py's gust_shader_uniforms() docstring for the full
// formula). Fog has no elastic/damped-spring variant (nothing currently
// asks for one), so gust_amp_a/gust_lag_a/gust_amp_b/gust_lag_b are
// always fixed at 1.0/0.0 (the plain gust_envelope() case) by fog.py.
uniform float gust_phase_offset;
uniform float gust_cycles_a;
uniform float gust_cycles_b;
uniform float gust_amp_a;
uniform float gust_lag_a;
uniform float gust_amp_b;
uniform float gust_lag_b;
uniform float gust_strength;
uniform float gust_spatial_delay;
uniform sampler2D depth_texture;   // 0.0 (far) - 1.0 (near), same convention as vfx_engine/depth.py
uniform float has_depth;           // 1.0 if depth_texture holds real depth data, 0.0 = none available (see fog.py)
uniform float ground_fog_strength;   // 0.0-1.0 - how much thinner fog gets toward the TOP of frame
uniform vec2 light_pos_uv;           // normalized (0..1) IMAGE-space (y=0 top) light position - see fog.py
uniform float has_light_pos;         // 1.0 if light_pos_uv was actually provided this render
uniform vec3 light_tint_color;       // color to brighten/tint fog toward, near the light position
uniform float light_coupling_strength;   // 0.0-1.0, how strongly fog brightens/tints around the light
uniform sampler2D luma_texture;      // lightly-blurred grayscale of the base photo, 0.0-1.0
uniform float has_luma;              // 1.0 if luma_texture holds a real photo, 0.0 = none available (see fog.py)
uniform float ground_texture_strength;   // 0.0-1.0, how strongly fog anchors to the photo's own ground detail

// A cheap, deterministic pseudo-random value for a 2D point - the building
// block every other noise function here is made from.
//
// This used to be the classic `fract(sin(dot(p, big_vec)) * huge_const)`
// trick. That version is only reliable for small |p| - discovered while
// implementing the design notes domain warp, whose extra fbm() sample point
// (p shifted by a warp offset) pushed coordinates into a range where
// sin()'s hardware range-reduction loses enough precision that many
// adjacent integer grid cells alias to near-identical hash values. The
// result isn't subtle: whole multi-cell regions of the noise field go
// flat, with hard axis-aligned edges exactly at the cells where the
// aliasing steps - real value noise, being smoothstep-interpolated
// between hashed corners, can never produce a genuine right-angle edge,
// so those edges were the tell that this was a precision bug, not an
// artistic noise-shape issue. This replacement (a well-known
// fract/multiply-chain hash - see Dave Hoskins' "hash without sine"
// family) never calls a transcendental function on the raw coordinate,
// so its output quality doesn't degrade as |p| grows - safe for the
// wider coordinate range domain warping and its extra fbm() calls need.
float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Smoothly interpolated ("value") noise between random grid points.
float value_noise(vec2 p) {
    vec2 cell = floor(p);
    vec2 frac_part = fract(p);

    float a = hash(cell);
    float b = hash(cell + vec2(1.0, 0.0));
    float c = hash(cell + vec2(0.0, 1.0));
    float d = hash(cell + vec2(1.0, 1.0));

    vec2 u = frac_part * frac_part * (3.0 - 2.0 * frac_part);   // smoothstep-style easing
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// Fractal Brownian motion: stack several octaves of noise at increasing
// frequency and decreasing strength - this is what gives it an organic,
// "many overlapping wisps" look instead of one blobby pattern.
float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 5; i++) {
        value += amplitude * value_noise(p);
        p *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

// domain warping ("fbm of fbm", the classic Inigo Quilez
// technique). Sampling fbm(p) directly makes the whole field one rigid
// pattern that only ever translates - every octave slides together, so
// motion instantly reads as "one texture scrolling," not gas actually
// moving. Real turbulence folds and shears: different regions of the
// field get pushed by a flow that itself varies in space.
//
// Here that flow is approximated by sampling a second, independent,
// low-frequency fbm to get a scalar "potential" at this point, turning
// that scalar into a unit direction (angle = potential * 2*pi), and
// nudging the sample point warp_strength units along that direction
// before the final fbm lookup. Because the direction itself comes from
// a smooth noise field, nearby points warp toward similar-but-not-
// identical directions - exactly the fold/shear look a rigid single
// fbm() call can't produce - at the cost of one extra fbm() evaluation
// per pixel (5 octaves -> effectively ~10).
float warped_fbm(vec2 p) {
    float warp_potential = fbm(p * 0.85 + vec2(4.2, 1.7));
    vec2 warp_dir = vec2(cos(warp_potential * 6.28318530718), sin(warp_potential * 6.28318530718));
    float warp_strength = 1.8;
    return fbm(p + warp_dir * warp_strength);
}

// position-aware gust value - see region_warp.glsl's
// identical function/vfx_engine.weather.gust_shader_uniforms()'s
// documented formula. gust_spatial_delay = 0.0 makes phase_delay always
// 0, reproducing this shader's old (pre-13.2) "every fragment feels the
// gust simultaneously" behavior exactly.
float gust_at(vec2 position_uv) {
    float distance_downwind = dot(position_uv - vec2(0.5), wind_bias);
    float phase_delay = -distance_downwind * gust_spatial_delay * 6.28318530718;
    float gust_theta = 6.28318530718 * (frame_i / max(num_frames, 1.0)) + gust_phase_offset + phase_delay;
    float wave = 0.6 * gust_amp_a * sin(gust_theta * gust_cycles_a - gust_lag_a)
               + 0.4 * gust_amp_b * sin(gust_theta * gust_cycles_b + 1.7 - gust_lag_b);
    return max(0.0, 1.0 + wave * gust_strength);
}

void main() {
    vec2 aspect_corrected = v_uv * vec2(resolution.x / resolution.y, 1.0);

    // Loop-safe drift: sample the noise field's TIME input around a
    // circle (radius `movement`) instead of sliding it linearly with
    // frame_i - the same trick particles_wind.comp uses for its wind
    // field (see that file's docstring for the full explanation). A
    // linear `time_offset * movement` offset (the original approach)
    // has no reason to land back on frame 0's value at frame num_frames,
    // so the fog pattern would visibly jump at the loop seam; cos/sin
    // are periodic by construction, so this closes the loop exactly.
    float theta = 6.28318530718 * (frame_i / max(num_frames, 1.0));
    // The base circular wander (unchanged) plus a periodic bias toward
    // the shared wind direction (see vfx_engine/weather.py) - ramps
    // 0..movement..0 across the loop via (0.5+0.5*cos(theta)), so it
    // stays exactly closed (frame 0 and frame num_frames land on the
    // identical value) instead of an unbounded one-way drift, the same
    // "closed loop, not infinite drift" pattern every cyclic effect in
    // this pipeline uses. Both terms scaled by this fragment's own
    // gust_at(v_uv) - stronger during a gust, calmer between them, in
    // step with every other wind-driven effect sampling the same
    // envelope (now genuinely per-position, not one
    // frame-global value every fragment shared identically).
    float gust = gust_at(v_uv);
    vec2 drift = gust * (movement * vec2(cos(theta), sin(theta) * 0.6)
        + wind_bias * movement * 0.6 * (0.5 + 0.5 * cos(theta)));

    // Depth-varying noise scale/drift: without this, near
    // and far fog share the exact same wisp scale and drift speed,
    // differing only in the post-multiplied alpha (fog.py's
    // depth_gradient) - the classic "one scrolling texture" look. Real
    // fog has parallax: nearer wisps are coarser (bigger apparent
    // features) and drift more per frame, like any nearer object panning
    // faster; farther wisps are finer and calmer. Sampling the actual
    // depth map to bend the noise-space coordinate ITSELF (not just its
    // final opacity) gives each pixel its own effective wisp scale/speed
    // instead of one flat sheet.
    // texture v is flipped (1.0 - v_uv.y): depth_texture is uploaded
    // directly from the depth array's own row order (row 0 = image top),
    // but OpenGL texture v=0 is the bottom - same flip
    // particles_rain.glsl's base_texture sampling already needs and does.
    float depth = has_depth > 0.5 ? texture(depth_texture, vec2(v_uv.x, 1.0 - v_uv.y)).r : 0.5;
    float local_scale = has_depth > 0.5 ? mix(4.4, 2.2, depth) : 3.0;
    vec2 local_drift = has_depth > 0.5 ? drift * mix(0.4, 1.15, depth) : drift;

    vec2 p = aspect_corrected * local_scale + local_drift;

    // sample through the domain warp instead of raw fbm(p) -
    // see warped_fbm()'s docstring above. This is the single highest-
    // impact fix for fog reading as "a scrolling texture" - it makes the
    // field fold and swirl instead of translating as one rigid pattern.
    float n = warped_fbm(p);
    // Contrast shaping: raw fbm
    // output is fairly evenly spread across its range, which reads as a
    // uniformly hazy field rather than real fog's distinct denser-clump/
    // clearer-gap billowing structure - so smoothstep carves that
    // structure out for free, purely numerically. But a single fixed,
    // fairly narrow smoothstep range (the original 0.32-0.68) produces
    // fairly distinct, separated density islands with a smooth-but-still-
    // visible edge around each one, especially on thin fog where the
    // "islands" read as isolated gray blobs rather than one continuous
    // atmospheric medium. Coupling the contrast range to `density` fixes
    // this without losing the billowing structure for heavy fog: thin fog
    // gets a wide, gentle range (barely more contrast than raw noise, so
    // it reads as soft continuous haze), while heavy fog gets a narrower
    // range than the original default (real, distinct clumping, since
    // there's enough density everywhere that "gaps" won't read as empty
    // holes). density_norm reaches 1.0 (fully "heavy") at density=0.6, a
    // typical heavy-fog config value.
    float density_norm = clamp(density / 0.6, 0.0, 1.0);
    float contrast_lo = mix(0.12, 0.35, density_norm);
    float contrast_hi = mix(0.88, 0.65, density_norm);
    float shaped = smoothstep(contrast_lo, contrast_hi, n);
    float alpha = clamp(shaped * density, 0.0, 1.0);

    // Large-scale horizontal banking: everything above (the
    // fine fbm texture) and everything below (depth's CPU-side alpha
    // multiply in fog.py, and height_factor just below) is a function of
    // either fine noise or "how far up/away" a pixel is - NOTHING here
    // varies primarily by COLUMN. For a typical landscape photo, depth's
    // iso-distance contours run roughly horizontal (farther = higher on
    // screen = where the hills/treeline are), so depth and height_factor
    // both collapse to the same "row-only" structure and multiply into a
    // flat horizontal band no matter how nice the fine noise inside it
    // looks. This adds a second, much-coarser, independently-offset fbm
    // sample (~5-8x lower frequency than local_scale, drifting slower than
    // the fine wisp texture riding inside it) purely to give the density
    // formula a source of large-scale left-right variation it otherwise
    // completely lacks - a few genuinely thicker/thinner drifting masses
    // spread across the frame's width, some columns clearly patchier than
    // others, instead of one uniform band with fine grain on top. The
    // vec2(17.3, -8.1) offset is arbitrary but fixed, chosen only so this
    // sample doesn't correlate with `p`'s own sample point.
    float bank_scale = 0.4;
    vec2 bank_drift = drift * 0.35;
    float bank_noise = fbm(aspect_corrected * bank_scale + bank_drift + vec2(17.3, -8.1));
    float bank_factor = mix(0.35, 1.0, smoothstep(0.25, 0.75, bank_noise));
    alpha *= bank_factor;

    // Vertical stratification: real ground fog/mist settles
    // low and thins with height, independent of near/far depth - v_uv.y=0
    // is the BOTTOM of the final image (see the vertex shader / fog.py's
    // FLIP_TOP_BOTTOM readback), v_uv.y=1 is the top. A power curve keeps
    // most of the frame at full density before curving off near the very
    // top, rather than a linear thin-out starting immediately.
    float height_factor = mix(1.0 - ground_fog_strength, 1.0, pow(1.0 - v_uv.y, 1.6));
    alpha *= height_factor;

    vec3 color = fog_color;

    // Ground-anchoring: everything above depends only on
    // the procedural noise field, the coarse depth gradient, and vertical
    // screen position - none of it has any relationship to the photo's
    // actual content, so fog can read as a translucent sheet floating IN
    // FRONT of the photo rather than something actually sitting in the
    // scene. Real ground fog interacts visibly with what it's sitting in:
    // it picks up a little ambient bounce brightness over already-bright
    // ground, and it pools AROUND solid objects (rocks, fences, treelines)
    // rather than smearing across their edges as if they weren't there.
    // luma_texture is a lightly-blurred (not edge-destroying) grayscale
    // of the base photo - blurred just enough to stop raw photo noise
    // from reading as fake fog texture, while keeping real object edges
    // intact for the edge term below.
    if (has_luma > 0.5) {
        vec2 luma_uv = vec2(v_uv.x, 1.0 - v_uv.y);
        float luma = texture(luma_texture, luma_uv).r;
        float luma_dx = textureOffset(luma_texture, luma_uv, ivec2(1, 0)).r
            - textureOffset(luma_texture, luma_uv, ivec2(-1, 0)).r;
        float luma_dy = textureOffset(luma_texture, luma_uv, ivec2(0, 1)).r
            - textureOffset(luma_texture, luma_uv, ivec2(0, -1)).r;
        float edge_strength = clamp(length(vec2(luma_dx, luma_dy)) * 3.0, 0.0, 1.0);

        // Scoped to "near the ground" via height_factor (already at its
        // full-density value low in frame, reduced high in frame when
        // ground_fog_strength > 0) - the far/high part of frame has no
        // reason to react to ground-level object edges, only the part of
        // frame ground fog actually sits in should.
        float ground_influence = height_factor * ground_texture_strength;
        // Slightly denser/brighter over already-bright ground - real fog
        // picks up ambient bounce light off bright surfaces (sunlit
        // grass, pale stone) more than it does over dark ones.
        alpha *= mix(1.0, 0.75 + 0.55 * luma, ground_influence);
        color = mix(color, color * (0.85 + 0.3 * luma), ground_influence * 0.6);
        // Thinner right along strong existing edges, so the fog reads as
        // pooling AROUND objects and partially revealing their silhouette
        // instead of ignoring them as one flat sheet with zero awareness
        // of what's actually underneath it.
        alpha *= mix(1.0, 1.0 - 0.5 * edge_strength, ground_influence);
    }

    // Fog x light coupling: real haze is brighter and
    // more visibly lit specifically AROUND a light source - that's
    // literally the mechanism that makes god rays visible in the first
    // place - not a flat, position-agnostic tint. light_pos_uv arrives in
    // IMAGE space (y=0 top, see fog.py/god_rays.detect_light_source) -
    // flipped here to match this shader's v_uv convention (v_uv.y=1 top)
    // before computing distance.
    if (has_light_pos > 0.5) {
        vec2 light_gl_uv = vec2(light_pos_uv.x, 1.0 - light_pos_uv.y);
        vec2 light_aspect = light_gl_uv * vec2(resolution.x / resolution.y, 1.0);
        float dist_to_light = length(aspect_corrected - light_aspect);
        float light_radius = 0.55;   // in the same aspect-corrected units as light_aspect/aspect_corrected
        float light_glow = exp(-(dist_to_light * dist_to_light) / (2.0 * light_radius * light_radius));
        // real light scattering through haze isn't a smooth
        // radial gradient centered on the light - it's modulated by the
        // same turbulent density variation the rest of the fog has,
        // brighter where haze happens to be denser along the light's
        // direction rather than a clean radial falloff independent of the
        // noise field. Multiplying by `shaped` (the same post-domain-warp
        // density value computed above) concentrates the glow in the
        // actual denser wisps near the light instead of painting one flat
        // "spotlight" disc on top of the fog - cheap, since `shaped` is
        // already computed. The mix() floor keeps a faint glow even in a
        // momentary gap right at the light position, rather than letting
        // the glow vanish completely and read as a flickering hole.
        light_glow *= mix(0.15, 1.0, shaped);
        color = mix(fog_color, light_tint_color * 1.25, light_glow * light_coupling_strength);
        // Fog also reads as genuinely brighter (not just tinted) right
        // around the light, same as real light-scattering haze.
        alpha = clamp(alpha * mix(1.0, 1.35, light_glow * light_coupling_strength), 0.0, 1.0);
    }

    frag_color = vec4(color, alpha);
}
