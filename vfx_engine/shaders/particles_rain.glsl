// particles_rain.glsl
//
// Dedicated rain shader - GPU-instanced quads, same per-instance
// attribute buffers as particles.glsl (see particles.py), but rain and
// snow need different-enough rendering (streak-vs-bokeh blending,
// refraction, dual-tone lighting for rain; ring bokeh, haze-to-sky-color,
// directional sparkle for snow, see particles_snow.glsl) that branching
// both inside one shared shader stopped being worth it - two dedicated
// files are clearer than one shader full of `if (particle_shape > 0.5)`.
//
// Split into #vertex/#fragment sections, same convention as particles.glsl
// - see gpu.py's split_shader_sections().

#vertex
#version 330

in vec2 in_corner;
in vec2 in_position;
in float in_rotation;
in float in_size;
in float in_alpha;
in float in_blur;         // defocus amount when focus_depth is used (see particles.py) - 0 = sharp, 1 = far out of focus
in float in_brightness;
in float in_stretch;      // elongation along the fall axis - see particles.py's velocity_stretch
in float in_shape_seed;   // per-instance: 0.0-1.0, hashed in the fragment shader into a per-drop
                          // curvature/core-offset/width variant - see
                          // particles.py's shape_seed
in float in_depth;        // per-drop persistent distance, 0.0 = far .. 1.0 = near (same convention as the scene depth map)

uniform vec2 viewport_size;
uniform float edge_fade;

out vec2 v_corner;
out float v_alpha;
out float v_blur;
out float v_brightness;
out vec2 v_screen_uv;
out float v_stretch;
out float v_shape_seed;
out float v_depth;

void main() {
    v_depth = in_depth;
    vec2 corner = in_corner;
    corner.y *= (1.0 + in_stretch);
    v_stretch = in_stretch;

    float c = cos(in_rotation);
    float s = sin(in_rotation);
    vec2 rotated_corner = vec2(
        corner.x * c - corner.y * s,
        corner.x * s + corner.y * c
    );
    vec2 world_pos = in_position + rotated_corner * in_size;

    vec2 ndc = (world_pos / viewport_size) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    gl_Position = vec4(ndc, 0.0, 1.0);
    v_corner = corner * 2.0;
    v_screen_uv = in_position / viewport_size;

    // Fade near ALL FOUR frame edges (not just top/bottom) - hides the
    // "teleport" pop when a particle's mod-wrapped center jumps from one
    // edge to the other in a single frame. The fade band scales with the
    // particle's own size (in_size), not just a fixed fraction of the
    // viewport - a large out-of-focus bokeh disc is much bigger than the
    // old fixed ~3%-of-height band, so a fixed band clipped it visibly;
    // scaling by size keeps the fade proportional to how big the disc
    // actually is.
    float fade = 1.0;
    if (edge_fade > 0.5) {
        float edge_band = max(viewport_size.y * 0.03, in_size * 2.5);
        float top_fade = smoothstep(0.0, edge_band, world_pos.y);
        float bottom_fade = smoothstep(0.0, edge_band, viewport_size.y - world_pos.y);
        float left_fade = smoothstep(0.0, edge_band, world_pos.x);
        float right_fade = smoothstep(0.0, edge_band, viewport_size.x - world_pos.x);
        fade = top_fade * bottom_fade * left_fade * right_fade;
    }

    v_alpha = in_alpha * fade;
    v_blur = in_blur;
    v_brightness = in_brightness;
    v_shape_seed = in_shape_seed;
}

#fragment
#version 330

in vec2 v_corner;
in float v_alpha;
in float v_blur;
in float v_brightness;
in vec2 v_screen_uv;
in float v_stretch;
in float v_shape_seed;
in float v_depth;

uniform sampler2D depth_texture;      // scene depth (0 far .. 1 near), single channel
uniform float occlusion_strength;     // 0.0 = drops ignore scene depth; 1.0 = drops behind a nearer surface are fully hidden
uniform vec3 particle_color;
uniform float atmosphere_enabled;
uniform sampler2D base_texture;
uniform float haze_strength;
uniform vec3 haze_color;          // sky/horizon color to blend far drops toward - see particles.py's sample_sky_color
uniform float backlight_strength;
uniform float shadow_strength;       // symmetric counterpart to backlight_strength - see particles.glsl's matching uniform comment
uniform float refraction_strength;   // 0.0 = off. >0.0 = drops bend the background behind them like real water
uniform vec2 light_pos_uv;           // normalized (0..1) screen position of the photo's detected dominant
                                      // light source - see vfx_engine/god_rays.py's detect_light_source() and
                                      // main.py's resolve_lighting_kwargs. (0.5, 0.0) (top-center) if unset.
uniform float has_light_pos;         // 1.0 if light_pos_uv was actually provided this render, 0.0 = use the old fixed "light comes from above" assumption
uniform sampler2D ambient_tint_texture; // tiny (e.g. 12x8) per-photo color grid, GL_LINEAR-filtered so
                                         // sampling it performs the "bilinear-interpolate between the 4 nearest grid
                                         // cells" lookup in hardware. See particles.py's build_ambient_tint_grid.
uniform float has_ambient_tint_grid;    // 1.0 if ambient_tint_texture was actually uploaded this render, 0.0 = fall
                                         // back to the old flat particle_color uniform for every drop regardless of
                                         // where it lands on screen

out vec4 frag_color;

// Cheap 1D hash - see particles.glsl's matching helper/comment.
float hash11(float p) {
    return fract(sin(p * 127.1) * 43758.5453123);
}

void main() {
    // each drop picks up the LOCAL color of the photo region
    // it's falling in front of, rather than every drop in the frame
    // sharing one global tint - a drop over a bright sunlit patch and one
    // over a dark shadowed patch shouldn't read as identical colors.
    vec3 base_color = particle_color;
    if (has_ambient_tint_grid > 0.5) {
        base_color = texture(ambient_tint_texture, vec2(v_screen_uv.x, 1.0 - v_screen_uv.y)).rgb;
    }
    vec3 color = base_color * v_brightness;

    // Per-particle shape variance: several
    // decorrelated values hashed from the single v_shape_seed input -
    // real rain streaks (or real motion-blur trails) have slightly
    // irregular width along their length and aren't perfectly
    // axis-symmetric; one mathematically perfect taper repeated
    // identically across hundreds of drops is a big part of what makes
    // rain read as "obviously the same primitive" up close.
    float kink_strength = (hash11(v_shape_seed + 11.0) - 0.5) * 0.5;   // per-drop bend direction/magnitude
    float core_offset = (hash11(v_shape_seed + 23.0) - 0.5) * 0.3;     // highlight core off-center, not dead-center
    float width_variant = mix(0.85, 1.2, hash11(v_shape_seed + 37.0)); // per-drop aspect-ratio variant

    float length_dist_raw = abs(v_corner.y) / (1.0 + v_stretch);
    // A gentle per-particle S-curve along the streak's length (not a
    // straight shear) - bends the WIDTH sample coordinate as a function
    // of position along the LENGTH, so the taper isn't perfectly straight
    // the way a pure smoothstep of a fixed axis always is.
    float kink = kink_strength * sin(clamp(length_dist_raw, 0.0, 1.0) * 3.14159265);
    float kinked_x = v_corner.x - kink;

    // Un-stretch v_corner.y back to a -1..1-ish range for an isotropic
    // (round) metric - see particles.glsl's matching comment for why
    // this has to divide by (1 + v_stretch).
    vec2 disc = vec2(kinked_x, v_corner.y / (1.0 + v_stretch));
    float disc_dist = length(disc);

    // Streak profile: crisp-ish width, soft-fading length - a real
    // in-focus, fast-moving drop.
    float width_dist = abs(kinked_x) * width_variant * 1.7;
    float length_dist = length_dist_raw;
    // Soft, tapered profile: a gaussian-like falloff across the width (no
    // hard-edged stick) and a long feathered taper along the length, so a
    // drop reads as a motion-blurred sliver of water, thicker in the
    // middle and fading out at both ends.
    float width_alpha = smoothstep(1.0, 0.15, width_dist);
    width_alpha *= width_alpha;
    float length_alpha = smoothstep(1.0, 0.25, length_dist);
    float streak_alpha = width_alpha * length_alpha * 0.85;
    // The bright highlight core sits off-center per drop (core_offset),
    // not dead-center on every single one - a real curved drop's specular
    // catch isn't perfectly centered on its own silhouette either.
    float streak_core = 1.0 - smoothstep(0.0, 0.45, abs(kinked_x - core_offset));
    // Real rain catches sky light, so it reads lighter than dark surroundings
    // (forest, water) - lift toward a pale sky-blue instead of staying
    // the (dark) ambient tint of whatever is behind it.
    color = mix(color, vec3(0.80, 0.86, 0.94), 0.3);
    vec3 streak_color = color + vec3(0.9, 0.95, 1.0) * streak_core * 0.18 * length_alpha;

    // Round bokeh-disc profile: soft core + gentle outer glow - a drop
    // far enough from the focus plane that it's no longer resolving as a
    // sharp line, just a soft translucent blob (real out-of-focus points).
    // Reuses the same irregular-lobe bump particles.glsl/particles_snow.glsl
    // apply to round shapes - a defocused drop is, at that
    // point, just another round particle, and deserves the same
    // circular-symmetry fix.
    float lobe_count = 3.0 + floor(hash11(v_shape_seed + 5.0) * 4.0);
    float shape_phase = v_shape_seed * 6.2831853;
    float disc_angle = atan(disc.y, disc.x);
    float disc_bump = 1.0
        + 0.14 * cos(disc_angle * lobe_count + shape_phase)
        + 0.07 * cos(disc_angle * (lobe_count + 2.0) - shape_phase * 1.7);
    float bumped_disc_dist = disc_dist / disc_bump;
    float disc_core = 1.0 - smoothstep(0.0, 0.5, bumped_disc_dist);
    float disc_glow = pow(clamp(1.0 - bumped_disc_dist, 0.0, 1.0), 1.6);
    float disc_alpha = max(disc_core * 0.55, disc_glow);

    // Blend between the two based on defocus (v_blur) - sharp/near-focus
    // drops stay thin streaks, far-out-of-focus ones relax into bokeh.
    // Rain never turns into a full round blob (that read as floating bubbles
    // rather than falling water): even heavily defocused drops stay
    // elongated, only slightly softened toward a disc.
    float bokeh_mix = smoothstep(0.55, 1.0, v_blur) * 0.3;
    float shape_alpha = mix(streak_alpha, disc_alpha, bokeh_mix);
    color = mix(streak_color, color, bokeh_mix);

    // Dual-tone ambient lighting: a warm highlight on the side facing the
    // light, a cool shadow tone on the opposite side - real drops aren't
    // flat-lit, they pick up two-tone light from their curved surface.
    // The light direction is computed PER-PARTICLE (from this particle's
    // own screen position toward the photo's actual detected light
    // source, see light_pos_uv above) rather than a fixed "light always
    // comes from directly above" assumption - a drop on the left of the
    // frame with a sunset light source on the right should catch its
    // highlight on its right-facing side, not its top.
    vec2 to_light = has_light_pos > 0.5 ? normalize(light_pos_uv - v_screen_uv) : vec2(0.0, -1.0);
    float light_facing = clamp(dot(disc, to_light) * 1.6, -1.0, 1.0);
    vec3 warm_tone = vec3(1.08, 1.02, 0.9);
    vec3 cool_tone = vec3(0.88, 0.93, 1.05);
    color *= mix(cool_tone, warm_tone, light_facing * 0.5 + 0.5);

    if (refraction_strength > 0.0) {
        // Refract the background through the drop - a real raindrop is a
        // tiny lens, it doesn't just sit as a flat tinted shape over
        // whatever's behind it. `disc` (the drop's own local coordinate)
        // stands in for a surface normal: pixels get pulled toward/away
        // from the drop's center, bending the sampled background like
        // light through curved water.
        vec2 uv = vec2(v_screen_uv.x, 1.0 - v_screen_uv.y);
        vec2 refract_uv = clamp(uv + disc * refraction_strength * 0.01, 0.0, 1.0);
        vec3 refracted = texture(base_texture, refract_uv).rgb;
        color = mix(color, refracted * 1.15, clamp(refraction_strength, 0.0, 0.85) * (0.35 + 0.65 * bokeh_mix));
    }

    if (atmosphere_enabled > 0.5) {
        // Blend toward the SKY color, not the exact pixel behind the
        // drop - blending into whatever's literally behind it (e.g. dark
        // trees) turns far drops into dark silhouettes instead of haze.
        color = mix(color, haze_color, v_blur * haze_strength);
        // the color mix above alone doesn't make a
        // hazy drop read as receding into mist - it just makes an
        // otherwise identically crisp, fully-opaque streak a paler color,
        // which at normal opacities still shows up as a distinct thin
        // dark-ish line against a light sky. Real rain genuinely fades in
        // opacity as it recedes into haze, not just in hue - so the same
        // v_blur * haze_strength term that drives the color blend also
        // drains shape_alpha toward a faint remnant (0.25 floor, not 0 -
        // a heavily-hazed drop should still be barely there, not vanish
        // outright and create a hole in the rain field).
        shape_alpha *= mix(1.0, 0.5, v_blur * haze_strength);
    }

    if (backlight_strength > 0.0 || shadow_strength > 0.0) {
        vec3 background = texture(base_texture, vec2(v_screen_uv.x, 1.0 - v_screen_uv.y)).rgb;
        float background_luminance = dot(background, vec3(0.299, 0.587, 0.114));

        if (backlight_strength > 0.0) {
            float backlight_boost = clamp((background_luminance - 0.55) * 2.0, 0.0, 1.0) * backlight_strength;
            color += vec3(1.0, 0.98, 0.92) * backlight_boost;
        }

        if (shadow_strength > 0.0) {
            // Symmetric counterpart to backlight above. Mirrored below backlight's 0.55 midpoint at
            // 0.45, leaving ordinary midtone backgrounds untouched.
            float shadow_dimming = clamp((0.45 - background_luminance) * 2.0, 0.0, 1.0) * shadow_strength;
            color = mix(color, color * vec3(0.55, 0.6, 0.68), shadow_dimming);
        }
    }

    if (occlusion_strength > 0.0) {
        // Occlusion by real scene geometry: if the scene surface under this
        // pixel is nearer than the drop's own distance, the drop is behind
        // it (e.g. a far drop behind a foreground tree) and fades out.
        float scene_near = texture(depth_texture, vec2(v_screen_uv.x, 1.0 - v_screen_uv.y)).r;
        float occluded = smoothstep(0.02, 0.16, scene_near - v_depth);
        shape_alpha *= 1.0 - occlusion_strength * occluded;
    }

    // 0.6: motion-blur averaging is now alpha-weighted (no longer dims streaks), so rain needs this to keep its earlier strength
    frag_color = vec4(color, shape_alpha * v_alpha * 0.6);
}
