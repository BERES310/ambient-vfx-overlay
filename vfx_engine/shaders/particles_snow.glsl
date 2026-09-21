// particles_snow.glsl
//
// Dedicated snow shader - see particles_rain.glsl's header comment for
// why rain and snow got split into separate shader files instead of one
// shared file with branches.

#vertex
#version 330

in vec2 in_corner;
in vec2 in_position;
in float in_rotation;
in float in_size;
in float in_alpha;
in float in_blur;         // defocus amount when focus_depth is used - 0 = sharp, 1 = far out of focus
in float in_brightness;
in float in_stretch;      // snow uses this too (velocity_stretch defaults to 0 for round flakes, but stays generic)
in float in_shape_seed;   // per-instance: 0.0-1.0, hashed in the fragment shader into a distinct
                          // per-flake silhouette - see particles.py's shape_seed

uniform vec2 viewport_size;
uniform float edge_fade;

out vec2 v_corner;
out float v_alpha;
out float v_blur;
out float v_brightness;
out vec2 v_screen_uv;
out float v_shape_seed;

void main() {
    vec2 corner = in_corner;
    corner.y *= (1.0 + in_stretch);

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

    // Fade near all four frame edges, band scaled by the flake's own
    // size - see particles_rain.glsl's matching comment: a fixed-size
    // fade band clips large out-of-focus bokeh discs visibly when they
    // cross it, since they're much bigger than the band itself.
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
in float v_shape_seed;

uniform vec3 particle_color;
uniform float atmosphere_enabled;
uniform sampler2D base_texture;
uniform float haze_strength;
uniform vec3 haze_color;         // sky/horizon color - see particles.py's sample_sky_color
uniform float backlight_strength;
uniform float shadow_strength;   // symmetric counterpart to backlight_strength - see particles.glsl's matching uniform comment
uniform float sparkle_strength;
uniform vec2 light_pos_uv;       // normalized (0..1) screen position of the photo's detected
                                  // dominant light source, mirroring particles_rain.glsl's own uniform. (0.5, 0.0)
                                  // (top-center) if unset.
uniform float has_light_pos;     // 1.0 if light_pos_uv was actually provided this render, 0.0 = fall back to
                                  // the old fixed "light comes from above" assumption

out vec4 frag_color;

// Cheap 1D hash - see particles.glsl's matching helper/comment.
float hash11(float p) {
    return fract(sin(p * 127.1) * 43758.5453123);
}

void main() {
    // Break circular symmetry: in_rotation is already
    // computed every frame from wind direction, but rotating a perfect
    // circle produces no visible change - a few angle-dependent lobes
    // (hashed from v_shape_seed, so every flake gets its own bump
    // pattern/orientation) finally makes that rotation do visible work,
    // and reads as "flake" instead of "dot" even at a glance. v_corner is
    // this flake's own LOCAL (pre-rotation) frame, but the whole quad is
    // rotated on-screen via in_rotation, so a shape defined here rotates
    // right along with it.
    float lobe_count = 3.0 + floor(hash11(v_shape_seed + 5.0) * 4.0);   // 3..6 lobes
    float shape_phase = v_shape_seed * 6.2831853;
    float angle = atan(v_corner.y, v_corner.x);
    // Out-of-focus flakes are lens discs, not crystals: the lobes fade out
    // with defocus so big near flakes read as round bokeh, not lumpy blobs.
    float lobe_amount = 1.0 - 0.8 * smoothstep(0.35, 0.85, v_blur);
    float bump = 1.0
        + lobe_amount * (0.14 * cos(angle * lobe_count + shape_phase)
        + 0.07 * cos(angle * (lobe_count + 2.0) - shape_phase * 1.7));
    float dist = length(v_corner) / bump;

    // Soft radial glow for in-focus flakes (bright core fading smoothly
    // to nothing - see particles.glsl's original comment for why this
    // beats a flat disc).
    float core_radius = mix(0.55, 0.3, v_blur);
    float core = 1.0 - smoothstep(0.0, core_radius, dist);
    float glow = pow(clamp(1.0 - dist, 0.0, 1.0), mix(1.3, 2.2, v_blur));
    float soft_shape = max(core, glow);

    // Ring-bokeh profile for far-out-of-focus flakes: real out-of-focus
    // points from a lens don't stay a soft gaussian blob as they get
    // bigger - they flatten into a mostly-solid disc with a distinctly
    // brighter rim (the classic "bokeh ring" look). The disc itself must
    // stay solid/opaque (not just the rim) - a rim-heavy, dim-centered
    // alpha reads as a hollow ring/bubble, not a snowflake.
    float flat_disc = 1.0 - smoothstep(0.8, 0.98, dist);
    float rim = smoothstep(0.6, 0.82, dist) * (1.0 - smoothstep(0.9, 1.0, dist));
    float ring_shape = flat_disc;   // alpha stays a solid disc - rim brightens color only, see below

    float defocus_mix = smoothstep(0.35, 0.85, v_blur);
    float shape_alpha = mix(soft_shape, ring_shape, defocus_mix);

    // Snow is white: depth-driven brightness only nudges it (a full multiply
    // turned far/big flakes dull grey against the bright overcast sky).
    vec3 color = particle_color * mix(1.0, v_brightness, 0.3);
    color += vec3(1.0) * rim * defocus_mix * 0.3;   // visibly brighten the bokeh ring itself

    // dual-tone ambient lighting, ported from
    // particles_rain.glsl's own hook - a warm tint on the side of the
    // flake facing the photo's actual detected light source, a cool tint
    // on the opposite side, computed per-particle from this flake's own
    // screen position toward light_pos_uv (see particles_rain.glsl's
    // matching comment for why that beats a fixed "light from above"
    // assumption). v_corner is this flake's own local (pre-rotation)
    // frame - the same coordinate the lobe-shape bump above already uses -
    // so the highlight rotates WITH the flake instead of staying
    // screen-locked. Snow scatters light far more diffusely than a wet
    // raindrop's near-mirror curved surface, so both the falloff (lower
    // dot-product multiplier - a broader, softer transition) and the tint
    // amplitude here are deliberately weaker than rain's tighter,
    // higher-contrast specular-like hook: a soft glow across the whole
    // flake, not a sharp two-tone split.
    vec2 to_light = has_light_pos > 0.5 ? normalize(light_pos_uv - v_screen_uv) : vec2(0.0, -1.0);
    float light_facing = clamp(dot(v_corner, to_light) * 0.9, -1.0, 1.0);
    vec3 warm_tone = vec3(1.05, 1.015, 0.94);
    vec3 cool_tone = vec3(0.95, 0.97, 1.03);
    color *= mix(cool_tone, warm_tone, light_facing * 0.5 + 0.5);

    if (sparkle_strength > 0.0) {
        // A small 4-point cross glint, biased toward particles near the
        // top of the frame (closer to sky/key light) - real glints
        // aren't uniformly likely across the whole image, they cluster
        // wherever the light source actually is.
        float directional_bias = mix(0.4, 1.3, clamp(1.0 - v_screen_uv.y, 0.0, 1.0));
        float sparkle_amount = sparkle_strength * directional_bias * clamp((v_brightness - 1.0) * 4.0, 0.0, 1.0);
        float cross_h = clamp(1.0 - abs(v_corner.y) * 14.0, 0.0, 1.0) * clamp(1.0 - abs(v_corner.x) * 0.7, 0.0, 1.0);
        float cross_v = clamp(1.0 - abs(v_corner.x) * 14.0, 0.0, 1.0) * clamp(1.0 - abs(v_corner.y) * 0.7, 0.0, 1.0);
        float sparkle = max(cross_h, cross_v) * sparkle_amount;
        shape_alpha = max(shape_alpha, sparkle * 0.85);
        color += vec3(1.0) * sparkle * 0.6;
    }

    if (atmosphere_enabled > 0.5) {
        // Blend toward the SKY color, not whatever's literally behind
        // the flake - blending into the actual background pixel turns a
        // far flake in front of dark trees into a dark green silhouette
        // instead of atmospheric haze.
        color = mix(color, haze_color, v_blur * haze_strength);
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
            // Snow is lit by the whole overcast sky, not one key light, so a
            // flake in front of dark trees/hills stays clearly lighter than
            // them - the dimming here is only a mild tint (rain keeps its
            // harder 0.55-0.68 dimming), otherwise big near flakes over a
            // dark hillside read as dark grey blobs against the bright sky.
            color = mix(color, color * vec3(0.82, 0.85, 0.9), shadow_dimming);
        }
    }

    frag_color = vec4(color, shape_alpha * v_alpha);
}
