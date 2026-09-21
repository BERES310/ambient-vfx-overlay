// particles.glsl
//
// Renders many small particles as GPU-instanced quads - used for snow,
// rain, sparks, fireflies and dust. One small quad (4 vertices, drawn as a
// TRIANGLE_STRIP) is instanced once per particle; per-instance attributes
// (position, rotation, size, alpha, blur, brightness, stretch) come from
// buffers written by either particles_wind.comp (GPU path) or particles.py's
// numpy fallback (CPU path) - see particles.py.
//
// This used to be rendered with gl_PointSize point sprites, but those are
// unreliable across GPU drivers (inconsistent max size, some drivers
// ignore gl_PointSize unless GL_PROGRAM_POINT_SIZE is enabled - which is
// exactly the bug that made snow/rain invisible before). Real geometry
// (instanced quads) has none of those caveats and also lets each particle
// rotate to face its motion/wind direction, and stretch along it.
//
// This file has two sections, split by the "#vertex"/"#fragment" markers
// below - vfx_engine/gpu.py splits on those markers and compiles each half
// separately (a plain text convention we made up for this project, not a
// real GLSL feature).

#vertex
#version 330

in vec2 in_corner;      // per-vertex (4 verts, TRIANGLE_STRIP): quad corner, -0.5..0.5
in vec2 in_position;    // per-instance: particle center, pixel coordinates (0,0 = top-left)
in float in_rotation;   // per-instance: radians
in float in_size;       // per-instance: quad half-width in pixels
in float in_alpha;      // per-instance: 0.0-1.0
in float in_blur;       // per-instance: 0.0 (sharp, close to camera) - 1.0 (soft, far away) -
                         // also reused as an atmospheric-perspective depth proxy in the
                         // fragment shader (farther/blurrier particles haze more)
in float in_brightness; // per-instance: brightness multiplier (~0.85-1.15) - per-particle
                         // variance so particles aren't identically-lit stamped copies
                         // of each other, see particles.py's brightness_jitter
in float in_stretch;    // per-instance: 0.0 = round particle, >0.0 elongates the quad along
                         // its local (pre-wind-rotation) fall axis, proportional to the
                         // particle's own fall speed - a real velocity-streak instead of
                         // the old fixed ghost-copy trail trick, see particles.py
in float in_shape_seed; // per-instance: 0.0-1.0, hashed in the fragment shader into 2-4+
                         // distinct silhouette variants per particle instead of one fixed
                         // shape formula shared identically by every particle - see
                         // particles.py's shape_seed and the fragment shader below.

uniform vec2 viewport_size;
uniform float edge_fade;   // 1.0 = fade particles near the top/bottom edges (hides the
                           // "pop" of a particle wrapping to the opposite edge when it
                           // loops), 0.0 = disabled - see particles.py's edge_fade param

out vec2 v_corner;
out float v_alpha;
out float v_blur;
out float v_brightness;
out vec2 v_screen_uv;
out float v_stretch;
out float v_shape_seed;

void main() {
    vec2 corner = in_corner;
    corner.y *= (1.0 + in_stretch);   // stretch along the local fall axis before rotating
    v_stretch = in_stretch;   // so the fragment shader can un-stretch v_corner.y back to -1..1 for its falloff math

    float c = cos(in_rotation);
    float s = sin(in_rotation);
    vec2 rotated_corner = vec2(
        corner.x * c - corner.y * s,
        corner.x * s + corner.y * c
    );
    vec2 world_pos = in_position + rotated_corner * in_size;

    // pixel coordinates -> normalized device coordinates (-1..1), flipping Y
    // because image coordinates count down from the top, OpenGL counts up.
    vec2 ndc = (world_pos / viewport_size) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    gl_Position = vec4(ndc, 0.0, 1.0);
    v_corner = corner * 2.0;   // rescale to -1..1 for the fragment shader's distance-from-center test
    v_screen_uv = in_position / viewport_size;   // for sampling the base-image/depth textures, see fragment shader

    // Position-based (not time-based) fade near the top/bottom edges, so
    // it's automatically seamless-loop-safe regardless of loop length -
    // a particle wrapping from y=height back to y=0 fades out just before
    // the wrap and fades back in just after, instead of popping at full
    // opacity right at the seam.
    float fade = 1.0;
    if (edge_fade > 0.5) {
        float edge_band = viewport_size.y * 0.03;   // outer ~3% of the frame height
        float top_fade = smoothstep(0.0, edge_band, world_pos.y);
        float bottom_fade = smoothstep(0.0, edge_band, viewport_size.y - world_pos.y);
        fade = top_fade * bottom_fade;
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

uniform vec3 particle_color;
uniform float atmosphere_enabled;   // 1.0 = blend far particles toward the scene's background
                                     // color behind them (atmospheric perspective), 0.0 = off
uniform sampler2D base_texture;     // the actual photo, sampled behind each particle
uniform float haze_strength;        // 0.0-1.0, how strongly far particles blend into the background
uniform float particle_shape;       // 0.0 = round soft glow (snow), 1.0 = thin tapered streak (rain) -
                                     // see particles.py's `shape` param. Without this, rain was just
                                     // a stretched version of the same flat circular blob snow uses,
                                     // which is exactly why they read as near-identical.
uniform float sparkle_strength;     // 0.0-1.0, round-shape only: adds an occasional cross-shaped
                                     // glint/bokeh highlight, brighter particles sparkle more -
                                     // real snow/dust catches light like this, a flat disk never does.
uniform float backlight_strength;   // 0.0-1.0, brightens a particle when whatever's directly behind
                                     // it (sampled from base_texture) is itself bright - a rough
                                     // approximation of real forward light scattering (rain/snow
                                     // glow more against a bright sky/light source than a dark one).
uniform float shadow_strength;      // 0.0-1.0, symmetric counterpart to backlight_strength - darkens
                                     // a particle when whatever's directly behind it is itself dark
                                     // (deep shade/shadow), fixing the one-directional "brighten only"
                                     // model that let particles render lighter than the shadow behind
                                     // them.

out vec4 frag_color;

// Cheap 1D hash - the building block for this shader's per-particle shape
// variance: several decorrelated values derived from the
// single v_shape_seed input by hashing it with different offsets.
float hash11(float p) {
    return fract(sin(p * 127.1) * 43758.5453123);
}

void main() {
    vec3 color = particle_color * v_brightness;
    float shape_alpha;

    if (particle_shape > 0.5) {
        // Thin streak: width and length are faded INDEPENDENTLY instead
        // of by radial distance - a radial falloff on a heavily elongated
        // quad just gives a blurry oval (indistinguishable from a
        // stretched snow blob), not a real streak. Width stays crisp-ish
        // (rain drops have a fairly defined edge at this scale); length
        // fades softly at both ends, like a real motion-blurred streak
        // trailing off instead of a hard-capped line.
        // Per-particle shape variance - see
        // particles_rain.glsl's matching (more heavily-used) copy of this
        // exact formula for the full explanation.
        float kink_strength = (hash11(v_shape_seed + 11.0) - 0.5) * 0.5;
        float core_offset = (hash11(v_shape_seed + 23.0) - 0.5) * 0.3;
        float width_variant = mix(0.85, 1.2, hash11(v_shape_seed + 37.0));

        // v_corner.y was pre-stretched by (1 + in_stretch) in the vertex
        // shader (see v_stretch), so it ranges up to +-(1+stretch), not
        // +-1 - un-stretch it back to a 0..1-ish range here, otherwise
        // this falloff (written assuming a +-1 range) only ever lights up
        // a tiny sliver near the center and the visible result barely
        // looks stretched at all despite the underlying geometry being
        // genuinely elongated.
        float length_dist = abs(v_corner.y) / (1.0 + v_stretch);
        float kink = kink_strength * sin(clamp(length_dist, 0.0, 1.0) * 3.14159265);
        float kinked_x = v_corner.x - kink;
        float width_dist = abs(kinked_x) * width_variant;
        float width_alpha = smoothstep(1.0, mix(0.35, 0.6, v_blur), width_dist);
        float length_alpha = smoothstep(1.0, 0.55, length_dist);
        shape_alpha = width_alpha * length_alpha;

        // A faint brighter core, offset per-particle rather than dead-
        // centered on every one - like a highlight catching the curved
        // surface of a falling drop.
        float core = 1.0 - smoothstep(0.0, 0.45, abs(kinked_x - core_offset));
        color += vec3(0.9, 0.95, 1.0) * core * 0.18 * length_alpha;
    } else {
        // Round soft glow: a true radial falloff (bright core fading
        // smoothly to nothing) instead of a flat disk with only its edge
        // antialiased - this alone is most of what makes real snow/dust
        // read as glowing points rather than pasted-on stickers.
        // A pure pow(1-dist, N) falloff dims fast well inside the nominal
        // radius (e.g. at N=2, half-radius is already down to 25%
        // alpha), so at the same nominal size it reads as visibly smaller
        // and dimmer than the old flat-disk-with-edge-AA shape did - a
        // real regression in visibility, not just a look change. Keep a
        // solid, mostly-opaque core (via the inner smoothstep) and only
        // apply the soft glow falloff in the outer portion, so it still
        // looks like a glowing point rather than a flat sticker, without
        // shrinking the particle's effective visible size.
        // Break circular symmetry: every round particle used
        // to be a perfectly radially-symmetric length(v_corner), so
        // in_rotation (already computed every frame from wind direction)
        // rotated a circle - no visible change. A few angle-dependent
        // lobes (a cheap irregular-polygon silhouette, 3-6 per particle,
        // phase/count both hashed from v_shape_seed so no two particles
        // share the same bump pattern) makes that rotation finally do
        // visible work, and reads immediately as "flake"/"drop" instead of
        // "dot". v_corner is in the particle's own LOCAL (pre-rotation)
        // frame, but since the whole quad is rotated on-screen via
        // in_rotation, a shape defined in this local frame rotates right
        // along with it.
        float lobe_count = 3.0 + floor(hash11(v_shape_seed + 5.0) * 4.0);   // 3..6 lobes
        float shape_phase = v_shape_seed * 6.2831853;
        float angle = atan(v_corner.y, v_corner.x);
        float bump = 1.0
            + 0.14 * cos(angle * lobe_count + shape_phase)
            + 0.07 * cos(angle * (lobe_count + 2.0) - shape_phase * 1.7);
        float dist = length(v_corner) / bump;
        float core_radius = mix(0.55, 0.3, v_blur);   // far/blurry flakes: smaller solid core, more glow
        float core = 1.0 - smoothstep(0.0, core_radius, dist);
        float glow = pow(clamp(1.0 - dist, 0.0, 1.0), mix(1.3, 2.2, v_blur));
        shape_alpha = max(core, glow);

        if (sparkle_strength > 0.0) {
            // A small 4-point cross glint, like light catching a facet -
            // stronger on brighter (per-particle jittered, see
            // particles.py's brightness_jitter) flakes, so only some of
            // them visibly sparkle rather than all of them uniformly.
            float sparkle_amount = sparkle_strength * clamp((v_brightness - 1.0) * 4.0, 0.0, 1.0);
            float cross_h = clamp(1.0 - abs(v_corner.y) * 14.0, 0.0, 1.0) * clamp(1.0 - abs(v_corner.x) * 0.7, 0.0, 1.0);
            float cross_v = clamp(1.0 - abs(v_corner.x) * 14.0, 0.0, 1.0) * clamp(1.0 - abs(v_corner.y) * 0.7, 0.0, 1.0);
            float sparkle = max(cross_h, cross_v) * sparkle_amount;
            shape_alpha = max(shape_alpha, sparkle * 0.85);
            color += vec3(1.0) * sparkle * 0.6;
        }
    }

    if (atmosphere_enabled > 0.5 || backlight_strength > 0.0 || shadow_strength > 0.0) {
        vec3 background = texture(base_texture, vec2(v_screen_uv.x, 1.0 - v_screen_uv.y)).rgb;

        if (atmosphere_enabled > 0.5) {
            // v_blur already encodes "how far away this particle is" (1.0
            // = farthest, see particles.py); farther particles desaturate/
            // blend toward whatever's actually behind them in the photo
            // instead of keeping a flat saturated color - this is what
            // sells distance, rather than just shrinking and dimming.
            color = mix(color, background, v_blur * haze_strength);
        }

        if (backlight_strength > 0.0 || shadow_strength > 0.0) {
            float background_luminance = dot(background, vec3(0.299, 0.587, 0.114));

            if (backlight_strength > 0.0) {
                // Forward scattering: real rain/snow glows brighter when
                // there's a bright light source or sky directly behind it -
                // approximated here by boosting brightness with the
                // background's own luminance sampled at this particle's
                // position, instead of every particle being lit identically
                // regardless of what's actually behind it.
                float backlight_boost = clamp((background_luminance - 0.55) * 2.0, 0.0, 1.0) * backlight_strength;
                color += vec3(1.0, 0.98, 0.92) * backlight_boost;
            }

            if (shadow_strength > 0.0) {
                // Symmetric counterpart to backlight above: backlight only ever brightens a particle;
                // nothing previously darkened one sitting in front of a
                // genuinely dark region, so a mid-brightness particle
                // straight-alpha-over composited onto deep shade always
                // rendered lighter than that shade. This pulls a
                // particle's own color DOWN (a cool, slightly desaturated
                // shadow tint, not pure black - it's still catching some
                // ambient light) proportional to how dark the sampled
                // background actually is, mirrored below backlight's own
                // 0.55 midpoint at 0.45 so ordinary midtone backgrounds are
                // left alone (neither brightened nor darkened) - a real
                // two-sided local exposure match instead of one-directional.
                float shadow_dimming = clamp((0.45 - background_luminance) * 2.0, 0.0, 1.0) * shadow_strength;
                color = mix(color, color * vec3(0.55, 0.6, 0.68), shadow_dimming);
            }
        }
    }

    frag_color = vec4(color, shape_alpha * v_alpha);
}
