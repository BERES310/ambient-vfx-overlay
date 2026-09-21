// region_warp.glsl
//
// Image-space displacement warp: resamples a photo's OWN pixels through a
// wind-driven UV offset field, within a masked region - the shared GPU
// engine behind vegetation.py's canopy sway and cloud.py's sky drift (see
// vfx_engine/scene_warp.py's render_region_warp). A single photo has no
// 3D geometry to actually simulate wind-blown motion on, so "wind moves
// this region" is approximated as "re-sample this region's existing
// pixels through a smoothly-varying offset field" instead - conceptually
// closer to a vertex/UV displacement than to particle rendering, but
// reuses the exact same curl-noise wind field math as
// particles_wind.comp/blur.glsl (see those files' docstrings for the full
// explanation of curl noise and the loop-safety trick), so the whole
// scene reads as one consistently-windy place instead of independent
// unrelated animations.
//
// Split into #vertex/#fragment sections, same convention as every other
// .glsl file here - see gpu.py's split_shader_sections().

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

uniform sampler2D base_texture;     // the photo itself - RGB, uploaded top-down like particles.py's atmosphere texture
uniform sampler2D region_texture;   // R = visibility/alpha mask (0..1), G = per-pixel displacement amplitude weight (0..1) - see scene_warp.py
uniform sampler2D depth_texture;    // 0.0 (far) - 1.0 (near), same convention as blur.glsl/fog.py
uniform float has_depth;            // 1.0 if depth_texture holds real depth data, 0.0 = none available (see scene_warp.py) -
                                     // a 1x1 dummy texture is still bound either way, same pattern as fog.py's depth_texture
uniform vec2 resolution;
uniform float frame_i;
uniform float num_frames;       // total frames in the loop - see the loop-safety note below
uniform float noise_scale;      // spatial scale of the main wind field (smaller = broader gusts)
uniform float noise_loop_radius;   // how much the wind noise varies over one loop (0 = static wind)
uniform float drift_strength;   // overall displacement magnitude, in pixels, at weight=1.0
uniform vec2 wind_bias;         // constant directional bias added to the curl noise (0,0 = none - pure localized
                                 // gusts, e.g. vegetation; nonzero = a dominant drift direction, e.g. cloud motion)
uniform float flutter_strength; // 0.0 = off. >0.0 adds a second, faster/finer, independent noise layer on top of
                                 // the main wind - vegetation's high-frequency "leaves flutter independently of
                                 // the branch they're on" detail; left at 0 for cloud drift, which has no equivalent leaf-scale detail.
uniform float flutter_scale;    // spatial scale of the flutter noise layer - finer (larger value) than the main wind
uniform float flutter_cycles;   // whole number of EXTRA noise-time cycles over one loop for the flutter layer,
                                 // on top of the main wind's one cycle - keeps it faster-looking while staying
                                 // exactly loop-safe (sin/cos of an integer multiple of theta is still periodic
                                 // over the same loop, see particles_wind.comp's matching trick)
uniform float wave_strength;    // 0.0 = off (default - unchanged turbulent-only sway/drift). >0.0 adds a genuine
                                 // TRAVELING WAVE displacement - a coherent ripple crossing the region in
                                 // wave_direction, distinct from the curl-noise field's in-place turbulent gusting
                                 // - see vegetation.py's render_grass_sway: grass has no rigid trunk
                                 // section the way a tree does, and reads as a "flag-wave" rippling across the
                                 // blade region, not a turbulent canopy gusting in place.
uniform vec2 wave_direction;    // unit vector - the direction the traveling wave ripples along
uniform float wave_frequency;   // spatial frequency of the traveling wave (radians per pixel)
uniform float wave_cycles;      // whole number of traveling-wave cycles over one loop - same loop-safety
                                 // requirement as flutter_cycles above (must be a whole number to close exactly)
uniform float gap_flicker_strength;   // 0.0 = off (default). >0.0 approximates light/gap flicker through foliage
                                       // - a fast, fine-scale, DISTINCT noise layer from flutter above,
                                       // modulating alpha more assertively than flutter's own subtle shimmer.
uniform float gap_flicker_scale;      // spatial scale of the gap-flicker noise - finer than flutter_scale

// see vfx_engine/weather.py's
// gust_shader_uniforms() docstring for the full formula these implement;
// gust_at() below evaluates it per-FRAGMENT using this fragment's own uv,
// instead of the CPU pre-multiplying drift_strength/flutter_strength/
// wave_strength by one frame-global scalar the way this shader used to
// receive them. gust_amp_a/gust_lag_a/gust_amp_b/gust_lag_b let this
// SAME function reproduce either the plain gust_envelope() (amp=1,
// lag=0) or the elastic/damped-spring elastic_gust_envelope() (real
// amp/lag) - see scene_warp.py's render_region_warp for which vegetation
// sway actually requests.
uniform float gust_phase_offset;
uniform float gust_cycles_a;
uniform float gust_cycles_b;
uniform float gust_amp_a;
uniform float gust_lag_a;
uniform float gust_amp_b;
uniform float gust_lag_b;
uniform float gust_strength;
uniform float gust_spatial_delay;
uniform vec2 gust_wind_bias;

// Same value-noise/curl building blocks as particles_wind.comp/blur.glsl -
// see those files for the full explanation of what curl noise is and why
// it's used instead of an independent per-pixel sine wobble.
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float value_noise(vec2 p) {
    vec2 cell = floor(p);
    vec2 f = fract(p);

    float a = hash(cell);
    float b = hash(cell + vec2(1.0, 0.0));
    float c = hash(cell + vec2(0.0, 1.0));
    float d = hash(cell + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// Reflects x into [0,1] via a triangle-wave fold, instead of clamping -
// fixes (clamping pins every sample beyond the edge to the
// SAME boundary pixel, which reads as a stretch/smear artifact once
// several neighboring output pixels all land on that one clamped source
// pixel) and doubles as.9's fix for cloud.py specifically
// (cloud drift and vegetation sway share this exact shader/mechanism):
// a mirrored sample always has SOME content to reveal at the trailing
// edge of the drift (a reflection of nearby real photo content) instead
// of clamping to the same static edge column/row forever, which is
// exactly the "content ceiling" 1.9 describes. Not literally new content
// (still real photo pixels, just reflected) but a smooth, seamless
// continuation rather than a hard clamp or an invented synthetic texture.
float mirror_coord(float x) {
    float t = mod(x, 2.0);
    if (t < 0.0) t += 2.0;
    return t <= 1.0 ? t : 2.0 - t;
}

vec2 curl(vec2 p) {
    float eps = 0.05;
    float n_up = value_noise(p + vec2(0.0, eps));
    float n_down = value_noise(p - vec2(0.0, eps));
    float n_right = value_noise(p + vec2(eps, 0.0));
    float n_left = value_noise(p - vec2(eps, 0.0));

    float dy = (n_up - n_down) / (2.0 * eps);
    float dx = (n_right - n_left) / (2.0 * eps);
    return vec2(dy, -dx);
}

// curl()'s finite-difference division by (2*eps = 0.1) means its raw
// output can swing up to roughly +-10 for a fully-saturated noise
// gradient (see particles_wind.comp, which uses this exact same curl()
// and compensates with its OWN small `drift` uniform, typically
// 0.08-0.4, before scaling by 40 - a tuning that lives entirely inside
// that file). This module has no equivalent per-caller compensation, so
// it's normalized once, here, to a roughly unit-scale vector - callers'
// drift_strength uniform can then be read directly as "approximate peak
// pixel displacement" without also having to know curl()'s internal
// scale. Confirmed necessary empirically: without this, vegetation sway
// at even a modest drift_strength visibly tore/ghosted the canopy
// (adjacent pixels landing tens of pixels apart in a single frame).
const float CURL_NORMALIZATION = 8.0;

// position-aware gust value, mirroring
// vfx_engine.weather.gust_shader_uniforms()'s documented formula exactly
// (see the uniforms above) - a position further downwind (along
// gust_wind_bias, relative to the frame's own center) sees the identical
// gust waveform, phase-delayed by gust_spatial_delay. gust_spatial_delay
// = 0.0 makes phase_delay always 0, so every fragment reads the same
// value regardless of position - exactly this shader's old (pre-13.2)
// behavior.
float gust_at(vec2 position_uv) {
    float distance_downwind = dot(position_uv - vec2(0.5), gust_wind_bias);
    float phase_delay = -distance_downwind * gust_spatial_delay * 6.28318530718;
    float gust_theta = 6.28318530718 * (frame_i / max(num_frames, 1.0)) + gust_phase_offset + phase_delay;
    float wave = 0.6 * gust_amp_a * sin(gust_theta * gust_cycles_a - gust_lag_a)
               + 0.4 * gust_amp_b * sin(gust_theta * gust_cycles_b + 1.7 - gust_lag_b);
    return max(0.0, 1.0 + wave * gust_strength);
}

void main() {
    // Flip Y to match the base/region textures' own top-down byte layout
    // (uploaded straight from a numpy/PIL array, row 0 = top) - same
    // convention particles.glsl's base_texture sampling uses.
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);

    vec2 region = texture(region_texture, uv).rg;
    float alpha = region.r;
    float weight = region.g;

    if (alpha <= 0.003) {
        frag_color = vec4(0.0);
        return;
    }

    // Loop-safe drift: sample the noise field's TIME input around a
    // circle instead of sliding it linearly with frame_i - see
    // particles_wind.comp's docstring for the full explanation of why
    // this closes the loop exactly.
    float theta = 6.28318530718 * (frame_i / max(num_frames, 1.0));
    vec2 noise_time = noise_loop_radius * vec2(cos(theta), sin(theta));
    vec2 noise_pos = uv * resolution * noise_scale + noise_time;
    vec2 wind = curl(noise_pos) / CURL_NORMALIZATION + wind_bias;

    // this fragment's OWN position-aware gust value - drift_
    // strength/flutter_strength/wave_strength below arrive UNSCALED (see
    // scene_warp.py); multiplying by gust HERE, per-fragment, is what
    // lets a gust reach the downwind side of a wide frame later than the
    // upwind side, instead of the whole region swaying in perfect
    // lockstep the way one CPU-precomputed frame-global scalar would.
    float gust = gust_at(uv);

    if (flutter_strength > 0.0) {
        // A second, independent noise sample - offset far from the main
        // wind's sampling coordinates (so it isn't just a scaled copy of
        // the same field) and cycling `flutter_cycles` times as fast,
        // still exactly loop-safe (see the uniform's comment above).
        float theta_fast = theta * flutter_cycles;
        vec2 flutter_time = noise_loop_radius * vec2(cos(theta_fast), sin(theta_fast));
        vec2 flutter_pos = uv * resolution * flutter_scale + flutter_time + vec2(500.0, 500.0);
        vec2 flutter = (curl(flutter_pos) / CURL_NORMALIZATION) * flutter_strength * gust;
        wind += flutter;
        // Leaf-scale flutter also jitters visibility slightly (real
        // leaves catching/losing the light as they flutter), not just
        // position - a subtle shimmer, not a strong flicker.
        alpha *= clamp(1.0 - length(flutter) * 0.05, 0.85, 1.0);
    }

    if (wave_strength > 0.0) {
        // far/low-depth pixels get a tighter (higher-
        // frequency) wave and near/high-depth pixels get a wider (lower-
        // frequency) one - the same depth-driven "farther = finer/slower-
        // reading, nearer = coarser/faster-reading" idea blur.glsl already
        // applies to its own fbm sampling via local_scale/local_drift (see
        // that file's matching depth_texture/has_depth uniforms), just
        // applied to this shader's traveling-wave term instead. This ONLY
        // touches wave_frequency (a purely spatial coefficient - no
        // periodicity constraint), never the wave_cycles*theta time term
        // below: a continuously depth-varying multiplier on THAT term
        // would generally stop being a whole multiple of 2*pi at
        // theta=2*pi, reopening a visible seam at the loop point (see
        // wave_cycles' own uniform comment for why it must stay a whole
        // number). Scaling only the frequency still reads as an "effective
        // speed" change on its own, since a traveling wave's phase
        // velocity is its time-frequency divided by its spatial frequency
        // (v = omega/k) - raising k (tighter spacing) while omega
        // (wave_cycles*theta's rate) stays fixed already lowers the
        // apparent speed for far pixels, and lowering k raises it for near
        // ones, with no separate time-term edit required.
        float depth = has_depth > 0.5 ? texture(depth_texture, uv).r : 0.5;
        float local_wave_frequency = has_depth > 0.5 ? wave_frequency * mix(1.6, 0.6, depth) : wave_frequency;

        // A genuine traveling wave - NOT another curl-noise sample. Phase
        // is a linear function of position along wave_direction (a real
        // spatial wavefront) minus a whole-cycle-count time term (loop-safe
        // the same way flutter_cycles is: wave_cycles * theta is a whole
        // multiple of 2*pi at theta=2*pi, so this returns to its exact
        // frame-0 value at the loop seam).
        float wave_phase = dot(uv * resolution, wave_direction) * local_wave_frequency - wave_cycles * theta;
        float wave = sin(wave_phase);
        // Displaces along wave_direction itself - like a grass blade
        // leaning further in the direction the current ripple is pushing,
        // not swirling in place the way curl-noise turbulence does.
        wind += wave_direction * wave * wave_strength * gust;
    }

    if (gap_flicker_strength > 0.0) {
        // Fake light-through-gaps flicker. Real
        // foliage in wind constantly opens/closes small gaps that let
        // through brief flickers of sky/light as leaf clusters move
        // relative to each other - modulating alpha with a fast,
        // fine-scale noise layer (sampled far from flutter's own
        // coordinates and cycling even faster, so it's a genuinely
        // distinct signal, not a scaled copy) is a believable stand-in
        // for the brightness flicker viewers actually associate with
        // this, synced to the same wind timing as everything else.
        // round(...) here (not flutter_cycles * 2.3 directly) - flutter_cycles
        // is itself always a whole number (enforced by its own callers), but
        // 2.3x a whole number isn't necessarily whole, which would desync
        // this layer's phase from theta and leave a visible seam at the loop
        // boundary (frame num_frames != frame 0). Rounding keeps this a
        // genuinely distinct, faster-cycling signal from flutter's own
        // theta_fast while still landing on a whole multiple of theta.
        float gap_theta = theta * round(flutter_cycles * 2.3);
        vec2 gap_time = noise_loop_radius * vec2(cos(gap_theta), sin(gap_theta));
        vec2 gap_pos = uv * resolution * gap_flicker_scale + gap_time + vec2(1500.0, 1500.0);
        float gap_noise = value_noise(gap_pos);
        // Contrast-shaped so it reads as brief, distinct "a gap just
        // opened" dips rather than a smooth wobble.
        float gap_openness = smoothstep(0.55, 0.85, gap_noise);
        alpha *= (1.0 - gap_flicker_strength * gap_openness);
    }

    // Multiplying the WHOLE summed wind vector by gust here (matching
    // this shader's original structure, where drift_strength arrived
    // CPU-pre-multiplied by one frame-global gust value) reproduces the
    // identical relative scaling flutter/wave already had against the
    // base curl/bias term - only WHERE the gust value itself now comes
    // from (this fragment's own gust_at(uv) instead of one shared
    // scalar) has changed, not the balance between these terms.
    vec2 offset = wind * drift_strength * gust * weight / resolution;
    vec2 raw_uv = uv + offset;
    vec2 warped_uv = vec2(mirror_coord(raw_uv.x), mirror_coord(raw_uv.y));

    vec3 color = texture(base_texture, warped_uv).rgb;
    frag_color = vec4(color, alpha);
}
