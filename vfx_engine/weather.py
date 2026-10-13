"""
weather.py

A single shared "weather state" - one wind DIRECTION and one GUST timing
envelope - sampled by every wind-driven effect in this pipeline (rain/
snow particle drift, fog drift, vegetation sway, cloud drift) instead of
each running off its own independent, uncoordinated noise field. See
the original design: a real gust of wind bends trees, sways
falling snow, and drifts fog all in the same direction at the same time -
before this module existed, each system's motion would have needed to be
hand-tuned to even COINCIDENTALLY look coordinated, because there was no
shared source of "which way is the wind blowing, and how hard, right now"
for them to agree on.

Kept intentionally simple rather than one literal shared noise texture:
a wind DIRECTION (config.WEATHER_WIND_DIRECTION_DEG, constant for the
whole loop - real prevailing wind doesn't reverse over a 20-30s clip) and
a GUST envelope (a periodic, loop-safe scalar describing how strong the
wind is at any moment in the loop - stronger during a gust, calmer
between them). Every consumer multiplies its OWN existing wind-strength
parameter by gust_envelope() and rotates its OWN existing directional
bias to wind_direction_vector(), rather than sampling one literal shared
per-pixel noise field - that would mean unifying very different rendering
paths (a GPU COMPUTE shader for particles, one FRAGMENT shader for fog,
another for vegetation/cloud) onto one identical noise field and
coordinate convention, a far larger rewrite for a benefit this module
already delivers: real directional and temporal coherence, which is what
the original design actually names as the goal ("the same direction at the same
time"), not bit-identical per-pixel noise sampling.

the original design ("the gust is a single global scalar - no spatial
propagation across the frame"): every pixel/particle used to "feel" the
gust at the exact same instant, with zero travel-time lag across the
scene - unrealistic for a wide landscape composition, where a gust
sweeping the whole visible frame in perfect lockstep is a real (if
subtle) simplification. Fixed WITHOUT needing one shared per-pixel noise
texture after all: gust_shader_uniforms() below returns the same small
set of scalar ingredients (phase offset, cycles, optional spring
response amplitude/lag, spatial delay, wind direction) to every
consumer's own rendering path (a GLSL fragment shader, a GLSL compute
shader, a numpy CPU fallback), each of which evaluates the position-
aware gust formula ITSELF, at its own native per-pixel/per-particle
resolution, using gust_at_position()'s exact math (mirrored, not
literally shared code, across GLSL/numpy - see
shaders/region_warp.glsl's/blur.glsl's/particles_wind.comp's own
gust_at() and _cpu_dynamic_data's vectorized equivalent). A position
further downwind sees the identical waveform, phase-delayed by
config.WEATHER_GUST_SPATIAL_DELAY (a fraction of one gust cycle to sweep
the full frame) - config.WEATHER_GUST_SPATIAL_DELAY = 0.0 reproduces the
old "every pixel feels it simultaneously" behavior exactly.
"""

import math
import random

import numpy as np

import config

# gust_envelope/elastic_gust_envelope used to be a PURE
# function of (frame_i, num_frames) plus fixed config.py constants - the
# SAME photo rendered twice, days apart, got a bit-for-bit identical wind
# gust pattern both times, and a live wallpaper looping for hours
# repeated the exact same gust timing every single cycle forever, with
# zero variation. _gust_phase_offset (see _get_gust_phase_offset) is a
# per-render random constant added to theta before the loop-safe
# whole-cycle-count sinusoids below - the same fix particles.py's own
# density-clumping noise already applied to an identical problem (a
# random offset sampled once per render, not per frame, specifically so
# "EVERY render of EVERY photo... would get the identical [pattern]" -
# see that module's own comment - stops being true). Cached at module
# level (not re-sampled per call) so it stays IDENTICAL for the whole
# render, shared by both gust_envelope and elastic_gust_envelope (so a
# gust's timing relative to its own lagged/damped echo - the whole point
# of 13.3/elastic_gust_envelope - is unaffected; only the pattern's
# ABSOLUTE position within the loop shifts per-render, not the relative
# phase between fast- and slow-reacting consumers).
_gust_phase_offset = None


def _get_gust_phase_offset():
    """
    Returns this process's per-render gust phase offset (radians),
    sampling it once via the stdlib (unseeded) `random` module on first
    call and caching it for every subsequent call in this same run.

    Adding a CONSTANT phase offset to theta before multiplying by a
    whole-cycle-count frequency (see _resolve_gust_cycles) does not
    break the closed-loop guarantee every other cyclic effect in this
    pipeline relies on: `sin((theta + offset) * cycles)` at theta=0 and
    at the loop-closing theta=2*pi differ by exactly
    `sin(2*pi*cycles + offset*cycles) - sin(offset*cycles)`, which is
    zero because `cycles` is a whole number and sin() is 2*pi-periodic -
    frame 0 and frame num_frames still land on the identical value,
    regardless of what the offset itself is.
    """
    global _gust_phase_offset
    if _gust_phase_offset is None:
        _gust_phase_offset = random.uniform(0.0, 2.0 * math.pi)
    return _gust_phase_offset


def wind_direction_vector():
    """
    Returns a (dx, dy) unit vector for config.WEATHER_WIND_DIRECTION_DEG -
    0 = due right (+x), 90 = straight up (-y in image space, since image Y
    grows downward) - the same screen-space convention cloud.py's drift
    direction already used before this module existed.
    """
    radians = math.radians(config.WEATHER_WIND_DIRECTION_DEG)
    return (math.cos(radians), -math.sin(radians))


def _resolve_gust_cycles(num_frames):
    """
    config.WEATHER_GUST_SECONDS_PER_CYCLE is a RATE (seconds
    per full gust pulse), not a fixed total-render cycle count - this
    converts it into whole cycle counts (see gust_envelope's docstring for
    why they must be whole numbers) sized to THIS render's actual duration
    (num_frames/config.FPS), so a long render gets proportionally more,
    naturally-paced gusts instead of the original (2, 3) stretched thin
    across whatever duration is configured. max(1, ...) keeps a very short
    render from collapsing a period to 0 whole cycles (which would flatten
    that term of the gust to a constant).
    """
    duration_seconds = max(num_frames, 1) / config.FPS
    period_a, period_b = config.WEATHER_GUST_SECONDS_PER_CYCLE
    return max(1, round(duration_seconds / period_a)), max(1, round(duration_seconds / period_b))


def gust_envelope(frame_i, num_frames):
    """
    Returns a scalar (nominally centered on 1.0, never negative) - this
    frame's overall wind gustiness for the WHOLE loop. Every wind-driven
    system multiplies its own wind-strength parameter by this SAME value
    for a given (frame_i, num_frames), so a gust visibly moves rain, snow,
    trees, and fog all at once instead of each system's own independent
    per-effect wobble happening to line up (or not) by chance.

    Built from TWO whole-cycle-count sinusoids (see _resolve_gust_cycles)
    at different frequencies and a phase offset, summed - a single sine
    wave reads as too mechanically regular ("the wind pulses like a
    metronome"); two summed at different frequencies gives a less
    perfectly-periodic-feeling gust pattern while still closing the loop
    EXACTLY (each term is a whole number of full cycles over num_frames,
    so frame 0 and frame num_frames land on the identical value - the same
    construction every other cyclic effect in this pipeline uses, see
    particles_wind.comp's docstring for the full explanation of why this
    has to be a whole number of cycles).

    config.WEATHER_GUST_STRENGTH = 0.0 makes this return exactly 1.0 for
    every frame - every consumer's motion is then numerically IDENTICAL to
    before this module existed (no coupling, no behavior change), not just
    approximately so.

    theta includes a per-render random phase offset (see
    _get_gust_phase_offset) so the same photo rendered again - or one
    live wallpaper's next loop cycle - doesn't repeat a bit-for-bit
    identical gust pattern; the loop still closes exactly regardless (see
    that function's docstring for why a constant additive phase doesn't
    break it).
    """
    theta = 2.0 * math.pi * (frame_i / max(num_frames, 1)) + _get_gust_phase_offset()
    cycles_a, cycles_b = _resolve_gust_cycles(num_frames)
    wave = 0.6 * math.sin(theta * cycles_a) + 0.4 * math.sin(theta * cycles_b + 1.7)
    return max(0.0, 1.0 + wave * config.WEATHER_GUST_STRENGTH)


def _spring_response(driving_cycles, natural_cycles, damping_ratio):
    """
    Standard 2nd-order damped-harmonic-oscillator frequency response: the
    (amplitude ratio, phase lag) a spring with natural frequency
    `natural_cycles` and damping ratio `damping_ratio` settles into, at
    STEADY STATE, when driven by a sinusoidal force at `driving_cycles`
    (same units - whole cycles over one loop, matching
    _resolve_gust_cycles's output). Both are in "cycles per loop" units, so
    the ratio between them is unitless (r = driving_cycles/natural_cycles).
    """
    ratio = driving_cycles / natural_cycles
    amplitude = 1.0 / math.sqrt((1.0 - ratio ** 2) ** 2 + (2.0 * damping_ratio * ratio) ** 2)
    phase_lag = math.atan2(2.0 * damping_ratio * ratio, 1.0 - ratio ** 2)
    return amplitude, phase_lag


def elastic_gust_envelope(frame_i, num_frames, damping_ratio=0.4, natural_freq_mult=2.2):
    """
    Elastic (lagged/damped) variant of gust_envelope() - fixes the
    confirmed "no elasticity, zero lag" gap: the plain
    envelope's gustiness applies to every consumer INSTANTLY, the same
    frame the noise says it should - real wood (or any driven spring
    system) doesn't track a driving force with zero lag, it responds like
    a damped spring: continuing to oscillate slightly AFTER a gust passes,
    not snapping back to neutral the instant the driving signal does.

    Implemented ANALYTICALLY, not by simulating a spring frame-by-frame
    from an arbitrary starting condition (which would need many "settle in"
    loops discarded before the output is even approximately periodic, and
    still wouldn't close the loop EXACTLY). gust_envelope()'s wave is
    already a sum of exactly two sinusoids at whole-cycle-count
    frequencies (see _resolve_gust_cycles) - a damped linear oscillator's
    steady-state response to a sinusoidal force is just another sinusoid
    at the SAME frequency, amplitude-scaled and phase-shifted by
    _spring_response()'s standard transfer function. Since no new
    frequency content is introduced (only existing whole-cycle components
    are rescaled/phase-shifted), the result is automatically EXACTLY
    loop-safe for any damping_ratio/natural_freq_mult - frame 0 and frame
    num_frames land on the identical value by construction, the same
    "whole number of cycles" guarantee every other cyclic effect in this
    pipeline relies on, with no simulation, settling period, or
    approximation needed.

    The spring's natural frequency is set relative to the driving gust's
    own faster component (natural_freq_mult * max(cycles_a, cycles_b)) -
    comfortably above both driving frequencies, so the response stays a
    subtle, believable phase lag (this is meant to read as "a fraction of
    a second behind the gust", not a dramatic, unstably-resonating
    overshoot).

    Reusable by any other "heavy, slow-reacting" consumer, not just
    vegetation - callers
    just swap gust_envelope for this in whatever per-frame call they
    already make.

    theta includes the SAME per-render phase offset
    gust_envelope uses (see _get_gust_phase_offset - one shared, cached
    value per process, not resampled here) - so a gust's timing relative
    to its own lagged/damped echo stays exactly as designed, while the
    pattern's absolute position within the loop still varies render to
    render.
    """
    theta = 2.0 * math.pi * (frame_i / max(num_frames, 1)) + _get_gust_phase_offset()
    cycles_a, cycles_b = _resolve_gust_cycles(num_frames)
    natural_cycles = max(cycles_a, cycles_b) * natural_freq_mult

    amp_a, lag_a = _spring_response(cycles_a, natural_cycles, damping_ratio)
    amp_b, lag_b = _spring_response(cycles_b, natural_cycles, damping_ratio)
    wave = (
        0.6 * amp_a * math.sin(theta * cycles_a - lag_a)
        + 0.4 * amp_b * math.sin(theta * cycles_b + 1.7 - lag_b)
    )
    return max(0.0, 1.0 + wave * config.WEATHER_GUST_STRENGTH)


def gust_shader_uniforms(frame_i, num_frames, elastic=False, damping_ratio=0.4, natural_freq_mult=2.2):
    """
    Returns a dict of scalar values a GPU shader (fragment or compute) or
    a numpy CPU fallback needs to compute the SAME position-aware gust
    value gust_envelope()/elastic_gust_envelope() would, entirely on its
    OWN side, at its own native per-pixel/per-particle resolution -
    the original design ("the gust is a single global scalar - no spatial
    propagation across the frame"). See this module's own docstring for
    the full explanation; see shaders/region_warp.glsl's/blur.glsl's/
    particles_wind.comp's own `gust_at()` GLSL function (and
    gust_multiplier_at below, its numpy CPU-fallback equivalent) for the
    position-aware formula every consumer implements using these same
    ingredients:

        distance_downwind = dot(position - 0.5, gust_wind_bias)
        phase_delay = -distance_downwind * gust_spatial_delay * 2*pi
        theta = 2*pi*(frame_i/num_frames) + gust_phase_offset + phase_delay
        wave = 0.6*gust_amp_a*sin(theta*gust_cycles_a - gust_lag_a)
             + 0.4*gust_amp_b*sin(theta*gust_cycles_b + 1.7 - gust_lag_b)
        gust = max(0, 1 + wave*gust_strength)

    (position is a 0.0-1.0 UV/screen fraction; a distance_downwind of 0
    at the frame's own center means the phase delay is symmetric around
    the center, not anchored to one edge - consistent regardless of
    frame aspect ratio or wind angle.)

    elastic: when True, includes the REAL damped-spring amplitude/lag
        values (see _spring_response) for the SAME (damping_ratio,
        natural_freq_mult) elastic_gust_envelope() would use - the
        caller's own gust_at() then reproduces the lagged/damped
        response exactly. False (default) returns amplitude=1.0/lag=0.0
        for both terms, making the position-aware formula above
        mathematically IDENTICAL to the plain gust_envelope() case.
    config.WEATHER_GUST_SPATIAL_DELAY = 0.0 makes phase_delay always 0 -
    every position sees the identical instantaneous value, exactly
    reproducing this module's pre-13.2 behavior.
    """
    cycles_a, cycles_b = _resolve_gust_cycles(num_frames)
    if elastic:
        natural_cycles = max(cycles_a, cycles_b) * natural_freq_mult
        amp_a, lag_a = _spring_response(cycles_a, natural_cycles, damping_ratio)
        amp_b, lag_b = _spring_response(cycles_b, natural_cycles, damping_ratio)
    else:
        amp_a, lag_a, amp_b, lag_b = 1.0, 0.0, 1.0, 0.0

    return dict(
        frame_i=float(frame_i),
        num_frames=float(max(num_frames, 1)),
        gust_phase_offset=_get_gust_phase_offset(),
        gust_cycles_a=float(cycles_a),
        gust_cycles_b=float(cycles_b),
        gust_amp_a=amp_a,
        gust_lag_a=lag_a,
        gust_amp_b=amp_b,
        gust_lag_b=lag_b,
        gust_strength=config.WEATHER_GUST_STRENGTH,
        gust_spatial_delay=config.WEATHER_GUST_SPATIAL_DELAY,
        gust_wind_bias=wind_direction_vector(),
    )


def gust_multiplier_at(x_frac, y_frac, frame_i, num_frames):
    """
    Numpy-vectorized CPU equivalent of the GLSL `gust_at()` function every
    shader-based consumer implements (see gust_shader_uniforms's
    docstring, the original design) - used by particles.py's `_cpu_dynamic_data`
    fallback path (no compute-shader support on this GPU/driver).
    x_frac/y_frac: numpy arrays (or plain floats), each particle's
    current position as a 0.0-1.0 fraction of the frame's width/height.
    Always uses the PLAIN (non-elastic) formula - no CPU-fallback
    consumer in this codebase currently needs the elastic variant (only
    vegetation.py's GPU-only region_warp.glsl path does).
    """
    wind_x, wind_y = wind_direction_vector()
    distance_downwind = (x_frac - 0.5) * wind_x + (y_frac - 0.5) * wind_y
    phase_delay = -distance_downwind * config.WEATHER_GUST_SPATIAL_DELAY * 2.0 * math.pi
    theta = 2.0 * math.pi * (frame_i / max(num_frames, 1)) + _get_gust_phase_offset() + phase_delay
    cycles_a, cycles_b = _resolve_gust_cycles(num_frames)
    wave = 0.6 * np.sin(theta * cycles_a) + 0.4 * np.sin(theta * cycles_b + 1.7)
    return np.clip(1.0 + wave * config.WEATHER_GUST_STRENGTH, 0.0, None)
