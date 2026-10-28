# 6-DOF Fixed-Wing UAV Flight Simulator (Python)

[![CI](https://github.com/jwlutz/uav-sim-py/actions/workflows/ci.yml/badge.svg)](https://github.com/jwlutz/uav-sim-py/actions/workflows/ci.yml)

A quaternion-based 6-DOF rigid-body simulator for a small fixed-wing UAV.
Includes stability-derivative aerodynamics, a trim solver, a cascaded PID
autopilot with first-order actuator dynamics, waypoint guidance with L1
loiter, Dryden gust turbulence, a Monte Carlo gust sweep, linearization with
classical mode identification, and an interactive 3D flight animation.

Pure NumPy / SciPy / Matplotlib. No MATLAB required.

![autopilot tracking](docs/figures/autopilot_tracking.png)

70-second slice of a waypoint mission (wp2 → wp3 → L1 loiter):

![flight animation](docs/figures/flight_animation.gif)

## Features

- **6-DOF rigid body EOM** with quaternion attitude and body-frame
  velocity / rate states.
- **Stability-derivative aero model** using Cessna-172-class derivatives
  (Nelson, Appendix B), scaled to a small UAV airframe.
- **Trim solver** that solves for (α, δe, δt) via `scipy.optimize.fsolve`
  for steady, wings-level, constant-altitude flight at any (V, h).
- **Linearization and mode analysis** by central-difference Jacobians of
  the full 13-state system. Recovers short period, phugoid, dutch roll,
  roll subsidence, and spiral with ω_n and ζ.
- **Stability sweep**: eigenvalue locus and damping ratio vs. airspeed.
- **7-loop cascaded PID autopilot**:
  altitude → pitch → elevator,
  heading → roll → aileron,
  sideslip + yaw-damper → rudder,
  airspeed → throttle.
  Each loop uses derivative-on-measurement, back-calculation anti-windup,
  a first-order command prefilter, and output saturation.
- **First-order actuator dynamics** with rate limiting on each surface.
- **Waypoint guidance** with arrival switching and an **L1 lookahead loiter**
  (Park, Deyst, How 2004). With the default gains, steady-state orbit
  radius matches the target to within 1 cm. A vector-field fallback mode
  is also available.
- **Dryden gust model**: MIL-F-8785C 2nd-order shaping filter
  `(1 + √3 τ s) / (1 + τ s)²` for the v and w components, plus a 1st-order
  Markov filter for u. The state-space realization is tuned via Lyapunov
  so the stationary variance equals σ² exactly.
- **Monte Carlo sweep**: N runs at each intensity, with RMS and peak
  tracking-error statistics.
- **3D flight animation** with interactive playback (play/pause,
  1× / 2× / 5× / 10×, timeline scrubber, drone tracking) and a side panel
  showing altitude, airspeed, Euler angles, and control surfaces synced to
  a time cursor.

## Install

```bash
cd uav-sim-py
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
pytest -q                                # unit tests
python examples/basic_trim.py            # trimmed open-loop flight
python examples/autopilot_demo.py        # altitude / heading / airspeed steps
python examples/waypoints.py             # 3-waypoint mission with loiter
python examples/gust_monte_carlo.py      # Monte Carlo gust sensitivity sweep
python examples/stability_sweep.py       # eigenvalue locus & damping vs V
python examples/animate_mission.py       # interactive 3D flight animation
python examples/demo.py                  # everything end-to-end; regenerates docs/figures/
```

A walkthrough notebook is at [docs/walkthrough.ipynb](docs/walkthrough.ipynb).

## Layout

```
src/uavsim/
  quaternion.py      Hamilton scalar-first quaternion math
  aircraft.py        Aircraft dataclass: mass, inertia, derivatives
  aero.py            Body-frame forces and moments
  dynamics.py        13-state 6-DOF EOM
  trim.py            fsolve trim
  sim.py             solve_ivp wrapper, SimResults dataclass
  controllers.py     PID with anti-windup / D-on-meas / prefilter, washout
  actuator.py        First-order actuator dynamics with rate limiting
  autopilot.py       Cascaded 7-loop autopilot, command schedule
  guidance.py        Waypoint guidance with L1 loiter
  closed_loop.py     Hybrid sim: continuous plant + discrete autopilot
  turbulence.py      Dryden gust model
  monte_carlo.py     Gust sensitivity sweep with statistics
  linearize.py       Numerical Jacobians + flight-mode classification
  stability_sweep.py Eigenvalue locus over a V_grid
  animation.py       Interactive 3D flight animation
  viz.py             State / control / ground-track plots
tests/               pytest checks for trim, autopilot, guidance,
                     turbulence, linearization, IO
examples/            Runnable demos and the end-to-end demo.py
```

## Conventions

- **NED earth frame**: x = north, y = east, z = down. Altitude = -p_d.
- **Body frame**: x forward, y right, z down.
- **ZYX (3-2-1) Euler angles**: yaw ψ, pitch θ, roll φ.
- **Quaternion** q = [w, x, y, z] rotates body vectors into NED:
  `v_ned = R(q) @ v_body`.
- **Control sign convention**:
  - +δe pitches nose down (`Cm_δe < 0`)
  - +δa rolls right (`Cl_δa > 0`)
  - +δr yaws nose left (`Cn_δr < 0`)
  - δt ∈ [0, 1] scales max thrust

## Sample outputs

Waypoint mission with L1 loiter at the final waypoint (radius 77.13 ± 0.007 m
in steady state, target 80 m):

![waypoint track](docs/figures/waypoint_track.png)

Same mission with Dryden gust turbulence (σ_w = 1.0 m/s, 2nd-order MIL-spec):

![mission with gusts](docs/figures/mission_with_gusts.png)

Eigenvalue locus over an airspeed sweep (color = V):

![stability locus](docs/figures/stability_locus.png)

Monte Carlo tracking error vs. Dryden gust intensity (15 runs each):

![monte carlo](docs/figures/monte_carlo.png)

## Sample mode-analysis output

At trim (V = 18 m/s, h = 100 m) with the default 4.5 kg UAV airframe:

```
phugoid         ω_n = 0.591 rad/s   ζ = -0.007
dutch_roll      ω_n = 3.766 rad/s   ζ = +0.220
short_period    ω_n = 4.359 rad/s   ζ = +0.842
spiral          λ   = -0.025  (slowly convergent)
roll            λ   = -14.733
```

The short period is heavily damped, dutch roll is moderately damped, the
phugoid is essentially neutrally stable, and the spiral mode is barely
stable. Roll subsidence is fast at -14.7 1/s. These are typical numbers for
a textbook small fixed-wing UAV.

## References

The implementation is independent and was written from textbook physics, but
the project is inspired by and follows the feature scope of:

- D. Smith, [`6dof-uav-sim`](https://github.com/dukesmith0/6dof-uav-sim)
  (MATLAB), the original concept and feature outline.

Technical references the implementation draws on:

- R. C. Nelson, *Flight Stability and Automatic Control*, 2nd ed.
  (McGraw-Hill, 1998). Stability-derivative aero model and Cessna-172-class
  numerical values, Appendix B.
- S. Park, J. Deyst, J. P. How, "A New Nonlinear Guidance Logic for
  Trajectory Tracking," *AIAA Guidance, Navigation, and Control
  Conference*, 2004. The L1 lookahead guidance used for loiter.
- *MIL-F-8785C*, "Flying Qualities of Piloted Airplanes," Department of
  Defense, 1980. Continuous Dryden turbulence model spectra.
- B. L. Stevens, F. L. Lewis, E. N. Johnson, *Aircraft Control and
  Simulation*, 3rd ed. (Wiley, 2015). Quaternion attitude kinematics and
  6-DOF EOM conventions.


