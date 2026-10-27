"""
Monte Carlo gust sensitivity sweep.

Runs N closed-loop simulations at each requested gust intensity (a scalar
multiplier on the Dryden sigma), with independent random seeds, and
aggregates statistics on tracking error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .aircraft import Aircraft
from .autopilot import Autopilot
from .closed_loop import simulate_closed_loop
from .trim import TrimState
from .turbulence import DrydenGust, DrydenWind


@dataclass
class MCStats:
    intensity:     float
    n_runs:        int
    altitude_rms:  np.ndarray   # per-run RMS altitude error (m)
    heading_rms:   np.ndarray
    Va_rms:        np.ndarray
    altitude_pk:   np.ndarray
    heading_pk:    np.ndarray
    Va_pk:         np.ndarray

    def summary(self) -> dict:
        return {
            "intensity": self.intensity,
            "n_runs": self.n_runs,
            "alt_rms_mean":  float(np.mean(self.altitude_rms)),
            "alt_rms_std":   float(np.std(self.altitude_rms)),
            "alt_peak_mean": float(np.mean(self.altitude_pk)),
            "hdg_rms_mean":  float(np.mean(self.heading_rms)),
            "Va_rms_mean":   float(np.mean(self.Va_rms)),
        }


def monte_carlo(
    ac: Aircraft,
    trim: TrimState,
    build_autopilot: Callable[[], Autopilot],
    intensities: list[float],
    n_runs: int,
    t_end: float = 60.0,
    dt_ctrl: float = 0.02,
    base_sigma=(1.06, 1.06, 0.70),
    base_L=(200.0, 200.0, 50.0),
    base_seed: int = 1234,
    settle_time: float = 5.0,
) -> list[MCStats]:
    """Run a gust sensitivity sweep.

    `build_autopilot` returns a fresh Autopilot (with its own command source)
    for each run. `intensities` scales the Dryden sigma vector. Statistics are
    computed on samples after `settle_time` to skip controller startup.
    """
    sigma_u, sigma_v, sigma_w = base_sigma
    Lu, Lv, Lw = base_L

    out: list[MCStats] = []
    rng = np.random.default_rng(base_seed)

    for intensity in intensities:
        alt_rms = np.zeros(n_runs); hdg_rms = np.zeros(n_runs); va_rms = np.zeros(n_runs)
        alt_pk  = np.zeros(n_runs); hdg_pk  = np.zeros(n_runs); va_pk  = np.zeros(n_runs)

        for run in range(n_runs):
            seed = int(rng.integers(0, 2**31 - 1))
            dryden = DrydenGust(
                sigma_u=sigma_u * intensity, sigma_v=sigma_v * intensity, sigma_w=sigma_w * intensity,
                Lu=Lu, Lv=Lv, Lw=Lw, seed=seed,
            )
            wind = DrydenWind(dryden=dryden)
            ap = build_autopilot()
            log = simulate_closed_loop(ac, trim, ap, t_end=t_end, dt_ctrl=dt_ctrl, wind=wind)
            r = log.res
            mask = r.t > settle_time

            # Reference signals (commanded) are sampled from the autopilot's
            # command source. Reuse the schedule from the AP itself.
            cmd_source = ap.cmd
            h_ref   = np.array([cmd_source.update(ti, log.res.x[i])[0]
                                for i, ti in enumerate(r.t)])
            psi_ref = np.array([cmd_source.update(ti, log.res.x[i])[1]
                                for i, ti in enumerate(r.t)])
            Va_ref  = np.array([cmd_source.update(ti, log.res.x[i])[2]
                                for i, ti in enumerate(r.t)])

            alt_err = r.altitude - h_ref
            hdg_err = ((r.euler[:, 2] - psi_ref + np.pi) % (2 * np.pi)) - np.pi
            Va_err  = r.Va - Va_ref

            alt_rms[run] = float(np.sqrt(np.mean(alt_err[mask] ** 2)))
            hdg_rms[run] = float(np.sqrt(np.mean(hdg_err[mask] ** 2)))
            va_rms[run]  = float(np.sqrt(np.mean(Va_err [mask] ** 2)))
            alt_pk[run]  = float(np.max(np.abs(alt_err[mask])))
            hdg_pk[run]  = float(np.max(np.abs(hdg_err[mask])))
            va_pk[run]   = float(np.max(np.abs(Va_err [mask])))

        out.append(MCStats(
            intensity=intensity, n_runs=n_runs,
            altitude_rms=alt_rms, heading_rms=hdg_rms, Va_rms=va_rms,
            altitude_pk=alt_pk,  heading_pk=hdg_pk,  Va_pk=va_pk,
        ))

    return out
