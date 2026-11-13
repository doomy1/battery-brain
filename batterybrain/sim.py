"""Simulation sessions that tie plant, sensors, estimators and controllers together.

These are the objects the browser demo drives: it calls ``step(n)`` and plots
whatever comes back. They are equally usable from a script or a notebook.
"""
import math

import numpy as np

from .cell import Cell, CellParams, NominalModel, Sensors
from .ekf import CoulombCounter, SocEKF
from .ocv import ocv
from .mpc import FastChargeMPC, TableCharger, ThermalObserver
from .profiles import Profile, C_RATE
from .residual_net import LearnedResidual

# Measurement noise variances, set from each model's own voltage residual on
# training data (see scripts/train_residual.py) and inflated because model
# error is correlated in time, not white.
R_PLAIN = 0.035 ** 2
R_HYBRID = 0.010 ** 2


def make_estimators(model, soc0, sigma=0.02):
    est = {
        "cc": CoulombCounter(model, soc0),
        "ekf": SocEKF(model, soc0, r_meas=R_PLAIN),
        "nn": SocEKF(model, soc0, r_meas=R_HYBRID, residual=LearnedResidual.load(model.q_ah)),
    }
    est["ekf"].reset(soc0, sigma)   # the session starts from a known state of charge
    est["nn"].reset(soc0, sigma)
    return est


class EstimationSession:
    """One cell under a load profile, three SOC estimators running side by side."""

    KEYS = ("t", "current", "voltage", "temp", "soc_true", "soc_cc", "soc_ekf", "soc_nn",
            "v_ekf", "v_nn", "sig_ekf", "sig_nn")

    def __init__(self, profile="drive", ambient_c=25.0, soc0=0.9, seed=1, dt=1.0):
        self.dt = dt
        self.t = 0.0
        self.model = NominalModel()
        self.cell = Cell(CellParams(), soc=soc0, temp_c=ambient_c, ambient_c=ambient_c)
        self.sensors = Sensors(seed=seed)
        self.est = make_estimators(self.model, soc0)
        self.seed = seed
        self.profile_kind = profile
        self.profile = Profile(profile, seed) if profile != "manual" else None
        self.manual_current = 0.0
        self.recharging = False
        self._sq = {"cc": 0.0, "ekf": 0.0, "nn": 0.0}
        self._n = 0

    # --- things the UI can poke --------------------------------------------
    def set_param(self, name, value):
        if name == "ambient":
            self.cell.ambient_c = float(value)
        elif name == "noise_mv":
            self.sensors.sigma_v = float(value) * 1e-3
        elif name == "bias_ma":
            self.sensors.bias_i = float(value) * 1e-3
        elif name == "fade":
            self.cell.fade = float(value)
        elif name == "manual_current":
            self.manual_current = float(value)
        elif name == "profile":
            self.profile_kind = value
            self.seed += 1
            self.profile = Profile(value, self.seed) if value != "manual" else None
            if self.profile is not None:
                self.profile._t_end = self.t
        else:
            raise KeyError(name)

    def scramble(self, error=0.3):
        """Give every estimator a badly wrong SOC, as after a memory loss."""
        wrong = self.cell.soc - error if self.cell.soc > 0.5 else self.cell.soc + error
        self.est["cc"].soc = wrong
        self.est["ekf"].reset(wrong)
        self.est["nn"].reset(wrong)
        self.reset_stats()

    def reset_stats(self):
        self._sq = {k: 0.0 for k in self._sq}
        self._n = 0

    # --- simulation ---------------------------------------------------------
    def _load_current(self):
        cell = self.cell
        if self.profile is None:
            amps = self.manual_current
        else:
            amps = self.profile.current(self.t)
            # Keep the demo running forever: recharge when nearly empty.
            if cell.soc < 0.07:
                self.recharging = True
            elif cell.soc > 0.90:
                self.recharging = False
            if self.recharging:
                amps = -0.8 * C_RATE
        if cell.soc >= 0.995 and amps < 0:
            amps = 0.0
        if cell.soc <= 0.005 and amps > 0:
            amps = 0.0
        return amps

    def step(self, n=1):
        out = {k: [] for k in self.KEYS}
        cell, est, dt = self.cell, self.est, self.dt
        for _ in range(int(n)):
            amps = self._load_current()
            cell.step(amps, dt)
            self.t += dt
            v, i, temp = self.sensors.read(cell, amps)
            for e in est.values():
                e.predict(i, dt)
            est["ekf"].update(v, i, temp)
            est["nn"].update(v, i, temp)

            self._n += 1
            for k, e in est.items():
                self._sq[k] += (e.soc - cell.soc) ** 2

            out["t"].append(self.t)
            out["current"].append(amps)
            out["voltage"].append(v)
            out["temp"].append(cell.temp_c)
            out["soc_true"].append(cell.soc)
            out["soc_cc"].append(est["cc"].soc)
            out["soc_ekf"].append(est["ekf"].soc)
            out["soc_nn"].append(est["nn"].soc)
            out["v_ekf"].append(est["ekf"].v_pred)
            out["v_nn"].append(est["nn"].v_pred)
            out["sig_ekf"].append(est["ekf"].soc_sigma)
            out["sig_nn"].append(est["nn"].soc_sigma)
        n_ = max(self._n, 1)
        out["rmse"] = {k: math.sqrt(v / n_) for k, v in self._sq.items()}
        out["recharging"] = self.recharging
        return out


class _Lane:
    """One cell being charged by one controller, with its own estimator."""

    def __init__(self, kind, model, ambient_c, soc0, i_max, v_max, t_max, target, seed):
        self.kind = kind
        self.model = model
        self.cell = Cell(CellParams(), soc=soc0, temp_c=ambient_c, ambient_c=ambient_c)
        self.sensors = Sensors(seed=seed)
        self.ekf = SocEKF(model, soc0, r_meas=R_HYBRID, residual=LearnedResidual.load(model.q_ah))
        self.ekf.P = np.diag([0.02 ** 2, 0.02 ** 2])
        self.target = target
        self.v_limit = v_max
        self.d_volt = 0.0
        self.u = 0.0
        self.done = False
        self.finish_time = None
        self.peak_temp = ambient_c
        self.peak_volt = 0.0
        if kind == "mpc":
            self.ctrl = FastChargeMPC(model, i_max=i_max, v_max=v_max, t_max=t_max, soc_target=target)
            self.thermal = ThermalObserver(model, ambient_c)
        else:
            self.ctrl = TableCharger(i_max=i_max, v_max=v_max, t_max=t_max)

    def tick(self, t, dt):
        cell = self.cell
        cell.step(-self.u, dt)
        v, i, temp = self.sensors.read(cell, -self.u)
        self.ekf.predict(i, dt)
        self.ekf.update(v, i, temp)
        self.peak_temp = max(self.peak_temp, cell.temp_c)
        self.peak_volt = max(self.peak_volt, cell.voltage(-self.u))

        if not self.done and cell.soc >= self.target:
            self.done = True
            self.finish_time = t
        if self.done or self.ekf.soc >= self.target + 0.01:
            self.u = 0.0
            return v, temp

        if self.kind == "mpc":
            ctrl, u_meas = self.ctrl, -i
            pol = -float(self.ekf.x[1])
            # Voltage model error: a fast additive offset plus a slow resistance
            # adaptation (normalised LMS). Keeping the two on different time
            # scales stops the adaptation from fighting the controller.
            err = v - (float(ocv(self.ekf.soc)) + pol + ctrl.r_eff * u_meas + self.d_volt)
            self.d_volt += 0.2 * err
            if u_meas > 3.0:
                ctrl.r_eff = min(0.15, max(0.005, ctrl.r_eff + 0.01 * err / u_meas))
            self.thermal.update(temp, ctrl.r_eff * u_meas ** 2 + pol * u_meas, cell.ambient_c)
            if round(t / dt) % round(ctrl.dt / dt) == 0:
                self.u = ctrl.control(self.ekf.soc, pol, self.thermal.temp, cell.ambient_c,
                                      d_heat=self.thermal.d, d_volt=self.d_volt)
            # Fast voltage limiter between MPC updates (runs every sample)
            if v > self.v_limit:
                self.u = max(0.0, self.u - (v - self.v_limit) / ctrl.r_eff)
        else:
            self.u = self.ctrl.control(v, temp, dt)
        return v, temp


class ChargeRace:
    """Two identical cells fast-charged side by side: table charger vs MPC."""

    LANES = ("table", "mpc")

    def __init__(self, ambient_c=25.0, soc0=0.1, target=0.8, i_max=15.0, v_max=4.2,
                 t_max=45.0, seed=3, dt=1.0):
        self.dt = dt
        self.t = 0.0
        self.limits = {"i_max": i_max, "v_max": v_max, "t_max": t_max, "target": target}
        model = NominalModel()
        self.lanes = {k: _Lane(k, model, ambient_c, soc0, i_max, v_max, t_max, target, seed + n)
                      for n, k in enumerate(self.LANES)}

    @property
    def finished(self):
        return all(lane.done for lane in self.lanes.values())

    def step(self, n=1):
        out = {"t": []}
        for k in self.LANES:
            out[k] = {"soc": [], "soc_est": [], "current": [], "voltage": [], "temp": []}
        for _ in range(int(n)):
            if self.finished:
                break
            self.t += self.dt
            out["t"].append(self.t)
            for k, lane in self.lanes.items():
                v, _ = lane.tick(self.t, self.dt)
                o = out[k]
                o["soc"].append(lane.cell.soc)
                o["soc_est"].append(lane.ekf.soc)
                o["current"].append(lane.u)
                o["voltage"].append(v)
                o["temp"].append(lane.cell.temp_c)
        out["finished"] = self.finished
        out["summary"] = {k: {"finish_time": lane.finish_time, "peak_temp": lane.peak_temp,
                              "peak_volt": lane.peak_volt, "done": lane.done}
                          for k, lane in self.lanes.items()}
        return out
