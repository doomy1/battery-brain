"""Generate training data for the residual network.

Mimics a lab characterisation campaign: the cell is cycled on a tester where
the true state of charge is known (from a reference coulomb count), at
several ambient temperatures. For every sample we record the network inputs
and the voltage error of the nominal model.
"""
import math

import numpy as np

from .cell import Cell, CellParams, NominalModel, Sensors
from .ocv import ocv
from .profiles import Profile, C_RATE
from .residual_net import ResidualFeatures, OUT_SCALE


def simulate_run(kind, ambient_c, seed, duration_s=7200, dt=1.0):
    """Return (X, y): features and scaled residual targets for one run."""
    rng = np.random.default_rng(seed)
    model = NominalModel()
    cell = Cell(CellParams(), soc=rng.uniform(0.3, 0.95), temp_c=ambient_c, ambient_c=ambient_c)
    sensors = Sensors(seed=seed + 1000)
    profile = Profile(kind, seed)
    feats = ResidualFeatures(model.q_ah)
    a = math.exp(-dt / model.tau1)
    v1_nom = 0.0
    temp = ambient_c
    recharging = False
    X, y = [], []
    for k in range(int(duration_s / dt)):
        amps = profile.current(k * dt)
        if cell.soc < 0.07:
            recharging = True
        elif cell.soc > 0.90:
            recharging = False
        if recharging:
            amps = -0.8 * C_RATE
        if (cell.soc >= 0.98 and amps < 0) or (cell.soc <= 0.02 and amps > 0):
            amps = 0.0
        cell.step(amps, dt)
        temp_prev = temp
        v, i, temp = sensors.read(cell, amps)
        feats.advance(i, dt)
        # Same ordering as the EKF: predict with the previous temperature reading
        v1_nom = a * v1_nom + model.r1_at(temp_prev) * (1.0 - a) * i
        v_model = float(ocv(cell.soc)) - v1_nom - model.r0_at(temp) * i
        X.append(feats.vector(cell.soc, i, temp))
        y.append((v - v_model) / OUT_SCALE)
    return np.array(X), np.array(y)


def build_dataset(temps, seeds, kinds=("drive", "home", "pulse", "cycle"), duration_s=7200):
    Xs, ys = [], []
    for kind in kinds:
        for temp in temps:
            for seed in seeds:
                X, y = simulate_run(kind, temp, seed + int(temp) * 7 + len(kind) * 131, duration_s)
                Xs.append(X)
                ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)
