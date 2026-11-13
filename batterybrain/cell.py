"""Battery cell models.

Two models live here on purpose:

* ``Cell`` is the "truth" plant used by the simulator. It has two RC pairs,
  resistances that change with temperature and state of charge, voltage
  hysteresis, a lumped thermal model and capacity fade.
* ``NominalModel`` is what the battery management system *believes*: a
  single-RC equivalent circuit with datasheet parameters and a simple
  temperature correction. It knows nothing about the second RC pair, the
  hysteresis, or how resistance changes with state of charge.

The gap between the two is the whole point of the project: the Kalman filter
has to cope with it, and the neural network learns it.

Sign convention: current > 0 discharges the cell.
"""
from dataclasses import dataclass
import math

import numpy as np

from .ocv import ocv

T_REF_K = 298.15


@dataclass
class CellParams:
    """Parameters of the truth plant (a 5 Ah cylindrical cell)."""
    q_ah: float = 5.0          # capacity when new [Ah]
    r0_ref: float = 0.022      # ohmic resistance at 25 degC, mid SOC [ohm]
    r1_ref: float = 0.012      # fast polarisation resistance [ohm]
    tau1: float = 12.0         # fast time constant [s]
    r2_ref: float = 0.010      # slow polarisation resistance [ohm]
    tau2: float = 240.0        # slow time constant [s]
    hyst_m: float = 0.018      # hysteresis magnitude [V]
    hyst_gamma: float = 60.0   # hysteresis rate constant [-]
    c_th: float = 70.0         # heat capacity [J/K]
    r_th: float = 4.0          # thermal resistance to ambient [K/W]
    ea_over_r: float = 2200.0  # Arrhenius constant for resistance [K]


@dataclass
class NominalModel:
    """The fixed-parameter model the BMS algorithms are designed with."""
    q_ah: float = 5.0
    r0: float = 0.024          # at 25 degC
    r1: float = 0.014          # at 25 degC
    tau1: float = 20.0
    c_th: float = 70.0
    r_th: float = 3.6
    ea_over_r: float = 1800.0  # datasheet temperature coefficient (not exact)

    def temp_factor(self, temp_c):
        """Datasheet-style temperature compensation for the resistances."""
        return math.exp(self.ea_over_r * (1.0 / (temp_c + 273.15) - 1.0 / T_REF_K))

    def r0_at(self, temp_c):
        return self.r0 * self.temp_factor(temp_c)

    def r1_at(self, temp_c):
        return self.r1 * self.temp_factor(temp_c)


class Cell:
    """Truth plant: 2-RC equivalent circuit + hysteresis + thermal + ageing."""

    def __init__(self, params=None, soc=0.9, temp_c=25.0, ambient_c=25.0, fade=0.0):
        self.p = params or CellParams()
        self.soc = float(soc)
        self.v1 = 0.0
        self.v2 = 0.0
        self.h = 0.0
        self.temp_c = float(temp_c)
        self.ambient_c = float(ambient_c)
        self.fade = float(fade)   # 0 = new cell, 0.2 = 20 % capacity lost

    @property
    def capacity_as(self):
        """Usable capacity in ampere-seconds."""
        return self.p.q_ah * 3600.0 * (1.0 - self.fade)

    def resistances(self):
        p, z = self.p, self.soc
        k = math.exp(p.ea_over_r * (1.0 / (self.temp_c + 273.15) - 1.0 / T_REF_K))
        k *= 1.0 + 2.0 * self.fade
        r0 = p.r0_ref * k * (1.0 + 0.6 * math.exp(-12.0 * z) + 0.15 * math.exp(-15.0 * (1.0 - z)))
        r1 = p.r1_ref * k * (1.0 + 0.8 * math.exp(-10.0 * z))
        r2 = p.r2_ref * k
        return r0, r1, r2

    def voltage(self, current):
        """Terminal voltage [V] while `current` [A] is flowing."""
        r0, _, _ = self.resistances()
        return float(ocv(self.soc)) + self.p.hyst_m * self.h - self.v1 - self.v2 - current * r0

    def step(self, current, dt=1.0):
        """Advance the cell by dt seconds with constant current."""
        p = self.p
        r0, r1, r2 = self.resistances()
        heat = current * current * r0 + current * (self.v1 + self.v2)

        a1 = math.exp(-dt / p.tau1)
        a2 = math.exp(-dt / p.tau2)
        self.v1 = a1 * self.v1 + r1 * (1.0 - a1) * current
        self.v2 = a2 * self.v2 + r2 * (1.0 - a2) * current

        if current != 0.0:
            kh = 1.0 - math.exp(-p.hyst_gamma * abs(current) * dt / self.capacity_as)
            self.h += kh * (-math.copysign(1.0, current) - self.h)

        self.soc = min(1.0, max(0.0, self.soc - current * dt / self.capacity_as))
        self.temp_c += dt / p.c_th * (heat - (self.temp_c - self.ambient_c) / p.r_th)


class Sensors:
    """Voltage, current and temperature sensors with noise and a current offset."""

    def __init__(self, sigma_v=0.002, sigma_i=0.02, sigma_t=0.1, bias_i=0.0, seed=0):
        self.sigma_v = sigma_v
        self.sigma_i = sigma_i
        self.sigma_t = sigma_t
        self.bias_i = bias_i
        self.rng = np.random.default_rng(seed)

    def read(self, cell, current):
        n = self.rng.standard_normal(3)
        v = cell.voltage(current) + self.sigma_v * n[0]
        i = current + self.bias_i + self.sigma_i * n[1]
        t = cell.temp_c + self.sigma_t * n[2]
        return v, i, t
