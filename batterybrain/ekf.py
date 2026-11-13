"""Extended Kalman filter for state of charge.

State        x = [z, v1]   (state of charge, polarisation voltage)
Input        i             (measured current, discharge positive)
Measurement  y = OCV(z) - v1 - R0(T)*i  (+ learned residual g, if supplied)

The process model is linear; the only nonlinearity is the OCV curve (and the
network) in the measurement equation, which is linearised at every step.
"""
import math

import numpy as np

from .ocv import ocv, docv_dz


class CoulombCounter:
    """Baseline: integrate the measured current. No feedback at all."""

    def __init__(self, model, soc0=0.5):
        self.model = model
        self.soc = soc0

    def predict(self, current, dt):
        self.soc = min(1.0, max(0.0, self.soc - current * dt / (self.model.q_ah * 3600.0)))


class SocEKF:
    def __init__(self, model, soc0=0.5, r_meas=0.02 ** 2, q_soc=(2e-5) ** 2,
                 q_v1=(1e-3) ** 2, residual=None):
        self.model = model
        self.residual = residual
        self.x = np.array([soc0, 0.0])
        self.P = np.diag([0.2 ** 2, 0.02 ** 2])
        self.Q = np.diag([q_soc, q_v1])   # per second
        self.R = r_meas
        self.temp_c = 25.0                # last measured temperature
        self.v_pred = float(ocv(soc0))

    @property
    def soc(self):
        return float(self.x[0])

    @property
    def soc_sigma(self):
        return math.sqrt(max(self.P[0, 0], 0.0))

    def reset(self, soc0, sigma=0.2):
        self.x[:] = (soc0, 0.0)
        self.P = np.diag([sigma ** 2, 0.02 ** 2])

    def predict(self, current, dt):
        """Time update over dt seconds during which `current` flowed."""
        m = self.model
        a = math.exp(-dt / m.tau1)
        self.x[0] -= current * dt / (m.q_ah * 3600.0)
        self.x[1] = a * self.x[1] + m.r1_at(self.temp_c) * (1.0 - a) * current
        F = np.array([[1.0, 0.0], [0.0, a]])
        self.P = F @ self.P @ F.T + self.Q * dt
        if self.residual is not None:
            self.residual.advance(current, dt)

    def measurement(self, current, temp_c):
        """Predicted voltage and measurement Jacobian H at the current state."""
        z, v1 = self.x
        y = float(ocv(z)) - v1 - self.model.r0_at(temp_c) * current
        dy_dz = float(docv_dz(z))
        if self.residual is not None:
            g, dg_dz = self.residual.eval(z, current, temp_c)
            y += g
            dy_dz += dg_dz
        # Keep the SOC channel observable even if the network's slope is wrong.
        return y, np.array([max(dy_dz, 0.05), -1.0])

    def update(self, v_meas, current, temp_c=25.0):
        """Measurement update. Returns the innovation (measured - predicted)."""
        self.temp_c = temp_c
        y, H = self.measurement(current, temp_c)
        self.v_pred = y
        S = float(H @ self.P @ H) + self.R
        K = self.P @ H / S
        innov = v_meas - y
        self.x += K * innov
        I_KH = np.eye(2) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * self.R   # Joseph form
        self.x[0] = min(1.0, max(0.0, self.x[0]))
        return innov
