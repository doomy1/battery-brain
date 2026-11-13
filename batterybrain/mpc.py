"""Fast-charging controllers.

``FastChargeMPC`` is a nonlinear model predictive controller solved with one
sequential-QP step per sample (real-time iteration). It pushes as much charge
current as it can while keeping predicted voltage and temperature inside
their limits.

``TableCharger`` is the conventional baseline: constant current, constant
voltage, with a stepped temperature derating table.

In this module the input u is *charging* current (u >= 0, amps) and
``pol`` is the polarisation voltage in the charging direction (pol = -v1).
"""
import math

import numpy as np

from .ocv import ocv, docv_dz
from .qp import solve_qp


def steady_state_kalman_gain(A, C, Qw, Rv, iters=2000):
    """Filter gain from iterating the Riccati difference equation to convergence."""
    P = Qw.copy()
    K = np.zeros((A.shape[0], C.shape[0]))
    for _ in range(iters):
        S = C @ P @ C.T + Rv
        K = P @ C.T @ np.linalg.inv(S)
        P = A @ (P - K @ C @ P) @ A.T + Qw
    return K


class ThermalObserver:
    """Kalman observer for temperature and an unknown heat disturbance.

    Model:  T+ = T + dt/C_th * (heat_model + d - (T - T_amb)/R_th),   d+ = d
    The disturbance state d [W] soaks up everything the nominal thermal model
    gets wrong, which is what lets the MPC hold the temperature limit without
    steady-state offset (internal model principle).
    """

    def __init__(self, model, temp_c, dt=1.0, sigma_t=0.1):
        self.model = model
        self.dt = dt
        beta = dt / model.c_th
        self.A = np.array([[1.0 - beta / model.r_th, beta], [0.0, 1.0]])
        self.C = np.array([[1.0, 0.0]])
        Qw = np.diag([1e-4, 1e-3])
        self.K = steady_state_kalman_gain(self.A, self.C, Qw, np.array([[sigma_t ** 2]]))[:, 0]
        self.temp = temp_c
        self.d = 0.0

    def update(self, temp_meas, heat_model, ambient_c):
        beta = self.dt / self.model.c_th
        t_pred = self.temp + beta * (heat_model + self.d - (self.temp - ambient_c) / self.model.r_th)
        err = temp_meas - t_pred
        self.temp = t_pred + self.K[0] * err
        self.d = self.d + self.K[1] * err


class FastChargeMPC:
    def __init__(self, model, horizon=30, dt=10.0, i_max=15.0, v_max=4.2, t_max=45.0,
                 soc_target=0.8, w_soc=1e4, w_du=5.0, v_margin=0.005, t_margin=0.3):
        self.model = model
        self.N = N = horizon
        self.dt = dt
        # Plan against slightly tightened limits to leave room for model error
        self.i_max, self.v_max, self.t_max = i_max, v_max - v_margin, t_max - t_margin
        self.soc_target = soc_target
        self.w_soc, self.w_du = w_soc, w_du
        self.r_eff = model.r0          # adapted online by the caller
        self.u_nom = np.zeros(N)
        self.u_prev = 0.0
        self.last_info = {}

        # SOC and polarisation are linear in u, so their sensitivities are fixed.
        # Row k is the state at step k (k = 0..N), column j is input u_j.
        c = dt / (model.q_ah * 3600.0)
        self.a = math.exp(-dt / model.tau1)
        b = model.r1 * (1.0 - self.a)
        self.Sz = np.zeros((N + 1, N))
        self.Sp = np.zeros((N + 1, N))
        for k in range(1, N + 1):
            for j in range(k):
                self.Sz[k, j] = c
                self.Sp[k, j] = b * self.a ** (k - 1 - j)
        self.D = np.eye(N) - np.eye(N, k=-1)   # first difference

    def control(self, soc, pol, temp, ambient_c, d_heat=0.0, d_volt=0.0, sqp_iters=2):
        """Return the charging current to apply for the next dt seconds.

        d_heat [W] and d_volt [V] are disturbance estimates supplied by the
        caller; they make the predictions match the plant at the current
        operating point (offset-free MPC).
        """
        m, N = self.model, self.N
        beta = self.dt / m.c_th
        a_t = 1.0 - beta / m.r_th
        r = self.r_eff
        ubar = np.clip(np.append(self.u_nom[1:], self.u_nom[-1]), 0.0, self.i_max)
        powers = self.a ** np.arange(N + 1)

        for _ in range(sqp_iters):
            # Nominal trajectory for the current guess ubar
            z = soc + self.Sz @ ubar
            p = powers * pol + self.Sp @ ubar
            T = np.empty(N + 1)
            T[0] = temp
            ST = np.zeros((N + 1, N))
            for k in range(N):
                T[k + 1] = T[k] + beta * (r * ubar[k] ** 2 + p[k] * ubar[k] + d_heat
                                          - (T[k] - ambient_c) / m.r_th)
                ST[k + 1] = a_t * ST[k] + beta * ubar[k] * self.Sp[k]
                ST[k + 1, k] += beta * (2.0 * r * ubar[k] + p[k])

            # Voltage at steps 0..N-1, linearised about ubar
            zk, pk = z[:-1], p[:-1]
            v_bar = ocv(zk) + pk + r * ubar + d_volt
            Gv = docv_dz(zk)[:, None] * self.Sz[:-1] + self.Sp[:-1] + r * np.eye(N)

            # Decision vector x = [u_0..u_{N-1}, eps_v, eps_T]
            Sz = self.Sz[1:]
            H = np.zeros((N + 2, N + 2))
            H[:N, :N] = 2.0 * (self.w_soc * Sz.T @ Sz + self.w_du * self.D.T @ self.D)
            H[:N, :N] += 1e-8 * np.eye(N)
            H[N, N] = 2.0 * 1e8          # voltage slack [V]
            H[N + 1, N + 1] = 2.0 * 1e5  # temperature slack [K]
            d0 = np.zeros(N)
            d0[0] = self.u_prev
            f = np.zeros(N + 2)
            f[:N] = 2.0 * self.w_soc * Sz.T @ np.full(N, soc - self.soc_target) \
                - 2.0 * self.w_du * self.D.T @ d0
            f[N], f[N + 1] = 1e6, 1e4

            eye, zero = np.eye(N), np.zeros((N, 1))
            one = np.ones((N, 1))
            G = np.block([
                [eye, zero, zero],                 # u <= i_max
                [-eye, zero, zero],                # u >= 0
                [Gv, -one, zero],                  # voltage limit (soft)
                [ST[1:], zero, -one],              # temperature limit (soft)
                [np.zeros((2, N)), -np.eye(2)],    # slacks >= 0
            ])
            h = np.concatenate([
                np.full(N, self.i_max),
                np.zeros(N),
                self.v_max - v_bar + Gv @ ubar,
                self.t_max - T[1:] + ST[1:] @ ubar,
                np.zeros(2),
            ])
            x, info = solve_qp(H, f, G, h)
            ubar = np.clip(x[:N], 0.0, self.i_max)

        self.last_info = info
        self.u_nom = ubar
        self.u_prev = float(ubar[0])
        return self.u_prev


class TableCharger:
    """Conventional charger: CC-CV with a stepped temperature derating table."""

    def __init__(self, i_max=15.0, v_max=4.2, t_max=45.0, ki=20.0):
        self.i_max, self.v_max, self.t_max = i_max, v_max, t_max
        self.ki = ki
        self.u_cv = i_max
        self.stage = 0
        self.t_filt = None

    def control(self, v_meas, temp_meas, dt=1.0):
        # Constant-voltage loop: integrate the voltage error
        self.u_cv = min(self.i_max, max(0.0, self.u_cv + self.ki * (self.v_max - v_meas) * dt))

        # Temperature derating with hysteresis so the stages do not chatter
        self.t_filt = temp_meas if self.t_filt is None else 0.9 * self.t_filt + 0.1 * temp_meas
        thresholds = (self.t_max - 5.0, self.t_max - 2.0, self.t_max - 0.5)
        factors = (1.0, 0.6, 0.3, 0.0)
        while self.stage < 3 and self.t_filt > thresholds[self.stage]:
            self.stage += 1
        while self.stage > 0 and self.t_filt < thresholds[self.stage - 1] - 1.0:
            self.stage -= 1
        return min(self.u_cv, self.i_max * factors[self.stage])
