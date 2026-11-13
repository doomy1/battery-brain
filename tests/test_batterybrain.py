import math

import numpy as np

from batterybrain import (Cell, CellParams, NominalModel, Sensors, SocEKF, MLP,
                          ChargeRace, EstimationSession, solve_qp)
from batterybrain.mpc import steady_state_kalman_gain
from batterybrain.ocv import ocv, docv_dz


def test_ocv_is_monotonic_and_slope_matches():
    z = np.linspace(0.0, 1.0, 2001)
    assert np.all(np.diff(ocv(z)) > 0)
    numeric = (ocv(z + 1e-6) - ocv(z - 1e-6)) / 2e-6
    assert np.allclose(numeric, docv_dz(z), atol=1e-5)


def test_cell_conserves_charge():
    cell = Cell(CellParams(), soc=0.8)
    for _ in range(3600):
        cell.step(1.0, 1.0)            # 1 A for one hour = 1 Ah out of 5 Ah
    assert abs(cell.soc - 0.6) < 1e-9


def test_qp_unconstrained_matches_linear_solve():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    H = A @ A.T + np.eye(6)
    f = rng.standard_normal(6)
    G = np.vstack([np.eye(6), -np.eye(6)])
    h = np.full(12, 1e3)               # bounds far away, so inactive
    x, _ = solve_qp(H, f, G, h)
    assert np.allclose(x, np.linalg.solve(H, -f), atol=1e-5)


def test_qp_satisfies_kkt_conditions():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n, m = 8, 20
        A = rng.standard_normal((n, n))
        H = A @ A.T + 0.1 * np.eye(n)
        f = 3.0 * rng.standard_normal(n)
        G = rng.standard_normal((m, n))
        h = rng.uniform(0.1, 1.0, m)   # x = 0 is strictly feasible
        x, info = solve_qp(H, f, G, h)
        lam = info["multipliers"]
        assert np.all(G @ x <= h + 1e-6)                       # primal feasible
        assert np.all(lam >= 0)                                # dual feasible
        assert np.allclose(H @ x + f + G.T @ lam, 0, atol=1e-5)  # stationarity
        assert abs(lam @ (h - G @ x)) < 1e-4                   # complementarity


def test_qp_box_constrained_scalar():
    # minimise (x - 3)^2 subject to x <= 1
    x, _ = solve_qp(np.array([[2.0]]), np.array([-6.0]), np.array([[1.0]]), np.array([1.0]))
    assert abs(x[0] - 1.0) < 1e-5


def test_network_gradient_and_input_derivative():
    net = MLP([6, 8, 8, 1], seed=3)
    rng = np.random.default_rng(3)
    X, y = rng.standard_normal((16, 6)), rng.standard_normal(16)
    _, gW, _ = net._gradients(X, y)
    eps = 1e-6
    for layer, (i, j) in ((0, (2, 3)), (1, (4, 1)), (2, (0, 5))):
        net.W[layer][i, j] += eps
        up = np.mean((net.forward(X) - y) ** 2)
        net.W[layer][i, j] -= 2 * eps
        down = np.mean((net.forward(X) - y) ** 2)
        net.W[layer][i, j] += eps
        assert abs((up - down) / (2 * eps) - gW[layer][i, j]) < 1e-6

    x = rng.standard_normal(6)
    value, slope = net.value_and_dx0(x)
    assert abs(value - net.forward(x[None])[0]) < 1e-12
    xp, xm = x.copy(), x.copy()
    xp[0] += eps
    xm[0] -= eps
    numeric = (net.forward(xp[None])[0] - net.forward(xm[None])[0]) / (2 * eps)
    assert abs(numeric - slope) < 1e-6


def test_network_can_learn_a_simple_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (2000, 2))
    y = np.sin(2 * X[:, 0]) * X[:, 1]
    net = MLP([2, 16, 16, 1], seed=0)
    net.fit(X, y, epochs=60, lr=5e-3, batch=64, verbose=False)
    assert np.mean((net.forward(X) - y) ** 2) < 2e-3


def test_ekf_converges_when_model_is_exact():
    """With a plant that matches the filter model, a bad initial guess must be corrected."""
    model = NominalModel()
    rng = np.random.default_rng(0)
    z, v1 = 0.7, 0.0
    ekf = SocEKF(model, soc0=0.3, r_meas=0.002 ** 2)
    a = math.exp(-1.0 / model.tau1)
    for k in range(1800):
        i = 5.0 if (k // 60) % 2 == 0 else -2.0
        z -= i / (model.q_ah * 3600.0)
        v1 = a * v1 + model.r1 * (1 - a) * i
        v = float(ocv(z)) - v1 - model.r0 * i + 0.002 * rng.standard_normal()
        ekf.predict(i, 1.0)
        ekf.update(v, i, 25.0)
    assert abs(ekf.soc - z) < 0.01
    assert abs(ekf.soc - z) < 4 * ekf.soc_sigma + 0.005


def test_kalman_gain_solves_riccati_equation():
    A = np.array([[0.95, 0.014], [0.0, 1.0]])
    C = np.array([[1.0, 0.0]])
    Qw, Rv = np.diag([1e-4, 1e-3]), np.array([[0.01]])
    K = steady_state_kalman_gain(A, C, Qw, Rv)
    assert np.all(np.abs(np.linalg.eigvals(A - A @ K @ C)) < 1.0)   # stable observer


def test_hybrid_estimator_beats_plain_ekf_in_the_cold():
    s = EstimationSession("drive", ambient_c=5.0, seed=21)
    s.scramble(0.2)
    s.step(900)
    s.reset_stats()
    rmse = s.step(2700)["rmse"]
    assert rmse["nn"] < 0.01
    assert rmse["nn"] < 0.5 * rmse["ekf"]
    assert rmse["ekf"] < rmse["cc"]


def test_mpc_charges_faster_and_respects_limits():
    race = ChargeRace(ambient_c=25.0)
    while not race.finished and race.t < 7200:
        out = race.step(120)
    summary = out["summary"]
    assert summary["mpc"]["done"] and summary["table"]["done"]
    assert summary["mpc"]["finish_time"] < summary["table"]["finish_time"]
    assert summary["mpc"]["peak_temp"] < 45.5
    assert summary["mpc"]["peak_volt"] < 4.22
