"""Dense quadratic program solver.

    minimise    0.5 x'Hx + f'x
    subject to  Gx <= h

Mehrotra predictor-corrector primal-dual interior point method, written from
scratch. It is the solver behind the model predictive controller.
"""
import numpy as np


def _max_step(v, dv):
    """Largest alpha in (0, 1] keeping v + alpha*dv >= 0."""
    neg = dv < 0
    if not np.any(neg):
        return 1.0
    return min(1.0, float(np.min(-v[neg] / dv[neg])))


def solve_qp(H, f, G, h, tol=1e-7, max_iter=60):
    """Solve the QP. Returns (x, info) where info has iterations and residuals."""
    n, m = len(f), len(h)
    x = np.zeros(n)
    s = np.maximum(h - G @ x, 1.0)   # slack: Gx + s = h, s > 0
    lam = np.ones(m)                 # multipliers, lam > 0
    scale = 1.0 + max(np.max(np.abs(f)), np.max(np.abs(h)))

    for it in range(1, max_iter + 1):
        r_dual = H @ x + f + G.T @ lam
        r_prim = G @ x + s - h
        mu = float(s @ lam) / m
        if max(np.max(np.abs(r_dual)), np.max(np.abs(r_prim))) < tol * scale and mu < tol:
            break

        d = lam / s
        M = H + G.T @ (d[:, None] * G)

        def newton(r_cent):
            dx = np.linalg.solve(M, -r_dual - G.T @ ((lam * r_prim - r_cent) / s))
            ds = -r_prim - G @ dx
            dlam = (-r_cent - lam * ds) / s
            return dx, ds, dlam

        # Predictor (affine scaling) step
        dx, ds, dlam = newton(s * lam)
        alpha = min(_max_step(s, ds), _max_step(lam, dlam))
        mu_aff = float((s + alpha * ds) @ (lam + alpha * dlam)) / m
        sigma = (mu_aff / mu) ** 3

        # Corrector step with centring
        dx, ds, dlam = newton(s * lam + ds * dlam - sigma * mu)
        alpha = 0.99 * min(_max_step(s, ds), _max_step(lam, dlam))
        x = x + alpha * dx
        s = s + alpha * ds
        lam = lam + alpha * dlam

    return x, {"iterations": it, "gap": mu,
               "primal_residual": float(np.max(np.abs(r_prim))),
               "dual_residual": float(np.max(np.abs(r_dual))), "multipliers": lam}
