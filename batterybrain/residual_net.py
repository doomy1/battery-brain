"""A small neural network that learns what the physics model gets wrong.

The nominal equivalent-circuit model predicts terminal voltage as

    v_model = OCV(z) - v1 - R0 * i

The real cell also has a second, slower polarisation, hysteresis, and
resistances that move with temperature and state of charge. This network
learns the leftover

    g(z, i, T, history) = v_measured - v_model

so the EKF can use ``v_model + g`` as its measurement equation.

Everything is plain NumPy (forward pass, backprop, Adam) so the exact same
code runs in the browser demo through Pyodide.
"""
import json
import math
from pathlib import Path

import numpy as np

OUT_SCALE = 0.1   # network output of 1.0 means 100 mV
WEIGHTS_PATH = Path(__file__).parent / "weights" / "residual_net.json"


class MLP:
    """Fully connected tanh network with a linear output."""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.standard_normal((o, i)) * math.sqrt(1.0 / i)
                  for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(o) for o in sizes[1:]]

    # --- inference ----------------------------------------------------------
    def forward(self, X):
        """Batch forward pass. X has shape (n, n_in); returns shape (n,)."""
        a = X
        for W, b in zip(self.W[:-1], self.b[:-1]):
            a = np.tanh(a @ W.T + b)
        return (a @ self.W[-1].T + self.b[-1])[:, 0]

    def value_and_dx0(self, x):
        """Output and its derivative with respect to the first input, for one sample."""
        a = x
        back = self.W[0][:, 0]            # d(pre-activation 1) / d x0
        for k, (W, b) in enumerate(zip(self.W[:-1], self.b[:-1])):
            a = np.tanh(W @ a + b)
            back = (1.0 - a * a) * back   # through tanh
            back = self.W[k + 1] @ back   # through next linear layer
        y = self.W[-1] @ a + self.b[-1]
        return float(y[0]), float(back[0])

    # --- training -----------------------------------------------------------
    def _gradients(self, X, y):
        acts = [X]
        for W, b in zip(self.W[:-1], self.b[:-1]):
            acts.append(np.tanh(acts[-1] @ W.T + b))
        pred = (acts[-1] @ self.W[-1].T + self.b[-1])[:, 0]
        err = pred - y
        delta = (2.0 / len(y)) * err[:, None]
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for k in range(len(self.W) - 1, -1, -1):
            gW[k] = delta.T @ acts[k]
            gb[k] = delta.sum(axis=0)
            if k > 0:
                delta = (delta @ self.W[k]) * (1.0 - acts[k] ** 2)
        return float(np.mean(err ** 2)), gW, gb

    def fit(self, X, y, X_val=None, y_val=None, epochs=30, lr=3e-3, batch=256,
            weight_decay=1e-5, seed=0, verbose=True):
        """Train with Adam on mean squared error. Returns the loss history."""
        rng = np.random.default_rng(seed)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        history = []
        for epoch in range(epochs):
            step_lr = lr * 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))
            order = rng.permutation(len(y))
            for start in range(0, len(y), batch):
                idx = order[start:start + batch]
                _, gW, gb = self._gradients(X[idx], y[idx])
                grads = [g + weight_decay * w for g, w in zip(gW, self.W)] + gb
                t += 1
                for p, g, mk, vk in zip(params, grads, m, v):
                    mk *= b1; mk += (1 - b1) * g
                    vk *= b2; vk += (1 - b2) * g * g
                    p -= step_lr * (mk / (1 - b1 ** t)) / (np.sqrt(vk / (1 - b2 ** t)) + eps)
            train = float(np.mean((self.forward(X) - y) ** 2))
            val = float(np.mean((self.forward(X_val) - y_val) ** 2)) if X_val is not None else float("nan")
            history.append((train, val))
            if verbose:
                print(f"epoch {epoch + 1:3d}  train rmse {math.sqrt(train) * OUT_SCALE * 1e3:6.2f} mV"
                      f"  val rmse {math.sqrt(val) * OUT_SCALE * 1e3:6.2f} mV")
        return history

    # --- persistence --------------------------------------------------------
    def to_dict(self):
        return {"W": [w.tolist() for w in self.W], "b": [b.tolist() for b in self.b]}

    @classmethod
    def from_dict(cls, d):
        net = cls.__new__(cls)
        net.W = [np.array(w, dtype=float) for w in d["W"]]
        net.b = [np.array(b, dtype=float) for b in d["b"]]
        return net


class ResidualFeatures:
    """Running features that summarise recent current history.

    The network is a static map, so memory has to come in through its inputs:
    two low-pass filtered currents (slow dynamics the 1-RC model misses) and a
    hysteresis-like state driven by charge throughput.
    """
    TAU_FAST = 30.0
    TAU_SLOW = 300.0
    GAMMA = 60.0

    def __init__(self, q_ah):
        self.q_as = q_ah * 3600.0
        self.f1 = 0.0
        self.f2 = 0.0
        self.s = 0.0

    def advance(self, current, dt):
        a1 = math.exp(-dt / self.TAU_FAST)
        a2 = math.exp(-dt / self.TAU_SLOW)
        self.f1 = a1 * self.f1 + (1.0 - a1) * current
        self.f2 = a2 * self.f2 + (1.0 - a2) * current
        if current != 0.0:
            k = 1.0 - math.exp(-self.GAMMA * abs(current) * dt / self.q_as)
            self.s += k * (-math.copysign(1.0, current) - self.s)

    def vector(self, z, current, temp_c):
        return np.array([z, current / 10.0, (temp_c - 25.0) / 20.0,
                         self.f1 / 10.0, self.f2 / 10.0, self.s])


class LearnedResidual:
    """Glue between the network, its features, and the EKF."""

    def __init__(self, net, q_ah):
        self.net = net
        self.features = ResidualFeatures(q_ah)

    @classmethod
    def load(cls, q_ah, path=WEIGHTS_PATH):
        with open(path) as fh:
            return cls(MLP.from_dict(json.load(fh)), q_ah)

    def advance(self, current, dt):
        self.features.advance(current, dt)

    def eval(self, z, current, temp_c):
        """Voltage correction [V] and its slope with respect to SOC."""
        y, dy = self.net.value_and_dx0(self.features.vector(z, current, temp_c))
        return OUT_SCALE * y, OUT_SCALE * dy
