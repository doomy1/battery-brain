"""Load current profiles (discharge positive, amps) for a 5 Ah cell.

Each profile is an endless, seeded, piecewise-constant sequence of segments
generated on demand, so the simulator can run for as long as you like and
every run is reproducible.
"""
import numpy as np

C_RATE = 5.0  # amps at 1C


class Profile:
    kinds = ("drive", "home", "pulse", "cycle")

    def __init__(self, kind="drive", seed=0):
        if kind not in self.kinds:
            raise ValueError(f"unknown profile {kind!r}")
        self.kind = kind
        self.rng = np.random.default_rng(seed)
        self._t_end = 0.0
        self._value = 0.0
        self._script = []   # queued (duration, amps) segments
        self._direction = 1.0

    def current(self, t):
        """Current at time t [s]. Call with non-decreasing t."""
        while t >= self._t_end:
            if not self._script:
                self._script = list(getattr(self, "_gen_" + self.kind)())
            dur, amps = self._script.pop(0)
            self._t_end += dur
            self._value = amps
        return self._value

    # --- segment generators -------------------------------------------------
    def _gen_drive(self):
        """Stop-and-go driving: acceleration bursts, cruising, regen braking."""
        r = self.rng
        yield r.uniform(5, 15), r.uniform(1.5, 3.2) * C_RATE      # accelerate
        yield r.uniform(20, 90), r.uniform(0.3, 1.0) * C_RATE     # cruise
        if r.random() < 0.7:
            yield r.uniform(4, 12), -r.uniform(0.5, 1.5) * C_RATE  # regen
        if r.random() < 0.4:
            yield r.uniform(5, 40), 0.02 * C_RATE                  # stopped

    def _gen_home(self):
        """Home backup: base load, appliance spikes, solar charging."""
        r = self.rng
        yield r.uniform(60, 300), r.uniform(0.10, 0.25) * C_RATE
        if r.random() < 0.6:
            yield r.uniform(20, 120), r.uniform(0.8, 1.6) * C_RATE
        if r.random() < 0.35:
            yield r.uniform(200, 600), -r.uniform(0.3, 0.7) * C_RATE

    def _gen_pulse(self):
        """Lab-style characterisation pulses with rests."""
        yield 10, 2.0 * C_RATE
        yield 40, 0.0
        yield 10, -1.5 * C_RATE
        yield 40, 0.0
        yield 300, 0.5 * C_RATE
        yield 60, 0.0

    def _gen_cycle(self):
        """Sustained charge / discharge blocks, including fast charging."""
        r = self.rng
        self._direction = -self._direction
        mag = r.uniform(0.5, 3.0) * C_RATE
        yield r.uniform(120, 900), self._direction * mag
        if r.random() < 0.3:
            yield r.uniform(30, 300), 0.0
