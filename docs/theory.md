# Theory notes

How each piece works, with the equations the code implements. Lecture numbers
refer to UIUC ME 462 (Advanced Computer Control), which covers most of this.

## 1. The plant and the model

Sign convention: current `i > 0` discharges the cell.

**Truth plant** (`cell.py`), state `[z, v1, v2, h, T]`:

```
z+   = z - i*dt / Q                       state of charge
v1+  = a1*v1 + R1*(1 - a1)*i              fast polarisation,  a1 = exp(-dt/tau1)
v2+  = a2*v2 + R2*(1 - a2)*i              slow polarisation
h+   = h + k*( -sign(i) - h )             hysteresis,  k = 1 - exp(-gamma*|i|*dt/Q)
T+   = T + dt/C_th * ( i^2*R0 + i*(v1+v2) - (T - T_amb)/R_th )
V    = OCV(z) + M*h - v1 - v2 - R0*i
```

`R0, R1, R2` follow an Arrhenius law in temperature and rise at low state of
charge. The RC updates are the exact zero-order-hold discretisation of
`dv/dt = -v/tau + i/C` (Lecture 1, sampling of continuous-time systems).

**Nominal model** (what the algorithms know): one RC pair, datasheet `R0, R1`
with a rough temperature correction, no hysteresis, no SOC dependence.

## 2. Extended Kalman filter (`ekf.py`, Lecture 10)

State `x = [z, v1]`, input `i`, measurement `y = V`.

```
predict:   x^- = f(x, i)             F = [[1, 0], [0, a1]]
           P^- = F P F' + Q
update:    H   = [ dOCV/dz (+ dg/dz),  -1 ]
           K   = P^- H' / (H P^- H' + R)
           x   = x^- + K (y - h(x^-))
           P   = (I - K H) P^- (I - K H)' + K R K'      (Joseph form)
```

The process model is linear; only the measurement is linearised. Why it beats
coulomb counting: the pair (F, H) is observable because `dOCV/dz > 0`, so
voltage carries information about `z` and a wrong initial condition decays.
Coulomb counting is the same model with `K = 0`: an integrator with no
feedback, so initial error and sensor bias never leave.

`R` is set from the measured voltage residual of each model and inflated,
because model error is correlated in time and a Kalman filter that treats it
as white becomes overconfident.

## 3. Learned residual (`residual_net.py`)

The network `g` (6 inputs, two tanh layers of 24, linear output) is trained to
predict `V_measured - V_nominal_model` from

```
[ z,  i,  T,  i filtered at 30 s,  i filtered at 300 s,  hysteresis proxy ]
```

The filtered currents give a static network the memory it needs for the slow
RC pair the nominal model lacks. Training data comes from simulated lab
cycling where true `z` is known. Loss is mean squared error, optimiser is
Adam, all written by hand; `tests/` checks the backprop gradients against
finite differences.

The filter then uses `h(x) = OCV(z) - v1 - R0*i + g(z, ...)` and adds `dg/dz`
(computed analytically through the network) to the Jacobian. The Kalman
structure, and its covariance, stay intact. This is a grey-box design: physics
for what is known, learning only for the remainder. A pure black-box network
mapping measurements to SOC would have no uncertainty estimate and no reason
to extrapolate sensibly.

## 4. Model predictive charging (`mpc.py`, Lectures 9, 20 to 23)

Decision: charging currents `u_0 .. u_{N-1}` over `N = 30` steps of 10 s.

```
minimise    sum_k  w_z (z_k - z_target)^2  +  w_du (u_k - u_{k-1})^2  +  slack penalties
subject to  0 <= u_k <= I_max
            OCV(z_k) + p_k + R*u_k + d_v  <=  V_max + eps_v
            T_k                            <=  T_max + eps_T
            eps_v, eps_T >= 0
```

`z` and the polarisation `p` are linear in `u`, so their sensitivities are
constant matrices built once (a condensed formulation: states are eliminated,
only inputs remain). Temperature is nonlinear through `R*u^2`; it is
linearised about the previous plan, giving one QP per sequential-QP iteration.
Only `u_0` is applied, then everything repeats (receding horizon).

Three things that mattered in practice:

- **Soft constraints (Lecture 22).** Slack variables with large linear and
  quadratic penalties keep the QP feasible if a limit is already violated.
- **Move penalty.** With heat linearised, the optimiser sees no cost in
  pulsing the current, and with a small `w_du` it produced a bang-bang plan
  that chattered. Because true heating is convex in `u`, pulsing is actually
  worse than a steady current. Raising `w_du` removed it.
- **Offset-free prediction (Lectures 1, 8).** A Kalman observer estimates a
  constant heat disturbance `d` on top of the nominal thermal model, and a
  filtered voltage offset `d_v` does the same for voltage. Feeding these into
  the prediction is the internal model principle: the controller holds the
  temperature limit with no steady-state offset despite a wrong model. The
  observer gain comes from iterating the Riccati difference equation.

## 5. Interior-point QP solver (`qp.py`, Lecture 21)

For `min 0.5 x'Hx + f'x  s.t.  Gx <= h` the KKT conditions with slack `s` and
multipliers `lam` are

```
Hx + f + G'lam = 0,     Gx + s = h,     s_i * lam_i = 0,     s, lam >= 0
```

Each iteration takes a Newton step on these equations with the last one
relaxed to `s_i * lam_i = sigma * mu`. Eliminating `ds` and `dlam` leaves one
`n x n` system `(H + G' diag(lam/s) G) dx = rhs`. Mehrotra's method solves it
twice per iteration: a predictor step with `sigma = 0` to see how far the
duality gap can drop, then a corrector that uses that information to choose
`sigma` and to cancel the second-order error. `tests/` verifies the KKT
conditions on random problems.

## What I would do next

1. Replace the simulated plant with a public dataset (CALCE, or the Panasonic
   18650PF set from McMaster) and retrain the network on real measurements.
2. Estimate capacity online (a slow dual filter) so ageing does not bias SOC.
3. Use the learned residual inside the MPC's voltage prediction as well.
4. Port the EKF and the QP solver to fixed-point-friendly C and measure
   execution time on a microcontroller.
5. Add a lithium-plating constraint, which is what really limits fast charging
   in the cold.
