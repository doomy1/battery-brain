"""Open-circuit voltage curve for an NMC-style lithium-ion cell.

A smooth analytic curve is used instead of a lookup table so that the slope
dOCV/dSOC (needed by the EKF measurement Jacobian and the MPC voltage
constraint) is exact and cheap.
"""
import numpy as np


def ocv(z):
    """Open-circuit voltage [V] at state of charge z in [0, 1]."""
    return 3.40 + 0.75 * z - 0.35 * z**2 + 0.38 * z**3 - 0.25 * np.exp(-25.0 * z)


def docv_dz(z):
    """Slope of the OCV curve [V per unit SOC]. Strictly positive on [0, 1]."""
    return 0.75 - 0.70 * z + 1.14 * z**2 + 6.25 * np.exp(-25.0 * z)
