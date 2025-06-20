import numpy as np


def debeye_waller_factor(T):
    """Temperature-dependent Debye-Waller factor from Santori et al."""
    DW0 = 0.03
    DW_inf = 0.6
    Tc = 77.0
    alpha = 2.0
    return DW0 + (DW_inf - DW0) / (1 + (T / Tc) ** alpha)
