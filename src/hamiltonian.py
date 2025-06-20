import numpy as np


def zeeman_hamiltonian(B_ext):
    """Simple Zeeman splitting for m_S=±1 levels."""
    g = 2.003
    mu_B = 1.3996e6  # MHz/T
    Sz = np.diag([1, 0, -1])
    return g * mu_B * B_ext * Sz
