from itertools import combinations
import numpy as np


def cce_level2_t2_star(spin_positions, couplings):
    """Estimate T2* from pairwise flip-flops in a cluster expansion."""
    pairs = list(combinations(range(len(spin_positions)), 2))
    decoh_rate = 0.0
    for i, j in pairs:
        Aij = couplings[i, j]
        delta = 1.0  # placeholder for energy mismatch
        decoh_rate += (Aij / 2.0) ** 2 / delta
    return 1.0 / np.sqrt(decoh_rate)
