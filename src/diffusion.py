import numpy as np


def spectral_diffusion_ou(Tcorr=0.005, sigma=0.2):
    """Return a callable sampling an Ornstein-Uhlenbeck process."""
    def sample_ou(prev, dt):
        return prev + (-prev / Tcorr) * dt + sigma * np.sqrt(2 / Tcorr) * np.random.randn()
    return sample_ou
