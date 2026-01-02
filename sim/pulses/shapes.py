# ==============================================================================
#  QUSIM - Quantum Simulator for NV Centers
#  Leon Kaiser, MSQC Goethe University, Frankfurt, Germany
#  https://msqc.cgi-host6.rz.uni-frankfurt.de
#  I.kaiser[at]em.uni-frankfurt.de
#
#  This software is provided for scientific and educational purposes.
#  Free to use, modify, and distribute with attribution.
# ==============================================================================
"""
Pulse envelope shapes for NV center control.

All shape functions return normalized envelopes (max = 1).
The actual amplitude is set when creating the pulse.

Available Shapes
----------------
rectangular : Constant amplitude, sharp edges
gaussian : Smooth Gaussian envelope
sinc : Sin(x)/x for broadband excitation
blackman : Window function with low side lobes
drag : DRAG pulse for leakage reduction
hermite : Hermite-Gaussian shapes
"""

import numpy as np
from typing import Optional, Tuple


def rectangular(t: np.ndarray, duration: float) -> np.ndarray:
    """
    Rectangular (square) pulse envelope.

    Parameters
    ----------
    t : np.ndarray
        Time array in seconds
    duration : float
        Pulse duration in seconds

    Returns
    -------
    np.ndarray
        Envelope values (1 inside pulse, 0 outside)
    """
    return np.where((t >= 0) & (t <= duration), 1.0, 0.0)


def gaussian(
    t: np.ndarray,
    duration: float,
    sigma: Optional[float] = None,
    truncate: float = 3.0
) -> np.ndarray:
    """
    Gaussian pulse envelope.

    Parameters
    ----------
    t : np.ndarray
        Time array in seconds
    duration : float
        Total pulse duration in seconds
    sigma : float, optional
        Standard deviation. Default: duration/6 (3σ truncation each side)
    truncate : float
        Truncation in units of sigma. Default: 3

    Returns
    -------
    np.ndarray
        Gaussian envelope (normalized to max 1)
    """
    if sigma is None:
        sigma = duration / (2 * truncate)

    t0 = duration / 2
    env = np.exp(-((t - t0) / sigma) ** 2 / 2)

    # Zero outside duration
    env = np.where((t >= 0) & (t <= duration), env, 0.0)

    return env


def sinc_pulse(
    t: np.ndarray,
    duration: float,
    zero_crossings: int = 3
) -> np.ndarray:
    """
    Sinc pulse envelope (sin(x)/x).

    Good for broadband excitation with controlled frequency profile.

    Parameters
    ----------
    t : np.ndarray
        Time array in seconds
    duration : float
        Total pulse duration in seconds
    zero_crossings : int
        Number of zero crossings on each side. Default: 3

    Returns
    -------
    np.ndarray
        Sinc envelope (normalized to max 1)
    """
    t0 = duration / 2
    x = zero_crossings * np.pi * (t - t0) / (duration / 2)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        env = np.where(x == 0, 1.0, np.sin(x) / x)

    # Zero outside duration
    env = np.where((t >= 0) & (t <= duration), env, 0.0)

    return env


def blackman(t: np.ndarray, duration: float) -> np.ndarray:
    """
    Blackman window pulse envelope.

    Very low side lobes, good for frequency selectivity.

    Parameters
    ----------
    t : np.ndarray
        Time array in seconds
    duration : float
        Total pulse duration in seconds

    Returns
    -------
    np.ndarray
        Blackman envelope (normalized to max 1)
    """
    # Blackman coefficients
    a0 = 0.42
    a1 = 0.5
    a2 = 0.08

    env = (a0 -
           a1 * np.cos(2 * np.pi * t / duration) +
           a2 * np.cos(4 * np.pi * t / duration))

    # Zero outside duration
    env = np.where((t >= 0) & (t <= duration), env, 0.0)

    # Normalize to max 1
    env = env / np.max(env) if np.max(env) > 0 else env

    return env


def drag(
    t: np.ndarray,
    duration: float,
    sigma: Optional[float] = None,
    beta: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DRAG (Derivative Removal by Adiabatic Gating) pulse.

    Reduces leakage to non-computational states by adding
    a derivative component in quadrature.

    Parameters
    ----------
    t : np.ndarray
        Time array in seconds
    duration : float
        Total pulse duration in seconds
    sigma : float, optional
        Gaussian width. Default: duration/6
    beta : float
        DRAG coefficient. Default: 0.5

    Returns
    -------
    tuple
        (I_component, Q_component) arrays

    Notes
    -----
    I channel: Gaussian
    Q channel: Derivative of Gaussian (scaled by beta)
    """
    if sigma is None:
        sigma = duration / 6

    t0 = duration / 2
    x = (t - t0) / sigma

    # I channel: Gaussian
    I = np.exp(-x**2 / 2)

    # Q channel: Derivative
    Q = -beta * x * I / sigma

    # Zero outside duration
    mask = (t >= 0) & (t <= duration)
    I = np.where(mask, I, 0.0)
    Q = np.where(mask, Q, 0.0)

    return I, Q


def hermite(
    t: np.ndarray,
    duration: float,
    order: int = 0,
    sigma: Optional[float] = None
) -> np.ndarray:
    """
    Hermite-Gaussian pulse envelope.

    Higher orders have more oscillations and are useful for
    selective excitation.

    Parameters
    ----------
    t : np.ndarray
        Time array in seconds
    duration : float
        Total pulse duration in seconds
    order : int
        Hermite polynomial order. Default: 0 (Gaussian)
    sigma : float, optional
        Gaussian width. Default: duration/6

    Returns
    -------
    np.ndarray
        Hermite-Gaussian envelope (normalized to max 1)
    """
    if sigma is None:
        sigma = duration / 6

    t0 = duration / 2
    x = (t - t0) / sigma

    # Hermite polynomials
    if order == 0:
        H = np.ones_like(x)
    elif order == 1:
        H = 2 * x
    elif order == 2:
        H = 4 * x**2 - 2
    elif order == 3:
        H = 8 * x**3 - 12 * x
    elif order == 4:
        H = 16 * x**4 - 48 * x**2 + 12
    else:
        # General case using scipy
        from scipy.special import hermite as hermite_poly
        H_func = hermite_poly(order)
        H = H_func(x)

    # Combine with Gaussian
    env = H * np.exp(-x**2 / 2)

    # Zero outside duration
    env = np.where((t >= 0) & (t <= duration), env, 0.0)

    # Normalize to max abs value = 1
    max_val = np.max(np.abs(env))
    if max_val > 0:
        env = env / max_val

    return env


def cosine(t: np.ndarray, duration: float) -> np.ndarray:
    """
    Cosine (Hann) window pulse envelope.

    Parameters
    ----------
    t : np.ndarray
        Time array in seconds
    duration : float
        Total pulse duration in seconds

    Returns
    -------
    np.ndarray
        Cosine envelope
    """
    env = 0.5 * (1 - np.cos(2 * np.pi * t / duration))
    env = np.where((t >= 0) & (t <= duration), env, 0.0)
    return env


def tanh_edges(
    t: np.ndarray,
    duration: float,
    rise_time: float = 0.1
) -> np.ndarray:
    """
    Pulse with smooth tanh rise and fall.

    Parameters
    ----------
    t : np.ndarray
        Time array in seconds
    duration : float
        Total pulse duration in seconds
    rise_time : float
        Rise/fall time as fraction of duration. Default: 0.1

    Returns
    -------
    np.ndarray
        Tanh-edged envelope
    """
    tr = duration * rise_time

    rise = 0.5 * (1 + np.tanh((t - tr) / (tr / 4)))
    fall = 0.5 * (1 - np.tanh((t - duration + tr) / (tr / 4)))

    env = rise * fall
    env = np.where((t >= 0) & (t <= duration), env, 0.0)

    # Normalize
    max_val = np.max(env)
    if max_val > 0:
        env = env / max_val

    return env
