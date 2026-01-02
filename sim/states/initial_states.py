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
Initial state preparation for NV center simulation.

Provides functions to create common initial states for NV center
experiments. The standard initialization is the |g, ms=0, mI=0⟩ state
achieved through optical pumping.

Basis Indexing
--------------
The 18-dimensional basis follows the convention:

    |idx⟩ = |ge⟩ ⊗ |ms⟩ ⊗ |mI⟩

    Index = 9 * ge + 3 * ms_idx + mI_idx

where:
    ge = 0 (ground) or 1 (excited)
    ms_idx = 0 (+1), 1 (0), 2 (-1)
    mI_idx = 0 (+1), 1 (0), 2 (-1)

Common indices:
    |g, +1, +1⟩ = 0
    |g, +1,  0⟩ = 1
    |g,  0,  0⟩ = 4  ← Standard initialization
    |g, -1,  0⟩ = 7
    |e,  0,  0⟩ = 13
"""

import numpy as np
from typing import Dict, Tuple, Optional

from .density_matrix import DensityMatrix


def state_index(ge: int, ms: int, mI: int) -> int:
    """
    Convert quantum numbers to state index.

    Parameters
    ----------
    ge : int
        Electronic state: 0 (ground) or 1 (excited)
    ms : int
        Electron spin: +1, 0, or -1
    mI : int
        Nuclear spin: +1, 0, or -1

    Returns
    -------
    int
        Index in 0-17 range
    """
    ms_idx = 1 - ms  # +1 -> 0, 0 -> 1, -1 -> 2
    mI_idx = 1 - mI  # +1 -> 0, 0 -> 1, -1 -> 2
    return 9 * ge + 3 * ms_idx + mI_idx


def ground_state_vector(ms: int = 0, mI: int = 0) -> np.ndarray:
    """
    Create pure state vector |g, ms, mI⟩.

    Parameters
    ----------
    ms : int
        Electron spin projection: +1, 0, or -1. Default: 0
    mI : int
        Nuclear spin projection: +1, 0, or -1. Default: 0

    Returns
    -------
    np.ndarray
        18-dimensional normalized state vector

    Examples
    --------
    >>> psi = ground_state_vector()  # |g, 0, 0⟩
    >>> print(f"Norm: {np.linalg.norm(psi):.1f}")
    Norm: 1.0
    >>> psi_m1 = ground_state_vector(ms=-1)  # |g, -1, 0⟩
    """
    psi = np.zeros(18, dtype=np.complex128)
    idx = state_index(0, ms, mI)
    psi[idx] = 1.0
    return psi


def ground_state(ms: int = 0, mI: int = 0) -> np.ndarray:
    """
    Create density matrix for |g, ms, mI⟩⟨g, ms, mI|.

    Parameters
    ----------
    ms : int
        Electron spin: +1, 0, or -1. Default: 0
    mI : int
        Nuclear spin: +1, 0, or -1. Default: 0

    Returns
    -------
    np.ndarray
        18×18 density matrix

    Examples
    --------
    >>> rho = ground_state()  # Standard NV initialization
    >>> print(f"Trace: {np.trace(rho):.1f}")
    Trace: 1.0
    """
    psi = ground_state_vector(ms, mI)
    return np.outer(psi, psi.conj())


def excited_state(ms: int = 0, mI: int = 0) -> np.ndarray:
    """
    Create density matrix for |e, ms, mI⟩⟨e, ms, mI|.

    Parameters
    ----------
    ms : int
        Electron spin: +1, 0, or -1. Default: 0
    mI : int
        Nuclear spin: +1, 0, or -1. Default: 0

    Returns
    -------
    np.ndarray
        18×18 density matrix
    """
    psi = np.zeros(18, dtype=np.complex128)
    idx = state_index(1, ms, mI)
    psi[idx] = 1.0
    return np.outer(psi, psi.conj())


def superposition(
    coefficients: Dict[Tuple[int, int, int], complex],
    normalize: bool = True
) -> np.ndarray:
    """
    Create superposition state from coefficients.

    Parameters
    ----------
    coefficients : dict
        Mapping (ge, ms, mI) -> coefficient
    normalize : bool
        Whether to normalize the state. Default: True

    Returns
    -------
    np.ndarray
        18×18 density matrix

    Examples
    --------
    >>> # |ψ⟩ = (|0⟩ + |−1⟩)/√2
    >>> coeffs = {
    ...     (0, 0, 0): 1.0,
    ...     (0, -1, 0): 1.0
    ... }
    >>> rho = superposition(coeffs)
    >>> print(f"Purity: {np.real(np.trace(rho @ rho)):.3f}")
    Purity: 1.000

    >>> # Mixed nuclear spin
    >>> coeffs = {
    ...     (0, 0, +1): 1/np.sqrt(3),
    ...     (0, 0, 0): 1/np.sqrt(3),
    ...     (0, 0, -1): 1/np.sqrt(3)
    ... }
    >>> rho = superposition(coeffs)
    """
    psi = np.zeros(18, dtype=np.complex128)

    for (ge, ms, mI), coeff in coefficients.items():
        idx = state_index(ge, ms, mI)
        psi[idx] = coeff

    if normalize:
        norm = np.linalg.norm(psi)
        if norm > 0:
            psi = psi / norm

    return np.outer(psi, psi.conj())


def thermal_state(
    H: np.ndarray,
    temperature: float
) -> np.ndarray:
    """
    Create thermal equilibrium state ρ = exp(-βH) / Z.

    Parameters
    ----------
    H : np.ndarray
        18×18 Hamiltonian matrix in rad/s
    temperature : float
        Temperature in Kelvin

    Returns
    -------
    np.ndarray
        18×18 density matrix

    Examples
    --------
    >>> from sim import HamiltonianBuilder
    >>> from sim.hamiltonian.terms import ZFS
    >>>
    >>> H = HamiltonianBuilder().add(ZFS(D=2.87)).build()
    >>> rho_300K = thermal_state(H, 300)  # Room temperature
    >>> rho_4K = thermal_state(H, 4)      # Cryogenic
    """
    dm = DensityMatrix.from_thermal(H, temperature)
    return dm.rho


def maximally_mixed() -> np.ndarray:
    """
    Create maximally mixed state ρ = I/18.

    Returns
    -------
    np.ndarray
        18×18 density matrix with equal populations

    Notes
    -----
    This state has maximum entropy S = log(18) and
    minimum purity Tr(ρ²) = 1/18.
    """
    return np.eye(18, dtype=np.complex128) / 18


def ms0_superposition(phase: float = 0.0, mI: int = 0) -> np.ndarray:
    """
    Create superposition of ms=+1 and ms=-1 states.

    |ψ⟩ = (|+1⟩ + e^(iφ)|−1⟩)/√2

    Parameters
    ----------
    phase : float
        Relative phase in radians. Default: 0
    mI : int
        Nuclear spin state. Default: 0

    Returns
    -------
    np.ndarray
        18×18 density matrix
    """
    coeffs = {
        (0, +1, mI): 1.0,
        (0, -1, mI): np.exp(1j * phase)
    }
    return superposition(coeffs)


def bell_state_electron_nuclear(
    state_type: str = "phi_plus",
    ge: int = 0
) -> np.ndarray:
    """
    Create Bell state between electron and nuclear spin.

    Parameters
    ----------
    state_type : str
        Type of Bell state:
        - "phi_plus": (|+1,+1⟩ + |−1,−1⟩)/√2
        - "phi_minus": (|+1,+1⟩ − |−1,−1⟩)/√2
        - "psi_plus": (|+1,−1⟩ + |−1,+1⟩)/√2
        - "psi_minus": (|+1,−1⟩ − |−1,+1⟩)/√2
    ge : int
        Electronic manifold: 0 (ground) or 1 (excited)

    Returns
    -------
    np.ndarray
        18×18 density matrix
    """
    if state_type == "phi_plus":
        coeffs = {(ge, +1, +1): 1, (ge, -1, -1): 1}
    elif state_type == "phi_minus":
        coeffs = {(ge, +1, +1): 1, (ge, -1, -1): -1}
    elif state_type == "psi_plus":
        coeffs = {(ge, +1, -1): 1, (ge, -1, +1): 1}
    elif state_type == "psi_minus":
        coeffs = {(ge, +1, -1): 1, (ge, -1, +1): -1}
    else:
        raise ValueError(f"Unknown Bell state: {state_type}")

    return superposition(coeffs)
