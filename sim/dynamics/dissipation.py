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
Dissipation operators for Lindblad master equation.

Lindblad operators describe irreversible processes like:
- T1 relaxation (spin-lattice relaxation)
- T2* dephasing (inhomogeneous broadening)
- T2 pure dephasing (homogeneous)
- Optical decay (spontaneous emission)

Lindblad Form
-------------
Each dissipation channel is described by a Lindblad operator L_k
that appears in the master equation as:

    D[ρ] = L ρ L† - ½{L†L, ρ}

The rate is incorporated into the operator: L = √γ × (bare operator)

Time Scales for NV Centers
--------------------------
- T1 (spin): 1-10 ms at room temperature
- T2* (inhomogeneous): 1-10 μs
- T2 (Hahn echo): 100-500 μs
- Optical T1: ~12 ns (excited state lifetime)
"""

import numpy as np
from typing import List

from ..core.operators import spin1_operators, extend_to_18x18, extend_electron_only


def t1_operators(
    gamma: float,
    manifold: str = "ground"
) -> List[np.ndarray]:
    """
    Create T1 relaxation (spin-lattice) Lindblad operators.

    Models incoherent transitions between spin states, driving
    the system toward thermal equilibrium.

    Parameters
    ----------
    gamma : float
        Relaxation rate 1/T1 in Hz
    manifold : str
        Which manifold: "ground", "excited", or "both"

    Returns
    -------
    list of np.ndarray
        List of 18×18 Lindblad operators [L_+, L_-]

    Notes
    -----
    For spin-1 systems, we use:
        L_+ = √γ · S_+  (excitation)
        L_- = √γ · S_-  (relaxation)

    At T=0, only L_- contributes (pure decay).
    At finite T, both contribute with rates determined by
    detailed balance.

    Examples
    --------
    >>> ops = t1_operators(gamma=1e3)  # T1 = 1 ms
    >>> len(ops)
    2
    """
    S = spin1_operators()

    # Scale by sqrt(gamma)
    sqrt_gamma = np.sqrt(gamma)

    # Create 3×3 operators
    L_plus_3x3 = sqrt_gamma * S.Sp
    L_minus_3x3 = sqrt_gamma * S.Sm

    # Extend to 18×18
    I3 = np.eye(3, dtype=np.complex128)

    if manifold == "both":
        L_plus = extend_to_18x18(L_plus_3x3)
        L_minus = extend_to_18x18(L_minus_3x3)
    elif manifold == "ground":
        L_plus = extend_electron_only(L_plus_3x3, "ground")
        L_minus = extend_electron_only(L_minus_3x3, "ground")
    elif manifold == "excited":
        L_plus = extend_electron_only(L_plus_3x3, "excited")
        L_minus = extend_electron_only(L_minus_3x3, "excited")
    else:
        raise ValueError(f"Unknown manifold: {manifold}")

    return [L_plus, L_minus]


def t2_dephasing_operator(
    gamma: float,
    manifold: str = "ground"
) -> np.ndarray:
    """
    Create T2* dephasing Lindblad operator.

    Models pure dephasing that destroys coherences without
    changing populations.

    Parameters
    ----------
    gamma : float
        Dephasing rate 1/T2* in Hz
    manifold : str
        Which manifold: "ground", "excited", or "both"

    Returns
    -------
    np.ndarray
        18×18 Lindblad operator

    Notes
    -----
    The dephasing operator is L = √γ · Sz.

    This causes decay of off-diagonal elements (coherences)
    at rate γ, while diagonal elements (populations) are unchanged.

    The total dephasing rate is:
        1/T2* = 1/(2T1) + 1/T_φ

    where T_φ is the pure dephasing time.

    Examples
    --------
    >>> L = t2_dephasing_operator(gamma=1e7)  # T2* = 100 ns
    >>> L.shape
    (18, 18)
    """
    S = spin1_operators()

    sqrt_gamma = np.sqrt(gamma)
    L_3x3 = sqrt_gamma * S.Sz

    if manifold == "both":
        return extend_to_18x18(L_3x3)
    elif manifold == "ground":
        return extend_electron_only(L_3x3, "ground")
    elif manifold == "excited":
        return extend_electron_only(L_3x3, "excited")
    else:
        raise ValueError(f"Unknown manifold: {manifold}")


def optical_decay_operators(gamma: float) -> List[np.ndarray]:
    """
    Create optical decay (spontaneous emission) operators.

    Models transitions from excited to ground state with
    photon emission. Spin-conserving transitions.

    Parameters
    ----------
    gamma : float
        Spontaneous emission rate in Hz (~1/12 ns ≈ 8×10⁷ Hz)

    Returns
    -------
    list of np.ndarray
        List of 9 Lindblad operators (one per |e,ms,mI⟩ → |g,ms,mI⟩)

    Notes
    -----
    Each spin state decays independently:
        L_{ms,mI} = √γ · |g,ms,mI⟩⟨e,ms,mI|

    The total emission rate from the excited state is:
        Γ = 9γ (summed over all channels)

    For NV: τ_rad ≈ 12 ns → γ ≈ 8×10⁷ Hz

    Examples
    --------
    >>> ops = optical_decay_operators(gamma=8e7)
    >>> len(ops)
    9
    """
    sqrt_gamma = np.sqrt(gamma)
    operators = []

    # For each spin state, create |g⟩⟨e| transition
    for ms_idx in range(3):  # ms = +1, 0, -1
        for mI_idx in range(3):  # mI = +1, 0, -1
            # State indices
            g_idx = 3 * ms_idx + mI_idx       # Ground: 0-8
            e_idx = 9 + 3 * ms_idx + mI_idx   # Excited: 9-17

            # Create |g⟩⟨e| operator
            L = np.zeros((18, 18), dtype=np.complex128)
            L[g_idx, e_idx] = sqrt_gamma

            operators.append(L)

    return operators


def nuclear_relaxation_operators(
    gamma: float,
    manifold: str = "ground"
) -> List[np.ndarray]:
    """
    Create nuclear spin T1 relaxation operators.

    Models incoherent transitions of the N14 nuclear spin.

    Parameters
    ----------
    gamma : float
        Nuclear relaxation rate in Hz (typically ~100 Hz)
    manifold : str
        Which manifold: "ground", "excited", or "both"

    Returns
    -------
    list of np.ndarray
        List of 18×18 Lindblad operators [L_+, L_-]

    Notes
    -----
    Nuclear T1 is typically very long (~seconds) but can be
    accelerated by certain processes.
    """
    I = spin1_operators()  # N14 is spin-1

    sqrt_gamma = np.sqrt(gamma)

    # Nuclear spin operators
    I_plus_3x3 = sqrt_gamma * I.Sp
    I_minus_3x3 = sqrt_gamma * I.Sm

    # Extend: I_ge ⊗ I_S ⊗ I_nuclear
    I3_S = np.eye(3, dtype=np.complex128)  # Electron spin identity

    I_plus_9x9 = np.kron(I3_S, I_plus_3x3)
    I_minus_9x9 = np.kron(I3_S, I_minus_3x3)

    if manifold == "both":
        I2 = np.eye(2, dtype=np.complex128)
        L_plus = np.kron(I2, I_plus_9x9)
        L_minus = np.kron(I2, I_minus_9x9)
    elif manifold == "ground":
        L_plus = np.zeros((18, 18), dtype=np.complex128)
        L_minus = np.zeros((18, 18), dtype=np.complex128)
        L_plus[:9, :9] = I_plus_9x9
        L_minus[:9, :9] = I_minus_9x9
    elif manifold == "excited":
        L_plus = np.zeros((18, 18), dtype=np.complex128)
        L_minus = np.zeros((18, 18), dtype=np.complex128)
        L_plus[9:, 9:] = I_plus_9x9
        L_minus[9:, 9:] = I_minus_9x9
    else:
        raise ValueError(f"Unknown manifold: {manifold}")

    return [L_plus, L_minus]


def intersystem_crossing_operators(
    gamma_isc: float,
    gamma_singlet: float
) -> List[np.ndarray]:
    """
    Create intersystem crossing (ISC) operators.

    Models the non-radiative decay pathway through singlet states
    that enables optical spin polarization.

    Parameters
    ----------
    gamma_isc : float
        ISC rate from excited state (spin-dependent)
    gamma_singlet : float
        Decay rate from singlet back to ground state

    Returns
    -------
    list of np.ndarray
        Lindblad operators for ISC process

    Notes
    -----
    This is a simplified model. The full ISC involves:
    - Decay from ³E (ms=±1) to singlet states
    - Decay from singlet to ³A₂ (preferentially ms=0)

    This is what enables optical spin polarization.
    """
    # Simplified: Direct |e,±1⟩ → |g,0⟩ transitions
    # This captures the essential physics of spin polarization

    sqrt_gamma = np.sqrt(gamma_isc * gamma_singlet / (gamma_isc + gamma_singlet))

    operators = []

    # From |e, +1, mI⟩ to |g, 0, mI⟩
    for mI_idx in range(3):
        e_p1_idx = 9 + 0 * 3 + mI_idx  # |e, +1, mI⟩
        g_0_idx = 1 * 3 + mI_idx        # |g, 0, mI⟩

        L = np.zeros((18, 18), dtype=np.complex128)
        L[g_0_idx, e_p1_idx] = sqrt_gamma
        operators.append(L)

    # From |e, -1, mI⟩ to |g, 0, mI⟩
    for mI_idx in range(3):
        e_m1_idx = 9 + 2 * 3 + mI_idx  # |e, -1, mI⟩
        g_0_idx = 1 * 3 + mI_idx        # |g, 0, mI⟩

        L = np.zeros((18, 18), dtype=np.complex128)
        L[g_0_idx, e_m1_idx] = sqrt_gamma
        operators.append(L)

    return operators
