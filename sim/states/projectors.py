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
Projectors and measurement operators for NV center simulation.

Projectors are used to calculate populations and implement measurements.
A projector P satisfies P² = P and P† = P.

Population Calculation
----------------------
The population in a subspace is:

    p = Tr(P ρ)

where P is the projector onto that subspace.

Basis Structure
---------------
The 18-dimensional basis:
    |idx⟩ = |ge⟩ ⊗ |ms⟩ ⊗ |mI⟩

Ground state manifold: indices 0-8
Excited state manifold: indices 9-17
"""

import numpy as np
from typing import Optional

from .initial_states import state_index


def projector_ground() -> np.ndarray:
    """
    Projector onto ground state manifold P_g.

    Returns
    -------
    np.ndarray
        18×18 projector (indices 0-8)

    Examples
    --------
    >>> P_g = projector_ground()
    >>> rho = ground_state()
    >>> pop = np.real(np.trace(P_g @ rho))
    >>> print(f"Ground population: {pop:.1f}")
    Ground population: 1.0
    """
    P = np.zeros((18, 18), dtype=np.complex128)
    P[:9, :9] = np.eye(9)
    return P


def projector_excited() -> np.ndarray:
    """
    Projector onto excited state manifold P_e.

    Returns
    -------
    np.ndarray
        18×18 projector (indices 9-17)
    """
    P = np.zeros((18, 18), dtype=np.complex128)
    P[9:, 9:] = np.eye(9)
    return P


def projector_ms(ms: int, manifold: str = "both") -> np.ndarray:
    """
    Projector onto electron spin state ms.

    Parameters
    ----------
    ms : int
        Electron spin projection: +1, 0, or -1
    manifold : str
        Which electronic manifold:
        - "ground": Only ground state
        - "excited": Only excited state
        - "both": Both manifolds (default)

    Returns
    -------
    np.ndarray
        18×18 projector

    Examples
    --------
    >>> P_ms0 = projector_ms(0)  # All ms=0 states
    >>> P_ms0_g = projector_ms(0, manifold="ground")  # Only ground
    """
    P = np.zeros((18, 18), dtype=np.complex128)

    # ms_idx: +1 -> 0, 0 -> 1, -1 -> 2
    ms_idx = 1 - ms

    for ge in [0, 1]:
        if manifold == "ground" and ge == 1:
            continue
        if manifold == "excited" and ge == 0:
            continue

        for mI in [+1, 0, -1]:
            idx = state_index(ge, ms, mI)
            P[idx, idx] = 1.0

    return P


def projector_mI(mI: int, manifold: str = "both") -> np.ndarray:
    """
    Projector onto nuclear spin state mI.

    Parameters
    ----------
    mI : int
        Nuclear spin projection: +1, 0, or -1
    manifold : str
        Which electronic manifold: "ground", "excited", or "both"

    Returns
    -------
    np.ndarray
        18×18 projector
    """
    P = np.zeros((18, 18), dtype=np.complex128)

    for ge in [0, 1]:
        if manifold == "ground" and ge == 1:
            continue
        if manifold == "excited" and ge == 0:
            continue

        for ms in [+1, 0, -1]:
            idx = state_index(ge, ms, mI)
            P[idx, idx] = 1.0

    return P


def projector_state(ge: int, ms: int, mI: int) -> np.ndarray:
    """
    Projector onto single basis state |ge, ms, mI⟩.

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
    np.ndarray
        18×18 rank-1 projector
    """
    P = np.zeros((18, 18), dtype=np.complex128)
    idx = state_index(ge, ms, mI)
    P[idx, idx] = 1.0
    return P


def spin_operator_18x18(component: str, manifold: str = "both") -> np.ndarray:
    """
    Electron spin operator in 18×18 space.

    Parameters
    ----------
    component : str
        Spin component: "x", "y", "z", "+", or "-"
    manifold : str
        Which manifold: "ground", "excited", or "both"

    Returns
    -------
    np.ndarray
        18×18 spin operator

    Examples
    --------
    >>> Sz = spin_operator_18x18("z")
    >>> rho = ground_state(ms=1)
    >>> sz = np.real(np.trace(Sz @ rho))
    >>> print(f"<Sz> = {sz:.1f}")
    <Sz> = 1.0
    """
    from ..core.operators import spin1_operators

    S = spin1_operators()

    if component == "x":
        S_op = S.Sx
    elif component == "y":
        S_op = S.Sy
    elif component == "z":
        S_op = S.Sz
    elif component == "+":
        S_op = S.Sp
    elif component == "-":
        S_op = S.Sm
    else:
        raise ValueError(f"Unknown component: {component}")

    # Extend to 18×18
    I3 = np.eye(3, dtype=np.complex128)  # Nuclear spin identity
    S_9x9 = np.kron(S_op, I3)

    if manifold == "both":
        I2 = np.eye(2, dtype=np.complex128)
        return np.kron(I2, S_9x9)
    elif manifold == "ground":
        H = np.zeros((18, 18), dtype=np.complex128)
        H[:9, :9] = S_9x9
        return H
    elif manifold == "excited":
        H = np.zeros((18, 18), dtype=np.complex128)
        H[9:, 9:] = S_9x9
        return H
    else:
        raise ValueError(f"Unknown manifold: {manifold}")


def nuclear_operator_18x18(component: str, manifold: str = "both") -> np.ndarray:
    """
    Nuclear spin operator (N14) in 18×18 space.

    Parameters
    ----------
    component : str
        Spin component: "x", "y", "z", "+", or "-"
    manifold : str
        Which manifold: "ground", "excited", or "both"

    Returns
    -------
    np.ndarray
        18×18 nuclear spin operator
    """
    from ..core.operators import spin1_operators

    I = spin1_operators()  # N14 is also spin-1

    if component == "x":
        I_op = I.Sx
    elif component == "y":
        I_op = I.Sy
    elif component == "z":
        I_op = I.Sz
    elif component == "+":
        I_op = I.Sp
    elif component == "-":
        I_op = I.Sm
    else:
        raise ValueError(f"Unknown component: {component}")

    # Extend to 18×18: I_ge ⊗ I_S ⊗ I_nuclear
    I3_S = np.eye(3, dtype=np.complex128)  # Electron spin identity
    I_9x9 = np.kron(I3_S, I_op)

    if manifold == "both":
        I2 = np.eye(2, dtype=np.complex128)
        return np.kron(I2, I_9x9)
    elif manifold == "ground":
        H = np.zeros((18, 18), dtype=np.complex128)
        H[:9, :9] = I_9x9
        return H
    elif manifold == "excited":
        H = np.zeros((18, 18), dtype=np.complex128)
        H[9:, 9:] = I_9x9
        return H
    else:
        raise ValueError(f"Unknown manifold: {manifold}")


def population_dict(rho: np.ndarray) -> dict:
    """
    Extract all populations from a density matrix.

    Parameters
    ----------
    rho : np.ndarray
        18×18 density matrix

    Returns
    -------
    dict
        Dictionary with keys like "g_ms0_mI0" and population values

    Examples
    --------
    >>> rho = ground_state()
    >>> pops = population_dict(rho)
    >>> print(f"Ground ms=0: {pops['g_ms0_mI0']:.2f}")
    Ground ms=0: 1.00
    """
    pops = {}

    for ge in [0, 1]:
        ge_label = "g" if ge == 0 else "e"
        for ms in [+1, 0, -1]:
            for mI in [+1, 0, -1]:
                idx = state_index(ge, ms, mI)
                pop = np.real(rho[idx, idx])

                # Format ms and mI labels
                ms_str = f"p{ms}" if ms > 0 else (f"m{-ms}" if ms < 0 else "0")
                mI_str = f"p{mI}" if mI > 0 else (f"m{-mI}" if mI < 0 else "0")

                key = f"{ge_label}_ms{ms_str}_mI{mI_str}"
                pops[key] = pop

    return pops
