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
Spin-1 operators for NV center simulation.

This module provides the spin-1 matrices required for describing
the electron spin (S=1) and N14 nuclear spin (I=1) in the NV center.

Physical Background
-------------------
The NV center ground state is a spin triplet (S=1) with states
|ms=+1>, |ms=0>, |ms=-1>. The spin operators satisfy the standard
commutation relations:

    [Sx, Sy] = i*Sz  (and cyclic permutations)
    S² = S(S+1) = 2  for S=1

Basis Convention
----------------
All matrices are represented in the {|+1>, |0>, |-1>} basis, where
Sz is diagonal with eigenvalues +1, 0, -1 on the diagonal.

18-Dimensional Hilbert Space
----------------------------
The full NV center Hilbert space is the tensor product:

    |ψ> = |g/e> ⊗ |ms> ⊗ |mI>

    - |g/e>: Ground/Excited state (2 dim)
    - |ms>:  Electron spin ms ∈ {+1, 0, -1} (3 dim)
    - |mI>:  N14 nuclear spin mI ∈ {+1, 0, -1} (3 dim)

    Total: 2 × 3 × 3 = 18 dimensions

The basis states are ordered as:
    |0>  = |g, +1, +1>
    |1>  = |g, +1,  0>
    |2>  = |g, +1, -1>
    |3>  = |g,  0, +1>
    ...
    |17> = |e, -1, -1>
"""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Spin1Ops:
    """
    Container for spin-1 operators.

    Attributes
    ----------
    Sx : np.ndarray
        Spin-x operator (3×3 complex)
    Sy : np.ndarray
        Spin-y operator (3×3 complex)
    Sz : np.ndarray
        Spin-z operator (3×3 complex, diagonal)
    Sp : np.ndarray
        Raising operator S+ = Sx + i*Sy
    Sm : np.ndarray
        Lowering operator S- = Sx - i*Sy
    I : np.ndarray
        Identity matrix (3×3)
    """
    Sx: np.ndarray
    Sy: np.ndarray
    Sz: np.ndarray
    Sp: np.ndarray
    Sm: np.ndarray
    I: np.ndarray


def spin1_operators() -> Spin1Ops:
    """
    Create spin-1 operators in the {|+1>, |0>, |-1>} basis.

    The matrices satisfy:
        - [Sx, Sy] = i*Sz (and cyclic permutations)
        - S² = Sx² + Sy² + Sz² = 2*I (for S=1)
        - Sz|m> = m|m> for m ∈ {+1, 0, -1}

    Returns
    -------
    Spin1Ops
        Dataclass containing Sx, Sy, Sz, S+, S-, and identity matrix

    Examples
    --------
    >>> ops = spin1_operators()
    >>> np.allclose(ops.Sx @ ops.Sy - ops.Sy @ ops.Sx, 1j * ops.Sz)
    True
    >>> np.linalg.eigvalsh(ops.Sz)
    array([-1.,  0.,  1.])
    """
    # Sx in the {|+1>, |0>, |-1>} basis
    # <m'|Sx|m> = (1/2) * sqrt(S(S+1) - m*m') * (δ_{m',m+1} + δ_{m',m-1})
    # For S=1 this gives the factor 1/√2
    Sx = (1.0 / np.sqrt(2.0)) * np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=np.complex128)

    # Sy = -i * [raising - lowering] / 2
    Sy = (1.0 / np.sqrt(2.0)) * np.array([
        [0, -1j, 0],
        [1j, 0, -1j],
        [0, 1j, 0]
    ], dtype=np.complex128)

    # Sz is diagonal with eigenvalues +1, 0, -1
    Sz = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, -1]
    ], dtype=np.complex128)

    # Raising and lowering operators
    # S+|m> = sqrt(S(S+1) - m(m+1)) |m+1>
    # S-|m> = sqrt(S(S+1) - m(m-1)) |m-1>
    Sp = Sx + 1j * Sy
    Sm = Sx - 1j * Sy

    # Identity matrix
    I = np.eye(3, dtype=np.complex128)

    return Spin1Ops(Sx=Sx, Sy=Sy, Sz=Sz, Sp=Sp, Sm=Sm, I=I)


def extend_to_18x18(H_3x3: np.ndarray) -> np.ndarray:
    """
    Extend a 3×3 electron spin Hamiltonian to the full 18×18 space.

    The operator acts on the electron spin in BOTH manifolds (ground
    and excited) and leaves the nuclear spin unchanged.

    Mathematically:
        H_18 = I_2 ⊗ H_3 ⊗ I_3

    where:
        - I_2: Identity on |g/e> (2×2)
        - H_3: Electron spin operator (3×3)
        - I_3: Identity on |mI> (3×3)

    Parameters
    ----------
    H_3x3 : np.ndarray
        3×3 Hamiltonian in the electron spin subspace

    Returns
    -------
    np.ndarray
        18×18 Hamiltonian in the full Hilbert space

    Notes
    -----
    This extension is correct for terms like ZFS and Zeeman that
    act only on the electron spin and are identical in both electronic
    states (g and e).

    Examples
    --------
    >>> from sim.hamiltonian.terms import ZFS
    >>> zfs = ZFS(D=2.87)
    >>> H_3 = zfs._build_3x3()
    >>> H_18 = extend_to_18x18(H_3)
    >>> H_18.shape
    (18, 18)
    """
    I3 = np.eye(3, dtype=np.complex128)  # Nuclear spin identity
    I2 = np.eye(2, dtype=np.complex128)  # g/e identity

    # Tensor product: I_2 ⊗ H_3 ⊗ I_3
    return np.kron(np.kron(I2, H_3x3), I3)


def extend_electron_only(H_3x3: np.ndarray, manifold: str = "ground") -> np.ndarray:
    """
    Extend 3×3 to 18×18, acting only on ONE electronic manifold.

    Unlike extend_to_18x18(), this operator acts ONLY on the
    ground state (|g>) OR the excited state (|e>), not both.

    This is needed for terms that differ between g and e,
    e.g., different ZFS parameters in the excited state.

    Parameters
    ----------
    H_3x3 : np.ndarray
        3×3 electron spin Hamiltonian
    manifold : str
        "ground" for |g> manifold (indices 0-8)
        "excited" for |e> manifold (indices 9-17)

    Returns
    -------
    np.ndarray
        18×18 Hamiltonian acting only on the chosen manifold

    Examples
    --------
    >>> # ZFS only in ground state
    >>> H_gs = extend_electron_only(H_zfs_3x3, manifold="ground")
    >>> # Different ZFS in excited state
    >>> H_es = extend_electron_only(H_zfs_excited_3x3, manifold="excited")
    >>> H_total = H_gs + H_es
    """
    I3 = np.eye(3, dtype=np.complex128)

    # First extend to 9×9 (electron spin ⊗ nuclear spin)
    H_9x9 = np.kron(H_3x3, I3)

    # Embed in 18×18
    H_18x18 = np.zeros((18, 18), dtype=np.complex128)

    if manifold == "ground":
        # Ground state: first 9 states (index 0-8)
        H_18x18[:9, :9] = H_9x9
    elif manifold == "excited":
        # Excited state: last 9 states (index 9-17)
        H_18x18[9:, 9:] = H_9x9
    else:
        raise ValueError(f"manifold must be 'ground' or 'excited', not '{manifold}'")

    return H_18x18


# Module-level operator instances for convenience
# Usage: from sim.core.operators import Sx, Sy, Sz
_ops = spin1_operators()
Sx = _ops.Sx
Sy = _ops.Sy
Sz = _ops.Sz
Sp = _ops.Sp
Sm = _ops.Sm
I3 = _ops.I

# Nuclear spin operators (also S=1)
_nuc = spin1_operators()
Ix = _nuc.Sx
Iy = _nuc.Sy
Iz = _nuc.Sz
Ip = _nuc.Sp
Im = _nuc.Sm
