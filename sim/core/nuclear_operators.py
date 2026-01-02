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
Nuclear spin operators for NV center simulation.

This module provides spin-1/2 operators for C13 nuclear spins and
utility functions for building tensor products in the extended
Hilbert space.

Hilbert Space Structure
-----------------------
The full NV center Hilbert space is:

    |ψ⟩ = |g/e⟩ ⊗ |ms⟩ ⊗ |mI_N14⟩

    Dimensions: 2 × 3 × 3 = 18

When including a C13 nuclear spin (I=1/2), this extends to:

    |ψ⟩ = |g/e⟩ ⊗ |ms⟩ ⊗ |mI_N14⟩ ⊗ |mI_C13⟩

    Dimensions: 2 × 3 × 3 × 2 = 36

However, for computational efficiency, C13 coupling is often
treated in a block-diagonal approximation within the 18D space.

Spin-1/2 Operators
------------------
The Pauli matrices σ are related to spin operators by:

    Ix = σx/2,  Iy = σy/2,  Iz = σz/2

The operators satisfy [Ii, Ij] = iε_ijk Ik.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SpinHalfOps:
    """
    Container for spin-1/2 operators.

    Attributes
    ----------
    Ix : np.ndarray
        Spin-x operator (2×2 complex)
    Iy : np.ndarray
        Spin-y operator (2×2 complex)
    Iz : np.ndarray
        Spin-z operator (2×2 complex, diagonal)
    Ip : np.ndarray
        Raising operator I+ = Ix + i*Iy
    Im : np.ndarray
        Lowering operator I- = Ix - i*Iy
    I : np.ndarray
        Identity matrix (2×2)

    Examples
    --------
    >>> ops = spin_half_operators()
    >>> np.allclose(ops.Ix @ ops.Iy - ops.Iy @ ops.Ix, 0.5j * ops.Iz)
    True
    >>> np.linalg.eigvalsh(ops.Iz)
    array([-0.5,  0.5])
    """
    Ix: np.ndarray
    Iy: np.ndarray
    Iz: np.ndarray
    Ip: np.ndarray
    Im: np.ndarray
    I: np.ndarray


def spin_half_operators() -> SpinHalfOps:
    """
    Create spin-1/2 operators in the {|+1/2⟩, |-1/2⟩} basis.

    The matrices are:
        Ix = (1/2) * σx = (1/2) * [[0, 1], [1, 0]]
        Iy = (1/2) * σy = (1/2) * [[0, -i], [i, 0]]
        Iz = (1/2) * σz = (1/2) * [[1, 0], [0, -1]]

    Returns
    -------
    SpinHalfOps
        Dataclass containing Ix, Iy, Iz, I+, I-, and identity

    Examples
    --------
    >>> ops = spin_half_operators()
    >>> # Check commutation relation [Ix, Iy] = i*Iz/2
    >>> comm = ops.Ix @ ops.Iy - ops.Iy @ ops.Ix
    >>> np.allclose(comm, 0.5j * ops.Iz)
    True

    >>> # Check I² = I(I+1) = 3/4 for I=1/2
    >>> I_squared = ops.Ix @ ops.Ix + ops.Iy @ ops.Iy + ops.Iz @ ops.Iz
    >>> np.allclose(I_squared, 0.75 * ops.I)
    True
    """
    # Pauli matrices divided by 2
    Ix = 0.5 * np.array([
        [0, 1],
        [1, 0]
    ], dtype=np.complex128)

    Iy = 0.5 * np.array([
        [0, -1j],
        [1j, 0]
    ], dtype=np.complex128)

    Iz = 0.5 * np.array([
        [1, 0],
        [0, -1]
    ], dtype=np.complex128)

    # Raising and lowering operators
    Ip = Ix + 1j * Iy  # I+ = [[0, 1], [0, 0]]
    Im = Ix - 1j * Iy  # I- = [[0, 0], [1, 0]]

    # Identity
    I = np.eye(2, dtype=np.complex128)

    return SpinHalfOps(Ix=Ix, Iy=Iy, Iz=Iz, Ip=Ip, Im=Im, I=I)


def extend_9x9_to_18x18(H_9x9: np.ndarray) -> np.ndarray:
    """
    Extend 9×9 matrix (electron ⊗ N14) to 18×18 full space.

    The 9×9 operator acts on |ms⟩ ⊗ |mI_N14⟩ and is embedded in the
    full space as acting identically on both g and e manifolds.

    Mathematically:
        H_18 = I_2 ⊗ H_9

    where I_2 is the identity on |g/e⟩.

    Parameters
    ----------
    H_9x9 : np.ndarray
        9×9 matrix in electron spin ⊗ N14 nuclear spin space

    Returns
    -------
    np.ndarray
        18×18 matrix in full Hilbert space

    Notes
    -----
    This is the correct extension for terms like hyperfine coupling
    that are identical in ground and excited states.

    Examples
    --------
    >>> H_9 = np.eye(9, dtype=complex)
    >>> H_18 = extend_9x9_to_18x18(H_9)
    >>> H_18.shape
    (18, 18)
    >>> np.allclose(H_18[:9, :9], H_9)
    True
    >>> np.allclose(H_18[9:, 9:], H_9)
    True
    """
    I2 = np.eye(2, dtype=np.complex128)
    return np.kron(I2, H_9x9)


def extend_9x9_to_18x18_manifold(
    H_9x9: np.ndarray,
    manifold: str = "ground"
) -> np.ndarray:
    """
    Extend 9×9 matrix to 18×18, acting only on one manifold.

    Unlike extend_9x9_to_18x18(), this operator acts ONLY on
    the ground state OR excited state, not both.

    Parameters
    ----------
    H_9x9 : np.ndarray
        9×9 matrix in electron spin ⊗ N14 space
    manifold : str
        "ground" for g manifold (indices 0-8)
        "excited" for e manifold (indices 9-17)

    Returns
    -------
    np.ndarray
        18×18 matrix acting only on chosen manifold

    Examples
    --------
    >>> H_9 = np.eye(9, dtype=complex)
    >>> H_g = extend_9x9_to_18x18_manifold(H_9, "ground")
    >>> np.allclose(H_g[:9, :9], H_9)
    True
    >>> np.allclose(H_g[9:, 9:], 0)
    True
    """
    H_18 = np.zeros((18, 18), dtype=np.complex128)

    if manifold == "ground":
        H_18[:9, :9] = H_9x9
    elif manifold == "excited":
        H_18[9:, 9:] = H_9x9
    else:
        raise ValueError(f"manifold must be 'ground' or 'excited', not '{manifold}'")

    return H_18


def build_spin_spin_coupling(
    S_ops: "tuple",
    I_ops: "tuple",
    A_tensor: np.ndarray
) -> np.ndarray:
    """
    Build spin-spin coupling Hamiltonian H = S·A·I.

    Computes the tensor product:
        H = Σ_ij A_ij (S_i ⊗ I_j)

    Parameters
    ----------
    S_ops : tuple
        Tuple of (Sx, Sy, Sz) electron spin operators (3×3 each)
    I_ops : tuple
        Tuple of (Ix, Iy, Iz) nuclear spin operators (2×2 or 3×3)
    A_tensor : np.ndarray
        3×3 hyperfine coupling tensor in rad/s

    Returns
    -------
    np.ndarray
        Coupling Hamiltonian of dimension (3*dim_I) × (3*dim_I)

    Notes
    -----
    The tensor product S_i ⊗ I_j creates a matrix of size
    (3 × dim_I) × (3 × dim_I).

    Examples
    --------
    >>> from sim.core.operators import spin1_operators
    >>> S = spin1_operators()
    >>> I = spin_half_operators()
    >>> A = np.diag([1e6, 1e6, 2e6])  # Axial tensor
    >>> H = build_spin_spin_coupling(
    ...     (S.Sx, S.Sy, S.Sz),
    ...     (I.Ix, I.Iy, I.Iz),
    ...     A
    ... )
    >>> H.shape
    (6, 6)
    """
    Sx, Sy, Sz = S_ops
    Ix, Iy, Iz = I_ops

    S_list = [Sx, Sy, Sz]
    I_list = [Ix, Iy, Iz]

    # Determine dimensions
    dim_S = Sx.shape[0]  # 3 for spin-1
    dim_I = Ix.shape[0]  # 2 for spin-1/2, 3 for spin-1
    dim_total = dim_S * dim_I

    H = np.zeros((dim_total, dim_total), dtype=np.complex128)

    for i in range(3):
        for j in range(3):
            if A_tensor[i, j] != 0:
                H += A_tensor[i, j] * np.kron(S_list[i], I_list[j])

    return H


def dipolar_tensor(
    r_vec: np.ndarray,
    gamma_1: float,
    gamma_2: float
) -> np.ndarray:
    """
    Compute dipolar coupling tensor between two spins.

    The dipolar interaction tensor is:

        A_dip = (μ₀/4π) × (γ₁γ₂ℏ²/r³) × (I - 3r̂r̂ᵀ)

    In frequency units (rad/s):

        A_dip = (μ₀γ₁γ₂ℏ)/(4πr³) × (I - 3r̂r̂ᵀ)

    Parameters
    ----------
    r_vec : np.ndarray
        Position vector from spin 1 to spin 2, in meters
    gamma_1 : float
        Gyromagnetic ratio of spin 1 in rad/(s·T)
    gamma_2 : float
        Gyromagnetic ratio of spin 2 in rad/(s·T)

    Returns
    -------
    np.ndarray
        3×3 dipolar tensor in rad/s

    Notes
    -----
    The tensor is traceless and symmetric.

    Examples
    --------
    >>> from sim.core.constants import GAMMA_E, GAMMA_C13, NM
    >>> r = np.array([0.154, 0, 0]) * NM  # C-C bond length
    >>> A = dipolar_tensor(r, GAMMA_E, GAMMA_C13)
    >>> np.isclose(np.trace(A), 0)  # Traceless
    True
    >>> np.allclose(A, A.T)  # Symmetric
    True
    """
    from .constants import HBAR

    MU_0 = 4 * np.pi * 1e-7  # Vacuum permeability in N/A²

    r = np.linalg.norm(r_vec)
    if r < 1e-15:
        raise ValueError("Distance r cannot be zero")

    r_hat = r_vec / r

    # Dipolar prefactor: (μ₀/4π) × (γ₁γ₂ℏ) / r³
    prefactor = (MU_0 / (4 * np.pi)) * gamma_1 * gamma_2 * HBAR / (r ** 3)

    # Tensor: I - 3 r̂ r̂ᵀ
    I3 = np.eye(3)
    r_outer = np.outer(r_hat, r_hat)
    tensor = I3 - 3 * r_outer

    return prefactor * tensor
