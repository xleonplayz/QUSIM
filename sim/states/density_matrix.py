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
Density matrix class for NV center quantum states.

The density matrix ρ describes the quantum state of the system,
including mixed states (statistical mixtures) and pure states.

Properties
----------
For a valid density matrix:
- Hermitian: ρ = ρ†
- Positive semi-definite: all eigenvalues ≥ 0
- Trace one: Tr(ρ) = 1

For a pure state |ψ⟩: ρ = |ψ⟩⟨ψ| with Tr(ρ²) = 1

Basis Convention
----------------
The 18-dimensional basis is ordered as:
    |g/e⟩ ⊗ |ms⟩ ⊗ |mI⟩

Index mapping:
    idx = 9 * ge + 3 * (1 - ms) + (1 - mI)

where ge = 0 (ground) or 1 (excited), ms ∈ {+1, 0, -1}, mI ∈ {+1, 0, -1}.
"""

import numpy as np
from typing import Optional, Union
from numpy.typing import ArrayLike


class DensityMatrix:
    """
    Density matrix for the 18-dimensional NV Hilbert space.

    Parameters
    ----------
    rho : np.ndarray, optional
        18×18 density matrix. If None, creates zero matrix.

    Attributes
    ----------
    rho : np.ndarray
        The 18×18 density matrix
    DIM : int
        Hilbert space dimension (18)

    Examples
    --------
    >>> from sim.states import DensityMatrix, ground_state_vector
    >>>
    >>> # Create from pure state
    >>> psi = ground_state_vector()
    >>> dm = DensityMatrix.from_pure_state(psi)
    >>> print(f"Purity: {dm.purity():.3f}")
    Purity: 1.000
    >>>
    >>> # Check properties
    >>> print(f"Trace: {dm.trace():.3f}")
    Trace: 1.000
    >>> print(f"Is valid: {dm.is_valid()}")
    Is valid: True

    Notes
    -----
    The density matrix is stored as a complex numpy array.
    All operations return copies to prevent accidental modification.
    """

    DIM = 18

    def __init__(self, rho: Optional[np.ndarray] = None):
        """Initialize density matrix."""
        if rho is None:
            self._rho = np.zeros((self.DIM, self.DIM), dtype=np.complex128)
        else:
            self._rho = np.array(rho, dtype=np.complex128)
            if self._rho.shape != (self.DIM, self.DIM):
                raise ValueError(f"Expected {self.DIM}×{self.DIM}, got {self._rho.shape}")

    @property
    def rho(self) -> np.ndarray:
        """Return copy of the density matrix."""
        return self._rho.copy()

    @classmethod
    def from_pure_state(cls, psi: np.ndarray) -> "DensityMatrix":
        """
        Create density matrix from pure state vector.

        Parameters
        ----------
        psi : np.ndarray
            18-dimensional state vector

        Returns
        -------
        DensityMatrix
            ρ = |ψ⟩⟨ψ|

        Examples
        --------
        >>> psi = np.zeros(18, dtype=complex)
        >>> psi[4] = 1.0  # |g, ms=0, mI=0⟩
        >>> dm = DensityMatrix.from_pure_state(psi)
        >>> dm.purity()
        1.0
        """
        psi = np.array(psi, dtype=np.complex128).flatten()
        if len(psi) != cls.DIM:
            raise ValueError(f"Expected {cls.DIM}-dim vector, got {len(psi)}")

        # Normalize
        norm = np.linalg.norm(psi)
        if norm > 0:
            psi = psi / norm

        rho = np.outer(psi, psi.conj())
        return cls(rho)

    @classmethod
    def from_thermal(
        cls,
        H: np.ndarray,
        temperature: float,
        zero_point: Optional[float] = None
    ) -> "DensityMatrix":
        """
        Create thermal equilibrium state.

        Parameters
        ----------
        H : np.ndarray
            18×18 Hamiltonian matrix in rad/s
        temperature : float
            Temperature in Kelvin
        zero_point : float, optional
            Energy zero point. If None, uses ground state energy.

        Returns
        -------
        DensityMatrix
            ρ = exp(-H/kT) / Z

        Notes
        -----
        At T → 0, returns the ground state.
        At T → ∞, returns the maximally mixed state.
        """
        from ..core.constants import KB, HBAR

        if temperature <= 0:
            # Ground state at T=0
            eigvals, eigvecs = np.linalg.eigh(H)
            psi_ground = eigvecs[:, 0]
            return cls.from_pure_state(psi_ground)

        # Shift to zero point
        if zero_point is None:
            zero_point = np.min(np.linalg.eigvalsh(H))

        H_shifted = H - zero_point * np.eye(cls.DIM)

        # β = 1 / (kB T), but H is in rad/s, so use ℏ
        # E = ℏω, so β = ℏ / (kB T)
        beta = HBAR / (KB * temperature)

        # Compute exp(-β H)
        rho_unnorm = np.linalg.matrix_power(
            np.eye(cls.DIM) - beta * H_shifted / 100, 100
        )  # Approximate exp

        # Better: use scipy
        from scipy.linalg import expm
        rho_unnorm = expm(-beta * H_shifted)

        # Normalize
        Z = np.trace(rho_unnorm)
        rho = rho_unnorm / Z

        return cls(rho)

    def trace(self) -> float:
        """Return trace of density matrix."""
        return np.real(np.trace(self._rho))

    def purity(self) -> float:
        """
        Return purity Tr(ρ²).

        Returns
        -------
        float
            Purity in [1/18, 1]. Pure state has purity 1.
        """
        return np.real(np.trace(self._rho @ self._rho))

    def entropy(self) -> float:
        """
        Return von Neumann entropy S = -Tr(ρ log ρ).

        Returns
        -------
        float
            Entropy in natural units (nats). Zero for pure states.
        """
        eigvals = np.linalg.eigvalsh(self._rho)
        # Filter out zero/negative eigenvalues
        eigvals = eigvals[eigvals > 1e-15]
        return -np.sum(eigvals * np.log(eigvals))

    def expectation(self, operator: np.ndarray) -> complex:
        """
        Compute expectation value ⟨O⟩ = Tr(O ρ).

        Parameters
        ----------
        operator : np.ndarray
            18×18 operator matrix

        Returns
        -------
        complex
            Expectation value
        """
        return np.trace(operator @ self._rho)

    def population(self, projector: np.ndarray) -> float:
        """
        Compute population Tr(P ρ) for a projector.

        Parameters
        ----------
        projector : np.ndarray
            18×18 projector matrix (P² = P)

        Returns
        -------
        float
            Population in [0, 1]
        """
        return np.real(np.trace(projector @ self._rho))

    def is_valid(self, tol: float = 1e-10) -> bool:
        """
        Check if this is a valid density matrix.

        Parameters
        ----------
        tol : float
            Tolerance for numerical checks

        Returns
        -------
        bool
            True if Hermitian, positive semi-definite, and trace 1
        """
        # Hermitian
        if not np.allclose(self._rho, self._rho.conj().T, atol=tol):
            return False

        # Trace 1
        if not np.isclose(self.trace(), 1.0, atol=tol):
            return False

        # Positive semi-definite
        eigvals = np.linalg.eigvalsh(self._rho)
        if np.any(eigvals < -tol):
            return False

        return True

    def fidelity(self, other: "DensityMatrix") -> float:
        """
        Compute fidelity F(ρ, σ) between two states.

        Parameters
        ----------
        other : DensityMatrix
            Another density matrix

        Returns
        -------
        float
            Fidelity in [0, 1]

        Notes
        -----
        F(ρ, σ) = [Tr(√(√ρ σ √ρ))]²

        For pure states: F = |⟨ψ|φ⟩|²
        """
        from scipy.linalg import sqrtm

        sqrt_rho = sqrtm(self._rho)
        product = sqrt_rho @ other._rho @ sqrt_rho
        sqrt_product = sqrtm(product)

        return np.real(np.trace(sqrt_product))**2

    def partial_trace(self, subsystem: str) -> np.ndarray:
        """
        Trace out a subsystem.

        Parameters
        ----------
        subsystem : str
            Which subsystem to trace out:
            - "electronic": Trace out |g/e⟩, returns 9×9
            - "electron_spin": Trace out |ms⟩, returns 6×6
            - "nuclear_spin": Trace out |mI⟩, returns 6×6

        Returns
        -------
        np.ndarray
            Reduced density matrix
        """
        rho = self._rho.reshape((2, 3, 3, 2, 3, 3))

        if subsystem == "electronic":
            # Trace out first index (g/e)
            # Result: 3×3×3×3 -> 9×9
            reduced = np.einsum('ijkijl->jkl', rho.reshape(2, 9, 2, 9))
            return reduced.reshape(9, 9)

        elif subsystem == "electron_spin":
            # Trace out second index (ms)
            reduced = np.einsum('iajibk->ijab', rho.reshape(2, 3, 3, 2, 3, 3))
            return reduced.reshape(6, 6)

        elif subsystem == "nuclear_spin":
            # Trace out third index (mI)
            reduced = np.einsum('ijaikb->ijab', rho.reshape(2, 3, 3, 2, 3, 3))
            return reduced.reshape(6, 6)

        else:
            raise ValueError(f"Unknown subsystem: {subsystem}")

    def __repr__(self) -> str:
        return f"DensityMatrix(purity={self.purity():.4f}, trace={self.trace():.4f})"
