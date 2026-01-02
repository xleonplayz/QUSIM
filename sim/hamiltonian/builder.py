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
Hamiltonian builder for NV center simulation.

The builder combines multiple Hamiltonian terms into a
total Hamiltonian and provides analysis methods.

Architecture
------------
The builder follows the Composite pattern:

    HamiltonianBuilder
    ├── ZFS(D=2.87)
    ├── Zeeman(B=10)
    ├── Hyperfine(...)
    └── ...

    H_total(t) = Σ term.build(t)

Units
-----
- Input: Practical units (GHz, mT, ...)
- Output: The matrix H is in rad/s (energy with ℏ=1)
- Eigenvalues can be retrieved in GHz via eigenvalues_ghz()

Example
-------
>>> from sim import HamiltonianBuilder
>>> from sim.hamiltonian.terms import ZFS, Zeeman
>>>
>>> H = HamiltonianBuilder()
>>> H.add(ZFS(D=2.87))        # 2.87 GHz ZFS
>>> H.add(Zeeman(B=10))       # 10 mT magnetic field
>>>
>>> # Matrix at t=0
>>> H_matrix = H.build(t=0.0)
>>> print(H_matrix.shape)
(18, 18)
>>>
>>> # Eigenvalues in GHz
>>> eigvals = H.eigenvalues_ghz()
>>> print(f"Levels: {np.unique(np.round(eigvals, 2))}")
"""

import numpy as np
from typing import List, Optional

from .terms.base import HamiltonianTerm


class HamiltonianBuilder:
    """
    Builder for constructing the NV center Hamiltonian.

    Collects individual terms (ZFS, Zeeman, Hyperfine, ...) and
    combines them into the total matrix.

    Attributes
    ----------
    terms : list
        List of added HamiltonianTerm objects

    Methods
    -------
    add(term)
        Add a term
    build(t=0.0)
        Build the 18×18 matrix at time t
    eigenvalues(t=0.0)
        Compute eigenvalues in rad/s
    eigenvalues_ghz(t=0.0)
        Compute eigenvalues in GHz

    Examples
    --------
    Simple NV center with ZFS and magnetic field:

    >>> H = HamiltonianBuilder()
    >>> H.add(ZFS(D=2.87))
    >>> H.add(Zeeman(B=10))
    >>> print(len(H))
    2
    >>> print(H)
    HamiltonianBuilder([ZFS, Zeeman])

    Analyze eigenvalues:

    >>> eigvals = H.eigenvalues_ghz()
    >>> print(f"Ground state: {min(eigvals):.3f} GHz")
    >>> print(f"Highest state: {max(eigvals):.3f} GHz")

    Method chaining:

    >>> H = HamiltonianBuilder().add(ZFS()).add(Zeeman(B=5))
    """

    # Hilbert space dimension (fixed for NV center)
    DIM = 18

    def __init__(self):
        """Initialize an empty builder."""
        self._terms: List[HamiltonianTerm] = []

    def add(self, term: HamiltonianTerm) -> "HamiltonianBuilder":
        """
        Add a Hamiltonian term.

        Parameters
        ----------
        term : HamiltonianTerm
            A term object (e.g., ZFS, Zeeman, Hyperfine)

        Returns
        -------
        HamiltonianBuilder
            self, for method chaining

        Raises
        ------
        TypeError
            If term is not a HamiltonianTerm

        Examples
        --------
        >>> H = HamiltonianBuilder()
        >>> H.add(ZFS(D=2.87)).add(Zeeman(B=10))
        HamiltonianBuilder([ZFS, Zeeman])
        """
        if not isinstance(term, HamiltonianTerm):
            raise TypeError(
                f"Expected HamiltonianTerm, got {type(term).__name__}. "
                f"Use terms like ZFS(...), Zeeman(...), etc."
            )
        self._terms.append(term)
        return self

    def clear(self) -> "HamiltonianBuilder":
        """
        Remove all terms.

        Returns
        -------
        HamiltonianBuilder
            self, for method chaining
        """
        self._terms.clear()
        return self

    @property
    def terms(self) -> List[HamiltonianTerm]:
        """
        List of added terms (copy).

        Returns
        -------
        list
            Copy of the term list
        """
        return self._terms.copy()

    @property
    def is_time_dependent(self) -> bool:
        """
        Whether any term is time-dependent.

        Returns
        -------
        bool
            True if at least one term is time-dependent
        """
        return any(term.is_time_dependent for term in self._terms)

    def build(self, t: float = 0.0) -> np.ndarray:
        """
        Build the total Hamiltonian matrix at time t.

        H_total(t) = Σ_i term_i.build(t)

        Parameters
        ----------
        t : float
            Time in seconds

        Returns
        -------
        np.ndarray
            18×18 complex Hermitian matrix in rad/s

        Notes
        -----
        The unit rad/s corresponds to energy with ℏ=1.
        To convert to Hz: f = ω / (2π)
        To GHz: f_GHz = ω / (2π × 10⁹)
        """
        H_total = np.zeros((self.DIM, self.DIM), dtype=np.complex128)

        for term in self._terms:
            H_total += term.build(t)

        return H_total

    def eigenvalues(self, t: float = 0.0) -> np.ndarray:
        """
        Compute eigenvalues of the Hamiltonian.

        Parameters
        ----------
        t : float
            Time in seconds

        Returns
        -------
        np.ndarray
            18 eigenvalues in rad/s, sorted ascending

        Notes
        -----
        Uses np.linalg.eigvalsh (for Hermitian matrices).
        The eigenvalues are real.
        """
        H = self.build(t)
        return np.linalg.eigvalsh(H)

    def eigenvalues_ghz(self, t: float = 0.0) -> np.ndarray:
        """
        Compute eigenvalues in GHz.

        Parameters
        ----------
        t : float
            Time in seconds

        Returns
        -------
        np.ndarray
            18 eigenvalues in GHz, sorted ascending

        Examples
        --------
        >>> H = HamiltonianBuilder().add(ZFS(D=2.87))
        >>> eigvals = H.eigenvalues_ghz()
        >>> np.unique(np.round(eigvals, 2))
        array([0.  , 2.87])
        """
        return self.eigenvalues(t) / (2 * np.pi * 1e9)

    def eigenvalues_mhz(self, t: float = 0.0) -> np.ndarray:
        """
        Compute eigenvalues in MHz.

        Parameters
        ----------
        t : float
            Time in seconds

        Returns
        -------
        np.ndarray
            18 eigenvalues in MHz, sorted ascending
        """
        return self.eigenvalues(t) / (2 * np.pi * 1e6)

    def eigenstates(self, t: float = 0.0) -> tuple:
        """
        Compute eigenvalues and eigenvectors.

        Parameters
        ----------
        t : float
            Time in seconds

        Returns
        -------
        eigenvalues : np.ndarray
            18 eigenvalues in rad/s
        eigenvectors : np.ndarray
            18×18 matrix, columns are eigenvectors
        """
        H = self.build(t)
        return np.linalg.eigh(H)

    def __repr__(self) -> str:
        if not self._terms:
            return "HamiltonianBuilder([])"
        term_names = ", ".join(term.name for term in self._terms)
        return f"HamiltonianBuilder([{term_names}])"

    def __len__(self) -> int:
        """Number of terms."""
        return len(self._terms)

    def __bool__(self) -> bool:
        """True if at least one term is present."""
        return len(self._terms) > 0
