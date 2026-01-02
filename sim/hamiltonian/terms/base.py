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
Abstract base class for Hamiltonian terms.

All Hamiltonian terms (ZFS, Zeeman, Hyperfine, etc.) inherit from this
class and implement the build() method.

Design Pattern
--------------
We use the Strategy pattern: Each term is interchangeable and the
HamiltonianBuilder combines them at runtime.

    HamiltonianBuilder
         |
         +-- add(term) --> [ZFS, Zeeman, Hyperfine, ...]
         |
         +-- build(t) --> Σ term.build(t) = H_total
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class HamiltonianTerm(ABC):
    """
    Abstract base class for all Hamiltonian terms.

    Each term must implement:
        - build(t) -> 18×18 numpy array (Hamiltonian matrix in rad/s)
        - is_time_dependent -> bool (whether the term depends on time)

    Attributes
    ----------
    name : str
        Name of the term for debugging/display

    Methods
    -------
    build(t=0.0)
        Build the 18×18 Hamiltonian matrix at time t
    is_time_dependent
        Property: True if the term depends on t

    Examples
    --------
    >>> class MyTerm(HamiltonianTerm):
    ...     @property
    ...     def is_time_dependent(self):
    ...         return False
    ...     def build(self, t=0.0):
    ...         return np.zeros((18, 18), dtype=complex)
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the Hamiltonian term.

        Parameters
        ----------
        name : str, optional
            Custom name. If not provided, the class name is used.
        """
        self._name = name if name is not None else self.__class__.__name__

    @property
    def name(self) -> str:
        """Name of this Hamiltonian term."""
        return self._name

    @property
    @abstractmethod
    def is_time_dependent(self) -> bool:
        """
        Whether this term is time-dependent.

        Returns
        -------
        bool
            True if build(t) returns different results for
            different t, otherwise False.

        Notes
        -----
        Time-independent terms (like ZFS without modulation) can
        be cached for better performance.
        """
        pass

    @abstractmethod
    def build(self, t: float = 0.0) -> np.ndarray:
        """
        Build the 18×18 Hamiltonian matrix at time t.

        Parameters
        ----------
        t : float
            Time in seconds. Only relevant for time-dependent terms.

        Returns
        -------
        np.ndarray
            18×18 complex matrix in units of rad/s.
            Must be Hermitian: H = H†

        Notes
        -----
        The matrix is in the 18-dimensional Hilbert space:
            |ψ> = |g/e> ⊗ |ms> ⊗ |mI>

        The unit rad/s corresponds to energy with ℏ=1.
        """
        pass

    def __repr__(self) -> str:
        """String representation of the term."""
        return f"{self.__class__.__name__}()"
