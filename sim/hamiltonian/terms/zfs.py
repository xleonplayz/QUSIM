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
Zero-Field Splitting (ZFS) term for the NV center Hamiltonian.

Physical Background
-------------------
The NV center has two unpaired electrons forming a triplet state
(S=1). The spin-spin interaction between these electrons leads to
Zero-Field Splitting (ZFS) - an energy splitting even without an
external magnetic field.

The ZFS Hamiltonian is:

    H_ZFS = D·Sz² + E·(Sx² - Sy²)

where:
    - D ≈ 2.87 GHz: Axial parameter (splitting ms=0 ↔ ms=±1)
    - E: Transverse parameter (strain-induced, lifts ms=±1 degeneracy)

Energy Levels
-------------
Without E-term (E=0):
    - E(ms=0) = 0
    - E(ms=±1) = D = 2.87 GHz  (degenerate)

With E-term (E≠0):
    - E(ms=0) = 0
    - E(ms=+1) = D + E
    - E(ms=-1) = D - E
    → Splitting of 2E between ms=±1

Typical Values
--------------
    - D_GS = 2.87 GHz (ground state at room temperature)
    - D_ES = 1.42 GHz (excited state)
    - E = 0 to ~10 MHz (depends on strain/defects)

Temperature Dependence
----------------------
D(T) ≈ D₀ + α·(T - 300K)  with α ≈ -74 kHz/K

References
----------
[1] Doherty et al., Physics Reports 528, 1-45 (2013)
[2] Maze et al., New J. Phys. 13, 025025 (2011)
"""

import numpy as np
from typing import Optional

from .base import HamiltonianTerm
from ...core.operators import spin1_operators, extend_to_18x18
from ...core.constants import GHZ


class ZFS(HamiltonianTerm):
    """
    Zero-Field Splitting Hamiltonian term.

    Implements H_ZFS = D·Sz² + E·(Sx² - Sy²)

    Parameters
    ----------
    D : float
        Axial ZFS parameter in GHz.
        Default: 2.87 GHz (NV ground state at room temperature)
    E : float
        Transverse ZFS parameter in GHz.
        Default: 0 (no strain splitting)
    name : str, optional
        Custom name for this term

    Attributes
    ----------
    D : float
        Axial parameter in GHz
    E : float
        Transverse parameter in GHz

    Examples
    --------
    Standard NV center (no strain):

    >>> zfs = ZFS(D=2.87)
    >>> H = zfs.build()
    >>> eigvals = np.linalg.eigvalsh(H) / (2*np.pi*1e9)  # in GHz
    >>> np.unique(np.round(eigvals, 2))
    array([0.  , 2.87])

    With strain splitting (E=5 MHz):

    >>> zfs = ZFS(D=2.87, E=0.005)
    >>> H = zfs.build()
    >>> eigvals = np.linalg.eigvalsh(H) / (2*np.pi*1e9)
    >>> # ms=±1 are now split at 2.865 and 2.875 GHz

    Notes
    -----
    The matrix is 18×18, but ZFS only acts on the electron spin.
    The 6-fold degeneracy at E(ms=0)=0 comes from:
        - 2× g/e states
        - 3× nuclear spin mI states

    The 12-fold degeneracy at E(ms=±1)=D comes from:
        - 2× ms=+1 and ms=-1
        - 2× g/e states
        - 3× nuclear spin mI states
    """

    def __init__(
        self,
        D: float = 2.87,
        E: float = 0.0,
        name: Optional[str] = None
    ):
        super().__init__(name=name)

        # Store parameters (in GHz, as given by user)
        self.D = D
        self.E = E

        # Pre-compute spin operators (3×3)
        self._ops = spin1_operators()

        # Cache for 18×18 matrix (since time-independent)
        self._H_18_cache: Optional[np.ndarray] = None

    @property
    def is_time_dependent(self) -> bool:
        """ZFS is time-independent (without external modulation)."""
        return False

    def _build_3x3(self) -> np.ndarray:
        """
        Build the 3×3 ZFS matrix in the electron spin subspace.

        The formula is:
            H = D·Sz² + E·(Sx² - Sy²)

        In the {|+1>, |0>, |-1>} basis, Sz² = diag(1, 0, 1),
        so D·Sz² gives diagonal elements (D, 0, D).

        The E-term mixes |+1> and |-1> and creates off-diagonal elements.

        Returns
        -------
        np.ndarray
            3×3 complex Hermitian matrix in rad/s
        """
        Sx = self._ops.Sx
        Sy = self._ops.Sy
        Sz = self._ops.Sz

        # Convert GHz -> rad/s
        D_rads = self.D * GHZ
        E_rads = self.E * GHZ

        # H = D·Sz² + E·(Sx² - Sy²)
        Sz_squared = Sz @ Sz  # = diag(1, 0, 1)
        Sx_sq_minus_Sy_sq = Sx @ Sx - Sy @ Sy

        H_zfs = D_rads * Sz_squared + E_rads * Sx_sq_minus_Sy_sq

        return H_zfs

    def build(self, t: float = 0.0) -> np.ndarray:
        """
        Build the 18×18 ZFS matrix.

        Parameters
        ----------
        t : float
            Time in seconds (ignored, since time-independent)

        Returns
        -------
        np.ndarray
            18×18 Hermitian matrix in rad/s
        """
        # Use cache since time-independent
        if self._H_18_cache is None:
            H_3x3 = self._build_3x3()
            self._H_18_cache = extend_to_18x18(H_3x3)

        return self._H_18_cache.copy()

    def __repr__(self) -> str:
        if self.E == 0:
            return f"ZFS(D={self.D} GHz)"
        else:
            return f"ZFS(D={self.D} GHz, E={self.E} GHz)"
