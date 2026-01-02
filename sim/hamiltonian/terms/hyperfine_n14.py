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
N14 hyperfine coupling for NV center.

The nitrogen-14 nucleus (I=1) at the vacancy site couples to the
electron spin through hyperfine interaction and has a nuclear
quadrupole moment.

Physics
-------
The N14 hyperfine Hamiltonian consists of two parts:

1. **Hyperfine coupling** (electron-nuclear spin interaction):

   H_hf = A_∥ Sz Iz + A_⊥ (Sx Ix + Sy Iy)

   where A_∥ ≈ -2.14 MHz and A_⊥ ≈ -2.70 MHz for NV centers.

2. **Quadrupole coupling** (nuclear electric quadrupole):

   H_Q = P (Iz² - 2/3)

   where P ≈ -5.0 MHz is the quadrupole parameter.

The total Hamiltonian is:

   H_N14 = A_∥ Sz Iz + A_⊥ (Sx Ix + Sy Iy) + P (Iz² - 2/3)

Matrix Dimension
----------------
The hyperfine coupling acts on the space |ms⟩ ⊗ |mI⟩ which has
dimension 3 × 3 = 9. This is then extended to the full 18-dimensional
space by tensor product with the electronic g/e identity.

Literature Values
-----------------
- A_∥ = -2.14 MHz (parallel hyperfine)
- A_⊥ = -2.70 MHz (perpendicular hyperfine)
- P = -5.0 MHz (quadrupole parameter)

References
----------
[1] Doherty et al., Physics Reports 528, 1-45 (2013)
[2] Smeltzer et al., New J. Phys. 13, 025021 (2011)
"""

import numpy as np
from typing import Optional

from .base import HamiltonianTerm
from ...core.operators import spin1_operators
from ...core.constants import MHZ


class HyperfineN14(HamiltonianTerm):
    """
    N14 hyperfine coupling with quadrupole interaction.

    Parameters
    ----------
    A_parallel : float
        Parallel hyperfine coupling A_∥ in MHz. Default: -2.14 MHz
    A_perp : float
        Perpendicular hyperfine coupling A_⊥ in MHz. Default: -2.70 MHz
    P_quad : float
        Quadrupole parameter P in MHz. Default: -5.0 MHz
    name : str, optional
        Custom name for the term

    Attributes
    ----------
    A_parallel : float
        Parallel coupling in MHz
    A_perp : float
        Perpendicular coupling in MHz
    P_quad : float
        Quadrupole parameter in MHz

    Examples
    --------
    >>> from sim import HamiltonianBuilder
    >>> from sim.hamiltonian.terms import ZFS, HyperfineN14
    >>>
    >>> H = HamiltonianBuilder()
    >>> H.add(ZFS(D=2.87))
    >>> H.add(HyperfineN14())  # Default N14 parameters
    >>>
    >>> # Check eigenvalues show hyperfine structure
    >>> eigvals = H.eigenvalues_mhz()
    >>> print(f"Hyperfine splitting visible in eigenvalues")

    Custom parameters:

    >>> hf = HyperfineN14(A_parallel=-2.2, A_perp=-2.8, P_quad=-4.9)
    >>> H_matrix = hf.build()
    >>> print(f"Matrix shape: {H_matrix.shape}")
    (18, 18)

    Notes
    -----
    The hyperfine structure splits each ms level into three mI levels.
    The quadrupole term causes asymmetric splitting of the mI = ±1 levels.

    The sign convention follows the standard where negative values
    indicate antiferromagnetic coupling.
    """

    def __init__(
        self,
        A_parallel: float = -2.14,  # MHz
        A_perp: float = -2.70,      # MHz
        P_quad: float = -5.0,       # MHz
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.A_parallel = A_parallel
        self.A_perp = A_perp
        self.P_quad = P_quad

        # Cache for the static Hamiltonian
        self._cache: Optional[np.ndarray] = None

    @property
    def is_time_dependent(self) -> bool:
        """N14 hyperfine coupling is time-independent."""
        return False

    def _build_9x9(self) -> np.ndarray:
        """
        Build the 9×9 hyperfine Hamiltonian in |ms⟩ ⊗ |mI⟩ space.

        Returns
        -------
        np.ndarray
            9×9 complex Hermitian matrix in rad/s
        """
        # Get spin-1 operators for both electron (S) and nuclear (I) spins
        S = spin1_operators()
        I = spin1_operators()  # N14 is also spin-1

        # Convert parameters from MHz to rad/s
        A_par = self.A_parallel * MHZ
        A_perp = self.A_perp * MHZ
        P = self.P_quad * MHZ

        # Build hyperfine coupling: A_∥ Sz⊗Iz + A_⊥ (Sx⊗Ix + Sy⊗Iy)
        # Note: I is Spin1Ops, use .Sx/.Sy/.Sz for nuclear operators
        H_hf = (
            A_par * np.kron(S.Sz, I.Sz) +
            A_perp * (np.kron(S.Sx, I.Sx) + np.kron(S.Sy, I.Sy))
        )

        # Build quadrupole term: P (Iz² - 2/3)
        # This only acts on the nuclear spin
        I3 = np.eye(3, dtype=np.complex128)
        Iz_squared = I.Sz @ I.Sz
        H_quad = P * np.kron(I3, Iz_squared - (2.0/3.0) * I.I)

        return H_hf + H_quad

    def build(self, t: float = 0.0) -> np.ndarray:
        """
        Build the 18×18 N14 hyperfine Hamiltonian.

        Parameters
        ----------
        t : float
            Time in seconds (not used, hyperfine is static)

        Returns
        -------
        np.ndarray
            18×18 complex Hermitian matrix in rad/s

        Notes
        -----
        The result is cached since the Hamiltonian is time-independent.
        The 9×9 matrix is extended to 18×18 by acting identically
        on both ground and excited electronic states.
        """
        if self._cache is None:
            H_9x9 = self._build_9x9()

            # Extend to 18×18: I_2 ⊗ H_9x9
            # Acts on both g and e manifolds identically
            I2 = np.eye(2, dtype=np.complex128)
            self._cache = np.kron(I2, H_9x9)

        return self._cache.copy()

    def hyperfine_tensor(self) -> np.ndarray:
        """
        Return the 3×3 hyperfine tensor in the NV frame.

        Returns
        -------
        np.ndarray
            3×3 diagonal tensor with [A_⊥, A_⊥, A_∥] in MHz

        Notes
        -----
        The NV axis is along z, so:
        - A_xx = A_yy = A_⊥ (perpendicular)
        - A_zz = A_∥ (parallel)
        """
        return np.diag([self.A_perp, self.A_perp, self.A_parallel])

    def splitting_mhz(self, ms: int = 0) -> dict:
        """
        Calculate the N14 hyperfine splitting for a given ms level.

        Parameters
        ----------
        ms : int
            Electron spin projection: +1, 0, or -1

        Returns
        -------
        dict
            Dictionary with keys 'mI_+1', 'mI_0', 'mI_-1' containing
            the energy shifts in MHz for each nuclear spin state

        Examples
        --------
        >>> hf = HyperfineN14()
        >>> split = hf.splitting_mhz(ms=0)
        >>> print(f"mI=0 shift: {split['mI_0']:.2f} MHz")
        """
        # For ms=0, only quadrupole contributes
        # For ms=±1, both hyperfine and quadrupole contribute

        P = self.P_quad
        A_par = self.A_parallel

        if ms == 0:
            # Only quadrupole: P(mI² - 2/3)
            shifts = {
                'mI_+1': P * (1 - 2/3),      # P/3
                'mI_0': P * (0 - 2/3),       # -2P/3
                'mI_-1': P * (1 - 2/3),      # P/3
            }
        else:
            # Hyperfine + quadrupole: A_∥·ms·mI + P(mI² - 2/3)
            shifts = {
                'mI_+1': A_par * ms * 1 + P * (1 - 2/3),
                'mI_0': A_par * ms * 0 + P * (0 - 2/3),
                'mI_-1': A_par * ms * (-1) + P * (1 - 2/3),
            }

        return shifts

    def __repr__(self) -> str:
        return (
            f"HyperfineN14(A_parallel={self.A_parallel} MHz, "
            f"A_perp={self.A_perp} MHz, P_quad={self.P_quad} MHz)"
        )
