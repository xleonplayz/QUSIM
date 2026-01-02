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
Zeeman term for the NV center Hamiltonian.

Physical Background
-------------------
The Zeeman effect describes the interaction of a magnetic moment
with an external magnetic field. For the electron spin:

    H_Zeeman = γ_e · (B_x·S_x + B_y·S_y + B_z·S_z)
             = γ_e · B⃗ · S⃗

where:
    - γ_e = g_e · μ_B / ℏ ≈ 1.76×10¹¹ rad/(s·T): Gyromagnetic ratio
    - g_e ≈ 2.0028: Electron g-factor
    - μ_B: Bohr magneton
    - B⃗: Magnetic field vector
    - S⃗: Spin operator vector

Energy Levels
-------------
For a field B along the z-axis (NV axis):

    E(ms) = γ_e · B_z · ms

The splitting between ms=+1 and ms=-1 is:
    ΔE = 2·γ_e·B_z = 2·g_e·μ_B·B_z/ℏ

In practical units:
    Δf ≈ 28 MHz/mT · B[mT]

Example: At B=10 mT, Δf ≈ 280 MHz between ms=+1 and ms=-1.

Interplay with ZFS
------------------
In the full picture with ZFS (D=2.87 GHz):
    - At B=0: ms=0 at 0 GHz, ms=±1 at 2.87 GHz (degenerate)
    - At B>0: ms=±1 split to 2.87 ± γB

This allows selective excitation of ms=0↔ms=+1 or ms=0↔ms=-1
at different microwave frequencies.

ODMR (Optically Detected Magnetic Resonance)
--------------------------------------------
The Zeeman splitting is the basis for NV magnetometry:
    - ODMR spectrum shows dips at 2.87 ± γB GHz
    - From the splitting, B can be determined

References
----------
[1] Doherty et al., Physics Reports 528, 1-45 (2013)
[2] Schirhagl et al., Annu. Rev. Phys. Chem. 65, 83-105 (2014)
"""

import numpy as np
from typing import Optional, Union, List, Tuple

from .base import HamiltonianTerm
from ...core.operators import spin1_operators, extend_to_18x18
from ...core.constants import GAMMA_E, G_E, MT, MHZ


class Zeeman(HamiltonianTerm):
    """
    Zeeman Hamiltonian for the electron spin in a magnetic field.

    Implements H_Zeeman = γ_e · (B_x·S_x + B_y·S_y + B_z·S_z)

    Parameters
    ----------
    B : float, list, tuple, or np.ndarray
        Magnetic field vector [B_x, B_y, B_z] in mT.
        If a single float is given, B_z=B, B_x=B_y=0 is assumed.
    g : float
        Electron g-factor. Default: 2.0028 (NV center)
    name : str, optional
        Custom name

    Attributes
    ----------
    B : np.ndarray
        Magnetic field vector [B_x, B_y, B_z] in mT
    g : float
        g-factor

    Examples
    --------
    Field along the NV axis (z):

    >>> zeeman = Zeeman(B=10)  # 10 mT in z-direction
    >>> zeeman = Zeeman(B=[0, 0, 10])  # equivalent
    >>> print(zeeman.splitting_mhz())
    560.6  # MHz splitting between ms=+1 and ms=-1

    Arbitrary field direction:

    >>> zeeman = Zeeman(B=[5, 0, 10])  # tilted field
    >>> H = zeeman.build()

    Together with ZFS:

    >>> from sim import HamiltonianBuilder
    >>> from sim.hamiltonian.terms import ZFS, Zeeman
    >>> H = HamiltonianBuilder()
    >>> H.add(ZFS(D=2.87))
    >>> H.add(Zeeman(B=10))
    >>> eigvals = H.eigenvalues_ghz()
    >>> # Shows splitting of ms=±1 levels

    Notes
    -----
    The typical Zeeman splitting is ~28 MHz/mT.

    More precisely: Δf = 2·g·μ_B·B/h ≈ 2·2.0028·9.274e-24·B / 6.626e-34
                      ≈ 56.06 MHz/mT · B[mT]

    I.e., the splitting between ms=+1 and ms=-1 is ~56 MHz per mT.
    Per ms level (e.g., ms=0 to ms=+1) it is ~28 MHz/mT.
    """

    def __init__(
        self,
        B: Union[float, List[float], Tuple[float, ...], np.ndarray] = 0.0,
        g: float = G_E,
        name: Optional[str] = None
    ):
        super().__init__(name=name)

        # Process magnetic field input
        if isinstance(B, (int, float)):
            # Single value: interpret as B_z
            self.B = np.array([0.0, 0.0, float(B)], dtype=np.float64)
        else:
            self.B = np.array(B, dtype=np.float64)
            if self.B.shape != (3,):
                raise ValueError(f"B must have 3 components, not {self.B.shape}")

        self.g = g

        # Compute gyromagnetic ratio
        # γ = g · μ_B / ℏ, scaled with the g-factor
        self._gamma = GAMMA_E * (self.g / G_E)

        # Cache spin operators
        self._ops = spin1_operators()

        # Matrix cache (since time-independent for static field)
        self._H_18_cache: Optional[np.ndarray] = None

    @property
    def is_time_dependent(self) -> bool:
        """Static Zeeman field is time-independent."""
        return False

    def _build_3x3(self) -> np.ndarray:
        """
        Build the 3×3 Zeeman matrix in the electron spin subspace.

        H = γ · (B_x·S_x + B_y·S_y + B_z·S_z)

        For B = (0, 0, B_z) this is simply:
            H = γ·B_z·S_z = γ·B_z·diag(+1, 0, -1)

        Returns
        -------
        np.ndarray
            3×3 Hermitian matrix in rad/s
        """
        Sx = self._ops.Sx
        Sy = self._ops.Sy
        Sz = self._ops.Sz

        # Convert mT -> Tesla
        B_tesla = self.B * MT

        # H = γ · (B_x·S_x + B_y·S_y + B_z·S_z)
        H_zeeman = self._gamma * (
            B_tesla[0] * Sx +
            B_tesla[1] * Sy +
            B_tesla[2] * Sz
        )

        return H_zeeman

    def build(self, t: float = 0.0) -> np.ndarray:
        """
        Build the 18×18 Zeeman matrix.

        Parameters
        ----------
        t : float
            Time in seconds (ignored for static field)

        Returns
        -------
        np.ndarray
            18×18 Hermitian matrix in rad/s
        """
        # Use cache
        if self._H_18_cache is None:
            H_3x3 = self._build_3x3()
            self._H_18_cache = extend_to_18x18(H_3x3)

        return self._H_18_cache.copy()

    def splitting_mhz(self) -> float:
        """
        Compute the Zeeman splitting in MHz.

        The splitting is defined as the energy difference
        between ms=+1 and ms=-1 for the B_z field.

        Δf = 2·γ·B_z / (2π)

        Returns
        -------
        float
            Splitting in MHz

        Examples
        --------
        >>> Zeeman(B=10).splitting_mhz()
        560.6...  # ~56 MHz/mT × 10 mT
        """
        B_z_tesla = self.B[2] * MT
        splitting_rads = 2 * self._gamma * B_z_tesla
        return splitting_rads / MHZ

    def larmor_frequency_mhz(self) -> float:
        """
        Compute the Larmor frequency in MHz.

        The Larmor frequency is the precession frequency of the spin
        in the magnetic field: f_L = γ·B / (2π)

        Returns
        -------
        float
            Larmor frequency for |B| in MHz
        """
        B_magnitude = np.linalg.norm(self.B) * MT
        return self._gamma * B_magnitude / MHZ

    def __repr__(self) -> str:
        if self.B[0] == 0 and self.B[1] == 0:
            return f"Zeeman(B_z={self.B[2]} mT)"
        else:
            return f"Zeeman(B={self.B.tolist()} mT)"
