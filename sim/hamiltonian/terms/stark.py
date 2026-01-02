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
Stark effect for NV center.

The NV center exhibits both DC and AC Stark effects when exposed
to electric fields. The quadratic Stark effect shifts the spin
levels through coupling to the orbital states.

Physics
-------
The Stark Hamiltonian is:

    H_stark = d_⊥ · E_⊥ · (S_x² - S_y²) + d_∥ · E_∥ · S_z²

where:
- d_⊥ ≈ 17 Hz/(V/m) is the perpendicular Stark coefficient
- d_∥ ≈ 3.5 Hz/(V/m) is the parallel Stark coefficient
- E_⊥, E_∥ are the electric field components

The full tensor form is:

    H_stark = Σ_{ij} D_{ij} E_j {S_i, S_i'}/2

Typical Values
--------------
- d_⊥ = 17 ± 3 Hz/(V/m)
- d_∥ = 3.5 ± 0.8 Hz/(V/m)

For typical lab fields of 10⁶ V/m, this gives shifts of ~10-20 kHz.

References
----------
[1] van Oort & Glasbeek, Chemical Physics Letters 168, 529 (1990)
[2] Dolde et al., Phys. Rev. Lett. 112, 097603 (2014)
"""

import numpy as np
from typing import Optional, Callable, Tuple
from numpy.typing import ArrayLike

from .base import HamiltonianTerm
from ...core.operators import spin1_operators, extend_to_18x18
from ...core.constants import HZ


class Stark(HamiltonianTerm):
    """
    Stark effect (DC and AC electric field coupling).

    Parameters
    ----------
    d_perp : float
        Perpendicular Stark coefficient in Hz/(V/m). Default: 17e-3
    d_par : float
        Parallel Stark coefficient in Hz/(V/m). Default: 3.5e-3
    E_field : array_like
        Static electric field vector [Ex, Ey, Ez] in V/m. Default: (0,0,0)
    E_field_func : callable, optional
        Time-dependent E(t) function returning [Ex, Ey, Ez].
        If provided, makes the term time-dependent.
    name : str, optional
        Custom name for the term

    Attributes
    ----------
    d_perp : float
        Perpendicular Stark coefficient in Hz/(V/m)
    d_par : float
        Parallel Stark coefficient in Hz/(V/m)
    E_field : np.ndarray
        Static electric field in V/m

    Examples
    --------
    >>> from sim import HamiltonianBuilder
    >>> from sim.hamiltonian.terms import ZFS, Stark
    >>>
    >>> # Static electric field along z
    >>> H = HamiltonianBuilder()
    >>> H.add(ZFS(D=2.87))
    >>> H.add(Stark(E_field=[0, 0, 1e6]))  # 1 MV/m along z
    >>>
    >>> # AC Stark effect (from optical field)
    >>> def ac_field(t):
    ...     return [1e6 * np.sin(2 * np.pi * 1e9 * t), 0, 0]
    >>>
    >>> H.add(Stark(E_field_func=ac_field))

    Notes
    -----
    The Stark effect is typically small compared to ZFS and Zeeman
    effects unless very strong electric fields are applied.

    The AC Stark shift from optical fields can be significant
    when the laser is detuned from the optical transition.
    """

    def __init__(
        self,
        d_perp: float = 17e-3,       # Hz/(V/m)
        d_par: float = 3.5e-3,       # Hz/(V/m)
        E_field: ArrayLike = (0, 0, 0),  # V/m
        E_field_func: Optional[Callable] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.d_perp = d_perp
        self.d_par = d_par
        self.E_field = np.array(E_field, dtype=float)
        self._E_func = E_field_func

        # Cache for static term
        self._cache: Optional[np.ndarray] = None

    @property
    def is_time_dependent(self) -> bool:
        """True if E_field_func is provided."""
        return self._E_func is not None

    def _build_3x3(self, E: np.ndarray) -> np.ndarray:
        """
        Build 3×3 Stark Hamiltonian for given electric field.

        Parameters
        ----------
        E : np.ndarray
            Electric field vector [Ex, Ey, Ez] in V/m

        Returns
        -------
        np.ndarray
            3×3 complex Hermitian matrix in rad/s
        """
        S = spin1_operators()

        Ex, Ey, Ez = E

        # Perpendicular coupling
        E_perp = np.sqrt(Ex**2 + Ey**2)

        # Calculate the azimuthal angle
        if E_perp > 0:
            phi = np.arctan2(Ey, Ex)
        else:
            phi = 0

        # H = d_⊥ · E_⊥ · (cos(2φ)(Sx² - Sy²) + sin(2φ){Sx, Sy})
        #   + d_∥ · Ez · Sz²

        H_perp = self.d_perp * E_perp * (
            np.cos(2 * phi) * (S.Sx @ S.Sx - S.Sy @ S.Sy) +
            np.sin(2 * phi) * (S.Sx @ S.Sy + S.Sy @ S.Sx)
        )

        H_par = self.d_par * Ez * (S.Sz @ S.Sz)

        # Convert to rad/s (d is in Hz/(V/m), E in V/m, so result is Hz)
        H = (H_perp + H_par) * HZ

        return H

    def build(self, t: float = 0.0) -> np.ndarray:
        """
        Build the 18×18 Stark Hamiltonian at time t.

        Parameters
        ----------
        t : float
            Time in seconds

        Returns
        -------
        np.ndarray
            18×18 complex Hermitian matrix in rad/s
        """
        if self.is_time_dependent:
            E_t = np.array(self._E_func(t))
            H_3x3 = self._build_3x3(E_t)
            return extend_to_18x18(H_3x3)

        else:
            if self._cache is None:
                H_3x3 = self._build_3x3(self.E_field)
                self._cache = extend_to_18x18(H_3x3)

            return self._cache.copy()

    def shift_khz(self, E_par: float = 0, E_perp: float = 0) -> Tuple[float, float]:
        """
        Calculate the Stark shift for given field components.

        Parameters
        ----------
        E_par : float
            Parallel electric field in V/m
        E_perp : float
            Perpendicular electric field in V/m

        Returns
        -------
        tuple
            (shift_par, shift_perp) in kHz
        """
        shift_par = self.d_par * E_par / 1000  # Convert to kHz
        shift_perp = self.d_perp * E_perp / 1000
        return (shift_par, shift_perp)

    def __repr__(self) -> str:
        if self.is_time_dependent:
            return f"Stark(time-dependent)"
        else:
            E_mag = np.linalg.norm(self.E_field)
            return f"Stark(|E|={E_mag:.2e} V/m)"
