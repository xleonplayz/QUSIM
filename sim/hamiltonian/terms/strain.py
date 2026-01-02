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
Strain-induced splitting for NV center.

Strain in the diamond lattice modifies the zero-field splitting
through spin-strain coupling. This can be caused by static stress
(e.g., from implantation damage) or dynamic strain (e.g., acoustic waves).

Physics
-------
The strain Hamiltonian is:

    H_strain = Σ_{ij} ε_{ij} · {S_i, S_j}/2

where ε is the strain tensor and {A, B} = AB + BA is the anticommutator.

For axially symmetric strain along the NV axis (z), this simplifies to:

    H_strain = E_x (S_x² - S_y²) + E_y (S_x S_y + S_y S_x)

where E_x and E_y are the transverse strain components in MHz.

This is similar to the E-term in ZFS but can be time-dependent.

Strain Tensor
-------------
The full 3×3 strain tensor has 6 independent components:
    ε = [[ε_xx, ε_xy, ε_xz],
         [ε_xy, ε_yy, ε_yz],
         [ε_xz, ε_yz, ε_zz]]

Only certain combinations couple to the NV spin:
- E₁ ~ ε_xx - ε_yy (transverse)
- E₂ ~ 2ε_xy (transverse)
- Shifts D ~ ε_zz (axial)

References
----------
[1] Doherty et al., Physics Reports 528, 1-45 (2013)
[2] MacQuarrie et al., Phys. Rev. Lett. 111, 227602 (2013)
"""

import numpy as np
from typing import Optional, Callable, Union
from numpy.typing import ArrayLike

from .base import HamiltonianTerm
from ...core.operators import spin1_operators, extend_to_18x18
from ...core.constants import MHZ


class Strain(HamiltonianTerm):
    """
    Strain-induced splitting (static or time-dependent).

    Parameters
    ----------
    Ex : float
        Transverse strain component E_x in MHz. Default: 0
    Ey : float
        Transverse strain component E_y in MHz. Default: 0
    epsilon : array_like, optional
        Full 3×3 strain tensor (dimensionless). Overrides Ex, Ey.
    time_func : callable, optional
        Function time_func(t) returning (Ex, Ey) or 3×3 tensor at time t.
        If provided, makes the term time-dependent.
    name : str, optional
        Custom name for the term

    Attributes
    ----------
    Ex : float
        Static E_x component in MHz
    Ey : float
        Static E_y component in MHz
    epsilon : np.ndarray or None
        Full strain tensor if provided
    is_time_dependent : bool
        True if time_func is provided

    Examples
    --------
    >>> from sim import HamiltonianBuilder
    >>> from sim.hamiltonian.terms import ZFS, Strain
    >>>
    >>> # Static strain
    >>> H = HamiltonianBuilder()
    >>> H.add(ZFS(D=2.87))
    >>> H.add(Strain(Ex=10, Ey=5))  # 10 MHz, 5 MHz transverse strain
    >>>
    >>> # Time-dependent strain (e.g., acoustic wave)
    >>> def acoustic_strain(t):
    ...     freq = 1e9  # 1 GHz acoustic wave
    ...     amplitude = 5  # 5 MHz
    ...     return (amplitude * np.sin(2 * np.pi * freq * t), 0)
    >>>
    >>> H.add(Strain(time_func=acoustic_strain))

    Notes
    -----
    The strain coupling is analogous to the E-term in ZFS:
    - Ex corresponds to the E_x splitting
    - Ey corresponds to additional symmetry breaking

    Typical strain values in NV experiments:
    - Bulk diamond: < 1 MHz
    - Near implantation damage: 1-50 MHz
    - Under mechanical stress: 10-100 MHz
    """

    def __init__(
        self,
        Ex: float = 0.0,           # MHz
        Ey: float = 0.0,           # MHz
        epsilon: Optional[ArrayLike] = None,
        time_func: Optional[Callable] = None,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.Ex = Ex
        self.Ey = Ey
        self.epsilon = np.array(epsilon) if epsilon is not None else None
        self._time_func = time_func

        # Cache for static term
        self._cache: Optional[np.ndarray] = None

    @property
    def is_time_dependent(self) -> bool:
        """True if time_func is provided."""
        return self._time_func is not None

    def _build_3x3(self, Ex: float, Ey: float) -> np.ndarray:
        """
        Build 3×3 strain Hamiltonian from transverse components.

        Parameters
        ----------
        Ex : float
            E_x component in rad/s
        Ey : float
            E_y component in rad/s

        Returns
        -------
        np.ndarray
            3×3 complex Hermitian matrix in rad/s
        """
        S = spin1_operators()

        # H = Ex * (Sx² - Sy²) + Ey * {Sx, Sy}
        H = (
            Ex * (S.Sx @ S.Sx - S.Sy @ S.Sy) +
            Ey * (S.Sx @ S.Sy + S.Sy @ S.Sx)
        )

        return H

    def _build_from_tensor(self, epsilon: np.ndarray) -> np.ndarray:
        """
        Build 3×3 strain Hamiltonian from full strain tensor.

        Parameters
        ----------
        epsilon : np.ndarray
            3×3 strain tensor (dimensionless)

        Returns
        -------
        np.ndarray
            3×3 complex Hermitian matrix in rad/s
        """
        S = spin1_operators()
        S_list = [S.Sx, S.Sy, S.Sz]

        H = np.zeros((3, 3), dtype=np.complex128)

        for i in range(3):
            for j in range(3):
                if epsilon[i, j] != 0:
                    # {S_i, S_j}/2 = (S_i S_j + S_j S_i)/2
                    anticomm = (S_list[i] @ S_list[j] + S_list[j] @ S_list[i]) / 2
                    H += epsilon[i, j] * anticomm

        return H

    def build(self, t: float = 0.0) -> np.ndarray:
        """
        Build the 18×18 strain Hamiltonian at time t.

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
            # Get time-dependent values
            result = self._time_func(t)

            if isinstance(result, tuple) and len(result) == 2:
                Ex_t, Ey_t = result
                H_3x3 = self._build_3x3(Ex_t * MHZ, Ey_t * MHZ)
            else:
                # Assume it returns a 3×3 tensor
                epsilon_t = np.array(result)
                H_3x3 = self._build_from_tensor(epsilon_t)

            return extend_to_18x18(H_3x3)

        else:
            # Static case - use cache
            if self._cache is None:
                if self.epsilon is not None:
                    H_3x3 = self._build_from_tensor(self.epsilon)
                else:
                    H_3x3 = self._build_3x3(self.Ex * MHZ, self.Ey * MHZ)

                self._cache = extend_to_18x18(H_3x3)

            return self._cache.copy()

    def __repr__(self) -> str:
        if self.epsilon is not None:
            return f"Strain(epsilon=<3×3 tensor>)"
        elif self.is_time_dependent:
            return f"Strain(time-dependent)"
        else:
            return f"Strain(Ex={self.Ex} MHz, Ey={self.Ey} MHz)"
