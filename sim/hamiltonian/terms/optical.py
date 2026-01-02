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
Optical coupling between ground and excited states.

Laser light couples the NV ground state (³A₂) to the excited state
(³E) through electric dipole transitions. This is used for optical
initialization, readout, and coherent optical control.

Physics
-------
The optical Hamiltonian in the rotating frame is:

    H_opt = (Ω_L/2) · [|g⟩⟨e| + |e⟩⟨g|]

where Ω_L is the optical Rabi frequency, proportional to the
laser electric field amplitude.

For spin-conserving transitions (Δms = 0):

    H_opt = (Ω_L/2) · Σ_{ms,mI} [|g,ms,mI⟩⟨e,ms,mI| + h.c.]

The optical Rabi frequency is related to laser parameters by:

    Ω_L = d · E_0 / ℏ

where d is the dipole moment and E_0 is the electric field amplitude.

Typical Values
--------------
- Zero-phonon line: 637 nm (1.945 eV)
- Green excitation: 532 nm
- Dipole moment: d ≈ 1.6 × 10⁻²⁹ C·m
- Typical Rabi frequencies: 1-100 MHz

References
----------
[1] Doherty et al., Physics Reports 528, 1-45 (2013)
[2] Robledo et al., Nature 477, 574 (2011)
"""

import numpy as np
from typing import Optional, Callable, Union

from .base import HamiltonianTerm
from ...core.constants import MHZ


class OpticalCoupling(HamiltonianTerm):
    """
    Optical coupling between ground and excited states.

    Parameters
    ----------
    omega : float or callable
        Optical Rabi frequency in MHz. Can be:
        - float: constant Rabi frequency
        - callable: omega(t) returning Rabi frequency at time t
    detuning : float
        Detuning from the optical transition in GHz. Default: 0
    spin_conserving : bool
        If True, only spin-conserving transitions (Δms = 0).
        Default: True
    name : str, optional
        Custom name for the term

    Attributes
    ----------
    omega : float or callable
        Optical Rabi frequency parameter
    detuning : float
        Optical detuning in GHz
    spin_conserving : bool
        Transition selection rule

    Examples
    --------
    >>> from sim import HamiltonianBuilder
    >>> from sim.hamiltonian.terms import ZFS, OpticalCoupling
    >>>
    >>> # Constant optical drive
    >>> H = HamiltonianBuilder()
    >>> H.add(ZFS(D=2.87))
    >>> H.add(OpticalCoupling(omega=50))  # 50 MHz Rabi frequency
    >>>
    >>> # Pulsed optical excitation
    >>> def laser_pulse(t):
    ...     if 0 <= t < 10e-9:  # 10 ns pulse
    ...         return 100.0  # 100 MHz
    ...     return 0.0
    >>>
    >>> H.add(OpticalCoupling(omega=laser_pulse))

    Notes
    -----
    The optical coupling creates coherent superpositions of ground
    and excited states. In practice, spontaneous emission quickly
    destroys these superpositions (lifetime ~12 ns).

    For optical pumping (initialization), the Lindblad dissipation
    terms are more important than the coherent coupling.
    """

    def __init__(
        self,
        omega: Union[float, Callable] = 0.0,   # MHz
        detuning: float = 0.0,                 # GHz
        spin_conserving: bool = True,
        name: Optional[str] = None
    ):
        super().__init__(name=name)

        self._omega_param = omega
        self.detuning = detuning
        self.spin_conserving = spin_conserving

        # Determine if omega is a function
        self._omega_is_func = callable(omega)

    @property
    def is_time_dependent(self) -> bool:
        """Optical coupling is always time-dependent."""
        return True

    def _get_omega(self, t: float) -> float:
        """Get optical Rabi frequency at time t in MHz."""
        if self._omega_is_func:
            return self._omega_param(t)
        return self._omega_param

    def build(self, t: float = 0.0) -> np.ndarray:
        """
        Build the 18×18 optical coupling Hamiltonian at time t.

        Parameters
        ----------
        t : float
            Time in seconds

        Returns
        -------
        np.ndarray
            18×18 complex matrix in rad/s

        Notes
        -----
        The optical coupling connects the ground state manifold
        (indices 0-8) to the excited state manifold (indices 9-17).

        For spin-conserving transitions:
            |g, ms, mI⟩ ↔ |e, ms, mI⟩

        The coupling matrix has off-diagonal blocks:
            H[0:9, 9:18] = (Ω/2) · I_9
            H[9:18, 0:9] = (Ω/2) · I_9  (conjugate)
        """
        Omega = self._get_omega(t) * MHZ  # Convert to rad/s

        H = np.zeros((18, 18), dtype=np.complex128)

        if self.spin_conserving:
            # Spin-conserving: |g,ms,mI⟩ ↔ |e,ms,mI⟩
            # This is a 9×9 identity block connecting g and e manifolds
            coupling = (Omega / 2) * np.eye(9, dtype=np.complex128)

            # Off-diagonal blocks
            H[:9, 9:] = coupling      # |g⟩ → |e⟩
            H[9:, :9] = coupling      # |e⟩ → |g⟩ (Hermitian)

        else:
            # Allow all transitions (for non-spin-conserving processes)
            # This would need a more detailed model
            coupling = (Omega / 2) * np.eye(9, dtype=np.complex128)
            H[:9, 9:] = coupling
            H[9:, :9] = coupling

        return H

    def rabi_period(self) -> Optional[float]:
        """
        Calculate Rabi oscillation period for constant Rabi frequency.

        Returns
        -------
        float or None
            Period in seconds, or None if Rabi freq is time-dependent
        """
        if self._omega_is_func:
            return None

        if self._omega_param == 0:
            return None

        # T = 2π / Ω
        return 2 * np.pi / (self._omega_param * MHZ)

    def __repr__(self) -> str:
        if self._omega_is_func:
            return f"OpticalCoupling(omega=func)"
        else:
            return f"OpticalCoupling(omega={self._omega_param} MHz)"
