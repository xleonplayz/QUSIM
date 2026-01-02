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
Microwave drive for NV center spin control.

Microwave fields at ~2.87 GHz drive transitions between the ms=0
and ms=±1 spin states. This is the primary control mechanism for
NV spin manipulation.

Physics
-------
In the rotating frame (rotating wave approximation), the MW drive
Hamiltonian is:

    H_MW = (Ω/2) · [S₊ e^(-iφ) + S₋ e^(+iφ)] + Δ · Sz

where:
- Ω(t) is the Rabi frequency (proportional to MW amplitude)
- φ(t) is the phase of the MW field
- Δ is the detuning from resonance
- S± = Sx ± i·Sy are the raising/lowering operators

The Rabi frequency is related to the MW magnetic field by:

    Ω = γ_e · B_MW

where γ_e ≈ 28 GHz/T is the electron gyromagnetic ratio.

Typical Values
--------------
- Rabi frequency: 1-50 MHz (depends on MW power)
- π-pulse duration: 10-500 ns
- Detuning: 0 for resonant driving

References
----------
[1] Childress & Hanson, MRS Bulletin 38, 134 (2013)
[2] Dréau et al., Phys. Rev. B 84, 195204 (2011)
"""

import numpy as np
from typing import Optional, Callable, Union
from numpy.typing import ArrayLike

from .base import HamiltonianTerm
from ...core.operators import spin1_operators, extend_electron_only
from ...core.constants import MHZ


class MicrowaveDrive(HamiltonianTerm):
    """
    Time-dependent microwave drive in rotating frame.

    Parameters
    ----------
    omega : float or callable
        Rabi frequency in MHz. Can be:
        - float: constant Rabi frequency
        - callable: omega(t) returning Rabi frequency at time t
    phase : float or callable
        Phase in radians. Can be:
        - float: constant phase
        - callable: phase(t) returning phase at time t
    detuning : float
        Frequency detuning from resonance in MHz. Default: 0
    manifold : str
        Which electronic manifold: "ground", "excited", or "both".
        Default: "ground"
    name : str, optional
        Custom name for the term

    Attributes
    ----------
    omega : float or callable
        Rabi frequency parameter
    phase : float or callable
        Phase parameter
    detuning : float
        Detuning in MHz
    manifold : str
        Target manifold

    Examples
    --------
    >>> from sim import HamiltonianBuilder
    >>> from sim.hamiltonian.terms import ZFS, MicrowaveDrive
    >>>
    >>> # Constant drive
    >>> H = HamiltonianBuilder()
    >>> H.add(ZFS(D=2.87))
    >>> H.add(MicrowaveDrive(omega=10, phase=0))  # 10 MHz Rabi frequency
    >>>
    >>> # Time-dependent Rabi frequency (pulsed)
    >>> def rabi_pulse(t):
    ...     if 0 <= t < 100e-9:  # 100 ns pulse
    ...         return 10.0  # 10 MHz
    ...     return 0.0
    >>>
    >>> H.add(MicrowaveDrive(omega=rabi_pulse))
    >>>
    >>> # Gaussian pulse shape
    >>> def gaussian_rabi(t):
    ...     t0, sigma = 50e-9, 20e-9
    ...     return 20 * np.exp(-(t - t0)**2 / (2 * sigma**2))
    >>>
    >>> H.add(MicrowaveDrive(omega=gaussian_rabi))

    Notes
    -----
    The MW drive is always considered time-dependent even if omega
    and phase are constants, because it represents an oscillating field.

    The rotating wave approximation (RWA) assumes near-resonant driving.
    For large detuning, the RWA breaks down.
    """

    def __init__(
        self,
        omega: Union[float, Callable] = 0.0,   # MHz
        phase: Union[float, Callable] = 0.0,   # radians
        detuning: float = 0.0,                 # MHz
        manifold: str = "ground",
        name: Optional[str] = None
    ):
        super().__init__(name=name)

        self._omega_param = omega
        self._phase_param = phase
        self.detuning = detuning
        self.manifold = manifold

        # Determine if omega/phase are functions or constants
        self._omega_is_func = callable(omega)
        self._phase_is_func = callable(phase)

    @property
    def is_time_dependent(self) -> bool:
        """MW drive is always time-dependent."""
        return True

    def _get_omega(self, t: float) -> float:
        """Get Rabi frequency at time t in MHz."""
        if self._omega_is_func:
            return self._omega_param(t)
        return self._omega_param

    def _get_phase(self, t: float) -> float:
        """Get phase at time t in radians."""
        if self._phase_is_func:
            return self._phase_param(t)
        return self._phase_param

    def _build_3x3(self, Omega: float, phi: float, Delta: float) -> np.ndarray:
        """
        Build 3×3 MW drive Hamiltonian.

        Parameters
        ----------
        Omega : float
            Rabi frequency in rad/s
        phi : float
            Phase in radians
        Delta : float
            Detuning in rad/s

        Returns
        -------
        np.ndarray
            3×3 complex matrix in rad/s
        """
        S = spin1_operators()

        # H = (Ω/2)(S₊ e^(-iφ) + S₋ e^(+iφ)) + Δ Sz
        # S₊ = Sx + i·Sy, S₋ = Sx - i·Sy

        H = (
            (Omega / 2) * (S.Sp * np.exp(-1j * phi) + S.Sm * np.exp(1j * phi)) +
            Delta * S.Sz
        )

        return H

    def build(self, t: float = 0.0) -> np.ndarray:
        """
        Build the 18×18 MW drive Hamiltonian at time t.

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
        The MW drive typically only acts on the ground state manifold
        in standard ODMR experiments.
        """
        # Get time-dependent parameters
        Omega = self._get_omega(t) * MHZ  # Convert to rad/s
        phi = self._get_phase(t)
        Delta = self.detuning * MHZ

        # Build 3×3 matrix
        H_3x3 = self._build_3x3(Omega, phi, Delta)

        # Extend to 18×18
        if self.manifold == "both":
            # Act on both g and e manifolds
            I2 = np.eye(2, dtype=np.complex128)
            I3 = np.eye(3, dtype=np.complex128)
            return np.kron(np.kron(I2, H_3x3), I3)
        else:
            # Act only on ground or excited manifold
            return extend_electron_only(H_3x3, self.manifold)

    def pi_time(self) -> Optional[float]:
        """
        Calculate π-pulse duration for constant Rabi frequency.

        Returns
        -------
        float or None
            π-pulse duration in seconds, or None if Rabi freq is time-dependent
        """
        if self._omega_is_func:
            return None

        if self._omega_param == 0:
            return None

        # π = Ω * t_π  =>  t_π = π / Ω
        return np.pi / (self._omega_param * MHZ)

    def __repr__(self) -> str:
        if self._omega_is_func:
            return f"MicrowaveDrive(omega=func, manifold={self.manifold})"
        else:
            return f"MicrowaveDrive(omega={self._omega_param} MHz, manifold={self.manifold})"
