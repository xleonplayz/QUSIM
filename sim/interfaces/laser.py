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
Laser interface for optical control.

The laser provides optical excitation for:
- Spin initialization (optical pumping)
- Spin readout (fluorescence detection)
- Coherent optical control (Rabi oscillations)

Laser Parameters
----------------
- Wavelength: 532 nm (green) or 637 nm (ZPL)
- Power: Typically 0.1-10 mW
- Rabi frequency: Depends on power and beam waist

The optical Rabi frequency is related to laser power by:
    Ω_L = d·E₀/ℏ = d·√(2P/(ε₀cπw₀²))/ℏ

where d is the dipole moment and w₀ is the beam waist.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class LaserPulse:
    """Single laser pulse."""
    start: float      # Start time in seconds
    duration: float   # Duration in seconds
    power: float      # Power in mW
    shape: str = "rect"  # Pulse shape


@dataclass
class LaserInterface:
    """
    Laser interface for optical control.

    Parameters
    ----------
    wavelength : float
        Laser wavelength in nm. Default: 532 (green)
    power_to_rabi : float
        Conversion factor from mW to MHz. Default: 10 MHz/mW

    Attributes
    ----------
    wavelength : float
        Laser wavelength in nm
    pulses : list
        List of scheduled laser pulses

    Examples
    --------
    >>> laser = LaserInterface()
    >>> laser.add_pulse(start=0, duration=1e-6, power=1.0)
    >>> laser.add_pulse(start=2e-6, duration=100e-9, power=0.5)
    >>>
    >>> # Get power at specific time
    >>> p = laser.get_power(500e-9)  # 1.0 mW
    """

    wavelength: float = 532.0  # nm
    power_to_rabi: float = 10.0  # MHz/mW
    linewidth: float = 0.0  # MHz (for noise simulation)
    _pulses: List[LaserPulse] = field(default_factory=list)

    def __post_init__(self):
        # Laser frequency
        c = 3e8  # m/s
        self.frequency = c / (self.wavelength * 1e-9)  # Hz

    @property
    def pulses(self) -> List[LaserPulse]:
        """List of scheduled pulses."""
        return self._pulses.copy()

    def clear(self):
        """Clear all pulses."""
        self._pulses.clear()

    def add_pulse(
        self,
        start: float,
        duration: float,
        power: float,
        shape: str = "rect"
    ):
        """
        Add a laser pulse.

        Parameters
        ----------
        start : float
            Start time in seconds
        duration : float
            Pulse duration in seconds
        power : float
            Laser power in mW
        shape : str
            Pulse shape: 'rect', 'gauss'. Default: 'rect'
        """
        pulse = LaserPulse(start, duration, power, shape)
        self._pulses.append(pulse)

    def add_cw(self, power: float, t_start: float = 0, t_end: float = 1e-3):
        """
        Add continuous wave operation.

        Parameters
        ----------
        power : float
            CW power in mW
        t_start : float
            Start time. Default: 0
        t_end : float
            End time. Default: 1 ms
        """
        self.add_pulse(t_start, t_end - t_start, power, "rect")

    def get_power(self, t: float) -> float:
        """
        Get laser power at time t.

        Parameters
        ----------
        t : float
            Time in seconds

        Returns
        -------
        float
            Power in mW (0 if laser is off)
        """
        for pulse in self._pulses:
            if pulse.start <= t < pulse.start + pulse.duration:
                if pulse.shape == "rect":
                    return pulse.power
                elif pulse.shape == "gauss":
                    t0 = pulse.start + pulse.duration / 2
                    sigma = pulse.duration / 6
                    return pulse.power * np.exp(-((t - t0) / sigma) ** 2)

        return 0.0

    def get_rabi(self, t: float) -> float:
        """
        Get optical Rabi frequency at time t.

        Parameters
        ----------
        t : float
            Time in seconds

        Returns
        -------
        float
            Rabi frequency in MHz
        """
        power = self.get_power(t)
        return power * self.power_to_rabi

    def is_on(self, t: float) -> bool:
        """Check if laser is on at time t."""
        return self.get_power(t) > 0

    def to_optical_coupling(self):
        """
        Create an OpticalCoupling term from this laser.

        Returns
        -------
        OpticalCoupling
            Hamiltonian term with this laser waveform
        """
        from ..hamiltonian.terms import OpticalCoupling

        return OpticalCoupling(omega=self.get_rabi)

    def schedule_initialization(
        self,
        t_start: float = 0,
        duration: float = 1e-6,
        power: float = 1.0
    ):
        """
        Schedule an initialization pulse.

        Typical initialization uses a few microseconds of green light
        to optically pump into ms=0.

        Parameters
        ----------
        t_start : float
            Start time in seconds
        duration : float
            Pulse duration in seconds. Default: 1 μs
        power : float
            Laser power in mW. Default: 1 mW
        """
        self.add_pulse(t_start, duration, power, "rect")

    def schedule_readout(
        self,
        t_start: float,
        duration: float = 300e-9,
        power: float = 1.0
    ):
        """
        Schedule a readout pulse.

        Readout typically uses a short pulse (~300 ns) to measure
        fluorescence before spin flips occur.

        Parameters
        ----------
        t_start : float
            Start time in seconds
        duration : float
            Pulse duration. Default: 300 ns
        power : float
            Laser power in mW. Default: 1 mW
        """
        self.add_pulse(t_start, duration, power, "rect")

    def __repr__(self) -> str:
        n_pulses = len(self._pulses)
        return f"LaserInterface(wavelength={self.wavelength} nm, {n_pulses} pulses)"
