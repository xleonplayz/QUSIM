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
Basic pulse definitions for NV center control.

Provides π and π/2 pulses for standard quantum gates.

Rotation Convention
-------------------
A pulse with Rabi frequency Ω, duration τ, and phase φ performs
the rotation:

    R_φ(θ) = exp(-i θ/2 (cos(φ)σ_x + sin(φ)σ_y))

where θ = Ω τ is the rotation angle.

For resonant driving:
    π-pulse: θ = π, causes spin flip |0⟩ ↔ |±1⟩
    π/2-pulse: θ = π/2, creates superposition
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Pulse:
    """
    Representation of a single MW pulse.

    Attributes
    ----------
    shape : str
        Pulse shape: 'rect', 'gauss', 'drag', etc.
    amplitude : float
        Peak Rabi frequency in MHz
    duration : float
        Pulse duration in seconds
    phase : float
        Pulse phase in radians
    kwargs : dict
        Additional shape-specific parameters
    """

    shape: str
    amplitude: float   # MHz (Rabi frequency)
    duration: float    # seconds
    phase: float       # radians
    kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}

    def rotation_angle(self) -> float:
        """Calculate total rotation angle θ = Ω τ."""
        return self.amplitude * 1e6 * 2 * np.pi * self.duration

    def axis_label(self) -> str:
        """Get rotation axis label (X, Y, -X, -Y)."""
        phase_deg = np.degrees(self.phase) % 360

        if abs(phase_deg) < 1 or abs(phase_deg - 360) < 1:
            return "X"
        elif abs(phase_deg - 90) < 1:
            return "Y"
        elif abs(phase_deg - 180) < 1:
            return "-X"
        elif abs(phase_deg - 270) < 1:
            return "-Y"
        else:
            return f"φ={phase_deg:.0f}°"


def pi_pulse(
    rabi_freq_mhz: float = 1.0,
    axis: str = "x",
    shape: str = "rect",
    **kwargs
) -> Pulse:
    """
    Create a π pulse (180° rotation).

    Parameters
    ----------
    rabi_freq_mhz : float
        Rabi frequency in MHz. Default: 1 MHz
    axis : str
        Rotation axis: 'x', 'y', '-x', '-y'. Default: 'x'
    shape : str
        Pulse shape: 'rect', 'gauss', etc. Default: 'rect'
    **kwargs
        Additional shape parameters (sigma, beta, etc.)

    Returns
    -------
    Pulse
        π pulse with calculated duration

    Examples
    --------
    >>> p = pi_pulse(rabi_freq_mhz=10)
    >>> print(f"Duration: {p.duration*1e9:.1f} ns")
    Duration: 50.0 ns

    >>> # Y-axis rotation
    >>> p_y = pi_pulse(rabi_freq_mhz=20, axis='y')
    """
    # Duration for π rotation: π = Ω * t_π => t_π = π / Ω
    omega = rabi_freq_mhz * 1e6 * 2 * np.pi  # rad/s
    duration = np.pi / omega

    # Phase for rotation axis
    phase_map = {
        'x': 0,
        'y': np.pi / 2,
        '-x': np.pi,
        '-y': 3 * np.pi / 2,
        'X': 0,
        'Y': np.pi / 2,
        '-X': np.pi,
        '-Y': 3 * np.pi / 2,
    }
    phase = phase_map.get(axis, 0)

    return Pulse(
        shape=shape,
        amplitude=rabi_freq_mhz,
        duration=duration,
        phase=phase,
        kwargs=kwargs
    )


def pi_half_pulse(
    rabi_freq_mhz: float = 1.0,
    axis: str = "x",
    shape: str = "rect",
    **kwargs
) -> Pulse:
    """
    Create a π/2 pulse (90° rotation).

    Parameters
    ----------
    rabi_freq_mhz : float
        Rabi frequency in MHz. Default: 1 MHz
    axis : str
        Rotation axis: 'x', 'y', '-x', '-y'. Default: 'x'
    shape : str
        Pulse shape. Default: 'rect'
    **kwargs
        Additional shape parameters

    Returns
    -------
    Pulse
        π/2 pulse with calculated duration

    Examples
    --------
    >>> p = pi_half_pulse(rabi_freq_mhz=10)
    >>> print(f"Duration: {p.duration*1e9:.1f} ns")
    Duration: 25.0 ns
    """
    # Duration for π/2 rotation
    omega = rabi_freq_mhz * 1e6 * 2 * np.pi  # rad/s
    duration = (np.pi / 2) / omega

    # Phase for rotation axis
    phase_map = {
        'x': 0,
        'y': np.pi / 2,
        '-x': np.pi,
        '-y': 3 * np.pi / 2,
    }
    phase = phase_map.get(axis.lower(), 0)

    return Pulse(
        shape=shape,
        amplitude=rabi_freq_mhz,
        duration=duration,
        phase=phase,
        kwargs=kwargs
    )


def arbitrary_rotation(
    theta: float,
    phi: float = 0,
    rabi_freq_mhz: float = 1.0,
    shape: str = "rect",
    **kwargs
) -> Pulse:
    """
    Create a pulse for arbitrary rotation R_φ(θ).

    Parameters
    ----------
    theta : float
        Rotation angle in radians
    phi : float
        Rotation axis azimuthal angle in radians. Default: 0 (X axis)
    rabi_freq_mhz : float
        Rabi frequency in MHz. Default: 1 MHz
    shape : str
        Pulse shape. Default: 'rect'
    **kwargs
        Additional shape parameters

    Returns
    -------
    Pulse
        Arbitrary rotation pulse

    Examples
    --------
    >>> # π/4 rotation around Y
    >>> p = arbitrary_rotation(theta=np.pi/4, phi=np.pi/2)
    """
    # Duration for rotation angle θ
    omega = rabi_freq_mhz * 1e6 * 2 * np.pi  # rad/s
    duration = theta / omega

    return Pulse(
        shape=shape,
        amplitude=rabi_freq_mhz,
        duration=duration,
        phase=phi,
        kwargs=kwargs
    )


def composite_pi_pulse(
    rabi_freq_mhz: float = 1.0,
    pulse_type: str = "BB1",
    shape: str = "rect",
    **kwargs
) -> list:
    """
    Create a composite π pulse for error compensation.

    Parameters
    ----------
    rabi_freq_mhz : float
        Rabi frequency in MHz
    pulse_type : str
        Type of composite pulse:
        - 'BB1': Broadband 1 (robust to amplitude errors)
        - 'CORPSE': (robust to frequency errors)
    shape : str
        Pulse shape

    Returns
    -------
    list of Pulse
        Sequence of pulses comprising the composite pulse
    """
    if pulse_type == "BB1":
        # BB1: 180_X - 360_phi1 - 180_phi2
        # where phi1 = arccos(-1/4), phi2 = 3*phi1
        phi1 = np.arccos(-0.25)
        phi2 = 3 * phi1

        return [
            pi_pulse(rabi_freq_mhz, axis='x', shape=shape, **kwargs),
            arbitrary_rotation(2*np.pi, phi1, rabi_freq_mhz, shape, **kwargs),
            arbitrary_rotation(np.pi, phi2, rabi_freq_mhz, shape, **kwargs),
        ]

    elif pulse_type == "CORPSE":
        # CORPSE: 420_X - 300_-X - 60_X (approximately)
        theta1 = 2 * np.pi + np.pi / 3
        theta2 = 2 * np.pi - np.pi / 3
        theta3 = np.pi / 3

        return [
            arbitrary_rotation(theta1, 0, rabi_freq_mhz, shape, **kwargs),
            arbitrary_rotation(theta2, np.pi, rabi_freq_mhz, shape, **kwargs),
            arbitrary_rotation(theta3, 0, rabi_freq_mhz, shape, **kwargs),
        ]

    else:
        raise ValueError(f"Unknown composite pulse type: {pulse_type}")
