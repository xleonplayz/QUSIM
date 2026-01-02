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
Standard pulse sequences for NV center experiments.

Implements common sequences for coherence measurements and
dynamical decoupling.

Sequence Types
--------------
Ramsey: Free induction decay (T2*)
    π/2 - τ - π/2

Spin Echo: Hahn echo (T2)
    π/2 - τ/2 - π - τ/2 - π/2

CPMG: Carr-Purcell-Meiboom-Gill (extended coherence)
    π/2 - [τ/2 - π_Y - τ/2]^N - π/2

XY4/XY8: Dynamical decoupling (robust to pulse errors)
    π/2 - [τ - π_X - τ - π_Y - τ - π_X - τ - π_Y]^N - π/2
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from .basic import Pulse, pi_pulse, pi_half_pulse


@dataclass
class PulseSequence:
    """
    Collection of pulses with timing.

    Attributes
    ----------
    pulses : list of Pulse
        List of pulses in the sequence
    delays : list of float
        Delay after each pulse in seconds

    Examples
    --------
    >>> seq = PulseSequence()
    >>> seq.add_pulse(pi_half_pulse(10), delay=1e-6)
    >>> seq.add_pulse(pi_pulse(10), delay=1e-6)
    >>> seq.add_pulse(pi_half_pulse(10))
    >>> print(f"Total duration: {seq.total_duration()*1e6:.1f} μs")
    """

    pulses: List[Pulse] = field(default_factory=list)
    delays: List[float] = field(default_factory=list)

    def add_pulse(self, pulse: Pulse, delay: float = 0.0):
        """
        Add a pulse to the sequence.

        Parameters
        ----------
        pulse : Pulse
            Pulse to add
        delay : float
            Delay after the pulse in seconds. Default: 0
        """
        self.pulses.append(pulse)
        self.delays.append(delay)

    def add_delay(self, delay: float):
        """Add a delay without a pulse."""
        # This extends the last delay
        if self.delays:
            self.delays[-1] += delay

    def total_duration(self) -> float:
        """Calculate total sequence duration."""
        t = 0.0
        for pulse, delay in zip(self.pulses, self.delays):
            t += pulse.duration + delay
        return t

    def to_awg(self, awg):
        """
        Load the sequence into an AWG interface.

        Parameters
        ----------
        awg : AWGInterface
            AWG interface to load into
        """
        for pulse, delay in zip(self.pulses, self.delays):
            awg.add_pulse(
                shape=pulse.shape,
                amplitude=pulse.amplitude,
                duration=pulse.duration,
                phase=pulse.phase,
                **pulse.kwargs
            )
            if delay > 0:
                awg.add_delay(delay)

    def __repr__(self) -> str:
        n = len(self.pulses)
        t = self.total_duration()
        return f"PulseSequence({n} pulses, {t*1e9:.1f} ns)"


def ramsey_sequence(
    tau: float,
    rabi_freq_mhz: float = 1.0,
    final_phase: float = 0.0,
    shape: str = "rect"
) -> PulseSequence:
    """
    Create a Ramsey sequence: π/2 - τ - π/2.

    Parameters
    ----------
    tau : float
        Free evolution time in seconds
    rabi_freq_mhz : float
        Rabi frequency for the pulses. Default: 1 MHz
    final_phase : float
        Phase of the final π/2 pulse in radians. Default: 0
    shape : str
        Pulse shape. Default: 'rect'

    Returns
    -------
    PulseSequence
        Ramsey sequence

    Examples
    --------
    >>> seq = ramsey_sequence(tau=1e-6, rabi_freq_mhz=10)
    >>> print(seq.total_duration())
    """
    seq = PulseSequence()

    # First π/2 pulse (X axis)
    seq.add_pulse(pi_half_pulse(rabi_freq_mhz, axis='x', shape=shape), delay=tau)

    # Second π/2 pulse (variable phase)
    final_pulse = pi_half_pulse(rabi_freq_mhz, axis='x', shape=shape)
    final_pulse.phase = final_phase
    seq.add_pulse(final_pulse)

    return seq


def spin_echo_sequence(
    tau: float,
    rabi_freq_mhz: float = 1.0,
    shape: str = "rect"
) -> PulseSequence:
    """
    Create a spin echo (Hahn echo) sequence: π/2 - τ/2 - π - τ/2 - π/2.

    Parameters
    ----------
    tau : float
        Total free evolution time in seconds
    rabi_freq_mhz : float
        Rabi frequency for the pulses. Default: 1 MHz
    shape : str
        Pulse shape. Default: 'rect'

    Returns
    -------
    PulseSequence
        Spin echo sequence
    """
    seq = PulseSequence()

    half_tau = tau / 2

    # π/2_X
    seq.add_pulse(pi_half_pulse(rabi_freq_mhz, axis='x', shape=shape), delay=half_tau)

    # π_Y (refocusing)
    seq.add_pulse(pi_pulse(rabi_freq_mhz, axis='y', shape=shape), delay=half_tau)

    # π/2_X (readout)
    seq.add_pulse(pi_half_pulse(rabi_freq_mhz, axis='x', shape=shape))

    return seq


def cpmg_sequence(
    tau: float,
    n_pulses: int,
    rabi_freq_mhz: float = 1.0,
    shape: str = "rect"
) -> PulseSequence:
    """
    Create a CPMG sequence: π/2 - [τ - π_Y - τ]^N - π/2.

    Parameters
    ----------
    tau : float
        Delay between refocusing pulses in seconds
    n_pulses : int
        Number of refocusing π pulses
    rabi_freq_mhz : float
        Rabi frequency. Default: 1 MHz
    shape : str
        Pulse shape. Default: 'rect'

    Returns
    -------
    PulseSequence
        CPMG sequence
    """
    seq = PulseSequence()

    half_tau = tau / 2

    # Initial π/2_X
    seq.add_pulse(pi_half_pulse(rabi_freq_mhz, axis='x', shape=shape), delay=half_tau)

    # N refocusing pulses
    for i in range(n_pulses):
        delay = tau if i < n_pulses - 1 else half_tau
        seq.add_pulse(pi_pulse(rabi_freq_mhz, axis='y', shape=shape), delay=delay)

    # Final π/2_X
    seq.add_pulse(pi_half_pulse(rabi_freq_mhz, axis='x', shape=shape))

    return seq


def xy4_sequence(
    tau: float,
    n_cycles: int = 1,
    rabi_freq_mhz: float = 1.0,
    shape: str = "rect"
) -> PulseSequence:
    """
    Create an XY4 dynamical decoupling sequence.

    Structure: π/2 - [τ - π_X - τ - π_Y - τ - π_X - τ - π_Y]^N - π/2

    Parameters
    ----------
    tau : float
        Delay between pulses in seconds
    n_cycles : int
        Number of XY4 cycles. Default: 1
    rabi_freq_mhz : float
        Rabi frequency. Default: 1 MHz
    shape : str
        Pulse shape. Default: 'rect'

    Returns
    -------
    PulseSequence
        XY4 sequence

    Notes
    -----
    XY4 is robust to pulse flip-angle errors due to the
    alternating X and Y phases.
    """
    seq = PulseSequence()

    # Initial π/2_X
    seq.add_pulse(pi_half_pulse(rabi_freq_mhz, axis='x', shape=shape), delay=tau)

    # XY4 cycles: X - Y - X - Y
    for _ in range(n_cycles):
        seq.add_pulse(pi_pulse(rabi_freq_mhz, axis='x', shape=shape), delay=tau)
        seq.add_pulse(pi_pulse(rabi_freq_mhz, axis='y', shape=shape), delay=tau)
        seq.add_pulse(pi_pulse(rabi_freq_mhz, axis='x', shape=shape), delay=tau)
        seq.add_pulse(pi_pulse(rabi_freq_mhz, axis='y', shape=shape), delay=tau)

    # Final π/2_X
    # Adjust last delay
    if seq.delays:
        seq.delays[-1] = 0  # Remove last delay before final pulse
    seq.add_pulse(pi_half_pulse(rabi_freq_mhz, axis='x', shape=shape))

    return seq


def xy8_sequence(
    tau: float,
    n_cycles: int = 1,
    rabi_freq_mhz: float = 1.0,
    shape: str = "rect"
) -> PulseSequence:
    """
    Create an XY8 dynamical decoupling sequence.

    Structure: π/2 - [τ - π_X - τ - π_Y - τ - π_X - τ - π_Y -
                      τ - π_Y - τ - π_X - τ - π_Y - τ - π_X]^N - π/2

    Parameters
    ----------
    tau : float
        Delay between pulses in seconds
    n_cycles : int
        Number of XY8 cycles. Default: 1
    rabi_freq_mhz : float
        Rabi frequency. Default: 1 MHz
    shape : str
        Pulse shape. Default: 'rect'

    Returns
    -------
    PulseSequence
        XY8 sequence

    Notes
    -----
    XY8 is more robust than XY4 due to the symmetric phase pattern.
    """
    seq = PulseSequence()

    # Initial π/2_X
    seq.add_pulse(pi_half_pulse(rabi_freq_mhz, axis='x', shape=shape), delay=tau)

    # XY8 cycles: X Y X Y Y X Y X
    for _ in range(n_cycles):
        seq.add_pulse(pi_pulse(rabi_freq_mhz, axis='x', shape=shape), delay=tau)
        seq.add_pulse(pi_pulse(rabi_freq_mhz, axis='y', shape=shape), delay=tau)
        seq.add_pulse(pi_pulse(rabi_freq_mhz, axis='x', shape=shape), delay=tau)
        seq.add_pulse(pi_pulse(rabi_freq_mhz, axis='y', shape=shape), delay=tau)
        seq.add_pulse(pi_pulse(rabi_freq_mhz, axis='y', shape=shape), delay=tau)
        seq.add_pulse(pi_pulse(rabi_freq_mhz, axis='x', shape=shape), delay=tau)
        seq.add_pulse(pi_pulse(rabi_freq_mhz, axis='y', shape=shape), delay=tau)
        seq.add_pulse(pi_pulse(rabi_freq_mhz, axis='x', shape=shape), delay=tau)

    # Adjust last delay and add final π/2
    if seq.delays:
        seq.delays[-1] = 0
    seq.add_pulse(pi_half_pulse(rabi_freq_mhz, axis='x', shape=shape))

    return seq


def rabi_sequence(
    duration: float,
    rabi_freq_mhz: float = 1.0,
    shape: str = "rect"
) -> PulseSequence:
    """
    Create a Rabi oscillation sequence (single pulse of variable duration).

    Parameters
    ----------
    duration : float
        Pulse duration in seconds
    rabi_freq_mhz : float
        Rabi frequency in MHz
    shape : str
        Pulse shape

    Returns
    -------
    PulseSequence
        Single-pulse Rabi sequence
    """
    seq = PulseSequence()

    pulse = Pulse(
        shape=shape,
        amplitude=rabi_freq_mhz,
        duration=duration,
        phase=0.0
    )
    seq.add_pulse(pulse)

    return seq
