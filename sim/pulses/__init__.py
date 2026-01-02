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
Pulse sequences for NV center experiments.

This module provides:
- Pulse shapes (Gaussian, DRAG, etc.)
- Basic pulses (π, π/2)
- Standard sequences (Ramsey, Echo, CPMG, DD)

Standard NV Experiments
-----------------------
Ramsey: π/2 - τ - π/2
    Measures free induction decay, T2*

Spin Echo: π/2 - τ/2 - π - τ/2 - π/2
    Refocuses static inhomogeneities, T2

CPMG: π/2 - [τ - π - τ]^N - π/2
    Multiple refocusing, extends coherence

XY4/XY8: Dynamical decoupling with phase cycling
    Robust to pulse errors

Example
-------
>>> from sim.pulses import ramsey_sequence
>>> from sim.interfaces import AWGInterface
>>>
>>> # Create Ramsey sequence with 1 μs delay
>>> seq = ramsey_sequence(tau=1e-6, rabi_freq_mhz=10)
>>>
>>> # Load into AWG
>>> awg = AWGInterface()
>>> seq.to_awg(awg)
"""

from .shapes import (
    rectangular,
    gaussian,
    sinc_pulse,
    blackman,
    drag,
    hermite,
)
from .basic import Pulse, pi_pulse, pi_half_pulse, arbitrary_rotation
from .sequences import (
    PulseSequence,
    ramsey_sequence,
    spin_echo_sequence,
    cpmg_sequence,
    xy4_sequence,
    xy8_sequence,
)

__all__ = [
    # Shapes
    "rectangular",
    "gaussian",
    "sinc_pulse",
    "blackman",
    "drag",
    "hermite",
    # Basic pulses
    "Pulse",
    "pi_pulse",
    "pi_half_pulse",
    "arbitrary_rotation",
    # Sequences
    "PulseSequence",
    "ramsey_sequence",
    "spin_echo_sequence",
    "cpmg_sequence",
    "xy4_sequence",
    "xy8_sequence",
]
