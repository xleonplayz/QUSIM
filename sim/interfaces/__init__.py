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
Hardware interfaces for NV center experiments.

This module provides software interfaces that model real hardware:
- AWG: Arbitrary Waveform Generator for microwave pulses
- Laser: Optical excitation and readout
- PhotonCounter: Detection and photon statistics

These interfaces can be connected to the simulation to provide
realistic time-dependent control of the NV center.

Example
-------
>>> from sim.interfaces import AWGInterface, LaserInterface
>>>
>>> # Create AWG for MW control
>>> awg = AWGInterface()
>>> awg.add_pulse("gauss", amplitude=10, duration=100e-9, phase=0)
>>>
>>> # Create laser for optical pumping
>>> laser = LaserInterface()
>>> laser.add_pulse(start=0, duration=1e-6, power=1.0)
"""

from .awg import AWGInterface
from .laser import LaserInterface
from .photon_counter import PhotonCounter

__all__ = ["AWGInterface", "LaserInterface", "PhotonCounter"]
