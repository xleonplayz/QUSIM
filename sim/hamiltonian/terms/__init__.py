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
Hamiltonian terms for NV center simulation.

This module provides individual physics terms that can be combined
using the HamiltonianBuilder.

Available Terms
---------------
Static Terms:
    ZFS : Zero-Field Splitting (~2.87 GHz)
    Zeeman : Magnetic field interaction
    HyperfineN14 : N14 hyperfine + quadrupole (~2.14 MHz)
    HyperfineC13 : C13 dipolar hyperfine
    Strain : Crystal strain effects
    Stark : Electric field effects
    JahnTeller : Vibronic coupling (excited state)

Time-Dependent Terms:
    MicrowaveDrive : MW spin control
    OpticalCoupling : Laser g↔e coupling

Example
-------
>>> from sim import HamiltonianBuilder
>>> from sim.hamiltonian.terms import ZFS, Zeeman, HyperfineN14, MicrowaveDrive
>>>
>>> H = HamiltonianBuilder()
>>> H.add(ZFS(D=2.87))
>>> H.add(Zeeman(B=10))
>>> H.add(HyperfineN14())
>>> H.add(MicrowaveDrive(omega=10))  # 10 MHz Rabi frequency
>>>
>>> H_matrix = H.build(t=0)
"""

from .base import HamiltonianTerm
from .zfs import ZFS
from .zeeman import Zeeman
from .hyperfine_n14 import HyperfineN14
from .hyperfine_c13 import HyperfineC13
from .strain import Strain
from .stark import Stark
from .mw_drive import MicrowaveDrive
from .optical import OpticalCoupling
from .jahn_teller import JahnTeller

__all__ = [
    "HamiltonianTerm",
    # Static terms
    "ZFS",
    "Zeeman",
    "HyperfineN14",
    "HyperfineC13",
    "Strain",
    "Stark",
    "JahnTeller",
    # Time-dependent terms
    "MicrowaveDrive",
    "OpticalCoupling",
]
