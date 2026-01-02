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
QUSIM - NV Center Quantum Simulator Library

A Python library for simulating nitrogen-vacancy (NV) centers in diamond.
Solves the Lindblad master equation for open quantum systems with realistic
NV physics including zero-field splitting, hyperfine coupling, strain,
and environmental interactions.

Quick Start
-----------
>>> from sim import HamiltonianBuilder, LindbladSolver
>>> from sim.hamiltonian.terms import ZFS, Zeeman, HyperfineN14
>>> from sim.states import ground_state
>>> from sim.pulses import ramsey_sequence
>>>
>>> # Build Hamiltonian
>>> H = HamiltonianBuilder()
>>> H.add(ZFS(D=2.87))        # 2.87 GHz zero-field splitting
>>> H.add(Zeeman(B=10))       # 10 mT magnetic field
>>> H.add(HyperfineN14())     # N14 hyperfine coupling
>>>
>>> # Setup dynamics
>>> solver = LindbladSolver(H)
>>> solver.add_t2_dephasing(gamma=1e6)
>>>
>>> # Run simulation
>>> rho0 = ground_state()
>>> result = solver.evolve(rho0, t_span=(0, 1e-6))

Features
--------
- 18×18 Hilbert space: |g/e⟩ ⊗ |ms⟩ ⊗ |mI⟩ (ground/excited × S=1 × I=1)
- Modular Hamiltonian terms: ZFS, Zeeman, Hyperfine (N14/C13), Strain, Stark
- Lindblad master equation solver with T1, T2, optical decay
- Hardware interfaces: AWG, Laser, PhotonCounter
- Pulse sequences: Ramsey, Spin Echo, CPMG, XY4, XY8
- Practical units: GHz, mT, MHz (converted internally to rad/s)

Modules
-------
sim.core
    Spin operators, physical constants
sim.hamiltonian
    HamiltonianBuilder and modular terms
sim.states
    Density matrices, initial states, projectors
sim.dynamics
    LindbladSolver, dissipation operators
sim.interfaces
    AWGInterface, LaserInterface, PhotonCounter
sim.pulses
    Pulse shapes, basic pulses, standard sequences

References
----------
[1] Doherty et al., Physics Reports 528, 1-45 (2013)
[2] Maze et al., New J. Phys. 13, 025025 (2011)
"""

# Core
from .core.constants import GHZ, MHZ, KHZ, MT, GAMMA_E, D_GS, D_ES
from .core.operators import Sx, Sy, Sz, Sp, Sm, Ix, Iy, Iz

# Hamiltonian
from .hamiltonian.builder import HamiltonianBuilder

# States
from .states import (
    DensityMatrix,
    ground_state,
    excited_state,
    thermal_state,
    superposition,
    projector_ground,
    projector_excited,
    projector_ms,
)

# Dynamics
from .dynamics import LindbladSolver, EvolutionResult

# Interfaces
from .interfaces import AWGInterface, LaserInterface, PhotonCounter

# Pulses
from .pulses import (
    Pulse,
    pi_pulse,
    pi_half_pulse,
    PulseSequence,
    ramsey_sequence,
    spin_echo_sequence,
    cpmg_sequence,
    xy4_sequence,
    xy8_sequence,
)

__all__ = [
    # Constants
    "GHZ", "MHZ", "KHZ", "MT", "GAMMA_E", "D_GS", "D_ES",
    # Operators
    "Sx", "Sy", "Sz", "Sp", "Sm", "Ix", "Iy", "Iz",
    # Hamiltonian
    "HamiltonianBuilder",
    # States
    "DensityMatrix",
    "ground_state",
    "excited_state",
    "thermal_state",
    "superposition",
    "projector_ground",
    "projector_excited",
    "projector_ms",
    # Dynamics
    "LindbladSolver",
    "EvolutionResult",
    # Interfaces
    "AWGInterface",
    "LaserInterface",
    "PhotonCounter",
    # Pulses
    "Pulse",
    "pi_pulse",
    "pi_half_pulse",
    "PulseSequence",
    "ramsey_sequence",
    "spin_echo_sequence",
    "cpmg_sequence",
    "xy4_sequence",
    "xy8_sequence",
]

__version__ = "1.0.0"
__author__ = "Leon Kaiser"
__email__ = "I.kaiser@em.uni-frankfurt.de"
