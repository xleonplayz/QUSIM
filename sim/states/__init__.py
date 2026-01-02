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
Quantum states module for NV center simulation.

This module provides tools for working with quantum states:
- Density matrices and pure states
- Initial state preparation
- Measurement projectors

Exports
-------
DensityMatrix : class
    Density matrix container with analysis methods
Initial States : functions
    ground_state, thermal_state, superposition, etc.
Projectors : functions
    projector_ground, projector_excited, projector_ms, etc.

Example
-------
>>> from sim.states import ground_state, projector_ms
>>>
>>> # Prepare initial state
>>> rho0 = ground_state()  # |g, ms=0, mI=0⟩⟨...|
>>>
>>> # Measure ms=0 population
>>> P = projector_ms(0)
>>> pop = np.real(np.trace(P @ rho0))
"""

from .density_matrix import DensityMatrix
from .initial_states import (
    ground_state,
    ground_state_vector,
    excited_state,
    thermal_state,
    maximally_mixed,
    superposition,
)
from .projectors import (
    projector_ground,
    projector_excited,
    projector_ms,
    projector_mI,
    projector_state,
)

__all__ = [
    # Density matrix
    "DensityMatrix",
    # Initial states
    "ground_state",
    "ground_state_vector",
    "excited_state",
    "thermal_state",
    "maximally_mixed",
    "superposition",
    # Projectors
    "projector_ground",
    "projector_excited",
    "projector_ms",
    "projector_mI",
    "projector_state",
]
