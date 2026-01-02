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
Dynamics module for NV center time evolution.

This module provides the Lindblad master equation solver for
simulating open quantum system dynamics including:
- Coherent evolution under the Hamiltonian
- Dissipation (T1 relaxation, spontaneous emission)
- Dephasing (T2*, T2)

The Master Equation
-------------------
The Lindblad master equation is:

    dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})

where:
- H is the system Hamiltonian
- L_k are Lindblad collapse operators
- ρ is the density matrix

Exports
-------
LindbladSolver : class
    Main solver class
EvolutionResult : class
    Container for evolution results
Dissipation operators : functions
    t1_operators, t2_dephasing_operator, optical_decay_operators

Example
-------
>>> from sim import HamiltonianBuilder
>>> from sim.hamiltonian.terms import ZFS, Zeeman
>>> from sim.dynamics import LindbladSolver
>>> from sim.states import ground_state
>>>
>>> # Build Hamiltonian
>>> H = HamiltonianBuilder()
>>> H.add(ZFS(D=2.87))
>>> H.add(Zeeman(B=10))
>>>
>>> # Setup solver with dissipation
>>> solver = LindbladSolver(H)
>>> solver.add_t2_dephasing(gamma=1e7)  # T2* = 100 ns
>>>
>>> # Evolve
>>> rho0 = ground_state()
>>> result = solver.evolve(rho0, t_span=(0, 1e-6), n_steps=100)
"""

from .lindblad import LindbladSolver, EvolutionResult
from .dissipation import (
    t1_operators,
    t2_dephasing_operator,
    optical_decay_operators,
    nuclear_relaxation_operators,
)

__all__ = [
    "LindbladSolver",
    "EvolutionResult",
    "t1_operators",
    "t2_dephasing_operator",
    "optical_decay_operators",
    "nuclear_relaxation_operators",
]
