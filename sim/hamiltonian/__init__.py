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
Hamiltonian module for NV center simulation.

This module provides the HamiltonianBuilder class for constructing
the total Hamiltonian from individual physics terms.

Example
-------
>>> from sim.hamiltonian import HamiltonianBuilder
>>> from sim.hamiltonian.terms import ZFS, Zeeman
>>>
>>> H = HamiltonianBuilder()
>>> H.add(ZFS(D=2.87))
>>> H.add(Zeeman(B=10))
>>> H_matrix = H.build(t=0.0)
"""

from .builder import HamiltonianBuilder

__all__ = ["HamiltonianBuilder"]
