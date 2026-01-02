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
Core module: Spin operators and physical constants.

This module provides the fundamental building blocks for
NV center simulation.

Exports
-------
Operators (Spin-1):
    spin1_operators : Creates spin-1 operators (Sx, Sy, Sz, S+, S-)
    Spin1Ops : Dataclass container for the operators
    extend_to_18x18 : Extends 3×3 to 18×18
    extend_electron_only : Extends only to g or e manifold

Operators (Spin-1/2, Nuclear):
    spin_half_operators : Creates spin-1/2 operators for C13
    SpinHalfOps : Dataclass container
    extend_9x9_to_18x18 : Extends 9×9 to 18×18
    dipolar_tensor : Computes dipolar coupling tensor

Units (conversion factors):
    GHZ : 1 GHz in rad/s
    MHZ : 1 MHz in rad/s
    KHZ : 1 kHz in rad/s
    MT : 1 mT in Tesla
    UM : 1 µm in meters
    NM : 1 nm in meters
"""

from .operators import (
    spin1_operators,
    Spin1Ops,
    extend_to_18x18,
    extend_electron_only,
)

from .nuclear_operators import (
    spin_half_operators,
    SpinHalfOps,
    extend_9x9_to_18x18,
    extend_9x9_to_18x18_manifold,
    build_spin_spin_coupling,
    dipolar_tensor,
)

from .constants import (
    # Units
    GHZ, MHZ, KHZ, HZ,
    MT, UT, GAUSS,
    UM, NM, ANGSTROM,
    # Fundamental constants
    HBAR, MU_B, MU_N, KB,
    # NV-specific
    G_E, GAMMA_E,
    G_N14, GAMMA_N14,
    G_C13, GAMMA_C13,
    D_GS, D_ES,
    A_PARALLEL_N14, A_PERP_N14, P_N14,
    # Helper functions
    ghz_to_rads, mhz_to_rads,
    rads_to_ghz, rads_to_mhz,
    mt_to_tesla,
)

__all__ = [
    # Spin-1 Operators
    "spin1_operators", "Spin1Ops",
    "extend_to_18x18", "extend_electron_only",
    # Spin-1/2 Nuclear Operators
    "spin_half_operators", "SpinHalfOps",
    "extend_9x9_to_18x18", "extend_9x9_to_18x18_manifold",
    "build_spin_spin_coupling", "dipolar_tensor",
    # Units
    "GHZ", "MHZ", "KHZ", "HZ",
    "MT", "UT", "GAUSS",
    "UM", "NM", "ANGSTROM",
    # Constants
    "HBAR", "MU_B", "MU_N", "KB",
    "G_E", "GAMMA_E",
    "G_N14", "GAMMA_N14",
    "G_C13", "GAMMA_C13",
    "D_GS", "D_ES",
    "A_PARALLEL_N14", "A_PERP_N14", "P_N14",
    # Functions
    "ghz_to_rads", "mhz_to_rads",
    "rads_to_ghz", "rads_to_mhz",
    "mt_to_tesla",
]
