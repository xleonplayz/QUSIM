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
Physical constants and unit conversions for NV center simulation.

Unit Convention
---------------
This library uses **practical laboratory units** for the API:
    - Frequencies: GHz, MHz, kHz (not Hz)
    - Magnetic fields: mT (not Tesla)
    - Lengths: µm, nm (not meters)

Internally, all values are converted to **rad/s** for the Hamiltonian.

Why rad/s?
----------
The Hamiltonian H has units of energy. In quantum mechanics we work
with ℏ=1, so that energy = frequency × 2π.

    E = ℏω = ω  (for ℏ=1)

Therefore, the Hamiltonian matrix elements are in rad/s.

Example
-------
>>> from sim.core.constants import GHZ, MHZ, MT
>>> D = 2.87 * GHZ  # ZFS in rad/s
>>> B = 10 * MT     # 10 mT in Tesla
"""

import numpy as np

# =============================================================================
# UNIT CONVERSIONS
# =============================================================================

# Frequency -> rad/s
# ------------------
# f [Hz] -> ω [rad/s] = 2π * f

GHZ = 2.0 * np.pi * 1e9   # 1 GHz in rad/s  (≈ 6.28e9)
"""1 GHz in rad/s. Usage: D = 2.87 * GHZ"""

MHZ = 2.0 * np.pi * 1e6   # 1 MHz in rad/s  (≈ 6.28e6)
"""1 MHz in rad/s. Usage: A = 2.14 * MHZ"""

KHZ = 2.0 * np.pi * 1e3   # 1 kHz in rad/s  (≈ 6.28e3)
"""1 kHz in rad/s."""

HZ = 2.0 * np.pi          # 1 Hz in rad/s   (≈ 6.28)
"""1 Hz in rad/s."""


# Magnetic field -> Tesla
# -----------------------
MT = 1e-3    # 1 mT = 0.001 T
"""1 millitesla in Tesla. Usage: B = 10 * MT"""

UT = 1e-6    # 1 µT = 0.000001 T
"""1 microtesla in Tesla."""

GAUSS = 1e-4  # 1 Gauss = 0.0001 T
"""1 Gauss in Tesla (CGS unit, sometimes still used)."""


# Length -> meters
# ----------------
UM = 1e-6    # 1 µm = 0.000001 m
"""1 micrometer in meters."""

NM = 1e-9    # 1 nm = 0.000000001 m
"""1 nanometer in meters."""

ANGSTROM = 1e-10  # 1 Å = 0.1 nm
"""1 Ångström in meters."""


# =============================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS
# =============================================================================

HBAR = 1.054571817e-34  # J·s
"""Reduced Planck constant ℏ = h/(2π) in J·s."""

H_PLANCK = 6.62607015e-34  # J·s
"""Planck constant h in J·s."""

MU_B = 9.274010078e-24  # J/T
"""Bohr magneton μ_B = eℏ/(2m_e) in J/T."""

MU_N = 5.050783746e-27  # J/T
"""Nuclear magneton μ_N = eℏ/(2m_p) in J/T."""

KB = 1.380649e-23  # J/K
"""Boltzmann constant k_B in J/K."""

E_CHARGE = 1.602176634e-19  # C
"""Elementary charge e in Coulomb."""


# =============================================================================
# NV CENTER SPECIFIC CONSTANTS
# =============================================================================

# g-factors
# ---------
G_E = 2.0028  # dimensionless
"""Electron g-factor for NV center (close to free electron g≈2.002)."""

G_N14 = 0.4038  # dimensionless
"""N14 nuclear g-factor."""

G_C13 = 1.4048  # dimensionless
"""C13 nuclear g-factor."""


# Gyromagnetic ratios (rad/s per Tesla)
# -------------------------------------
# γ = g * μ / ℏ

GAMMA_E = G_E * MU_B / HBAR  # ≈ 1.761e11 rad/(s·T)
"""
Electron gyromagnetic ratio γ_e in rad/(s·T).

For a magnetic field B, the Larmor frequency is:
    ω_L = γ_e * B

At B = 1 mT:
    ω_L = 1.761e11 * 1e-3 = 1.761e8 rad/s ≈ 28 MHz
"""

GAMMA_N14 = G_N14 * MU_N / HBAR  # ≈ 1.93e7 rad/(s·T)
"""N14 nuclear gyromagnetic ratio in rad/(s·T)."""

GAMMA_C13 = G_C13 * MU_N / HBAR  # ≈ 6.73e7 rad/(s·T)
"""C13 nuclear gyromagnetic ratio in rad/(s·T)."""


# Zero-Field Splitting (Typical values)
# -------------------------------------
D_GS = 2.87 * GHZ  # Ground state
"""
ZFS D-parameter in ground state: 2.87 GHz (in rad/s).

This is the energy difference between ms=0 and ms=±1.
"""

D_ES = 1.42 * GHZ  # Excited state
"""ZFS D-parameter in excited state: 1.42 GHz (in rad/s)."""


# Hyperfine couplings (Typical values for N14)
# --------------------------------------------
A_PARALLEL_N14 = -2.14 * MHZ  # A_|| (parallel to NV axis)
"""Parallel hyperfine coupling for N14: -2.14 MHz (in rad/s)."""

A_PERP_N14 = -2.70 * MHZ  # A_⊥ (perpendicular to NV axis)
"""Perpendicular hyperfine coupling for N14: -2.70 MHz (in rad/s)."""

P_N14 = -5.0 * MHZ  # Quadrupole parameter
"""N14 quadrupole parameter: -5.0 MHz (in rad/s)."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ghz_to_rads(freq_ghz: float) -> float:
    """
    Convert frequency from GHz to rad/s.

    Parameters
    ----------
    freq_ghz : float
        Frequency in GHz

    Returns
    -------
    float
        Frequency in rad/s

    Examples
    --------
    >>> ghz_to_rads(2.87)  # ZFS
    18032327871.53...
    """
    return freq_ghz * GHZ


def mhz_to_rads(freq_mhz: float) -> float:
    """
    Convert frequency from MHz to rad/s.

    Parameters
    ----------
    freq_mhz : float
        Frequency in MHz

    Returns
    -------
    float
        Frequency in rad/s
    """
    return freq_mhz * MHZ


def rads_to_ghz(omega: float) -> float:
    """
    Convert rad/s back to GHz.

    Parameters
    ----------
    omega : float
        Angular frequency in rad/s

    Returns
    -------
    float
        Frequency in GHz
    """
    return omega / GHZ


def rads_to_mhz(omega: float) -> float:
    """
    Convert rad/s back to MHz.

    Parameters
    ----------
    omega : float
        Angular frequency in rad/s

    Returns
    -------
    float
        Frequency in MHz
    """
    return omega / MHZ


def mt_to_tesla(field_mt: float) -> float:
    """
    Convert magnetic field from mT to Tesla.

    Parameters
    ----------
    field_mt : float
        Magnetic field in mT

    Returns
    -------
    float
        Magnetic field in Tesla
    """
    return field_mt * MT


def zeeman_splitting_mhz(B_mt: float, g: float = G_E) -> float:
    """
    Calculate the Zeeman splitting in MHz for a given B-field.

    The splitting between ms=+1 and ms=-1 is:
        Δf = 2 * γ * B / (2π) = 2 * g * μ_B * B / h

    Parameters
    ----------
    B_mt : float
        Magnetic field in mT
    g : float
        g-factor (default: 2.0028 for NV)

    Returns
    -------
    float
        Splitting in MHz

    Examples
    --------
    >>> zeeman_splitting_mhz(10)  # 10 mT
    560.6...  # approximately 56 MHz/mT * 10 mT
    """
    B_tesla = B_mt * MT
    gamma = g * MU_B / HBAR
    splitting_rads = 2 * gamma * B_tesla
    return splitting_rads / MHZ
