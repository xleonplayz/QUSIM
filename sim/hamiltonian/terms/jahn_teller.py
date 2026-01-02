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
Jahn-Teller coupling in the NV center excited state.

The excited state (³E) of the NV center is orbitally degenerate and
couples to vibrational modes through the dynamic Jahn-Teller effect.
This causes mixing between the orbital states and affects the
optical properties.

Physics
-------
The Jahn-Teller Hamiltonian describes electron-phonon coupling:

    H_JT = ω(a†a + 1/2) + λ(a + a†) · O_orbital

where:
- ω is the phonon mode frequency
- a, a† are phonon creation/annihilation operators
- λ is the electron-phonon coupling strength
- O_orbital is an orbital operator

For the E-symmetry modes coupling to the ³E state:

    H_JT = ωx(ax†ax + 1/2) + ωy(ay†ay + 1/2)
         + λx(ax + ax†)(Sx² - Sy²)
         + λy(ay + ay†){Sx, Sy}

Typical Values
--------------
- Phonon frequency: ω ~ 65 meV (~ 16 THz)
- Coupling strength: λ ~ 0.5 (dimensionless)
- Huang-Rhys factor: S ~ λ²/2

Simplified Model
----------------
For the 18D Hilbert space, we use an effective model that captures
the main effects without explicit phonon states. This is valid when:
- Temperature is low (kT << ℏω)
- Phonon dynamics are fast compared to spin dynamics

References
----------
[1] Fu et al., Phys. Rev. Lett. 103, 256404 (2009)
[2] Abtew et al., Phys. Rev. Lett. 107, 146403 (2011)
"""

import numpy as np
from typing import Optional

from .base import HamiltonianTerm
from ...core.operators import spin1_operators
from ...core.constants import MHZ


class JahnTeller(HamiltonianTerm):
    """
    Jahn-Teller vibronic coupling in excited state (simplified model).

    This is a simplified effective model that captures the main effects
    of Jahn-Teller coupling without explicit phonon states.

    Parameters
    ----------
    omega_x : float
        Phonon mode frequency (x-mode) in MHz. Default: 1000 (1 GHz)
    omega_y : float
        Phonon mode frequency (y-mode) in MHz. Default: 1200 (1.2 GHz)
    lambda_x : float
        Coupling strength (x-mode), dimensionless. Default: 0.05
    lambda_y : float
        Coupling strength (y-mode), dimensionless. Default: 0.04
    temperature : float
        Temperature in Kelvin. Default: 300 K
    name : str, optional
        Custom name for the term

    Attributes
    ----------
    omega_x, omega_y : float
        Phonon frequencies in MHz
    lambda_x, lambda_y : float
        Coupling strengths (dimensionless)
    temperature : float
        Temperature in Kelvin

    Examples
    --------
    >>> from sim import HamiltonianBuilder
    >>> from sim.hamiltonian.terms import ZFS, JahnTeller
    >>>
    >>> H = HamiltonianBuilder()
    >>> H.add(ZFS(D=2.87))
    >>> H.add(JahnTeller())  # Default parameters
    >>>
    >>> # Low temperature (cryogenic)
    >>> H.add(JahnTeller(temperature=4.0))

    Notes
    -----
    The Jahn-Teller effect is primarily important for:
    - Understanding the excited state fine structure
    - Calculating optical transition rates
    - Modeling temperature dependence

    At room temperature, the JT effect averages out due to fast
    phonon dynamics. At low temperature, it can cause strain-like
    splitting in the excited state.

    This simplified model only acts on the excited state manifold.
    """

    def __init__(
        self,
        omega_x: float = 1000.0,   # MHz (~1 GHz)
        omega_y: float = 1200.0,   # MHz
        lambda_x: float = 0.05,    # dimensionless
        lambda_y: float = 0.04,    # dimensionless
        temperature: float = 300.0,  # Kelvin
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.temperature = temperature

        # Cache for the static Hamiltonian
        self._cache: Optional[np.ndarray] = None

    @property
    def is_time_dependent(self) -> bool:
        """Jahn-Teller coupling is time-independent (effective model)."""
        return False

    def _thermal_occupation(self, omega_mhz: float) -> float:
        """
        Calculate thermal phonon occupation number.

        Parameters
        ----------
        omega_mhz : float
            Phonon frequency in MHz

        Returns
        -------
        float
            Average phonon occupation <n>
        """
        from ...core.constants import KB, HBAR

        if self.temperature <= 0:
            return 0.0

        # ω in rad/s
        omega = omega_mhz * MHZ

        # Bose-Einstein distribution: <n> = 1 / (exp(ℏω/kT) - 1)
        x = HBAR * omega / (KB * self.temperature)

        if x > 40:  # Avoid overflow
            return 0.0

        return 1.0 / (np.exp(x) - 1)

    def _build_excited_3x3(self) -> np.ndarray:
        """
        Build effective 3×3 JT Hamiltonian for excited state electron spin.

        Returns
        -------
        np.ndarray
            3×3 complex Hermitian matrix in rad/s
        """
        S = spin1_operators()

        # Effective coupling strength includes thermal averaging
        # <Q²> = ℏ/(2mω) × (2<n> + 1) ~ (2<n> + 1)
        n_x = self._thermal_occupation(self.omega_x)
        n_y = self._thermal_occupation(self.omega_y)

        # Effective strain-like terms from JT coupling
        # H_eff ~ λ² × ω × (Sx² - Sy²) for x-mode
        # H_eff ~ λ² × ω × {Sx, Sy} for y-mode

        # At low temperature: static JT distortion
        # At high temperature: dynamic averaging

        eff_x = self.lambda_x**2 * self.omega_x * MHZ * (2 * n_x + 1)
        eff_y = self.lambda_y**2 * self.omega_y * MHZ * (2 * n_y + 1)

        H = (
            eff_x * (S.Sx @ S.Sx - S.Sy @ S.Sy) +
            eff_y * (S.Sx @ S.Sy + S.Sy @ S.Sx)
        )

        return H

    def build(self, t: float = 0.0) -> np.ndarray:
        """
        Build the 18×18 Jahn-Teller Hamiltonian.

        Parameters
        ----------
        t : float
            Time in seconds (not used)

        Returns
        -------
        np.ndarray
            18×18 complex Hermitian matrix in rad/s

        Notes
        -----
        The JT effect only acts on the excited state manifold
        (indices 9-17). The ground state is not affected.
        """
        if self._cache is None:
            H_3x3 = self._build_excited_3x3()

            # Extend to 9×9 (electron spin ⊗ N14 nuclear spin)
            I3_nuclear = np.eye(3, dtype=np.complex128)
            H_9x9 = np.kron(H_3x3, I3_nuclear)

            # Embed in 18×18, only in excited state (indices 9-17)
            H_18x18 = np.zeros((18, 18), dtype=np.complex128)
            H_18x18[9:, 9:] = H_9x9

            self._cache = H_18x18

        return self._cache.copy()

    def huang_rhys_factor(self) -> tuple:
        """
        Calculate the Huang-Rhys factors for both modes.

        Returns
        -------
        tuple
            (S_x, S_y) Huang-Rhys factors (dimensionless)

        Notes
        -----
        The Huang-Rhys factor S = λ²/2 characterizes the strength
        of electron-phonon coupling. S ~ 0.5 corresponds to moderate
        coupling where the Stokes shift equals the phonon energy.
        """
        S_x = self.lambda_x**2 / 2
        S_y = self.lambda_y**2 / 2
        return (S_x, S_y)

    def __repr__(self) -> str:
        return f"JahnTeller(T={self.temperature} K, λ_x={self.lambda_x}, λ_y={self.lambda_y})"
