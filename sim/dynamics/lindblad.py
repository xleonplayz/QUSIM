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
Lindblad master equation solver for NV center dynamics.

Solves the Lindblad master equation for open quantum systems:

    dρ/dt = -i[H(t), ρ] + Σ_k D[L_k](ρ)

where the Lindblad dissipator is:

    D[L](ρ) = L ρ L† - ½{L†L, ρ}

The solver uses scipy.integrate.solve_ivp with adaptive
step size control (RK45 method by default).

Vectorization
-------------
For numerical integration, the density matrix is vectorized:

    ρ (18×18) → ρ_vec (324)

The Lindblad equation becomes:

    dρ_vec/dt = L ρ_vec

where L is the Liouvillian superoperator (324×324).

For time-dependent Hamiltonians, L is recomputed at each time step.
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable

from ..hamiltonian.builder import HamiltonianBuilder
from .dissipation import (
    t1_operators,
    t2_dephasing_operator,
    optical_decay_operators,
)


@dataclass
class EvolutionResult:
    """
    Container for time evolution results.

    Attributes
    ----------
    times : np.ndarray
        Array of time points in seconds
    rho_t : list of np.ndarray
        List of density matrices at each time point
    success : bool
        Whether the integration succeeded

    Methods
    -------
    expectation(operator)
        Compute ⟨O⟩(t) for all times
    population(projector)
        Compute population for all times
    final_state()
        Return the final density matrix
    """

    times: np.ndarray
    rho_t: List[np.ndarray]
    success: bool = True

    def expectation(self, operator: np.ndarray) -> np.ndarray:
        """
        Compute expectation value ⟨O⟩(t) for all times.

        Parameters
        ----------
        operator : np.ndarray
            18×18 operator matrix

        Returns
        -------
        np.ndarray
            Array of expectation values (complex)
        """
        return np.array([np.trace(operator @ rho) for rho in self.rho_t])

    def population(self, projector: np.ndarray) -> np.ndarray:
        """
        Compute population Tr(P ρ) for all times.

        Parameters
        ----------
        projector : np.ndarray
            18×18 projector matrix

        Returns
        -------
        np.ndarray
            Array of populations (real, in [0,1])
        """
        return np.real(self.expectation(projector))

    def final_state(self) -> np.ndarray:
        """Return the final density matrix."""
        return self.rho_t[-1]

    def purity(self) -> np.ndarray:
        """Compute purity Tr(ρ²) for all times."""
        return np.array([np.real(np.trace(rho @ rho)) for rho in self.rho_t])


class LindbladSolver:
    """
    Lindblad master equation solver.

    Parameters
    ----------
    hamiltonian : HamiltonianBuilder
        The system Hamiltonian (can be time-dependent)

    Attributes
    ----------
    hamiltonian : HamiltonianBuilder
        The Hamiltonian builder
    dissipators : list
        List of Lindblad operators

    Examples
    --------
    >>> from sim import HamiltonianBuilder
    >>> from sim.hamiltonian.terms import ZFS, Zeeman, MicrowaveDrive
    >>> from sim.dynamics import LindbladSolver
    >>> from sim.states import ground_state
    >>>
    >>> # Build time-dependent Hamiltonian
    >>> H = HamiltonianBuilder()
    >>> H.add(ZFS(D=2.87))
    >>> H.add(Zeeman(B=10))
    >>> H.add(MicrowaveDrive(omega=10))  # 10 MHz Rabi
    >>>
    >>> # Setup solver with dissipation
    >>> solver = LindbladSolver(H)
    >>> solver.add_t1_relaxation(gamma=1e3)    # T1 = 1 ms
    >>> solver.add_t2_dephasing(gamma=1e6)     # T2* = 1 μs
    >>>
    >>> # Evolve from ground state
    >>> rho0 = ground_state()
    >>> result = solver.evolve(rho0, t_span=(0, 1e-6), n_steps=100)
    >>>
    >>> # Analyze
    >>> from sim.states import projector_ms
    >>> pop_ms0 = result.population(projector_ms(0))
    >>> print(f"Final ms=0 population: {pop_ms0[-1]:.3f}")
    """

    DIM = 18

    def __init__(self, hamiltonian: HamiltonianBuilder):
        self.hamiltonian = hamiltonian
        self._dissipators: List[np.ndarray] = []

    @property
    def dissipators(self) -> List[np.ndarray]:
        """List of Lindblad operators."""
        return self._dissipators.copy()

    def add_dissipator(self, L: np.ndarray):
        """
        Add a Lindblad operator.

        Parameters
        ----------
        L : np.ndarray
            18×18 Lindblad operator (already includes √γ)
        """
        self._dissipators.append(L)

    def add_t1_relaxation(self, gamma: float, manifold: str = "ground"):
        """
        Add T1 spin-lattice relaxation.

        Parameters
        ----------
        gamma : float
            Relaxation rate 1/T1 in Hz
        manifold : str
            "ground", "excited", or "both"
        """
        ops = t1_operators(gamma, manifold)
        for L in ops:
            self._dissipators.append(L)

    def add_t2_dephasing(self, gamma: float, manifold: str = "ground"):
        """
        Add T2* pure dephasing.

        Parameters
        ----------
        gamma : float
            Dephasing rate 1/T2* in Hz
        manifold : str
            "ground", "excited", or "both"
        """
        L = t2_dephasing_operator(gamma, manifold)
        self._dissipators.append(L)

    def add_optical_decay(self, gamma: float):
        """
        Add optical spontaneous emission.

        Parameters
        ----------
        gamma : float
            Emission rate in Hz (~8×10⁷ for NV)
        """
        ops = optical_decay_operators(gamma)
        for L in ops:
            self._dissipators.append(L)

    def clear_dissipators(self):
        """Remove all dissipation channels."""
        self._dissipators.clear()

    def _lindblad_rhs(self, t: float, rho_vec: np.ndarray) -> np.ndarray:
        """
        Right-hand side of Lindblad equation for ODE solver.

        Parameters
        ----------
        t : float
            Current time
        rho_vec : np.ndarray
            Flattened density matrix (324 elements)

        Returns
        -------
        np.ndarray
            d(rho_vec)/dt
        """
        # Reshape to matrix
        rho = rho_vec.reshape((self.DIM, self.DIM))

        # Hamiltonian term: -i[H, ρ]
        H = self.hamiltonian.build(t)
        drho = -1j * (H @ rho - rho @ H)

        # Dissipation terms: Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
        for L in self._dissipators:
            L_dag = L.conj().T
            L_dag_L = L_dag @ L

            drho += (
                L @ rho @ L_dag -
                0.5 * (L_dag_L @ rho + rho @ L_dag_L)
            )

        return drho.flatten()

    def evolve(
        self,
        rho0: np.ndarray,
        t_span: Tuple[float, float],
        n_steps: int = 100,
        method: str = "RK45",
        rtol: float = 1e-8,
        atol: float = 1e-10
    ) -> EvolutionResult:
        """
        Evolve the density matrix in time.

        Parameters
        ----------
        rho0 : np.ndarray
            Initial density matrix (18×18)
        t_span : tuple
            (t_start, t_end) in seconds
        n_steps : int
            Number of output time points. Default: 100
        method : str
            ODE solver method. Default: "RK45"
        rtol : float
            Relative tolerance. Default: 1e-8
        atol : float
            Absolute tolerance. Default: 1e-10

        Returns
        -------
        EvolutionResult
            Container with times and density matrices
        """
        # Time points for output
        t_eval = np.linspace(t_span[0], t_span[1], n_steps)

        # Initial condition
        rho0_vec = rho0.flatten()

        # Solve ODE
        sol = solve_ivp(
            self._lindblad_rhs,
            t_span,
            rho0_vec,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            vectorized=False
        )

        # Extract results
        times = sol.t
        rho_t = [sol.y[:, i].reshape((self.DIM, self.DIM)) for i in range(len(times))]

        return EvolutionResult(
            times=times,
            rho_t=rho_t,
            success=sol.success
        )

    def steady_state(
        self,
        t_max: float = 1e-3,
        tol: float = 1e-10
    ) -> Optional[np.ndarray]:
        """
        Find the steady state by long-time evolution.

        Parameters
        ----------
        t_max : float
            Maximum evolution time in seconds
        tol : float
            Convergence tolerance

        Returns
        -------
        np.ndarray or None
            Steady state density matrix, or None if not converged
        """
        # Start from maximally mixed state
        rho0 = np.eye(self.DIM, dtype=np.complex128) / self.DIM

        # Evolve in steps
        t_current = 0
        dt = t_max / 10

        while t_current < t_max:
            result = self.evolve(rho0, (0, dt), n_steps=10)
            rho_new = result.final_state()

            # Check convergence
            diff = np.linalg.norm(rho_new - rho0)
            if diff < tol:
                return rho_new

            rho0 = rho_new
            t_current += dt

        return None  # Not converged

    def __repr__(self) -> str:
        n_diss = len(self._dissipators)
        td = "time-dependent" if self.hamiltonian.is_time_dependent else "static"
        return f"LindbladSolver({td} H, {n_diss} dissipators)"
