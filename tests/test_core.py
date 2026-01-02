# ==============================================================================
#  QUSIM - Core Tests
# ==============================================================================
"""Basic tests for QUSIM core functionality."""

import numpy as np
import pytest


class TestImports:
    """Test that all modules can be imported."""

    def test_import_sim(self):
        from sim import HamiltonianBuilder
        assert HamiltonianBuilder is not None

    def test_import_terms(self):
        from sim.hamiltonian.terms import ZFS, Zeeman, MicrowaveDrive, OpticalCoupling
        assert ZFS is not None
        assert Zeeman is not None
        assert MicrowaveDrive is not None
        assert OpticalCoupling is not None

    def test_import_dynamics(self):
        from sim.dynamics import LindbladSolver
        assert LindbladSolver is not None

    def test_import_states(self):
        from sim.states import ground_state, projector_excited
        assert ground_state is not None
        assert projector_excited is not None


class TestOperators:
    """Test spin operators."""

    def test_spin1_operators(self):
        from sim.core.operators import spin1_operators
        S = spin1_operators()

        # Check dimensions
        assert S.Sx.shape == (3, 3)
        assert S.Sy.shape == (3, 3)
        assert S.Sz.shape == (3, 3)

    def test_sz_eigenvalues(self):
        from sim.core.operators import spin1_operators
        S = spin1_operators()

        eigenvalues = np.linalg.eigvalsh(S.Sz)
        expected = np.array([-1, 0, 1])
        np.testing.assert_allclose(sorted(eigenvalues), expected)


class TestHamiltonian:
    """Test Hamiltonian building."""

    def test_zfs_shape(self):
        from sim import HamiltonianBuilder
        from sim.hamiltonian.terms import ZFS

        H = HamiltonianBuilder()
        H.add(ZFS(D=2.87))

        H_mat = H.build(t=0)
        assert H_mat.shape == (18, 18)

    def test_zfs_hermitian(self):
        from sim import HamiltonianBuilder
        from sim.hamiltonian.terms import ZFS

        H = HamiltonianBuilder()
        H.add(ZFS(D=2.87))

        H_mat = H.build(t=0)
        np.testing.assert_allclose(H_mat, H_mat.conj().T, atol=1e-12)


class TestStates:
    """Test initial states."""

    def test_ground_state_shape(self):
        from sim.states import ground_state

        rho = ground_state(ms=0, mI=0)
        assert rho.shape == (18, 18)

    def test_ground_state_trace(self):
        from sim.states import ground_state

        rho = ground_state(ms=0, mI=0)
        assert np.isclose(np.trace(rho), 1.0)

    def test_ground_state_positive(self):
        from sim.states import ground_state

        rho = ground_state(ms=0, mI=0)
        eigenvalues = np.linalg.eigvalsh(rho)
        assert all(eigenvalues >= -1e-12)


class TestLindblad:
    """Test Lindblad solver."""

    def test_evolution_preserves_trace(self):
        from sim import HamiltonianBuilder
        from sim.hamiltonian.terms import ZFS
        from sim.dynamics import LindbladSolver
        from sim.states import ground_state

        H = HamiltonianBuilder()
        H.add(ZFS(D=2.87))

        solver = LindbladSolver(H)
        rho0 = ground_state(ms=0, mI=0)

        result = solver.evolve(rho0, t_span=(0, 1e-9), n_steps=10)

        for rho in result.rho_t:
            assert np.isclose(np.trace(rho), 1.0, atol=1e-6)
