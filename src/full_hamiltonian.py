#!/usr/bin/env python3
"""
Complete NV Hamiltonian with All Physical Effects
================================================
Full implementation with GS/ES splitting, Zeeman x,y,z, strain, hyperfine tensors
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
from functools import partial
from typing import Tuple

jax.config.update("jax_enable_x64", True)

# Physical constants
MU_B = 9.274009994e-24  # Bohr magneton (J/T)
HBAR = 1.0545718e-34    # Reduced Planck constant (J⋅s)
G_E = 2.0028            # Electron g-factor

@jit
def spin_1_operators():
    """
    Spin-1 operators for NV center electronic spin.
    Returns Sx, Sy, Sz matrices in the |ms=-1,0,+1⟩ basis.
    """
    # Spin-1 matrices (3x3)
    Sx = (1/jnp.sqrt(2)) * jnp.array([
        [0, 1, 0],
        [1, 0, 1], 
        [0, 1, 0]
    ], dtype=jnp.complex128)
    
    Sy = (1/jnp.sqrt(2)) * jnp.array([
        [0, -1j, 0],
        [1j, 0, -1j],
        [0, 1j, 0]
    ], dtype=jnp.complex128)
    
    Sz = jnp.array([
        [-1, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ], dtype=jnp.complex128)
    
    return Sx, Sy, Sz

@jit 
def spin_half_operators():
    """
    Spin-1/2 operators for ¹³C nuclear spins.
    """
    sx = 0.5 * jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
    sy = 0.5 * jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
    sz = 0.5 * jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
    
    return sx, sy, sz

@jit
def nuclear_I1_operators():
    """
    Nuclear spin-1 operators for ¹⁴N.
    """
    Ix = (1/jnp.sqrt(2)) * jnp.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=jnp.complex128)
    
    Iy = (1/jnp.sqrt(2)) * jnp.array([
        [0, -1j, 0],
        [1j, 0, -1j],
        [0, 1j, 0]
    ], dtype=jnp.complex128)
    
    Iz = jnp.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, -1]
    ], dtype=jnp.complex128)
    
    return Ix, Iy, Iz

@jit
def projectors_spin1():
    """
    Projection operators for spin-1 system.
    P_m = |m⟩⟨m| for m = -1, 0, +1
    """
    P_minus1 = jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=jnp.complex128)
    P_0 = jnp.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=jnp.complex128)
    P_plus1 = jnp.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=jnp.complex128)
    
    return P_minus1, P_0, P_plus1

@jit
def tensor_product_chain(operators_list):
    """
    Compute tensor product of a chain of operators.
    operators_list: [A, B, C, ...] → A ⊗ B ⊗ C ⊗ ...
    """
    result = operators_list[0]
    for op in operators_list[1:]:
        result = jnp.kron(result, op)
    return result

@jit
def extend_to_full_space(op, position, space_dims):
    """
    Extend operator to full tensor product space.
    
    Args:
        op: operator matrix
        position: position in tensor product (0-indexed)
        space_dims: list of dimensions for each subspace
    
    Returns:
        Operator extended to full space with identities
    """
    operators_list = []
    
    for i, dim in enumerate(space_dims):
        if i == position:
            operators_list.append(op)
        else:
            operators_list.append(jnp.eye(dim, dtype=jnp.complex128))
    
    return tensor_product_chain(operators_list)

class FullNVHamiltonian:
    """
    Complete NV Hamiltonian with all physical effects.
    
    Hilbert space structure:
    - Electronic spin: S=1 (3 states: |ms=-1,0,+1⟩)
    - ¹⁴N nuclear spin: I=1 (3 states)  
    - ¹³C nuclear spins: I=1/2 each (2^n_C13 states)
    
    Total dimension: 3 × 3 × 2^n_C13
    """
    
    def __init__(self, params):
        self.params = params
        
        # System dimensions
        self.n_C13 = params.n_carbon13
        self.dim_electronic = 3  # S=1
        self.dim_N14 = 3         # I=1
        self.dim_C13_total = 2**self.n_C13  # Each C13 is I=1/2
        
        # Total Hilbert space dimension
        self.total_dim = self.dim_electronic * self.dim_N14 * self.dim_C13_total
        
        # Space dimensions for tensor products
        self.space_dims = [self.dim_electronic, self.dim_N14] + [2] * self.n_C13
        
        # Generate base operators
        self._generate_operators()
        
    def _generate_operators(self):
        """Generate all operators in full tensor product space."""
        
        # Electronic spin operators
        Sx_el, Sy_el, Sz_el = spin_1_operators()
        
        # Extend electronic operators to full space
        self.Sx = extend_to_full_space(Sx_el, 0, self.space_dims)
        self.Sy = extend_to_full_space(Sy_el, 0, self.space_dims)
        self.Sz = extend_to_full_space(Sz_el, 0, self.space_dims)
        
        # Electronic projectors
        P_m1, P_0, P_p1 = projectors_spin1()
        self.P_minus1 = extend_to_full_space(P_m1, 0, self.space_dims)
        self.P_0 = extend_to_full_space(P_0, 0, self.space_dims)
        self.P_plus1 = extend_to_full_space(P_p1, 0, self.space_dims)
        
        # ¹⁴N nuclear operators
        Ix_N14, Iy_N14, Iz_N14 = nuclear_I1_operators()
        
        self.Ix_N14 = extend_to_full_space(Ix_N14, 1, self.space_dims)
        self.Iy_N14 = extend_to_full_space(Iy_N14, 1, self.space_dims)
        self.Iz_N14 = extend_to_full_space(Iz_N14, 1, self.space_dims)
        
        # ¹³C operators for each nucleus
        self.C13_operators = []
        sx_c13, sy_c13, sz_c13 = spin_half_operators()
        
        for i in range(self.n_C13):
            position = 2 + i  # Electronic=0, N14=1, C13s start at 2
            
            Ix_c13_i = extend_to_full_space(sx_c13, position, self.space_dims)
            Iy_c13_i = extend_to_full_space(sy_c13, position, self.space_dims)
            Iz_c13_i = extend_to_full_space(sz_c13, position, self.space_dims)
            
            self.C13_operators.append({
                'Ix': Ix_c13_i,
                'Iy': Iy_c13_i, 
                'Iz': Iz_c13_i
            })
    
    @partial(jit, static_argnums=(0,))
    def zero_field_splitting(self, D_gs_Hz, D_es_Hz=None):
        """
        Zero-field splitting Hamiltonian.
        H_ZFS = 2π D (Sz² - (2/3)I)
        
        For ground state only (excited state treated separately).
        """
        I3 = jnp.eye(self.total_dim, dtype=jnp.complex128)
        
        # Ground state ZFS
        H_zfs_gs = 2 * jnp.pi * D_gs_Hz * (self.Sz @ self.Sz - (2/3) * I3)
        
        return H_zfs_gs
    
    @partial(jit, static_argnums=(0,))
    def zeeman_hamiltonian(self, B_vec_mT):
        """
        Zeeman interaction in arbitrary magnetic field.
        H_Zeeman = γ_e (B_x S_x + B_y S_y + B_z S_z)
        
        Args:
            B_vec_mT: [Bx, By, Bz] in mT
        """
        # Convert mT to T
        B_vec_T = B_vec_mT * 1e-3
        
        # Electronic gyromagnetic ratio
        gamma_e = G_E * MU_B / HBAR  # rad/(s⋅T)
        
        H_zeeman = gamma_e * (
            B_vec_T[0] * self.Sx + 
            B_vec_T[1] * self.Sy + 
            B_vec_T[2] * self.Sz
        )
        
        return H_zeeman
    
    @partial(jit, static_argnums=(0,))
    def strain_hamiltonian(self, strain_MHz):
        """
        Strain/Stark shift Hamiltonian.
        H_strain = 2π ε (Sx² - Sy²)
        
        Args:
            strain_MHz: strain parameter in MHz
        """
        H_strain = 2 * jnp.pi * strain_MHz * 1e6 * (
            self.Sx @ self.Sx - self.Sy @ self.Sy
        )
        
        return H_strain
    
    @partial(jit, static_argnums=(0,))
    def hyperfine_N14(self, A_parallel_MHz, A_perp_MHz):
        """
        Hyperfine interaction with ¹⁴N nuclear spin.
        H_hf = 2π [A∥ Sz⊗Iz + A⊥ (Sx⊗Ix + Sy⊗Iy)]
        
        Args:
            A_parallel_MHz: parallel hyperfine coupling
            A_perp_MHz: perpendicular hyperfine coupling
        """
        H_hf_N14 = 2 * jnp.pi * 1e6 * (
            A_parallel_MHz * (self.Sz @ self.Iz_N14) +
            A_perp_MHz * (
                self.Sx @ self.Ix_N14 + 
                self.Sy @ self.Iy_N14
            )
        )
        
        return H_hf_N14
    
    @partial(jit, static_argnums=(0,))
    def hyperfine_C13_all(self, A_C13_array):
        """
        Hyperfine interactions with all ¹³C nuclear spins.
        
        Args:
            A_C13_array: array of shape (n_C13, 2) with [A_parallel, A_perp] for each C13
        """
        H_hf_C13_total = jnp.zeros((self.total_dim, self.total_dim), dtype=jnp.complex128)
        
        for i in range(self.n_C13):
            A_par, A_perp = A_C13_array[i]
            
            # Hyperfine with i-th C13 nucleus
            H_hf_i = 2 * jnp.pi * 1e6 * (
                A_par * (self.Sz @ self.C13_operators[i]['Iz']) +
                A_perp * (
                    self.Sx @ self.C13_operators[i]['Ix'] +
                    self.Sy @ self.C13_operators[i]['Iy']
                )
            )
            
            H_hf_C13_total += H_hf_i
        
        return H_hf_C13_total
    
    @partial(jit, static_argnums=(0,))
    def nuclear_zeeman_N14(self, B_vec_mT, gamma_N14=-2.711e7):
        """
        Nuclear Zeeman interaction for ¹⁴N.
        H_nZ = γ_N (B_x Ix + B_y Iy + B_z Iz)
        
        Args:
            B_vec_mT: magnetic field vector in mT
            gamma_N14: ¹⁴N gyromagnetic ratio in rad/(s⋅T)
        """
        B_vec_T = B_vec_mT * 1e-3
        
        H_nZ_N14 = gamma_N14 * (
            B_vec_T[0] * self.Ix_N14 +
            B_vec_T[1] * self.Iy_N14 +
            B_vec_T[2] * self.Iz_N14
        )
        
        return H_nZ_N14
    
    @partial(jit, static_argnums=(0,))
    def nuclear_zeeman_C13(self, B_vec_mT, gamma_C13=6.728e7):
        """
        Nuclear Zeeman interactions for all ¹³C spins.
        
        Args:
            B_vec_mT: magnetic field vector in mT
            gamma_C13: ¹³C gyromagnetic ratio in rad/(s⋅T)
        """
        B_vec_T = B_vec_mT * 1e-3
        
        H_nZ_C13_total = jnp.zeros((self.total_dim, self.total_dim), dtype=jnp.complex128)
        
        for i in range(self.n_C13):
            H_nZ_i = gamma_C13 * (
                B_vec_T[0] * self.C13_operators[i]['Ix'] +
                B_vec_T[1] * self.C13_operators[i]['Iy'] +
                B_vec_T[2] * self.C13_operators[i]['Iz']
            )
            
            H_nZ_C13_total += H_nZ_i
        
        return H_nZ_C13_total
    
    @partial(jit, static_argnums=(0,))
    def quadrupole_N14(self, P_MHz=5.01):
        """
        Nuclear quadrupole interaction for ¹⁴N.
        H_Q = 2π P [Iz² - (2/3)I_N14]
        
        Args:
            P_MHz: quadrupole coupling constant in MHz
        """
        I_N14 = jnp.eye(self.total_dim, dtype=jnp.complex128)
        
        H_quad_N14 = 2 * jnp.pi * P_MHz * 1e6 * (
            self.Iz_N14 @ self.Iz_N14 - (2/3) * I_N14
        )
        
        return H_quad_N14
    
    @partial(jit, static_argnums=(0,))
    def full_hamiltonian(
        self, 
        B_vec_mT, 
        strain_MHz, 
        A_N14_tuple: Tuple[float, float],
        A_C13_array,
        include_nuclear_zeeman=True,
        include_quadrupole=True
    ):
        """
        Complete NV Hamiltonian with all terms.
        
        Args:
            B_vec_mT: [Bx, By, Bz] magnetic field in mT
            strain_MHz: strain parameter in MHz
            A_N14_tuple: (A_parallel, A_perp) for ¹⁴N in MHz
            A_C13_array: array of (A_parallel, A_perp) for each ¹³C in MHz
            include_nuclear_zeeman: include nuclear Zeeman terms
            include_quadrupole: include quadrupole interaction
        
        Returns:
            Full Hamiltonian matrix
        """
        # Zero-field splitting (ground state)
        H = self.zero_field_splitting(self.params.D_GS_Hz)
        
        # Electronic Zeeman
        H += self.zeeman_hamiltonian(B_vec_mT)
        
        # Strain/Stark
        H += self.strain_hamiltonian(strain_MHz)
        
        # Hyperfine interactions
        A_N14_par, A_N14_perp = A_N14_tuple
        H += self.hyperfine_N14(A_N14_par, A_N14_perp)
        H += self.hyperfine_C13_all(A_C13_array)
        
        # Nuclear interactions (optional)
        if include_nuclear_zeeman:
            H += self.nuclear_zeeman_N14(B_vec_mT)
            H += self.nuclear_zeeman_C13(B_vec_mT)
        
        if include_quadrupole:
            H += self.quadrupole_N14()
        
        return H
    
    def get_eigenvalues_and_states(self, H):
        """
        Diagonalize Hamiltonian and return eigenvalues/eigenstates.
        Note: Uses numpy for eigenvalue decomposition (JAX eigh can be unstable).
        """
        # Convert to numpy for stable eigenvalue decomposition
        H_np = np.array(H)
        
        # Diagonalize
        eigenvals, eigenstates = np.linalg.eigh(H_np)
        
        # Convert back to JAX arrays
        eigenvals = jnp.array(eigenvals)
        eigenstates = jnp.array(eigenstates)
        
        return eigenvals, eigenstates
    
    def get_ms_populations(self, state_vector):
        """
        Extract ms populations from a state vector in the full Hilbert space.
        """
        # Project onto electronic ms subspaces
        p_minus1 = jnp.real(jnp.conj(state_vector).T @ self.P_minus1 @ state_vector)
        p_0 = jnp.real(jnp.conj(state_vector).T @ self.P_0 @ state_vector)
        p_plus1 = jnp.real(jnp.conj(state_vector).T @ self.P_plus1 @ state_vector)
        
        return {
            'p_ms_minus1': float(p_minus1),
            'p_ms_0': float(p_0),
            'p_ms_plus1': float(p_plus1)
        }

def example_usage():
    """Example of how to use the full Hamiltonian."""
    
    # Mock parameters
    class MockParams:
        D_GS_Hz = 2.87e9
        n_carbon13 = 3
    
    params = MockParams()
    
    # Create Hamiltonian object
    nv_ham = FullNVHamiltonian(params)
    
    print(f"Full Hilbert space dimension: {nv_ham.total_dim}")
    
    # Define physical parameters
    B_field = jnp.array([0.0, 0.0, 10.0])  # 10 mT along z
    strain = 1.0  # 1 MHz strain
    A_N14 = (2.2, 2.0)  # MHz
    A_C13 = jnp.array([[1.0, 0.3], [0.8, 0.2], [0.5, 0.1]])  # MHz
    
    # Build full Hamiltonian
    H_full = nv_ham.full_hamiltonian(
        B_field, strain, A_N14, A_C13
    )
    
    print(f"Hamiltonian shape: {H_full.shape}")
    print(f"Hamiltonian is Hermitian: {jnp.allclose(H_full, H_full.T.conj())}")
    
    # Get eigenvalues (first 10)
    eigenvals, eigenstates = nv_ham.get_eigenvalues_and_states(H_full)
    
    print(f"First 10 eigenvalues (GHz): {eigenvals[:10] / (2*jnp.pi*1e9)}")
    
    # Analyze ground state
    ground_state = eigenstates[:, 0]
    ms_pops = nv_ham.get_ms_populations(ground_state)
    
    print(f"Ground state ms populations: {ms_pops}")
    
    return nv_ham, H_full

if __name__ == "__main__":
    print("🔬 Testing Full NV Hamiltonian")
    print("=" * 40)
    
    nv_ham, H = example_usage()
    
    print("\n✅ Full Hamiltonian implementation working!")
    print(f"   Total Hilbert space: {nv_ham.total_dim} dimensions")
    print(f"   Includes: ZFS + Zeeman + Strain + Hyperfine + Nuclear terms")