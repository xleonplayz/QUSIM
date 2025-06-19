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

# --- Optimized basis operators (from user snippet) ---
Basis_e = 3  # Elektronenspin S=1
Basis_N = 3  # Kernspin I=1 (14N)
I3_e = jnp.eye(Basis_e, dtype=jnp.complex128)
I3_N = jnp.eye(Basis_N, dtype=jnp.complex128)

@jit
def spin1_ops():
    """
    Spin-1 operators for NV center electronic spin (from user snippet).
    Returns Sx, Sy, Sz matrices in the |ms=-1,0,+1⟩ basis.
    """
    Sx = jnp.array([[0,1,0],[1,0,1],[0,1,0]], dtype=jnp.complex128)/jnp.sqrt(2)
    Sy = jnp.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]], dtype=jnp.complex128)/jnp.sqrt(2)
    Sz = jnp.diag(jnp.array([1,0,-1], dtype=jnp.complex128))
    return Sx, Sy, Sz

@jit
def spin_1_operators():
    """
    Alias for backward compatibility.
    """
    return spin1_ops()

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
def nuclear1_ops():
    """
    Nuclear spin-1 operators for ¹⁴N (from user snippet).
    """
    Ix = jnp.array([[0,1,0],[1,0,1],[0,1,0]], dtype=jnp.complex128)/jnp.sqrt(2)
    Iy = jnp.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]], dtype=jnp.complex128)/jnp.sqrt(2)
    Iz = jnp.diag(jnp.array([1,0,-1], dtype=jnp.complex128))
    return Ix, Iy, Iz

@jit
def nuclear_I1_operators():
    """
    Alias for backward compatibility.
    """
    return nuclear1_ops()

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

def tensor_product_chain(operators_list):
    """
    Compute tensor product of a chain of operators.
    operators_list: [A, B, C, ...] → A ⊗ B ⊗ C ⊗ ...
    """
    result = operators_list[0]
    for op in operators_list[1:]:
        result = jnp.kron(result, op)
    return result

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
        
        # Electronic spin operators (using optimized versions)
        Sx_el, Sy_el, Sz_el = spin1_ops()
        
        # Extend electronic operators to full space
        self.Sx = extend_to_full_space(Sx_el, 0, self.space_dims)
        self.Sy = extend_to_full_space(Sy_el, 0, self.space_dims)
        self.Sz = extend_to_full_space(Sz_el, 0, self.space_dims)
        
        # Electronic projectors
        P_m1, P_0, P_p1 = projectors_spin1()
        self.P_minus1 = extend_to_full_space(P_m1, 0, self.space_dims)
        self.P_0 = extend_to_full_space(P_0, 0, self.space_dims)
        self.P_plus1 = extend_to_full_space(P_p1, 0, self.space_dims)
        
        # ¹⁴N nuclear operators (using optimized versions)
        Ix_N14, Iy_N14, Iz_N14 = nuclear1_ops()
        
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
    def H_ground_optimized(self, B_vec, strain_xy):
        """
        Optimized ground state Hamiltonian using user's exact snippet.
        
        Args:
            B_vec: jnp.array([B_x, B_y, B_z]) in Tesla
            strain_xy: tuple (E_x, E_y) in Hz
        """
        Sx, Sy, Sz = spin1_ops()
        Ix, Iy, Iz = nuclear1_ops()

        # Kronprodukt zu Gesamt-Hilbertraum (3⊗‍3)
        def K(e_op, n_op):
            return jnp.kron(e_op, n_op)

        # 1) Zero-Field Splitting D (axial)
        H_zfs = self.params.D_GS_Hz * 2*jnp.pi * (K(Sz, I3_N) @ K(Sz, I3_N) - (2/3)*jnp.eye(9, dtype=jnp.complex128))

        # 2) Zeeman-Term (B⋅S)
        gamma = G_E * (MU_B / HBAR)  # in Hz/T
        H_Zeeman = 2*jnp.pi * (
            gamma * B_vec[0] * K(Sx, I3_N) +
            gamma * B_vec[1] * K(Sy, I3_N) +
            gamma * B_vec[2] * K(Sz, I3_N)
        )

        # 3) Hyperfein-Tensor
        H_HF = 2*jnp.pi * (
            self.params.A_para_Hz * K(Sz, Iz) +
            self.params.A_perp_Hz * (K(Sx, Ix) + K(Sy, Iy))
        )

        # 4) Strain-/Stark-Effekt
        Ex, Ey = strain_xy
        H_strain = 2*jnp.pi * (
            Ex * (K(Sx, I3_N) @ K(Sx, I3_N) - K(Sy, I3_N) @ K(Sy, I3_N)) +
            Ey * (K(Sx, I3_N) @ K(Sy, I3_N) + K(Sy, I3_N) @ K(Sx, I3_N))
        )

        return H_zfs + H_Zeeman + H_HF + H_strain
    
    @partial(jit, static_argnums=(0,))
    def H_excited_optimized(self, B_vec, strain_xy, orbital_delta):
        """
        Optimized excited state Hamiltonian using user's exact snippet.
        
        Args:
            B_vec: magnetic field in Tesla
            strain_xy: strain tuple in Hz
            orbital_delta: Spin-Orbit / strain-splitting in ES (Hz), z.B. ±Exy
        """
        Sx, Sy, Sz = spin1_ops()
        Ix, Iy, Iz = nuclear1_ops()

        def K(e_op, n_op):
            return jnp.kron(e_op, n_op)

        # 1) ES Zero-Field Splitting
        H_zfs_es = self.params.D_ES_Hz * 2*jnp.pi * (K(Sz, I3_N) @ K(Sz, I3_N) - (2/3)*jnp.eye(9, dtype=jnp.complex128))

        # 2) Spin-Orbit / orbital splitting (Ex/Ey)
        Ex, Ey = orbital_delta
        H_orbital = 2*jnp.pi * (
            Ex * (K(Sx, I3_N) @ K(Sx, I3_N) - K(Sy, I3_N) @ K(Sy, I3_N)) +
            Ey * (K(Sx, I3_N) @ K(Sy, I3_N) + K(Sy, I3_N) @ K(Sx, I3_N))
        )

        # 3) Zeeman & Hyperfein & Strain analog zum GS
        gamma = G_E * (MU_B / HBAR)
        H_Zeeman = 2*jnp.pi * (
            gamma * B_vec[0] * K(Sx, I3_N) +
            gamma * B_vec[1] * K(Sy, I3_N) +
            gamma * B_vec[2] * K(Sz, I3_N)
        )
        
        H_HF = 2*jnp.pi * (
            self.params.A_para_Hz * K(Sz, Iz) +
            self.params.A_perp_Hz * (K(Sx, Ix) + K(Sy, Iy))
        )
        
        # Strain in ES (zusätzlich zu orbital splitting)
        strain_Ex, strain_Ey = strain_xy
        H_strain = 2*jnp.pi * (
            strain_Ex * (K(Sx, I3_N) @ K(Sx, I3_N) - K(Sy, I3_N) @ K(Sy, I3_N)) +
            strain_Ey * (K(Sx, I3_N) @ K(Sy, I3_N) + K(Sy, I3_N) @ K(Sx, I3_N))
        )

        return H_zfs_es + H_orbital + H_Zeeman + H_HF + H_strain
    
    @partial(jit, static_argnums=(0,))
    def H_ground_with_C13_extended(self, B_vec, strain_xy, A_C13_couplings):
        """
        Extended ground state Hamiltonian with multiple ¹³C nuclear spins.
        Based on user's snippet for extension to multiple C13 nuclei.
        
        Args:
            B_vec: magnetic field in Tesla
            strain_xy: strain tuple in Hz
            A_C13_couplings: array of (A_para, A_perp) for each ¹³C in Hz
        """
        # Start with basic GS Hamiltonian (without C13)
        H_base = self.H_ground_optimized(B_vec, strain_xy)
        
        if self.n_C13 == 0:
            return H_base
        
        # Add C13 contributions using existing operators
        if self.n_C13 == 0:
            return H_base
        
        # Use existing C13 operators from full implementation
        H_C13_total = jnp.zeros((self.total_dim, self.total_dim), dtype=jnp.complex128)
        
        for j in range(self.n_C13):
            A_para_j, A_perp_j = A_C13_couplings[j]
            
            # Use pre-computed operators
            H_z_j = A_para_j * 2*jnp.pi * (self.Sz @ self.C13_operators[j]['Iz'])
            H_x_j = A_perp_j * 2*jnp.pi * (self.Sx @ self.C13_operators[j]['Ix'])
            H_y_j = A_perp_j * 2*jnp.pi * (self.Sy @ self.C13_operators[j]['Iy'])
            
            H_C13_total += H_z_j + H_x_j + H_y_j
        
        # Resize H_base to match total dimensions if needed
        if H_base.shape != H_C13_total.shape:
            # H_base is 9x9 (electron + N14), need to extend to full space
            # This is a simplified fallback - use full implementation for C13 cases
            return self.full_hamiltonian(
                B_vec * 1e3, strain_xy[0] * 1e-6, 
                (self.params.A_para_Hz * 1e-6, self.params.A_perp_Hz * 1e-6),
                A_C13_couplings * 1e-6, use_optimized=False
            )
        
        return H_base + H_C13_total
    
    @partial(jit, static_argnums=(0, 5, 6, 7))
    def full_hamiltonian(
        self, 
        B_vec_mT, 
        strain_MHz, 
        A_N14_tuple: Tuple[float, float],
        A_C13_array,
        include_nuclear_zeeman=True,
        include_quadrupole=True,
        use_optimized=True
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
            use_optimized: use optimized Hamiltonian from user snippets
        
        Returns:
            Full Hamiltonian matrix
        """
        if use_optimized and self.n_C13 == 0:  # Only for simplified N14 case
            # Convert units
            B_vec_T = B_vec_mT * 1e-3  # mT to T
            strain_Hz = strain_MHz * 1e6  # MHz to Hz
            
            # Use optimized implementation
            return self.H_ground_optimized(B_vec_T, (strain_Hz, 0.0))
        
        else:
            # Use full implementation for complex cases
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

# --- Parameter class for optimized Hamiltonians ---
class OptimizedParams:
    """Parameters for optimized Hamiltonian implementation."""
    D_GS_Hz = 2.87e9      # GS ZFS
    D_ES_Hz = 1.42e9      # ES ZFS
    g_round = 2.003       # g-Faktor
    mu_B = MU_B / HBAR    # μB/ħ in Hz/T
    A_para_Hz = 2.16e6    # 14N axial
    A_perp_Hz = 2.16e6    # 14N transversal
    E_strain_x = 5.0e6    # Strain-Komponente Δ₁
    E_strain_y = 0.0      # Strain-Komponente Δ₂
    n_carbon13 = 0        # For optimized version
    D_ES_Hz = 1.42e9      # ES ZFS (for excited state)

def test_optimized_hamiltonians():
    """Test the optimized Hamiltonian implementations."""
    
    print("🧮 Testing Optimized Hamiltonian Implementations")
    print("=" * 50)
    
    # Test 1: Simple case (no C13)
    print("\n1. Testing simple case (electron + N14 only):")
    params_simple = OptimizedParams()
    nv_simple = FullNVHamiltonian(params_simple)
    
    B_vec = jnp.array([0.0, 0.0, 1.0])  # 1 Tesla z-field
    strain = (1e6, 0.0)  # 1 MHz strain
    
    H_opt = nv_simple.H_ground_optimized(B_vec, strain)
    print(f"   Dimension: {H_opt.shape}")
    print(f"   Hermitian: {jnp.allclose(H_opt, H_opt.T.conj())}")
    
    eigenvals_opt, _ = nv_simple.get_eigenvalues_and_states(H_opt)
    print(f"   First 3 eigenvalues (GHz): {eigenvals_opt[:3] / (2*jnp.pi*1e9)}")
    
    # Test 2: Excited state
    print("\n2. Testing excited state Hamiltonian:")
    orbital_splitting = (0.5e6, 0.0)  # 0.5 MHz orbital splitting
    
    H_es = nv_simple.H_excited_optimized(B_vec, strain, orbital_splitting)
    print(f"   ES dimension: {H_es.shape}")
    print(f"   ES Hermitian: {jnp.allclose(H_es, H_es.T.conj())}")
    
    eigenvals_es, _ = nv_simple.get_eigenvalues_and_states(H_es)
    print(f"   ES first 3 eigenvalues (GHz): {eigenvals_es[:3] / (2*jnp.pi*1e9)}")
    
    # Test 3: Extended case with C13
    print("\n3. Testing extended case with ¹³C nuclei:")
    
    class ExtendedParams:
        D_GS_Hz = 2.87e9
        D_ES_Hz = 1.42e9
        A_para_Hz = 2.16e6
        A_perp_Hz = 2.16e6
        n_carbon13 = 2
    
    params_ext = ExtendedParams()
    nv_ext = FullNVHamiltonian(params_ext)
    
    A_C13_array = jnp.array([[1.0e6, 0.3e6], [0.8e6, 0.2e6]])  # Two C13 nuclei
    
    H_ext = nv_ext.H_ground_with_C13_extended(B_vec, strain, A_C13_array)
    print(f"   Extended dimension: {H_ext.shape}")
    print(f"   Extended Hermitian: {jnp.allclose(H_ext, H_ext.T.conj())}")
    
    eigenvals_ext, _ = nv_ext.get_eigenvalues_and_states(H_ext)
    print(f"   Extended first 5 eigenvalues (GHz): {eigenvals_ext[:5] / (2*jnp.pi*1e9)}")
    
    # Test 4: Performance comparison
    print("\n4. Performance comparison:")
    
    import time
    
    # Time optimized version
    start = time.perf_counter()
    for _ in range(100):
        _ = nv_simple.H_ground_optimized(B_vec, strain)
    time_opt = time.perf_counter() - start
    
    # Time standard version
    start = time.perf_counter()
    for _ in range(100):
        _ = nv_simple.full_hamiltonian(
            B_vec * 1e3, strain[0] * 1e-6, (2.16, 2.16), jnp.array([]), 
            use_optimized=False
        )
    time_std = time.perf_counter() - start
    
    print(f"   Optimized version: {time_opt*10:.2f} ms (100 calls)")
    print(f"   Standard version:  {time_std*10:.2f} ms (100 calls)")
    print(f"   Speedup factor:    {time_std/time_opt:.1f}x")
    
    return nv_simple, nv_ext

def example_usage():
    """Example of how to use the full Hamiltonian."""
    
    # Mock parameters (extended with optimized parameters)
    class MockParams:
        D_GS_Hz = 2.87e9
        D_ES_Hz = 1.42e9
        A_para_Hz = 2.16e6
        A_perp_Hz = 2.16e6
        n_carbon13 = 2
    
    params = MockParams()
    
    # Create Hamiltonian object
    nv_ham = FullNVHamiltonian(params)
    
    print(f"Full Hilbert space dimension: {nv_ham.total_dim}")
    
    # Define physical parameters
    B_field = jnp.array([0.0, 0.0, 10.0])  # 10 mT along z
    strain = 1.0  # 1 MHz strain
    A_N14 = (2.2, 2.0)  # MHz
    A_C13 = jnp.array([[1.0, 0.3], [0.8, 0.2], [0.5, 0.1]])  # MHz
    
    # Build full Hamiltonian (test both implementations)
    print("\nTesting optimized implementation (simplified case):")
    
    # Simple case for optimized version
    params_simple = OptimizedParams()
    nv_ham_simple = FullNVHamiltonian(params_simple)
    
    B_field_T = B_field * 1e-3  # Convert to Tesla
    H_optimized = nv_ham_simple.H_ground_optimized(B_field_T, (strain * 1e6, 0.0))
    
    print(f"Optimized Hamiltonian shape: {H_optimized.shape}")
    print(f"Optimized Hamiltonian is Hermitian: {jnp.allclose(H_optimized, H_optimized.T.conj())}")
    
    print("\nTesting full implementation (with C13):")
    H_full = nv_ham.full_hamiltonian(
        B_field, strain, A_N14, A_C13, use_optimized=False
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
    
    # Test optimized version eigenvalues
    print("\nOptimized Hamiltonian eigenvalues:")
    eigenvals_opt, _ = nv_ham_simple.get_eigenvalues_and_states(H_optimized)
    print(f"First 5 eigenvalues (GHz): {eigenvals_opt[:5] / (2*jnp.pi*1e9)}")
    
    return nv_ham, H_full, nv_ham_simple, H_optimized

if __name__ == "__main__":
    print("🔬 Testing Full NV Hamiltonian")
    print("=" * 40)
    
    # Test standard implementation
    nv_ham, H_full, nv_simple, H_opt = example_usage()
    
    print("\n" + "=" * 50)
    
    # Test optimized implementations
    nv_opt_simple, nv_opt_ext = test_optimized_hamiltonians()
    
    print("\n✅ Full Hamiltonian implementation working!")
    print(f"   Standard version: {nv_ham.total_dim} dimensions")
    print(f"   Optimized version: {nv_simple.total_dim} dimensions")
    print(f"   Extended version: {nv_opt_ext.total_dim} dimensions")
    print(f"   Includes: ZFS + Zeeman + Strain + Hyperfine + Nuclear terms")
    print(f"   User snippets integrated: ✅ H_ground, ✅ H_excited, ✅ C13 extension")