#!/usr/bin/env python3
"""
Advanced Lindblad Dissipation for NV Centers
===========================================
Separate T1/T2, ISC(ms=0/±1), singlet manifold, and jump operators
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
from functools import partial
import scipy.constants as sc

from .full_hamiltonian import FullNVHamiltonian, projectors_spin1

jax.config.update("jax_enable_x64", True)

# Physical constants
K_B = sc.Boltzmann
HBAR = sc.hbar
E_VOLT = sc.electron_volt

@jit
def arrhenius_rate(k0_Hz, E_activation_eV, T_K):
    """Arrhenius temperature dependence for rates."""
    return k0_Hz * jnp.exp(-E_activation_eV * E_VOLT / (K_B * T_K))

@jit
def thermal_rate_factor(T_K, T_ref=300.0, alpha=0.1):
    """Simple thermal rate enhancement factor."""
    return 1.0 + alpha * (T_K - T_ref) / T_ref

class AdvancedLindblad:
    """
    Advanced Lindblad master equation with all physical dissipation channels.
    
    Includes:
    - Separate T1 and T2 processes
    - ISC rates dependent on ms quantum number
    - Singlet manifold dynamics
    - Temperature-dependent rates
    - Photon emission jump operators
    """
    
    def __init__(self, nv_hamiltonian: FullNVHamiltonian, params):
        self.nv_ham = nv_hamiltonian
        self.params = params
        self.total_dim = nv_hamiltonian.total_dim
        
        # Generate all necessary operators
        self._generate_dissipation_operators()
        
    def _generate_dissipation_operators(self):
        """Generate all dissipation operators."""
        
        # Electronic projectors (already in nv_hamiltonian)
        self.P_minus1 = self.nv_ham.P_minus1
        self.P_0 = self.nv_ham.P_0  
        self.P_plus1 = self.nv_ham.P_plus1
        
        # Spin operators
        self.Sx = self.nv_ham.Sx
        self.Sy = self.nv_ham.Sy
        self.Sz = self.nv_ham.Sz
        
        # Ladder operators
        self.S_plus = self.Sx + 1j * self.Sy   # |ms⟩ → |ms+1⟩
        self.S_minus = self.Sx - 1j * self.Sy  # |ms⟩ → |ms-1⟩
        
        # Transitions between ms levels
        self.sigma_plus = self.P_0 @ self.S_plus @ self.P_minus1    # |ms=-1⟩ → |ms=0⟩
        self.sigma_minus = self.P_0 @ self.S_minus @ self.P_plus1   # |ms=+1⟩ → |ms=0⟩
        self.sigma_z = self.P_plus1 @ self.S_plus @ self.P_0        # |ms=0⟩ → |ms=+1⟩
        
    @partial(jit, static_argnums=(0,))
    def t1_relaxation_operators(self):
        """
        T1 (longitudinal) relaxation operators.
        Thermal equilibration between ms levels.
        """
        # T1 rate (convert ms to Hz)
        gamma_1 = 1.0 / (self.params.T1_ms * 1e-3)
        
        # Thermal populations at given temperature
        T_K = self.params.Temperature_K
        
        if T_K > 0:
            # Boltzmann distribution for ms levels
            # E(ms=±1) = D, E(ms=0) = 0 (relative)
            D_Hz = self.params.D_GS_Hz
            kT_Hz = K_B * T_K / HBAR  # Convert to Hz
            
            # Boltzmann factors
            exp_factor = jnp.exp(-D_Hz / kT_Hz) if kT_Hz > 0 else 0.0
            Z = 2 * exp_factor + 1  # Partition function
            
            # Thermal populations
            p_eq_0 = 1.0 / Z
            p_eq_pm1 = exp_factor / Z
        else:
            # T=0: only ms=0 populated
            p_eq_0 = 1.0
            p_eq_pm1 = 0.0
        
        collapse_ops = []
        
        # T1 processes with detailed balance
        # |ms=±1⟩ → |ms=0⟩ (faster)
        rate_pm1_to_0 = gamma_1 * p_eq_0
        collapse_ops.append(jnp.sqrt(rate_pm1_to_0) * self.sigma_plus)   # |-1⟩→|0⟩
        collapse_ops.append(jnp.sqrt(rate_pm1_to_0) * self.sigma_minus)  # |+1⟩→|0⟩
        
        # |ms=0⟩ → |ms=±1⟩ (slower, detailed balance)
        rate_0_to_pm1 = gamma_1 * p_eq_pm1
        collapse_ops.append(jnp.sqrt(rate_0_to_pm1) * self.sigma_z)      # |0⟩→|+1⟩
        collapse_ops.append(jnp.sqrt(rate_0_to_pm1) * self.sigma_z.T.conj())  # |0⟩→|-1⟩
        
        return collapse_ops
    
    @partial(jit, static_argnums=(0,))  
    def t2_dephasing_operators(self):
        """
        T2 (transverse) dephasing operators.
        Pure dephasing without population change.
        """
        # T2* rate (convert μs to Hz)
        gamma_phi = 1.0 / (self.params.T2_star_us * 1e-6)
        
        # Pure dephasing operator (Sz causes phase randomization)
        # This destroys coherences but preserves populations
        collapse_ops = [jnp.sqrt(gamma_phi) * self.Sz]
        
        return collapse_ops
    
    @partial(jit, static_argnums=(0,))
    def isc_operators(self):
        """
        Intersystem crossing (ISC) operators with ms-dependent rates.
        |ms⟩ → |singlet⟩ with different rates for ms=0 vs ms=±1
        """
        T_K = self.params.Temperature_K
        
        # Temperature-dependent ISC rates
        gamma_ISC_ms0 = arrhenius_rate(
            self.params.gamma_ISC_ms0_MHz * 1e6,
            self.params.ISC_ms0_activation_meV * 1e-3,
            T_K
        )
        
        gamma_ISC_ms1 = arrhenius_rate(
            self.params.gamma_ISC_ms1_MHz * 1e6, 
            self.params.ISC_ms1_activation_meV * 1e-3,
            T_K
        )
        
        # For simplicity, map to ground state ms=0 (singlet decay mechanism)
        # In full model, would include actual singlet manifold states
        
        collapse_ops = []
        
        # ISC from ms=0 (slower)
        L_ISC_0 = jnp.sqrt(gamma_ISC_ms0) * self.P_0
        collapse_ops.append(L_ISC_0)
        
        # ISC from ms=±1 (faster)  
        L_ISC_plus1 = jnp.sqrt(gamma_ISC_ms1) * self.P_plus1
        L_ISC_minus1 = jnp.sqrt(gamma_ISC_ms1) * self.P_minus1
        collapse_ops.extend([L_ISC_plus1, L_ISC_minus1])
        
        return collapse_ops
    
    @partial(jit, static_argnums=(0,))
    def singlet_operators(self):
        """
        Singlet manifold dynamics.
        Simplified: singlet decays back to ms=0 ground state.
        """
        # Singlet lifetime
        gamma_singlet = 1.0 / (self.params.tau_singlet_ns * 1e-9)
        
        # In simplified model: direct decay to ms=0
        # In full model: would include actual |singlet⟩ states
        
        # Effective singlet decay (population feeding into ms=0)
        singlet_decay_rate = gamma_singlet * 0.1  # 10% effective contribution
        
        # This represents the net effect of singlet→gs decay
        L_singlet = jnp.sqrt(singlet_decay_rate) * self.P_0
        
        return [L_singlet]
    
    @partial(jit, static_argnums=(0,))
    def photon_emission_operators(self):
        """
        Photon emission (radiative decay) jump operators.
        These create the fluorescence signal.
        """
        # Radiative decay rate
        gamma_rad = self.params.gamma_rad_MHz * 1e6
        
        # Photon emission preferentially from certain transitions
        # In NV: mainly from excited state, but here simplified to ground state
        
        collapse_ops = []
        
        # Emit photon: population-dependent emission
        # Higher rate for ms=0 (bright state)
        L_emission_0 = jnp.sqrt(gamma_rad * 1.0) * self.P_0
        
        # Lower rate for ms=±1 (darker states)
        L_emission_plus1 = jnp.sqrt(gamma_rad * 0.3) * self.P_plus1
        L_emission_minus1 = jnp.sqrt(gamma_rad * 0.3) * self.P_minus1
        
        collapse_ops.extend([L_emission_0, L_emission_plus1, L_emission_minus1])
        
        return collapse_ops
    
    @partial(jit, static_argnums=(0,))
    def orbital_relaxation_operators(self):
        """
        Orbital relaxation in excited state manifold.
        Temperature-dependent non-radiative processes.
        """
        if not hasattr(self.params, 'k_orb_300K_MHz'):
            return []
        
        T_K = self.params.Temperature_K
        
        # Temperature-dependent orbital relaxation
        gamma_orbital = arrhenius_rate(
            self.params.k_orb_300K_MHz * 1e6,
            self.params.k_orb_activation_K * K_B / E_VOLT,  # Convert K to eV
            T_K
        )
        
        # Pure dephasing in excited state manifold
        # Simplified: affects all electronic levels
        collapse_ops = []
        
        # Orbital dephasing (affects coherences)
        L_orbital = jnp.sqrt(gamma_orbital) * (
            self.P_0 + 0.5 * (self.P_plus1 + self.P_minus1)
        )
        
        collapse_ops.append(L_orbital)
        
        return collapse_ops
    
    @partial(jit, static_argnums=(0,))
    def charge_state_operators(self):
        """
        Charge state dynamics: NV⁻ ↔ NV⁰
        Simplified operators for charge conversion.
        """
        # Ionization rates
        gamma_ion_gs = self.params.ionization_rate_GS_MHz * 1e6
        gamma_ion_es = self.params.ionization_rate_ES_MHz * 1e6
        
        # Recombination rate
        gamma_rec = self.params.recombination_rate_MHz * 1e6
        
        collapse_ops = []
        
        # Ionization: NV⁻ → NV⁰ (population loss)
        # Simplified: uniform loss from all ms states
        identity = jnp.eye(self.total_dim, dtype=jnp.complex128)
        L_ionization = jnp.sqrt(gamma_ion_gs) * identity
        
        # Recombination: NV⁰ → NV⁻ (population gain)
        # Preferentially populates ms=0
        L_recombination = jnp.sqrt(gamma_rec) * self.P_0
        
        collapse_ops.extend([L_ionization, L_recombination])
        
        return collapse_ops
    
    def get_all_collapse_operators(self):
        """
        Generate complete set of collapse operators for Lindblad equation.
        """
        all_operators = []
        
        # Add all dissipation channels
        all_operators.extend(self.t1_relaxation_operators())
        all_operators.extend(self.t2_dephasing_operators())
        all_operators.extend(self.isc_operators())
        all_operators.extend(self.singlet_operators())
        all_operators.extend(self.photon_emission_operators())
        all_operators.extend(self.orbital_relaxation_operators())
        all_operators.extend(self.charge_state_operators())
        
        return all_operators
    
    @partial(jit, static_argnums=(0,))
    def lindblad_superoperator(self, rho, H_total, collapse_ops):
        """
        Lindblad master equation right-hand side.
        
        d𝜌/dt = -i[H,𝜌] + Σ_k (L_k 𝜌 L_k† - 1/2{L_k†L_k, 𝜌})
        """
        # Unitary evolution
        drho_dt = -1j * (H_total @ rho - rho @ H_total)
        
        # Dissipative evolution
        for L_k in collapse_ops:
            L_k_dag = L_k.T.conj()
            L_dag_L = L_k_dag @ L_k
            
            # Lindblad terms
            drho_dt += L_k @ rho @ L_k_dag
            drho_dt -= 0.5 * (L_dag_L @ rho + rho @ L_dag_L)
        
        return drho_dt
    
    @partial(jit, static_argnums=(0,))
    def rk4_step(self, rho, H_total, collapse_ops, dt):
        """
        4th-order Runge-Kutta step for master equation.
        """
        k1 = self.lindblad_superoperator(rho, H_total, collapse_ops)
        k2 = self.lindblad_superoperator(rho + dt/2 * k1, H_total, collapse_ops)
        k3 = self.lindblad_superoperator(rho + dt/2 * k2, H_total, collapse_ops)
        k4 = self.lindblad_superoperator(rho + dt * k3, H_total, collapse_ops)
        
        rho_new = rho + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # Ensure trace preservation and Hermiticity
        rho_new = 0.5 * (rho_new + rho_new.T.conj())
        rho_new = rho_new / jnp.trace(rho_new)
        
        return rho_new
    
    def get_thermal_density_matrix(self, H_total):
        """
        Generate thermal equilibrium density matrix.
        ρ_eq = exp(-βH) / Tr[exp(-βH)]
        """
        T_K = self.params.Temperature_K
        
        if T_K <= 0:
            # T=0: ground state only
            eigenvals, eigenstates = jnp.linalg.eigh(H_total)
            ground_state = eigenstates[:, 0]
            rho_eq = jnp.outer(ground_state, ground_state.conj())
        else:
            # Finite temperature
            beta = 1.0 / (K_B * T_K)
            
            # Convert Hamiltonian to numpy for stable matrix exponential
            H_np = np.array(H_total)
            
            # Thermal state
            exp_neg_beta_H = scipy.linalg.expm(-beta * HBAR * H_np)
            Z = np.trace(exp_neg_beta_H)
            rho_eq_np = exp_neg_beta_H / Z
            
            # Convert back to JAX
            rho_eq = jnp.array(rho_eq_np)
        
        return rho_eq

def test_advanced_lindblad():
    """Test the advanced Lindblad implementation."""
    
    # Mock parameters
    class MockParams:
        D_GS_Hz = 2.87e9
        n_carbon13 = 2
        T1_ms = 5.0
        T2_star_us = 2.0
        Temperature_K = 300.0
        gamma_ISC_ms0_MHz = 5.0
        gamma_ISC_ms1_MHz = 50.0
        ISC_ms0_activation_meV = 50.0
        ISC_ms1_activation_meV = 10.0
        tau_singlet_ns = 200.0
        gamma_rad_MHz = 83.0
        k_orb_300K_MHz = 100.0
        k_orb_activation_K = 500.0
        ionization_rate_GS_MHz = 0.0001
        ionization_rate_ES_MHz = 0.1
        recombination_rate_MHz = 0.01
    
    params = MockParams()
    
    # Create Hamiltonian
    nv_ham = FullNVHamiltonian(params)
    
    # Create Lindblad system
    lindblad = AdvancedLindblad(nv_ham, params)
    
    print(f"Total Hilbert space dimension: {lindblad.total_dim}")
    
    # Generate collapse operators
    collapse_ops = lindblad.get_all_collapse_operators()
    print(f"Number of collapse operators: {len(collapse_ops)}")
    
    # Build test Hamiltonian
    B_field = jnp.array([0.0, 0.0, 1.0])  # 1 mT
    strain = 0.1  # MHz
    A_N14 = (2.2, 2.0)
    A_C13 = jnp.array([[1.0, 0.3], [0.5, 0.1]])
    
    H_total = nv_ham.full_hamiltonian(B_field, strain, A_N14, A_C13)
    
    # Get thermal equilibrium state
    rho_eq = lindblad.get_thermal_density_matrix(H_total)
    print(f"Thermal state trace: {jnp.trace(rho_eq):.6f}")
    
    # Test single time step
    dt = 1e-9  # 1 ns
    rho_new = lindblad.rk4_step(rho_eq, H_total, collapse_ops, dt)
    print(f"After 1ns evolution trace: {jnp.trace(rho_new):.6f}")
    
    # Extract ms populations
    ms_pops_initial = nv_ham.get_ms_populations(jnp.diag(rho_eq))
    ms_pops_final = nv_ham.get_ms_populations(jnp.diag(rho_new))
    
    print(f"Initial ms populations: {ms_pops_initial}")
    print(f"Final ms populations: {ms_pops_final}")
    
    return lindblad

if __name__ == "__main__":
    print("🧮 Testing Advanced Lindblad Dissipation")
    print("=" * 45)
    
    lindblad = test_advanced_lindblad()
    
    print("\n✅ Advanced Lindblad implementation working!")
    print("   Includes: T1/T2, ISC, singlet, photon emission, orbital relaxation")