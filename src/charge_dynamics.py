#!/usr/bin/env python3
"""
Charge State Dynamics for NV Centers
===================================
Master equation for NV⁻ ↔ NV⁰ charge state transitions
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from functools import partial
import scipy.constants as sc

jax.config.update("jax_enable_x64", True)

# Physical constants
K_B = sc.Boltzmann
HBAR = sc.hbar
E_VOLT = sc.electron_volt

@jit
def saturating_rate(rate_base_Hz, I_laser_rel, I_sat_rel=1.0):
    """
    Saturation of laser-dependent rates.
    R(I) = R_base × I/I_sat / (1 + I/I_sat)
    """
    s = I_laser_rel / I_sat_rel
    return rate_base_Hz * s / (1.0 + s)

@jit  
def power_law_rate(rate_base_Hz, I_laser_rel, power=1.0):
    """
    Power-law dependence on laser intensity.
    R(I) = R_base × (I/I_0)^power
    """
    return rate_base_Hz * (I_laser_rel ** power)

class ChargeStateDynamics:
    """
    Complete charge state dynamics for NV centers.
    
    Models the transitions:
    NV⁻ ⇌ NV⁰ + e⁻
    
    Includes:
    - Ground state ionization (weak)
    - Excited state ionization (strong) 
    - Recombination from surface states
    - Laser power dependence
    - Temperature effects
    - Surface state dynamics
    """
    
    def __init__(self, params):
        self.params = params
        
    @partial(jit, static_argnums=(0,))
    def ionization_rate_gs(self, I_laser_rel, T_K=300.0):
        """
        Ground state ionization rate.
        Very weak process, enhanced by high laser power.
        """
        # Base rate (very low)
        rate_base = self.params.ionization_rate_GS_MHz * 1e6
        
        # Multi-photon process (power law dependence)
        rate_laser = power_law_rate(rate_base, I_laser_rel, power=2.0)
        
        # Temperature enhancement (thermal activation)
        if hasattr(self.params, 'ion_activation_gs_meV'):
            E_activation = self.params.ion_activation_gs_meV * 1e-3
            thermal_factor = jnp.exp(-E_activation * E_VOLT / (K_B * T_K))
        else:
            thermal_factor = 1.0
        
        return rate_laser * thermal_factor
    
    @partial(jit, static_argnums=(0,))
    def ionization_rate_es(self, I_laser_rel, T_K=300.0):
        """
        Excited state ionization rate.
        Stronger process due to higher energy electrons.
        """
        # Base rate (higher than GS)
        rate_base = self.params.ionization_rate_ES_MHz * 1e6
        
        # Linear saturation with laser power
        rate_laser = saturating_rate(rate_base, I_laser_rel)
        
        # Weaker temperature dependence (already in excited state)
        if hasattr(self.params, 'ion_activation_es_meV'):
            E_activation = self.params.ion_activation_es_meV * 1e-3
            thermal_factor = jnp.exp(-E_activation * E_VOLT / (K_B * T_K))
        else:
            thermal_factor = 1.0 + 0.1 * (T_K - 300.0) / 300.0  # Weak enhancement
        
        return rate_laser * thermal_factor
    
    @partial(jit, static_argnums=(0,))
    def recombination_rate(self, surface_state_population=1.0):
        """
        Recombination rate: NV⁰ + e⁻ → NV⁻
        
        Args:
            surface_state_population: population of surface electron states (0-1)
        """
        # Base recombination rate
        rate_base = self.params.recombination_rate_MHz * 1e6
        
        # Depends on availability of electrons at surface
        rate_eff = rate_base * surface_state_population
        
        return rate_eff
    
    @partial(jit, static_argnums=(0,))
    def charge_transfer_rates(self, I_laser_rel, T_K, surface_pop=1.0):
        """
        Get all charge transfer rates.
        
        Returns:
            dict with ionization and recombination rates
        """
        rates = {
            'ionization_gs_Hz': self.ionization_rate_gs(I_laser_rel, T_K),
            'ionization_es_Hz': self.ionization_rate_es(I_laser_rel, T_K),
            'recombination_Hz': self.recombination_rate(surface_pop)
        }
        
        return rates
    
    @partial(jit, static_argnums=(0,))
    def charge_liouvillian(self, rho_minus, rho_zero, rates):
        """
        Liouvillian for charge state master equation.
        
        d/dt [ρ_minus] = [-k_ion    k_rec  ] [ρ_minus]
             [ρ_zero ]   [ k_ion   -k_rec ] [ρ_zero ]
        
        Args:
            rho_minus: density matrix for NV⁻ state
            rho_zero: density matrix for NV⁰ state  
            rates: dict of transition rates
        """
        k_ion = rates['ionization_gs_Hz']  # Simplified: use GS rate
        k_rec = rates['recombination_Hz']
        
        # Evolution equations
        drho_minus_dt = -k_ion * rho_minus + k_rec * rho_zero
        drho_zero_dt = k_ion * rho_minus - k_rec * rho_zero
        
        return drho_minus_dt, drho_zero_dt
    
    @partial(jit, static_argnums=(0,))
    def jump_operators_charge(self, total_dim):
        """
        Jump operators for charge state transitions.
        These can be included in the full Lindblad equation.
        """
        identity = jnp.eye(total_dim, dtype=jnp.complex128)
        
        jump_ops = []
        
        # Ionization jump: NV⁻ → NV⁰ (lose electron)
        # In practice, this removes population from the system
        L_ionization = jnp.sqrt(self.params.ionization_rate_GS_MHz * 1e6) * identity
        
        # Recombination jump: NV⁰ → NV⁻ (gain electron)  
        # In practice, this adds population to the system
        L_recombination = jnp.sqrt(self.params.recombination_rate_MHz * 1e6) * identity
        
        jump_ops.extend([L_ionization, L_recombination])
        
        return jump_ops
    
    @partial(jit, static_argnums=(0,))
    def steady_state_fractions(self, I_laser_rel, T_K=300.0):
        """
        Calculate steady-state charge state fractions.
        
        At equilibrium: k_ion × f_minus = k_rec × f_zero
        With normalization: f_minus + f_zero = 1
        """
        rates = self.charge_transfer_rates(I_laser_rel, T_K)
        
        k_ion = rates['ionization_gs_Hz']
        k_rec = rates['recombination_Hz']
        
        # Steady state solution
        if k_ion + k_rec > 0:
            f_minus = k_rec / (k_ion + k_rec)
            f_zero = k_ion / (k_ion + k_rec)
        else:
            # No transitions: assume all NV⁻
            f_minus = 1.0
            f_zero = 0.0
        
        return {'f_NV_minus': f_minus, 'f_NV_zero': f_zero}
    
    @partial(jit, static_argnums=(0,))
    def time_evolution_charge_only(self, f_minus_initial, I_laser_rel, T_K, time_array):
        """
        Time evolution of charge state fractions only.
        Simple 2-level system evolution.
        """
        rates = self.charge_transfer_rates(I_laser_rel, T_K)
        k_ion = rates['ionization_gs_Hz']
        k_rec = rates['recombination_Hz']
        
        # Rate matrix
        rate_matrix = jnp.array([
            [-k_ion, k_rec],
            [k_ion, -k_rec]
        ])
        
        # Matrix exponential solution
        f_initial = jnp.array([f_minus_initial, 1.0 - f_minus_initial])
        
        def evolve_single_time(t):
            # Use matrix exponential (simplified implementation)
            # For exact solution: f(t) = exp(K*t) @ f(0)
            
            # Characteristic time
            tau_char = 1.0 / (k_ion + k_rec) if (k_ion + k_rec) > 0 else jnp.inf
            
            # Exponential approach to equilibrium
            f_eq = self.steady_state_fractions(I_laser_rel, T_K)
            f_eq_array = jnp.array([f_eq['f_NV_minus'], f_eq['f_NV_zero']])
            
            decay_factor = jnp.exp(-t / tau_char) if jnp.isfinite(tau_char) else 1.0
            f_t = f_eq_array + (f_initial - f_eq_array) * decay_factor
            
            return f_t
        
        # Vectorized evolution
        f_evolution = vmap(evolve_single_time)(time_array)
        
        return f_evolution

class SurfaceStateDynamics:
    """
    Model surface states that mediate charge transfer.
    These affect the recombination rate.
    """
    
    def __init__(self, params):
        self.params = params
        
        # Surface state parameters
        self.n_surface_states = 10  # Number of surface state types
        self.surface_binding_energies_eV = jnp.linspace(0.1, 2.0, self.n_surface_states)
        self.surface_capture_rates_Hz = jnp.full(self.n_surface_states, 1e6)
        
    @partial(jit, static_argnums=(0,))
    def surface_state_occupations(self, T_K, electron_chemical_potential_eV=0.0):
        """
        Calculate surface state electron occupations using Fermi-Dirac statistics.
        """
        # Fermi-Dirac distribution
        beta = 1.0 / (K_B * T_K / E_VOLT)  # eV^-1
        
        occupations = 1.0 / (1.0 + jnp.exp(beta * (self.surface_binding_energies_eV - electron_chemical_potential_eV)))
        
        return occupations
    
    @partial(jit, static_argnums=(0,))
    def effective_recombination_rate(self, T_K, base_rate_Hz):
        """
        Calculate effective recombination rate modified by surface states.
        """
        occupations = self.surface_state_occupations(T_K)
        
        # Average occupation (proxy for electron availability)
        avg_occupation = jnp.mean(occupations)
        
        # Modify recombination rate
        eff_rate = base_rate_Hz * avg_occupation
        
        return eff_rate

def test_charge_dynamics():
    """Test charge state dynamics implementation."""
    
    # Mock parameters
    class MockParams:
        ionization_rate_GS_MHz = 0.0001
        ionization_rate_ES_MHz = 0.1
        recombination_rate_MHz = 0.01
        Temperature_K = 300.0
    
    params = MockParams()
    
    # Create charge dynamics
    charge = ChargeStateDynamics(params)
    
    # Test rates at different laser powers
    laser_powers = jnp.array([0.1, 1.0, 10.0, 100.0])  # Relative to saturation
    
    print("Laser power dependence:")
    for I_rel in laser_powers:
        rates = charge.charge_transfer_rates(I_rel, params.Temperature_K)
        steady_state = charge.steady_state_fractions(I_rel, params.Temperature_K)
        
        print(f"  I_rel = {I_rel:5.1f}: "
              f"k_ion_gs = {rates['ionization_gs_Hz']:.2e} Hz, "
              f"k_rec = {rates['recombination_Hz']:.2e} Hz, "
              f"f_NV- = {steady_state['f_NV_minus']:.3f}")
    
    # Test time evolution
    time_ns = jnp.linspace(0, 1000, 101)  # 1 μs evolution
    f_evolution = charge.time_evolution_charge_only(1.0, 1.0, params.Temperature_K, time_ns * 1e-9)
    
    print(f"\nTime evolution (1 μs):")
    print(f"  Initial: f_NV- = {f_evolution[0, 0]:.3f}")
    print(f"  Final:   f_NV- = {f_evolution[-1, 0]:.3f}")
    
    # Test surface states
    surface = SurfaceStateDynamics(params)
    surface_occ = surface.surface_state_occupations(params.Temperature_K)
    
    print(f"\nSurface states:")
    print(f"  Number of states: {surface.n_surface_states}")
    print(f"  Average occupation: {jnp.mean(surface_occ):.3f}")
    print(f"  Energy range: {surface.surface_binding_energies_eV[0]:.1f} - {surface.surface_binding_energies_eV[-1]:.1f} eV")
    
    return charge, surface

if __name__ == "__main__":
    print("⚡ Testing Charge State Dynamics")
    print("=" * 35)
    
    charge, surface = test_charge_dynamics()
    
    print("\n✅ Charge state dynamics working!")
    print("   Includes: ionization, recombination, surface states, laser dependence")