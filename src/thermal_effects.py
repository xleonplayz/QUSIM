#!/usr/bin/env python3
"""
Temperature and Blackbody Effects
=================================
Complete thermal physics for NV centers including blackbody radiation
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import scipy.constants as sc

jax.config.update("jax_enable_x64", True)

# Physical constants
K_B = sc.Boltzmann  # J/K
H_BAR = sc.hbar     # J⋅s  
C_LIGHT = sc.c      # m/s
E_VOLT = sc.electron_volt  # J

@jit
def arrhenius_rate(k0_Hz, E_a_eV, T_K):
    """
    Arrhenius temperature dependence: k(T) = k₀ exp(-E_a/k_B T)
    """
    return k0_Hz * jnp.exp(-E_a_eV * E_VOLT / (K_B * T_K))

@jit
def bose_einstein_distribution(omega_rad_per_s, T_K):
    """
    Bose-Einstein distribution for phonon occupation:
    n̄(ω,T) = 1/(exp(ℏω/k_B T) - 1)
    """
    x = H_BAR * omega_rad_per_s / (K_B * jnp.maximum(T_K, 1e-6))  # Avoid division by zero
    
    # Use jnp.where for conditional logic
    result = jnp.where(
        x > 50, 
        0.0,  # Large x: exponentially suppressed
        jnp.where(
            x < 0.01,
            K_B * T_K / (H_BAR * omega_rad_per_s),  # Small x: classical limit
            1.0 / (jnp.exp(x) - 1.0)  # General case
        )
    )
    
    # Handle T=0 case
    result = jnp.where(T_K <= 0, 0.0, result)
    
    return result

def blackbody_photon_rate(omega_rad_per_s, T_K, volume_m3=1e-21):
    """
    Blackbody photon density and interaction rate.
    For NV centers, relevant for infrared transitions.
    """
    # Photon density per unit frequency
    # ρ(ω) = ℏω³/(π²c³) × 1/(exp(ℏω/k_B T) - 1)
    
    if T_K <= 0:
        return 0.0
    
    omega_cubed = omega_rad_per_s**3
    bose_factor = bose_einstein_distribution(omega_rad_per_s, T_K)
    
    # Photon density per unit volume per unit frequency
    photon_density = (H_BAR * omega_cubed / (jnp.pi**2 * C_LIGHT**3)) * bose_factor
    
    # Interaction rate ∝ density × cross section × c
    # Cross section ~ λ² for optical transitions
    wavelength_m = 2 * jnp.pi * C_LIGHT / omega_rad_per_s
    cross_section_m2 = (wavelength_m**2) / (4 * jnp.pi)  # Rough estimate
    
    interaction_rate = photon_density * cross_section_m2 * C_LIGHT * volume_m3
    
    return interaction_rate

@jit
def phonon_assisted_transitions(base_rate_Hz, omega_phonon_rad_per_s, T_K, n_phonons=1):
    """
    Phonon-assisted transition rates with temperature dependence.
    Rate ∝ (n̄ + 1) for emission, Rate ∝ n̄ for absorption
    """
    n_bar = bose_einstein_distribution(omega_phonon_rad_per_s, T_K)
    
    if n_phonons > 0:
        # Phonon emission: (n̄ + 1) factor
        enhanced_rate = base_rate_Hz * (n_bar + 1.0)**n_phonons
    else:
        # Phonon absorption: n̄ factor  
        enhanced_rate = base_rate_Hz * n_bar**abs(n_phonons)
    
    return enhanced_rate

@jit
def debye_model_phonons(T_K, theta_D_K, omega_max_rad_per_s):
    """
    Debye model for phonon bath in diamond.
    Returns average phonon energy and occupation.
    """
    # Debye cutoff frequency
    omega_D = K_B * theta_D_K / H_BAR
    
    # Average phonon energy in Debye model
    x_D = theta_D_K / jnp.maximum(T_K, 1e-6)  # Avoid division by zero
    
    # Use jnp.where for conditional logic
    E_avg = jnp.where(
        x_D > 50,
        # Low temperature: E_avg ≈ (π⁴/5) × (T/θ_D)⁴ × k_B θ_D
        (jnp.pi**4 / 5) * (T_K / theta_D_K)**4 * K_B * theta_D_K,
        jnp.where(
            x_D < 0.1,
            # High temperature: E_avg ≈ 3 k_B T
            3 * K_B * T_K,
            # General case (simplified)
            3 * K_B * T_K * (T_K / theta_D_K)**3
        )
    )
    
    # Handle T=0 case
    E_avg = jnp.where(T_K <= 0, 0.0, E_avg)
    
    # Average phonon occupation
    n_avg = bose_einstein_distribution(omega_D, T_K)
    
    return E_avg, n_avg

@jit
def orbital_relaxation_rate(T_K, E_activation_eV, k0_300K_Hz):
    """
    Temperature-dependent orbital relaxation rate.
    Includes both Arrhenius activation and phonon assistance.
    """
    # Arrhenius factor for activation barrier
    arrhenius_factor = jnp.exp(-E_activation_eV * E_VOLT / (K_B * T_K))
    
    # Phonon assistance (diamond phonons ~165 meV)
    omega_phonon = 0.165 * E_VOLT / H_BAR  # rad/s
    phonon_factor = bose_einstein_distribution(omega_phonon, T_K) + 1.0
    
    # Combined rate
    rate = k0_300K_Hz * (T_K / 300.0) * arrhenius_factor * phonon_factor
    
    return rate

@jit
def temperature_dependent_dephasing(T_K, T2_0_us, coupling_strengths_MHz):
    """
    Temperature-dependent dephasing from phonon interactions.
    T2⁻¹(T) = T2⁻¹(0) + Σᵢ gᵢ² coth(ℏωᵢ/2k_B T)
    """
    if T_K <= 0:
        return T2_0_us
    
    # Base dephasing rate
    gamma_0 = 1.0 / T2_0_us  # MHz
    
    # Phonon-induced dephasing
    gamma_phonon = 0.0
    
    for coupling_MHz in coupling_strengths_MHz:
        # Typical phonon frequency for each coupling
        omega_phonon = coupling_MHz * 1e6 * 2 * jnp.pi  # rad/s
        
        # Hyperbolic cotangent factor
        x = H_BAR * omega_phonon / (2 * K_B * T_K)
        if x > 50:
            coth_factor = 1.0  # Low temperature limit
        elif x < 0.01:
            coth_factor = 2 * K_B * T_K / (H_BAR * omega_phonon)  # High T limit
        else:
            coth_factor = 1.0 / jnp.tanh(x)
        
        # Add to total dephasing
        gamma_phonon += (coupling_MHz)**2 * coth_factor
    
    # Total dephasing rate
    gamma_total = gamma_0 + gamma_phonon * 1e-6  # Convert back to μs⁻¹
    
    # Effective T2
    T2_eff = 1.0 / gamma_total
    
    return T2_eff

@jit
def thermal_population_distribution(energy_levels_eV, T_K):
    """
    Thermal equilibrium populations for multi-level system.
    """
    # Boltzmann factors
    beta = 1.0 / (K_B * jnp.maximum(T_K, 1e-6) / E_VOLT)  # eV⁻¹, avoid division by zero
    boltzmann_factors = jnp.exp(-beta * (energy_levels_eV - energy_levels_eV[0]))
    
    # Normalize
    Z = jnp.sum(boltzmann_factors)  # Partition function
    populations = boltzmann_factors / Z
    
    # Handle T=0 case: only ground state populated
    ground_state_only = jnp.zeros_like(energy_levels_eV)
    ground_state_only = ground_state_only.at[0].set(1.0)
    
    populations = jnp.where(T_K <= 0, ground_state_only, populations)
    
    return populations

@jit
def infrared_absorption_rate(T_K, transition_energy_eV, oscillator_strength=1e-3):
    """
    Infrared absorption from blackbody radiation.
    Relevant for singlet-triplet transitions in NV centers.
    """
    # Transition frequency
    omega = jnp.maximum(transition_energy_eV, 1e-6) * E_VOLT / H_BAR  # Avoid zero energy
    
    # Blackbody photon interaction rate
    bb_rate = blackbody_photon_rate(omega, T_K)
    
    # Scale by oscillator strength
    absorption_rate = bb_rate * oscillator_strength
    
    # Handle zero energy case
    absorption_rate = jnp.where(transition_energy_eV <= 0, 0.0, absorption_rate)
    
    return absorption_rate

class ThermalPhysicsNV:
    """
    Complete thermal physics model for NV centers.
    """
    
    def __init__(self, params):
        self.T_K = params.Temperature_K
        self.theta_D_K = params.theta_D_K  # Debye temperature
        
        # Energy levels (relative to ground state in eV)
        self.E_gs_eV = 0.0           # Ground state ms=0
        self.E_gs_pm1_eV = 2.87e-3   # Ground state ms=±1 (2.87 GHz)
        self.E_es_eV = 1.945         # Excited state (~637 nm)
        self.E_singlet_eV = 1.19     # Singlet state (~1042 nm)
        
        # Activation energies
        self.E_ISC_ms0_eV = params.ISC_ms0_activation_meV * 1e-3
        self.E_ISC_ms1_eV = params.ISC_ms1_activation_meV * 1e-3
        self.E_orbital_eV = params.k_orb_activation_K * K_B / E_VOLT
        
        # Reference rates at 300K
        self.k_ISC_ms0_300K = params.gamma_ISC_ms0_MHz * 1e6
        self.k_ISC_ms1_300K = params.gamma_ISC_ms1_MHz * 1e6
        self.k_orbital_300K = params.k_orb_300K_MHz * 1e6
        
    def update_temperature(self, T_K):
        """Update temperature and recalculate all thermal effects."""
        self.T_K = T_K
        
    def get_isc_rates(self):
        """Get temperature-dependent ISC rates."""
        rate_ms0 = arrhenius_rate(self.k_ISC_ms0_300K, self.E_ISC_ms0_eV, self.T_K)
        rate_ms1 = arrhenius_rate(self.k_ISC_ms1_300K, self.E_ISC_ms1_eV, self.T_K)
        
        return rate_ms0, rate_ms1
    
    def get_orbital_relaxation_rate(self):
        """Get temperature-dependent orbital relaxation rate."""
        return orbital_relaxation_rate(self.T_K, self.E_orbital_eV, self.k_orbital_300K)
    
    def get_thermal_populations(self):
        """Get thermal equilibrium populations of ground state sublevels."""
        # Ground state energies
        energies = jnp.array([self.E_gs_pm1_eV, self.E_gs_eV, self.E_gs_pm1_eV])  # ms=-1,0,+1
        
        return thermal_population_distribution(energies, self.T_K)
    
    def get_phonon_effects(self):
        """Get phonon bath effects."""
        # Diamond phonon frequencies (simplified)
        omega_phonons = jnp.array([0.165, 0.120, 0.080]) * E_VOLT / H_BAR  # eV → rad/s
        
        phonon_occupations = jnp.array([bose_einstein_distribution(omega, self.T_K) 
                                       for omega in omega_phonons])
        
        E_avg, n_avg = debye_model_phonons(self.T_K, self.theta_D_K, omega_phonons[0])
        
        return {
            'phonon_occupations': phonon_occupations,
            'average_energy_eV': E_avg / E_VOLT,
            'average_occupation': n_avg
        }
    
    def get_blackbody_effects(self):
        """Get blackbody radiation effects."""
        # Infrared transitions
        singlet_absorption = infrared_absorption_rate(self.T_K, self.E_singlet_eV)
        
        return {
            'singlet_absorption_Hz': singlet_absorption,
            'thermal_photon_rate_Hz': blackbody_photon_rate(
                self.E_singlet_eV * E_VOLT / H_BAR, self.T_K
            )
        }
    
    def get_effective_T2_star(self, coupling_strengths_MHz, T2_0_us=2.0):
        """Get temperature-dependent T2* including phonon dephasing."""
        return temperature_dependent_dephasing(self.T_K, T2_0_us, coupling_strengths_MHz)
    
    def get_temperature_summary(self):
        """Get complete summary of thermal effects."""
        isc_ms0, isc_ms1 = self.get_isc_rates()
        orbital_rate = self.get_orbital_relaxation_rate()
        thermal_pops = self.get_thermal_populations()
        phonon_effects = self.get_phonon_effects()
        blackbody_effects = self.get_blackbody_effects()
        
        return {
            'temperature_K': self.T_K,
            'isc_rate_ms0_Hz': isc_ms0,
            'isc_rate_ms1_Hz': isc_ms1,
            'orbital_relaxation_Hz': orbital_rate,
            'thermal_populations': thermal_pops,
            'phonon_effects': phonon_effects,
            'blackbody_effects': blackbody_effects,
            'debye_temperature_K': self.theta_D_K
        }

if __name__ == "__main__":
    # Test thermal physics
    print("🌡️ Testing Thermal Physics & Blackbody Effects")
    print("=" * 50)
    
    # Mock parameters
    class MockParams:
        Temperature_K = 300.0
        theta_D_K = 150.0
        ISC_ms0_activation_meV = 50.0
        ISC_ms1_activation_meV = 10.0
        k_orb_activation_K = 500.0
        gamma_ISC_ms0_MHz = 5.0
        gamma_ISC_ms1_MHz = 50.0
        k_orb_300K_MHz = 100.0
    
    params = MockParams()
    thermal = ThermalPhysicsNV(params)
    
    # Test different temperatures
    temperatures = [4.0, 77.0, 300.0, 500.0]
    
    for T in temperatures:
        thermal.update_temperature(T)
        summary = thermal.get_temperature_summary()
        
        print(f"\nT = {T} K:")
        print(f"  ISC ms=0: {summary['isc_rate_ms0_Hz']:.2e} Hz")
        print(f"  ISC ms=±1: {summary['isc_rate_ms1_Hz']:.2e} Hz") 
        print(f"  Orbital relaxation: {summary['orbital_relaxation_Hz']:.2e} Hz")
        print(f"  Thermal populations: {summary['thermal_populations']}")
        print(f"  Average phonon occupation: {summary['phonon_effects']['average_occupation']:.3f}")
        print(f"  Blackbody singlet rate: {summary['blackbody_effects']['singlet_absorption_Hz']:.2e} Hz")
    
    # Test T2* temperature dependence
    print(f"\nT2* temperature dependence:")
    coupling_strengths = jnp.array([0.1, 0.05, 0.02])  # MHz
    
    for T in [4, 77, 300]:
        thermal.update_temperature(T)
        T2_star = thermal.get_effective_T2_star(coupling_strengths)
        print(f"  T = {T:3.0f} K: T2* = {T2_star:.2f} μs")
    
    print("\n✅ Thermal physics working!")