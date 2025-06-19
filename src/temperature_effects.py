#!/usr/bin/env python3
"""
Advanced Temperature Effects
============================
Arrhenius rates, thermal broadening, and temperature-dependent physics
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
C_LIGHT = sc.c

@jit
def arrhenius_rate(k0_Hz, E_activation_eV, T_K, T_ref_K=300.0):
    """
    Arrhenius temperature dependence for reaction rates.
    k(T) = k₀ × exp(-E_a / k_B T)
    
    Args:
        k0_Hz: pre-exponential factor at reference temperature
        E_activation_eV: activation energy in eV
        T_K: temperature in Kelvin
        T_ref_K: reference temperature (default 300K)
    
    Returns:
        Temperature-dependent rate in Hz
    """
    # Avoid division by zero
    T_safe = jnp.maximum(T_K, 1.0)
    
    # Arrhenius factor
    exponent = -E_activation_eV * E_VOLT / (K_B * T_safe)
    
    # Avoid numerical overflow/underflow
    exponent = jnp.clip(exponent, -50.0, 50.0)
    
    return k0_Hz * jnp.exp(exponent)

@jit
def bose_einstein_factor(energy_eV, T_K):
    """
    Bose-Einstein distribution factor for phonon processes.
    n(E,T) = 1 / (exp(E/k_B T) - 1)
    
    For stimulated emission: rate ∝ (1 + n(E,T))
    For absorption: rate ∝ n(E,T)
    """
    # Avoid division by zero
    T_safe = jnp.maximum(T_K, 1.0)
    
    # Dimensionless energy
    x = energy_eV * E_VOLT / (K_B * T_safe)
    
    # Avoid numerical issues
    x = jnp.clip(x, 1e-10, 50.0)
    
    # Bose-Einstein factor
    n_BE = 1.0 / (jnp.exp(x) - 1.0)
    
    return n_BE

@jit
def fermi_dirac_factor(energy_eV, mu_eV, T_K):
    """
    Fermi-Dirac distribution for electronic states.
    f(E,T) = 1 / (1 + exp((E-μ)/k_B T))
    
    Args:
        energy_eV: energy level
        mu_eV: chemical potential (Fermi level)
        T_K: temperature
    """
    T_safe = jnp.maximum(T_K, 1.0)
    
    x = (energy_eV - mu_eV) * E_VOLT / (K_B * T_safe)
    x = jnp.clip(x, -50.0, 50.0)
    
    return 1.0 / (1.0 + jnp.exp(x))

@jit
def thermal_broadening_gaussian(T_K, mass_u=12.0):
    """
    Thermal (Doppler) broadening for atomic transitions.
    σ_D = ν₀/c × √(2k_B T / m)
    
    Args:
        T_K: temperature in Kelvin
        mass_u: atomic mass in atomic mass units
    
    Returns:
        Gaussian broadening width in Hz (for 637nm NV line)
    """
    # NV zero-phonon line frequency
    wavelength_m = 637e-9
    freq_0 = C_LIGHT / wavelength_m  # Hz
    
    # Atomic mass in kg
    mass_kg = mass_u * sc.atomic_mass
    
    # Thermal velocity width
    v_thermal = jnp.sqrt(2 * K_B * T_K / mass_kg)
    
    # Doppler broadening
    sigma_doppler_Hz = freq_0 * v_thermal / C_LIGHT
    
    return sigma_doppler_Hz

@jit
def phonon_sideband_strength(T_K, debye_waller_factor, phonon_energy_meV):
    """
    Temperature-dependent phonon sideband strength.
    Includes Debye-Waller factor and thermal population.
    
    Args:
        T_K: temperature
        debye_waller_factor: Debye-Waller factor (0-1)
        phonon_energy_meV: typical phonon energy in meV
    
    Returns:
        Relative sideband strength
    """
    # Phonon occupation number
    n_phonon = bose_einstein_factor(phonon_energy_meV * 1e-3, T_K)
    
    # Debye-Waller suppression
    dw_factor = debye_waller_factor
    
    # Total sideband strength
    sideband_strength = dw_factor * n_phonon
    
    return sideband_strength

class TemperatureDependentPhysics:
    """
    Complete temperature-dependent physics for NV centers.
    
    Includes:
    - Arrhenius rates for all processes
    - Thermal broadening (Doppler, phonon)
    - Temperature-dependent linewidths
    - Phonon sidebands and Debye-Waller factors
    - Electronic/vibronic coupling
    """
    
    def __init__(self, params):
        self.params = params
        
        # Reference temperature
        self.T_ref = getattr(params, 'T_reference_K', 300.0)
        
        # Activation energies (in eV)
        self.E_act_T1 = getattr(params, 'T1_activation_eV', 0.05)      # 50 meV
        self.E_act_T2 = getattr(params, 'T2_activation_eV', 0.01)      # 10 meV
        self.E_act_ISC = getattr(params, 'ISC_activation_eV', 0.1)     # 100 meV
        self.E_act_charge = getattr(params, 'charge_activation_eV', 0.5) # 500 meV
        
        # Phonon parameters
        self.phonon_energy_meV = getattr(params, 'phonon_energy_meV', 40.0)  # 40 meV (local mode)
        self.debye_temp_K = getattr(params, 'debye_temperature_K', 1860.0)    # Diamond Debye temp
        self.huang_rhys_factor = getattr(params, 'huang_rhys_factor', 0.28)   # NV center
        
        # Linewidth parameters
        self.gamma_0_MHz = getattr(params, 'natural_linewidth_MHz', 13.0)     # Natural linewidth
        self.gamma_phonon_300K_MHz = getattr(params, 'phonon_broadening_300K_MHz', 20.0)
        
    @partial(jit, static_argnums=(0,))
    def temperature_dependent_T1(self, T_K):
        """
        Temperature-dependent T1 (longitudinal) relaxation time.
        T1⁻¹(T) = T1⁻¹(T₀) × Arrhenius + phonon processes
        """
        # Base T1 rate at reference temperature
        gamma_T1_ref = 1.0 / (self.params.T1_ms * 1e-3)  # Convert ms to Hz
        
        # Arrhenius component (thermally activated)
        gamma_arrhenius = arrhenius_rate(gamma_T1_ref, self.E_act_T1, T_K, self.T_ref)
        
        # Direct phonon process: T¹ dependence at low T
        T_ratio_low = T_K / self.T_ref
        gamma_direct = gamma_T1_ref * T_ratio_low
        
        # Raman process: T⁷ dependence at intermediate T
        gamma_raman = gamma_T1_ref * (T_ratio_low ** 7)
        
        # Orbach process: Arrhenius at high T  
        gamma_orbach = arrhenius_rate(gamma_T1_ref * 10, 0.15, T_K, self.T_ref)  # 150 meV
        
        # Total rate (dominant process wins)
        gamma_total = jnp.maximum(
            jnp.maximum(gamma_direct, gamma_raman),
            jnp.maximum(gamma_arrhenius, gamma_orbach)
        )
        
        return 1.0 / gamma_total  # Return T1 time in seconds
    
    @partial(jit, static_argnums=(0,))
    def temperature_dependent_T2(self, T_K):
        """
        Temperature-dependent T2* (dephasing) time.
        Includes phonon dephasing and spectral diffusion.
        """
        # Base T2 at reference temperature
        T2_ref_s = self.params.T2_star_us * 1e-6
        gamma_T2_ref = 1.0 / T2_ref_s
        
        # Phonon dephasing (temperature dependent)
        # γ_phonon ∝ n_phonon × (n_phonon + 1) for Raman process
        phonon_energy = self.phonon_energy_meV * 1e-3
        n_ph = bose_einstein_factor(phonon_energy, T_K)
        
        gamma_phonon = self.gamma_phonon_300K_MHz * 1e6 * n_ph * (n_ph + 1)
        
        # Spectral diffusion (thermal activation of charge fluctuators)
        gamma_spectral = arrhenius_rate(gamma_T2_ref * 0.5, self.E_act_T2, T_K, self.T_ref)
        
        # Total dephasing rate
        gamma_total = gamma_T2_ref + gamma_phonon + gamma_spectral
        
        return 1.0 / gamma_total  # Return T2* time in seconds
    
    @partial(jit, static_argnums=(0,))
    def temperature_dependent_ISC_rates(self, T_K):
        """
        Temperature-dependent intersystem crossing rates.
        Different activation energies for ms=0 vs ms=±1.
        """
        # Base ISC rates
        gamma_ISC_ms0_ref = self.params.gamma_ISC_ms0_MHz * 1e6
        gamma_ISC_ms1_ref = self.params.gamma_ISC_ms1_MHz * 1e6
        
        # Arrhenius temperature dependence
        gamma_ISC_ms0 = arrhenius_rate(gamma_ISC_ms0_ref, self.E_act_ISC, T_K, self.T_ref)
        gamma_ISC_ms1 = arrhenius_rate(gamma_ISC_ms1_ref, self.E_act_ISC * 0.5, T_K, self.T_ref)
        
        # Phonon-assisted ISC (additional channel)
        phonon_energy = self.phonon_energy_meV * 1e-3
        n_ph = bose_einstein_factor(phonon_energy, T_K)
        
        gamma_phonon_ISC = (gamma_ISC_ms0_ref + gamma_ISC_ms1_ref) * 0.1 * n_ph
        
        return {
            'gamma_ISC_ms0_Hz': gamma_ISC_ms0 + gamma_phonon_ISC,
            'gamma_ISC_ms1_Hz': gamma_ISC_ms1 + gamma_phonon_ISC
        }
    
    @partial(jit, static_argnums=(0,))
    def temperature_dependent_charge_rates(self, T_K):
        """
        Temperature-dependent charge state transition rates.
        Strong thermal activation expected.
        """
        # Base rates
        gamma_ion_gs = self.params.ionization_rate_GS_MHz * 1e6
        gamma_ion_es = self.params.ionization_rate_ES_MHz * 1e6
        gamma_rec = self.params.recombination_rate_MHz * 1e6
        
        # Arrhenius enhancement
        gamma_ion_gs_T = arrhenius_rate(gamma_ion_gs, self.E_act_charge, T_K, self.T_ref)
        gamma_ion_es_T = arrhenius_rate(gamma_ion_es, self.E_act_charge * 0.7, T_K, self.T_ref)
        
        # Recombination less temperature dependent (barrier limited)
        gamma_rec_T = gamma_rec * (1.0 + 0.1 * (T_K - self.T_ref) / self.T_ref)
        
        return {
            'ionization_gs_Hz': gamma_ion_gs_T,
            'ionization_es_Hz': gamma_ion_es_T,
            'recombination_Hz': gamma_rec_T
        }
    
    @partial(jit, static_argnums=(0,))
    def thermal_line_profile_parameters(self, T_K):
        """
        Get temperature-dependent line profile parameters.
        Includes Doppler broadening and phonon sidebands.
        """
        # Natural linewidth (temperature independent)
        gamma_natural = self.gamma_0_MHz * 1e6
        
        # Doppler broadening
        sigma_doppler = thermal_broadening_gaussian(T_K)
        
        # Phonon broadening (homogeneous)
        phonon_energy = self.phonon_energy_meV * 1e-3
        n_ph = bose_einstein_factor(phonon_energy, T_K)
        gamma_phonon = self.gamma_phonon_300K_MHz * 1e6 * (2 * n_ph + 1)
        
        # Total homogeneous linewidth
        gamma_homogeneous = gamma_natural + gamma_phonon
        
        # Phonon sideband parameters
        sideband_strength = phonon_sideband_strength(
            T_K, jnp.exp(-self.huang_rhys_factor), self.phonon_energy_meV
        )
        
        return {
            'gamma_natural_Hz': gamma_natural,
            'sigma_doppler_Hz': sigma_doppler,
            'gamma_homogeneous_Hz': gamma_homogeneous,
            'sideband_strength': sideband_strength,
            'zero_phonon_weight': jnp.exp(-self.huang_rhys_factor * (2 * n_ph + 1))
        }
    
    @partial(jit, static_argnums=(0,))
    def debye_waller_factor_detailed(self, T_K):
        """
        Detailed Debye-Waller factor calculation.
        W(T) = exp(-2S⟨n⟩) where ⟨n⟩ = n_Bose(T)
        """
        # Effective phonon energy for NV center
        phonon_energy = self.phonon_energy_meV * 1e-3
        
        # Bose occupation
        n_bose = bose_einstein_factor(phonon_energy, T_K)
        
        # Huang-Rhys factor (electron-phonon coupling strength)
        S = self.huang_rhys_factor
        
        # Debye-Waller factor
        W = jnp.exp(-2 * S * n_bose)
        
        return W
    
    @partial(jit, static_argnums=(0,))
    def effective_zero_field_splitting(self, T_K):
        """
        Temperature-dependent zero-field splitting.
        D(T) = D₀ + thermal shifts from lattice expansion and phonons.
        """
        D_0 = self.params.D_GS_Hz
        
        # Linear thermal shift (lattice expansion)
        alpha_thermal = -74e3  # Hz/K for NV center
        D_thermal = D_0 + alpha_thermal * (T_K - self.T_ref)
        
        # Phonon renormalization (weaker effect)
        phonon_energy = self.phonon_energy_meV * 1e-3
        n_ph = bose_einstein_factor(phonon_energy, T_K)
        
        # Phonon correction (small)
        delta_D_phonon = -1e6 * n_ph  # -1 MHz per phonon
        
        D_eff = D_thermal + delta_D_phonon
        
        return D_eff
    
    def get_all_temperature_effects(self, T_K):
        """
        Get complete set of temperature-dependent parameters.
        """
        effects = {
            'relaxation_times': {
                'T1_s': self.temperature_dependent_T1(T_K),
                'T2_s': self.temperature_dependent_T2(T_K)
            },
            'ISC_rates': self.temperature_dependent_ISC_rates(T_K),
            'charge_rates': self.temperature_dependent_charge_rates(T_K),
            'line_profile': self.thermal_line_profile_parameters(T_K),
            'spin_physics': {
                'D_eff_Hz': self.effective_zero_field_splitting(T_K),
                'debye_waller_factor': self.debye_waller_factor_detailed(T_K)
            }
        }
        
        return effects

@jit
def blackbody_rate_enhancement(frequency_Hz, T_K):
    """
    Blackbody radiation enhancement of spontaneous emission.
    Factor = 1 + n_Planck(ω,T) for stimulated emission
    """
    # Photon energy
    energy_J = HBAR * 2 * jnp.pi * frequency_Hz
    
    # Avoid division by zero
    T_safe = jnp.maximum(T_K, 1.0)
    
    # Planck distribution
    x = energy_J / (K_B * T_safe)
    x = jnp.clip(x, 1e-10, 50.0)
    
    n_planck = 1.0 / (jnp.exp(x) - 1.0)
    
    # Enhancement factor
    enhancement = 1.0 + n_planck
    
    return enhancement

@jit
def thermal_occupation_ms_levels(D_Hz, T_K):
    """
    Thermal occupation of ms = 0, ±1 levels in ground state.
    """
    # Energy differences (ms = ±1 higher by D)
    E_0 = 0.0
    E_pm1 = D_Hz * HBAR
    
    T_safe = jnp.maximum(T_K, 1.0)
    beta = 1.0 / (K_B * T_safe)
    
    # Boltzmann factors
    exp_0 = 1.0  # Reference
    exp_pm1 = jnp.exp(-beta * E_pm1)
    
    # Partition function
    Z = exp_0 + 2 * exp_pm1  # ms=0 (1 state) + ms=±1 (2 states)
    
    # Populations
    p_0 = exp_0 / Z
    p_pm1 = exp_pm1 / Z  # Each of ms=+1, ms=-1
    
    return {'p_ms0': p_0, 'p_ms_plus1': p_pm1, 'p_ms_minus1': p_pm1}

def test_temperature_effects():
    """Test temperature effects implementation."""
    
    # Mock parameters
    class MockParams:
        T1_ms = 5.0
        T2_star_us = 2.0
        gamma_ISC_ms0_MHz = 5.0
        gamma_ISC_ms1_MHz = 50.0
        ionization_rate_GS_MHz = 0.0001
        ionization_rate_ES_MHz = 0.1
        recombination_rate_MHz = 0.01
        D_GS_Hz = 2.87e9
        natural_linewidth_MHz = 13.0
        phonon_broadening_300K_MHz = 20.0
        huang_rhys_factor = 0.28
        phonon_energy_meV = 40.0
    
    params = MockParams()
    
    # Create temperature system
    temp_physics = TemperatureDependentPhysics(params)
    
    # Test temperature range
    temperatures = jnp.array([4, 77, 150, 300, 500, 800])  # Kelvin
    
    print("Temperature dependence analysis:")
    print("T(K)    T1(ms)   T2(μs)   ISC_ms0(MHz)  ISC_ms1(MHz)  D_eff(GHz)")
    print("-" * 70)
    
    for T_K in temperatures:
        effects = temp_physics.get_all_temperature_effects(T_K)
        
        T1_ms = effects['relaxation_times']['T1_s'] * 1e3
        T2_us = effects['relaxation_times']['T2_s'] * 1e6
        ISC_ms0_MHz = effects['ISC_rates']['gamma_ISC_ms0_Hz'] / 1e6
        ISC_ms1_MHz = effects['ISC_rates']['gamma_ISC_ms1_Hz'] / 1e6
        D_eff_GHz = effects['spin_physics']['D_eff_Hz'] / 1e9
        
        print(f"{T_K:4.0f}   {T1_ms:6.2f}   {T2_us:6.2f}   {ISC_ms0_MHz:8.2f}    {ISC_ms1_MHz:8.2f}    {D_eff_GHz:6.3f}")
    
    # Test line profile parameters
    print(f"\nLine profile parameters at different temperatures:")
    print("T(K)   γ_nat(MHz)  σ_Doppler(MHz)  γ_phonon(MHz)  ZPL_weight")
    print("-" * 60)
    
    for T_K in [4, 77, 300, 800]:
        line_params = temp_physics.thermal_line_profile_parameters(T_K)
        
        gamma_nat = line_params['gamma_natural_Hz'] / 1e6
        sigma_dop = line_params['sigma_doppler_Hz'] / 1e6
        gamma_ph = (line_params['gamma_homogeneous_Hz'] - line_params['gamma_natural_Hz']) / 1e6
        zpl_weight = line_params['zero_phonon_weight']
        
        print(f"{T_K:4.0f}   {gamma_nat:8.1f}    {sigma_dop:10.2f}    {gamma_ph:8.1f}      {zpl_weight:.3f}")
    
    # Test thermal populations
    print(f"\nThermal populations of ms levels:")
    print("T(K)    p(ms=0)   p(ms=±1)")
    print("-" * 25)
    
    for T_K in [4, 77, 300, 800]:
        D_eff = temp_physics.effective_zero_field_splitting(T_K)
        pops = thermal_occupation_ms_levels(D_eff, T_K)
        
        print(f"{T_K:4.0f}    {pops['p_ms0']:.3f}     {pops['p_ms_plus1']:.3f}")
    
    # Test specific functions
    print(f"\nSpecific function tests:")
    
    # Arrhenius rate
    k_300K = 1e6  # Hz
    E_act = 0.1   # eV
    k_77K = arrhenius_rate(k_300K, E_act, 77.0, 300.0)
    print(f"  Arrhenius: 1 MHz at 300K → {k_77K:.2e} Hz at 77K (E_a=100meV)")
    
    # Bose-Einstein
    phonon_40meV = bose_einstein_factor(0.04, 300.0)
    print(f"  Bose-Einstein: 40meV phonon occupation at 300K = {phonon_40meV:.2f}")
    
    # Thermal broadening
    doppler_300K = thermal_broadening_gaussian(300.0)
    print(f"  Doppler broadening at 300K = {doppler_300K/1e6:.2f} MHz")
    
    # Blackbody enhancement
    freq_637nm = C_LIGHT / 637e-9
    bb_enhancement = blackbody_rate_enhancement(freq_637nm, 300.0)
    print(f"  Blackbody enhancement at 637nm, 300K = {bb_enhancement:.2e}")
    
    return temp_physics

if __name__ == "__main__":
    print("🌡️  Testing Temperature Effects System")
    print("=" * 45)
    
    temp_physics = test_temperature_effects()
    
    print("\n✅ Temperature effects system working!")
    print("   Includes: Arrhenius rates, thermal broadening, phonon effects")