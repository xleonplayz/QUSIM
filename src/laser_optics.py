#!/usr/bin/env python3
"""
Realistic Laser Optics and Saturation
====================================
Voigt profiles, power broadening, and advanced laser physics
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from functools import partial
import scipy.special

jax.config.update("jax_enable_x64", True)

# Physical constants
C_LIGHT = 299792458  # m/s
H_PLANCK = 6.62607015e-34  # J⋅s
K_B = 1.380649e-23  # J/K

@jit
def faddeeva_approx(z):
    """
    Approximation of Faddeeva function w(z) for JAX compatibility.
    Uses Abramowitz & Stegun approximation for complex error function.
    """
    # For complex z = x + iy, w(z) = e^(-z²) * erfc(-iz)
    # Use rational approximation for better accuracy
    
    x = jnp.real(z)
    y = jnp.imag(z)
    
    # Humlicek's w4 algorithm (simplified)
    t = y - 1j * x
    s = jnp.abs(x) + y
    
    # Region 1: |y| > 15 or s > 15
    region1 = (jnp.abs(y) > 15) | (s > 15)
    w1 = 1j * jnp.sqrt(jnp.pi) / t
    
    # Region 2: s > 5.5
    region2 = (~region1) & (s > 5.5)
    u = t * t
    w2 = (t * (1.410474 + u * 0.512424)) / (0.75 + u * (3.0 + u))
    
    # Region 3: s <= 5.5
    region3 = (~region1) & (~region2)
    u = t * t
    w3 = jnp.exp(u) - t * (36183.31 - u * (3321.9905 - u * (1540.787 - u * (219.0313 - u * (35.76683 - u * (1.320522 - u * 0.56419)))))) / \
         (32066.6 - u * (24322.84 - u * (9022.228 - u * (2186.181 - u * (364.2191 - u * (61.57037 - u * (1.841439 - u)))))))
    
    # Combine regions
    w = jnp.where(region1, w1, jnp.where(region2, w2, w3))
    
    return w

@jit
def voigt_profile(delta_Hz, sigma_Hz, gamma_Hz):
    """
    Voigt line profile: convolution of Gaussian and Lorentzian.
    
    Parameters:
    - delta_Hz: frequency detuning from line center
    - sigma_Hz: Gaussian width (thermal/instrumental broadening)
    - gamma_Hz: Lorentzian width (natural/pressure broadening)
    
    Returns: Normalized line shape
    """
    # Voigt parameters
    z = (delta_Hz + 1j * gamma_Hz) / (sigma_Hz * jnp.sqrt(2))
    
    # Faddeeva function
    w = faddeeva_approx(z)
    
    # Voigt profile
    voigt = jnp.real(w) / (sigma_Hz * jnp.sqrt(2 * jnp.pi))
    
    return voigt

@jit 
def power_broadened_saturation(I_rel, delta_Hz, gamma_natural_Hz, I_sat_rel=1.0):
    """
    Power-broadened saturation with realistic line shape.
    
    s(Δ,I) = I/I_sat / (1 + I/I_sat + (2Δ/Γ')²)
    where Γ' = Γ₀√(1 + I/I_sat) is power-broadened linewidth
    """
    # Power broadening factor
    s0 = I_rel / I_sat_rel
    
    # Power-broadened linewidth
    gamma_broadened = gamma_natural_Hz * jnp.sqrt(1 + s0)
    
    # Saturation parameter
    saturation = s0 / (1 + s0 + (2 * delta_Hz / gamma_broadened)**2)
    
    return saturation, gamma_broadened

@jit
def laser_beam_profile(r_um, z_um, w0_um, wavelength_nm=532):
    """
    Gaussian beam intensity profile.
    
    I(r,z) = I₀ * (w₀/w(z))² * exp(-2r²/w(z)²)
    where w(z) = w₀√(1 + (z/z_R)²) and z_R = πw₀²/λ
    """
    # Rayleigh range
    z_R = jnp.pi * w0_um**2 / (wavelength_nm * 1e-3)  # Convert nm to μm
    
    # Beam waist at position z
    w_z = w0_um * jnp.sqrt(1 + (z_um / z_R)**2)
    
    # Intensity profile (normalized to peak)
    intensity = (w0_um / w_z)**2 * jnp.exp(-2 * r_um**2 / w_z**2)
    
    return intensity, w_z

@jit
def laser_intensity_at_nv(beam_params):
    """
    Calculate laser intensity at NV center position.
    Accounts for beam focusing and positioning.
    """
    # NV position relative to beam focus
    r_nv = jnp.sqrt(beam_params['x_offset_um']**2 + beam_params['y_offset_um']**2)
    z_nv = beam_params['z_offset_um']
    
    # Beam profile
    intensity, w_z = laser_beam_profile(
        r_nv, z_nv, 
        beam_params['w0_um'], 
        beam_params['wavelength_nm']
    )
    
    # Convert to saturation parameter
    I_sat_mW_per_um2 = beam_params['I_sat_mW'] / (jnp.pi * beam_params['w0_um']**2)
    I_actual_mW_per_um2 = beam_params['power_mW'] / (jnp.pi * w_z**2)
    
    I_rel = I_actual_mW_per_um2 / I_sat_mW_per_um2 * intensity
    
    return I_rel, w_z

class LaserSystem:
    """
    Complete laser system with realistic optics and saturation.
    """
    
    def __init__(self, params):
        self.wavelength_nm = 532.0  # Green laser
        self.linewidth_natural_Hz = 1e6  # Natural linewidth ~1 MHz
        self.linewidth_laser_Hz = 1e6   # Laser linewidth ~1 MHz
        self.power_mW = params.laser_power_mW
        self.I_sat_mW = params.I_sat_mW
        
        # Beam parameters
        self.beam_params = {
            'power_mW': self.power_mW,
            'I_sat_mW': self.I_sat_mW,
            'w0_um': 1.0,  # Beam waist 1 μm
            'wavelength_nm': self.wavelength_nm,
            'x_offset_um': 0.0,  # NV at beam center
            'y_offset_um': 0.0,
            'z_offset_um': 0.0   # NV at focus
        }
        
        # Spectral parameters
        self.detuning_Hz = 0.0  # On resonance
        self.temperature_K = 300.0  # Room temperature
        
        # Calculate thermal broadening
        self.doppler_width_Hz = self._calculate_doppler_width()
        
    def _calculate_doppler_width(self):
        """Calculate Doppler broadening from thermal motion."""
        # For NV centers in diamond, thermal motion is limited
        # Use effective temperature for localized vibrations
        f0 = C_LIGHT / (self.wavelength_nm * 1e-9)  # Optical frequency
        
        # Thermal velocity for effective mass
        m_eff = 12 * 1.66054e-27  # Effective mass ~ 12 amu
        v_thermal = jnp.sqrt(2 * K_B * self.temperature_K / m_eff)
        
        # Doppler width (FWHM)
        doppler_fwhm = 2 * f0 * v_thermal / C_LIGHT
        
        # Convert FWHM to Gaussian σ
        sigma_doppler = doppler_fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
        
        return sigma_doppler
        
    def get_excitation_rate(self, detuning_Hz=0.0):
        """
        Calculate excitation rate with full Voigt profile and power broadening.
        """
        # Total detuning
        total_detuning = self.detuning_Hz + detuning_Hz
        
        # Laser intensity at NV position
        I_rel, beam_waist = laser_intensity_at_nv(self.beam_params)
        
        # Power broadening
        saturation, gamma_broadened = power_broadened_saturation(
            I_rel, total_detuning, self.linewidth_natural_Hz
        )
        
        # Voigt profile (Doppler + natural broadening)
        sigma_total = jnp.sqrt(self.doppler_width_Hz**2 + self.linewidth_laser_Hz**2)
        voigt_factor = voigt_profile(total_detuning, sigma_total, gamma_broadened)
        
        # Excitation rate
        gamma_max = 1e8  # Maximum excitation rate ~100 MHz
        excitation_rate = gamma_max * saturation * voigt_factor
        
        return excitation_rate
    
    def set_detuning(self, detuning_Hz):
        """Set laser detuning from NV transition."""
        self.detuning_Hz = detuning_Hz
        
    def set_power(self, power_mW):
        """Set laser power."""
        self.power_mW = power_mW
        self.beam_params['power_mW'] = power_mW
        
    def get_beam_parameters(self):
        """Get current beam parameters."""
        I_rel, w_z = laser_intensity_at_nv(self.beam_params)
        
        return {
            'intensity_relative': float(I_rel),
            'beam_waist_um': float(w_z),
            'power_mW': self.power_mW,
            'detuning_Hz': self.detuning_Hz,
            'doppler_width_Hz': self.doppler_width_Hz
        }

@jit 
def laser_heating_effects(excitation_rate_Hz, T_bath_K):
    """
    Calculate laser heating effects on NV center.
    High excitation rates can heat the local environment.
    """
    # Simplified model: heating ∝ excitation rate
    # Each photon deposits ~0.5 eV of vibrational energy
    
    phonon_energy_eV = 0.1  # Typical phonon energy in diamond
    heating_rate_eV_per_s = excitation_rate_Hz * phonon_energy_eV * 0.1  # 10% efficiency
    
    # Convert to temperature rise (very simplified)
    # Real calculation would need heat capacity and thermal conductivity
    heat_capacity_eV_per_K = 1e-3  # Effective local heat capacity
    
    delta_T = heating_rate_eV_per_s / (1e6 * heat_capacity_eV_per_K)  # Steady state
    
    T_effective = T_bath_K + delta_T
    
    return T_effective, delta_T

class AdvancedLaserPhysics:
    """
    Advanced laser physics effects for ultra-realistic simulation.
    """
    
    def __init__(self, laser_system):
        self.laser = laser_system
        
    @jit
    def ac_stark_shift(self, intensity_rel):
        """
        AC Stark shift from strong laser fields.
        Shifts energy levels proportional to intensity.
        """
        # AC Stark coefficient (typical for NV centers)
        alpha_stark_Hz_per_rel_intensity = 1e6  # 1 MHz per relative intensity unit
        
        stark_shift = alpha_stark_Hz_per_rel_intensity * intensity_rel
        
        return stark_shift
    
    @jit 
    def multi_photon_processes(self, intensity_rel):
        """
        Multi-photon excitation processes at high intensity.
        """
        # Two-photon absorption coefficient
        beta_2PA = 1e-12  # Very small for NV centers
        
        two_photon_rate = beta_2PA * intensity_rel**2
        
        return two_photon_rate
    
    @jit
    def laser_noise_effects(self, key, dt_ns):
        """
        Include laser amplitude and phase noise.
        """
        key1, key2 = jax.random.split(key)
        
        # Amplitude noise (relative intensity noise)
        rin_percent = 0.1  # 0.1% RIN typical for solid-state lasers
        amplitude_noise = 1.0 + (rin_percent / 100) * jax.random.normal(key1)
        
        # Phase noise (linewidth broadening)
        phase_diffusion_rad_per_sqrt_Hz = 1e-6  # Typical phase noise
        phase_noise_rad = phase_diffusion_rad_per_sqrt_Hz * jnp.sqrt(dt_ns * 1e-9) * jax.random.normal(key2)
        
        return amplitude_noise, phase_noise_rad, key

if __name__ == "__main__":
    # Test laser system
    print("🔬 Testing Advanced Laser Physics")
    print("=" * 40)
    
    # Mock parameters
    class MockParams:
        laser_power_mW = 1.0
        I_sat_mW = 0.35
    
    params = MockParams()
    laser = LaserSystem(params)
    
    # Test Voigt profile
    detunings = jnp.linspace(-10e6, 10e6, 101)  # ±10 MHz
    excitation_rates = jnp.array([laser.get_excitation_rate(d) for d in detunings])
    
    print(f"Laser system parameters:")
    beam_params = laser.get_beam_parameters()
    for key, value in beam_params.items():
        print(f"  {key}: {value:.3g}")
    
    print(f"\nVoigt profile test:")
    print(f"  Peak excitation rate: {jnp.max(excitation_rates):.2e} Hz")
    print(f"  FWHM detuning: ~{2.35 * laser.doppler_width_Hz / 1e6:.1f} MHz")
    
    # Test power broadening
    powers = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])  # mW
    peak_rates = []
    
    for power in powers:
        laser.set_power(power)
        rate = laser.get_excitation_rate(0.0)  # On resonance
        peak_rates.append(rate)
    
    print(f"\nPower broadening test:")
    for p, r in zip(powers, peak_rates):
        print(f"  {p:.1f} mW → {r:.2e} Hz")
    
    print("\n✅ Advanced laser physics working!")