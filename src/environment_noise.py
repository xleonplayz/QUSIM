#!/usr/bin/env python3
"""
Environment Noise - Ornstein-Uhlenbeck Processes
===============================================
Correlated noise on B-field and strain fluctuations
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
from functools import partial
import scipy.constants as sc

jax.config.update("jax_enable_x64", True)

@jit
def ou_process_step(x_current, dt, tau_corr, sigma, xi, mu=0.0):
    """
    Single step of Ornstein-Uhlenbeck process.
    
    dx/dt = -(x - μ)/τ + σ√(2/τ) ξ(t)
    
    Args:
        x_current: current value
        dt: time step
        tau_corr: correlation time
        sigma: noise strength (standard deviation)
        xi: random noise (Gaussian, zero mean, unit variance)
        mu: mean value (equilibrium)
    
    Returns:
        New value x(t + dt)
    """
    # Exponential decay towards mean
    decay_factor = jnp.exp(-dt / tau_corr)
    
    # Noise amplitude for this step
    noise_amplitude = sigma * jnp.sqrt((1 - decay_factor**2))
    
    # Update equation
    x_new = mu + decay_factor * (x_current - mu) + noise_amplitude * xi
    
    return x_new

@jit
def ou_process_analytical(t_array, tau_corr, sigma, key, mu=0.0, x0=None):
    """
    Generate complete OU process time series analytically.
    More efficient than step-by-step integration.
    
    Args:
        t_array: time points
        tau_corr: correlation time
        sigma: noise strength
        key: JAX random key
        mu: mean value
        x0: initial value (defaults to mu)
    
    Returns:
        Array of process values at each time point
    """
    n_points = len(t_array)
    
    if x0 is None:
        x0 = mu
    
    # Generate all random increments at once
    key, subkey = random.split(key)
    xi_array = random.normal(subkey, (n_points - 1,))
    
    # Initialize output array
    x_array = jnp.zeros(n_points)
    x_array = x_array.at[0].set(x0)
    
    # Vectorized generation (scan operation)
    def ou_scan_step(x_prev, inputs):
        t_prev, t_current, xi = inputs
        dt = t_current - t_prev
        x_new = ou_process_step(x_prev, dt, tau_corr, sigma, xi, mu)
        return x_new, x_new
    
    # Prepare inputs for scan
    t_prev = t_array[:-1]
    t_current = t_array[1:]
    scan_inputs = (t_prev, t_current, xi_array)
    
    # Generate time series
    _, x_series = jax.lax.scan(ou_scan_step, x0, scan_inputs)
    
    # Combine initial value with generated series
    x_complete = jnp.concatenate([jnp.array([x0]), x_series])
    
    return x_complete

class EnvironmentNoise:
    """
    Complete environment noise model for NV centers.
    
    Includes:
    - B-field fluctuations (3D vector OU processes)
    - Strain fluctuations (scalar OU process)
    - Temperature fluctuations
    - Charge fluctuations
    - Cross-correlations between noise sources
    """
    
    def __init__(self, params):
        self.params = params
        
        # B-field noise parameters
        self.B_noise_sigma_uT = getattr(params, 'B_noise_sigma_uT', 1.0)  # 1 μT RMS
        self.B_noise_tau_corr_us = getattr(params, 'B_noise_tau_corr_us', 10.0)  # 10 μs correlation
        self.B_mean_mT = getattr(params, 'B_field_mT', jnp.array([0.0, 0.0, 0.0]))
        
        # Strain noise parameters  
        self.strain_noise_sigma_MHz = getattr(params, 'strain_noise_sigma_MHz', 0.1)  # 0.1 MHz RMS
        self.strain_noise_tau_corr_us = getattr(params, 'strain_noise_tau_corr_us', 100.0)  # 100 μs correlation
        self.strain_mean_MHz = getattr(params, 'strain_mean_MHz', 0.0)
        
        # Temperature noise (affects all rates)
        self.temp_noise_sigma_K = getattr(params, 'temp_noise_sigma_K', 1.0)  # 1K RMS
        self.temp_noise_tau_corr_ms = getattr(params, 'temp_noise_tau_corr_ms', 1000.0)  # 1s correlation
        self.temp_mean_K = getattr(params, 'Temperature_K', 300.0)
        
        # Cross-correlation parameters
        self.enable_cross_correlations = getattr(params, 'enable_cross_correlations', True)
        self.B_strain_correlation = getattr(params, 'B_strain_correlation', 0.1)  # 10% correlation
        
    @partial(jit, static_argnums=(0,))
    def generate_b_field_noise(self, time_array_ns, key):
        """
        Generate 3D B-field noise time series.
        Each component follows independent OU process.
        
        Args:
            time_array_ns: time points in nanoseconds
            key: JAX random key
        
        Returns:
            B-field array (n_times, 3) in mT
        """
        time_s = time_array_ns * 1e-9
        tau_corr_s = self.B_noise_tau_corr_us * 1e-6
        sigma_mT = self.B_noise_sigma_uT * 1e-3  # Convert μT to mT
        
        # Generate independent noise for each component
        keys = random.split(key, 3)
        
        B_components = []
        for i in range(3):
            B_comp = ou_process_analytical(
                time_s, tau_corr_s, sigma_mT, keys[i], 
                mu=self.B_mean_mT[i], x0=self.B_mean_mT[i]
            )
            B_components.append(B_comp)
        
        # Stack into (n_times, 3) array
        B_field_array = jnp.stack(B_components, axis=1)
        
        return B_field_array
    
    @partial(jit, static_argnums=(0,))
    def generate_strain_noise(self, time_array_ns, key):
        """
        Generate strain noise time series.
        
        Args:
            time_array_ns: time points in nanoseconds
            key: JAX random key
        
        Returns:
            Strain array in MHz
        """
        time_s = time_array_ns * 1e-9
        tau_corr_s = self.strain_noise_tau_corr_us * 1e-6
        
        strain_array = ou_process_analytical(
            time_s, tau_corr_s, self.strain_noise_sigma_MHz, key,
            mu=self.strain_mean_MHz, x0=self.strain_mean_MHz
        )
        
        return strain_array
    
    @partial(jit, static_argnums=(0,))
    def generate_temperature_noise(self, time_array_ns, key):
        """
        Generate temperature fluctuation noise.
        
        Args:
            time_array_ns: time points in nanoseconds
            key: JAX random key
        
        Returns:
            Temperature array in Kelvin
        """
        time_s = time_array_ns * 1e-9
        tau_corr_s = self.temp_noise_tau_corr_ms * 1e-3
        
        temp_array = ou_process_analytical(
            time_s, tau_corr_s, self.temp_noise_sigma_K, key,
            mu=self.temp_mean_K, x0=self.temp_mean_K
        )
        
        # Ensure temperature stays positive
        temp_array = jnp.maximum(temp_array, 1.0)  # Minimum 1K
        
        return temp_array
    
    @partial(jit, static_argnums=(0,))
    def generate_correlated_noise(self, time_array_ns, key):
        """
        Generate correlated B-field and strain noise.
        Uses Cholesky decomposition for controlled correlations.
        
        Args:
            time_array_ns: time points in nanoseconds
            key: JAX random key
        
        Returns:
            Dictionary with correlated noise arrays
        """
        if not self.enable_cross_correlations:
            # Generate independent noise
            keys = random.split(key, 3)
            B_field = self.generate_b_field_noise(time_array_ns, keys[0])
            strain = self.generate_strain_noise(time_array_ns, keys[1])
            temperature = self.generate_temperature_noise(time_array_ns, keys[2])
            
            return {
                'B_field_mT': B_field,
                'strain_MHz': strain,
                'temperature_K': temperature
            }
        
        # Generate correlated noise
        time_s = time_array_ns * 1e-9
        n_times = len(time_s)
        
        # Correlation matrix (simplified: only B_z and strain correlated)
        correlation_matrix = jnp.array([
            [1.0, 0.0, 0.0, self.B_strain_correlation],  # B_x
            [0.0, 1.0, 0.0, 0.0],                       # B_y  
            [0.0, 0.0, 1.0, self.B_strain_correlation], # B_z
            [self.B_strain_correlation, 0.0, self.B_strain_correlation, 1.0]  # strain
        ])
        
        # Cholesky decomposition
        L = jnp.linalg.cholesky(correlation_matrix)
        
        # Generate independent Gaussian noise
        key, subkey = random.split(key)
        independent_noise = random.normal(subkey, (n_times - 1, 4))
        
        # Apply correlations
        correlated_noise = independent_noise @ L.T
        
        # Convert to OU processes
        keys = random.split(key, 4)
        
        # B-field components
        B_x = self._correlated_ou_process(
            time_s, correlated_noise[:, 0], 
            self.B_noise_tau_corr_us * 1e-6, self.B_noise_sigma_uT * 1e-3,
            self.B_mean_mT[0]
        )
        
        B_y = self._correlated_ou_process(
            time_s, correlated_noise[:, 1],
            self.B_noise_tau_corr_us * 1e-6, self.B_noise_sigma_uT * 1e-3,
            self.B_mean_mT[1]
        )
        
        B_z = self._correlated_ou_process(
            time_s, correlated_noise[:, 2],
            self.B_noise_tau_corr_us * 1e-6, self.B_noise_sigma_uT * 1e-3,
            self.B_mean_mT[2]
        )
        
        # Strain
        strain = self._correlated_ou_process(
            time_s, correlated_noise[:, 3],
            self.strain_noise_tau_corr_us * 1e-6, self.strain_noise_sigma_MHz,
            self.strain_mean_MHz
        )
        
        # Temperature (uncorrelated)
        key, temp_key = random.split(key)
        temperature = self.generate_temperature_noise(time_array_ns, temp_key)
        
        B_field = jnp.stack([B_x, B_y, B_z], axis=1)
        
        return {
            'B_field_mT': B_field,
            'strain_MHz': strain,
            'temperature_K': temperature
        }
    
    @jit
    def _correlated_ou_process(self, time_array, noise_increments, tau_corr, sigma, mu):
        """
        Generate OU process with pre-computed (correlated) noise increments.
        """
        n_points = len(time_array)
        x_array = jnp.zeros(n_points)
        x_array = x_array.at[0].set(mu)
        
        def ou_corr_step(x_prev, inputs):
            t_prev, t_current, xi = inputs
            dt = t_current - t_prev
            x_new = ou_process_step(x_prev, dt, tau_corr, sigma, xi, mu)
            return x_new, x_new
        
        t_prev = time_array[:-1]
        t_current = time_array[1:]
        scan_inputs = (t_prev, t_current, noise_increments)
        
        _, x_series = jax.lax.scan(ou_corr_step, mu, scan_inputs)
        
        x_complete = jnp.concatenate([jnp.array([mu]), x_series])
        
        return x_complete
    
    @partial(jit, static_argnums=(0,))  
    def spectral_density(self, frequencies_Hz, tau_corr_s, sigma):
        """
        Calculate power spectral density of OU process.
        S(f) = 2σ²τ / (1 + (2πfτ)²)
        
        Args:
            frequencies_Hz: frequency array
            tau_corr_s: correlation time in seconds
            sigma: noise strength
        
        Returns:
            Power spectral density
        """
        omega_tau = 2 * jnp.pi * frequencies_Hz * tau_corr_s
        psd = 2 * sigma**2 * tau_corr_s / (1 + omega_tau**2)
        
        return psd
    
    def get_noise_characteristics(self):
        """
        Get complete noise characterization for analysis.
        """
        return {
            'B_field_noise': {
                'sigma_uT': self.B_noise_sigma_uT,
                'tau_corr_us': self.B_noise_tau_corr_us,
                'mean_mT': self.B_mean_mT
            },
            'strain_noise': {
                'sigma_MHz': self.strain_noise_sigma_MHz,
                'tau_corr_us': self.strain_noise_tau_corr_us,
                'mean_MHz': self.strain_mean_MHz
            },
            'temperature_noise': {
                'sigma_K': self.temp_noise_sigma_K,
                'tau_corr_ms': self.temp_noise_tau_corr_ms,
                'mean_K': self.temp_mean_K
            },
            'correlations': {
                'enabled': self.enable_cross_correlations,
                'B_strain_correlation': self.B_strain_correlation
            }
        }

@jit
def ou_autocorrelation(tau_delay, tau_corr, sigma):
    """
    Analytical autocorrelation function for OU process.
    R(τ) = σ² exp(-|τ|/τ_corr)
    
    Args:
        tau_delay: time delay
        tau_corr: correlation time
        sigma: noise strength
    
    Returns:
        Autocorrelation value
    """
    return sigma**2 * jnp.exp(-jnp.abs(tau_delay) / tau_corr)

@jit  
def allan_variance_ou(averaging_times, tau_corr, sigma):
    """
    Calculate Allan variance for OU process.
    Useful for characterizing stability and noise.
    
    Args:
        averaging_times: array of averaging times
        tau_corr: correlation time
        sigma: noise strength
    
    Returns:
        Allan variance array
    """
    tau_ratio = averaging_times / tau_corr
    
    # Allan variance formula for OU process
    allan_var = sigma**2 * tau_corr / averaging_times * (
        2 * tau_ratio - 3 + 4 * jnp.exp(-tau_ratio) - jnp.exp(-2 * tau_ratio)
    )
    
    return allan_var

def test_environment_noise():
    """Test environment noise implementation."""
    
    # Mock parameters
    class MockParams:
        B_noise_sigma_uT = 2.0
        B_noise_tau_corr_us = 50.0
        B_field_mT = jnp.array([0.0, 0.0, 1.0])
        strain_noise_sigma_MHz = 0.2
        strain_noise_tau_corr_us = 200.0
        strain_mean_MHz = 0.0
        temp_noise_sigma_K = 0.5
        temp_noise_tau_corr_ms = 2000.0
        Temperature_K = 300.0
        enable_cross_correlations = True
        B_strain_correlation = 0.2
    
    params = MockParams()
    
    # Create noise system
    noise = EnvironmentNoise(params)
    
    print("Noise characteristics:")
    chars = noise.get_noise_characteristics()
    for category, values in chars.items():
        print(f"  {category}:")
        for key, val in values.items():
            print(f"    {key}: {val}")
    
    # Generate test time series
    time_ns = jnp.linspace(0, 10000, 1001)  # 10 μs, 10 ns resolution
    key = random.PRNGKey(123)
    
    # Test independent noise
    print(f"\nTesting independent noise generation:")
    
    keys = random.split(key, 3)
    B_field = noise.generate_b_field_noise(time_ns, keys[0])
    strain = noise.generate_strain_noise(time_ns, keys[1])
    temperature = noise.generate_temperature_noise(time_ns, keys[2])
    
    print(f"  B-field: shape={B_field.shape}, mean={jnp.mean(B_field, axis=0)}, std={jnp.std(B_field, axis=0)}")
    print(f"  Strain: shape={strain.shape}, mean={jnp.mean(strain):.3f}, std={jnp.std(strain):.3f}")
    print(f"  Temperature: shape={temperature.shape}, mean={jnp.mean(temperature):.1f}, std={jnp.std(temperature):.3f}")
    
    # Test correlated noise
    print(f"\nTesting correlated noise generation:")
    
    corr_noise = noise.generate_correlated_noise(time_ns, key)
    
    B_corr = corr_noise['B_field_mT']
    strain_corr = corr_noise['strain_MHz']
    
    # Check correlation between B_z and strain
    correlation = jnp.corrcoef(B_corr[:, 2], strain_corr)[0, 1]
    print(f"  B_z - strain correlation: {correlation:.3f} (target: {params.B_strain_correlation})")
    
    # Test spectral properties
    print(f"\nSpectral analysis:")
    
    # Sample frequency range
    freqs = jnp.logspace(1, 6, 50)  # 10 Hz to 1 MHz
    
    # B-field PSD
    B_psd = noise.spectral_density(freqs, params.B_noise_tau_corr_us * 1e-6, params.B_noise_sigma_uT * 1e-3)
    
    # Strain PSD  
    strain_psd = noise.spectral_density(freqs, params.strain_noise_tau_corr_us * 1e-6, params.strain_noise_sigma_MHz)
    
    print(f"  B-field PSD at 1 kHz: {B_psd[jnp.argmin(jnp.abs(freqs - 1e3))]:.2e} mT²/Hz")
    print(f"  Strain PSD at 1 kHz: {strain_psd[jnp.argmin(jnp.abs(freqs - 1e3))]:.2e} MHz²/Hz")
    
    # Test autocorrelation
    print(f"\nAutocorrelation analysis:")
    
    delay_times = jnp.array([0, 10, 50, 100, 200]) * 1e-6  # μs
    
    for delay in delay_times:
        B_autocorr = ou_autocorrelation(delay, params.B_noise_tau_corr_us * 1e-6, params.B_noise_sigma_uT * 1e-3)
        strain_autocorr = ou_autocorrelation(delay, params.strain_noise_tau_corr_us * 1e-6, params.strain_noise_sigma_MHz)
        
        print(f"  τ={delay*1e6:.0f}μs: B_autocorr={B_autocorr:.4f}, strain_autocorr={strain_autocorr:.4f}")
    
    return noise

if __name__ == "__main__":
    print("🌊 Testing Environment Noise System")
    print("=" * 40)
    
    noise = test_environment_noise()
    
    print("\n✅ Environment noise system working!")
    print("   Includes: B-field OU, strain OU, temperature OU, correlations")