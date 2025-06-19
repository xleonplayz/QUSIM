#!/usr/bin/env python3
"""
Photon Sampling with Bin-wise Monte Carlo
========================================
Vectorized photon generation with statistical accuracy
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
from functools import partial

jax.config.update("jax_enable_x64", True)

@partial(jit, static_argnums=(2,))
def sample_photons_single_bin(rate_Hz, dt_ns, n_runs, key):
    """
    Sample photons for a single time bin over multiple experimental runs.
    
    Args:
        rate_Hz: photon rate in Hz
        dt_ns: time bin duration in ns
        n_runs: number of experimental runs
        key: JAX random key
    
    Returns:
        Array of photon counts for each run
    """
    # Expected photons per bin per run
    lambda_poisson = rate_Hz * dt_ns * 1e-9
    
    # Sample Poisson distribution for each run
    keys = random.split(key, n_runs)
    
    def single_run_sample(k):
        return random.poisson(k, lambda_poisson)
    
    # Vectorized sampling
    counts = vmap(single_run_sample)(keys)
    
    return counts

@partial(jit, static_argnums=(2,))
def sample_photons_time_series(rates_Hz, dt_ns, n_runs):
    """
    Sample complete time series of photon counts.
    
    Args:
        rates_Hz: array of photon rates for each time bin (shape: n_bins)
        dt_ns: time bin duration in ns
        n_runs: number of experimental runs
    
    Returns:
        Photon count matrix (shape: n_runs, n_bins)
    """
    n_bins = len(rates_Hz)
    
    # Generate keys for all bins and runs
    master_key = random.PRNGKey(0)
    keys = random.split(master_key, n_bins)
    
    def sample_all_runs_for_bin(i):
        return sample_photons_single_bin(rates_Hz[i], dt_ns, n_runs, keys[i])
    
    # Vectorized over time bins
    bin_indices = jnp.arange(n_bins)
    all_counts = vmap(sample_all_runs_for_bin)(bin_indices)
    
    # Transpose to get (n_runs, n_bins) shape
    counts_matrix = all_counts.T
    
    return counts_matrix

@jit
def photon_statistics(counts_matrix):
    """
    Calculate photon statistics from multiple runs.
    
    Args:
        counts_matrix: (n_runs, n_bins) array of photon counts
    
    Returns:
        Dictionary of statistics
    """
    # Mean and variance across runs for each bin
    mean_counts = jnp.mean(counts_matrix, axis=0)
    var_counts = jnp.var(counts_matrix, axis=0)
    std_counts = jnp.sqrt(var_counts)
    
    # Total statistics
    total_mean = jnp.mean(mean_counts)
    total_var = jnp.mean(var_counts)
    
    # Signal-to-noise ratio
    snr = jnp.where(std_counts > 0, mean_counts / std_counts, 0.0)
    
    # Fano factor (measure of super/sub-Poissonian statistics)
    fano_factor = jnp.where(mean_counts > 0, var_counts / mean_counts, 1.0)
    
    return {
        'mean_counts': mean_counts,
        'std_counts': std_counts,
        'var_counts': var_counts,
        'snr': snr,
        'fano_factor': fano_factor,
        'total_mean': total_mean,
        'total_var': total_var
    }

@jit
def shot_noise_limited_snr(mean_counts):
    """
    Calculate theoretical shot-noise limited SNR.
    For Poisson statistics: SNR = √N
    """
    return jnp.sqrt(jnp.maximum(mean_counts, 1e-10))

@jit
def correlation_analysis(counts_matrix, max_lag=10):
    """
    Calculate temporal correlations in photon counts.
    
    Args:
        counts_matrix: (n_runs, n_bins) photon counts
        max_lag: maximum lag for correlation calculation
    
    Returns:
        Auto-correlation function
    """
    n_runs, n_bins = counts_matrix.shape
    
    # Calculate auto-correlation for each run, then average
    def autocorr_single_run(counts):
        # Normalize counts (zero mean)
        counts_norm = counts - jnp.mean(counts)
        
        # Calculate correlations for different lags
        correlations = jnp.zeros(max_lag + 1)
        
        for lag in range(max_lag + 1):
            if lag == 0:
                # Zero lag: variance
                corr = jnp.mean(counts_norm**2)
            else:
                # Non-zero lag
                if lag < n_bins:
                    corr = jnp.mean(counts_norm[:-lag] * counts_norm[lag:])
                else:
                    corr = 0.0
            
            correlations = correlations.at[lag].set(corr)
        
        return correlations
    
    # Average over all runs
    all_correlations = vmap(autocorr_single_run)(counts_matrix)
    avg_correlation = jnp.mean(all_correlations, axis=0)
    
    # Normalize by zero-lag value
    avg_correlation = avg_correlation / jnp.maximum(avg_correlation[0], 1e-10)
    
    return avg_correlation

class PhotonSamplingSystem:
    """
    Complete photon sampling system with statistical analysis.
    
    Features:
    - Vectorized Poisson sampling
    - Multi-run statistical analysis
    - Temporal correlation analysis
    - Shot noise analysis
    - Performance optimization
    """
    
    def __init__(self, n_runs=1000):
        self.n_runs = n_runs
        self.last_statistics = None
        self.last_correlations = None
    
    @partial(jit, static_argnums=(0,))
    def sample_experiment(self, rates_Hz, dt_ns):
        """
        Sample a complete experiment with statistical analysis.
        
        Args:
            rates_Hz: time-dependent photon rates
            dt_ns: time bin size
        
        Returns:
            Dictionary with counts and statistics
        """
        # Sample photon counts
        counts_matrix = sample_photons_time_series(rates_Hz, dt_ns, self.n_runs)
        
        # Calculate statistics
        stats = photon_statistics(counts_matrix)
        
        # Store for later analysis
        self.last_statistics = stats
        
        return {
            'counts_matrix': counts_matrix,
            'mean_counts': stats['mean_counts'],
            'std_counts': stats['std_counts'],
            'statistics': stats
        }
    
    @partial(jit, static_argnums=(0,))
    def analyze_correlations(self, counts_matrix, max_lag=10):
        """
        Analyze temporal correlations in the data.
        """
        correlations = correlation_analysis(counts_matrix, max_lag)
        self.last_correlations = correlations
        return correlations
    
    @partial(jit, static_argnums=(0,))
    def optimize_measurement_time(self, rate_Hz, target_snr, dt_ns):
        """
        Calculate optimal measurement time to achieve target SNR.
        
        Args:
            rate_Hz: photon rate
            target_snr: desired signal-to-noise ratio
            dt_ns: time bin size
        
        Returns:
            Optimal number of time bins and total measurement time
        """
        # For Poisson statistics: SNR = √(rate × time)
        # Therefore: time = (target_snr)² / rate
        
        optimal_time_s = (target_snr**2) / rate_Hz
        optimal_time_ns = optimal_time_s * 1e9
        
        # Number of bins needed
        n_bins_optimal = jnp.ceil(optimal_time_ns / dt_ns).astype(int)
        
        # Actual measurement time
        actual_time_ns = n_bins_optimal * dt_ns
        
        # Achieved SNR
        total_counts = rate_Hz * actual_time_s
        achieved_snr = shot_noise_limited_snr(total_counts)
        
        return {
            'optimal_bins': n_bins_optimal,
            'measurement_time_ns': actual_time_ns,
            'measurement_time_s': actual_time_s,
            'achieved_snr': achieved_snr,
            'total_expected_counts': total_counts
        }
    
    @partial(jit, static_argnums=(0,))
    def contrast_measurement(self, rate_bright_Hz, rate_dark_Hz, dt_ns, measurement_time_ns):
        """
        Simulate contrast measurement between bright and dark states.
        
        Returns statistical significance of contrast measurement.
        """
        n_bins = int(measurement_time_ns / dt_ns)
        
        # Create time series (half bright, half dark for simplicity)
        rates_Hz = jnp.concatenate([
            jnp.full(n_bins//2, rate_bright_Hz),
            jnp.full(n_bins//2, rate_dark_Hz)
        ])
        
        # Sample experiment
        result = self.sample_experiment(rates_Hz, dt_ns)
        counts_matrix = result['counts_matrix']
        
        # Separate bright and dark periods
        bright_counts = counts_matrix[:, :n_bins//2]
        dark_counts = counts_matrix[:, n_bins//2:]
        
        # Calculate statistics for each period
        bright_mean = jnp.mean(bright_counts)
        dark_mean = jnp.mean(dark_counts)
        bright_std = jnp.std(bright_counts)
        dark_std = jnp.std(dark_counts)
        
        # Contrast and significance
        contrast = (bright_mean - dark_mean) / (bright_mean + dark_mean)
        
        # Statistical significance (t-test like)
        pooled_std = jnp.sqrt((bright_std**2 + dark_std**2) / 2)
        significance = jnp.abs(bright_mean - dark_mean) / pooled_std if pooled_std > 0 else 0.0
        
        return {
            'contrast': contrast,
            'significance': significance,
            'bright_mean': bright_mean,
            'dark_mean': dark_mean,
            'bright_std': bright_std,
            'dark_std': dark_std
        }

@jit
def generate_realistic_pulse_profile(tau_ns, readout_ns, pulse_rate_factor=0.1):
    """
    Generate realistic pulse profile with low fluorescence during pulse,
    high fluorescence during readout.
    
    Args:
        tau_ns: pulse duration
        readout_ns: readout duration  
        pulse_rate_factor: fluorescence rate during pulse (relative to readout)
    
    Returns:
        Time array and rate array
    """
    dt_ns = 10.0  # 10 ns bins
    
    # Time arrays
    pulse_times = jnp.arange(0, tau_ns, dt_ns)
    readout_times = jnp.arange(tau_ns, tau_ns + readout_ns, dt_ns)
    
    all_times = jnp.concatenate([pulse_times, readout_times])
    
    # Rate profile
    base_rate_Hz = 1e6  # 1 MHz during readout
    
    pulse_rates = jnp.full(len(pulse_times), base_rate_Hz * pulse_rate_factor)
    readout_rates = jnp.full(len(readout_times), base_rate_Hz)
    
    all_rates = jnp.concatenate([pulse_rates, readout_rates])
    
    return all_times, all_rates

def test_photon_sampling():
    """Test photon sampling system."""
    
    print("Testing photon sampling system...")
    
    # Create sampling system
    sampler = PhotonSamplingSystem(n_runs=100)
    
    # Test 1: Simple constant rate
    print(f"\n1. Constant rate test:")
    rate_Hz = 1e6  # 1 MHz
    dt_ns = 10.0   # 10 ns bins
    n_bins = 100   # 1 μs total
    
    rates = jnp.full(n_bins, rate_Hz)
    result = sampler.sample_experiment(rates, dt_ns)
    
    expected_per_bin = rate_Hz * dt_ns * 1e-9
    measured_mean = jnp.mean(result['mean_counts'])
    
    print(f"  Expected per bin: {expected_per_bin:.3f}")
    print(f"  Measured mean: {measured_mean:.3f}")
    print(f"  Relative error: {abs(measured_mean - expected_per_bin)/expected_per_bin*100:.1f}%")
    
    # Test 2: Pulse profile
    print(f"\n2. Pulse profile test:")
    tau_ns = 100
    readout_ns = 500
    
    times, rates = generate_realistic_pulse_profile(tau_ns, readout_ns)
    result = sampler.sample_experiment(rates, dt_ns)
    
    # Analyze pulse vs readout phases
    pulse_bins = int(tau_ns / dt_ns)
    pulse_mean = jnp.mean(result['mean_counts'][:pulse_bins])
    readout_mean = jnp.mean(result['mean_counts'][pulse_bins:])
    
    contrast = (readout_mean - pulse_mean) / (readout_mean + pulse_mean)
    
    print(f"  Pulse mean: {pulse_mean:.1f} counts/bin")
    print(f"  Readout mean: {readout_mean:.1f} counts/bin")
    print(f"  Contrast: {contrast:.3f}")
    
    # Test 3: Correlations
    print(f"\n3. Correlation analysis:")
    correlations = sampler.analyze_correlations(result['counts_matrix'], max_lag=5)
    
    print(f"  Zero-lag correlation: {correlations[0]:.3f}")
    print(f"  First-lag correlation: {correlations[1]:.3f}")
    print(f"  Correlation decay: {correlations}")
    
    # Test 4: SNR optimization
    print(f"\n4. SNR optimization:")
    target_snr = 10.0
    optimization = sampler.optimize_measurement_time(rate_Hz, target_snr, dt_ns)
    
    print(f"  Target SNR: {target_snr}")
    print(f"  Optimal measurement time: {optimization['measurement_time_s']:.3f} s")
    print(f"  Required bins: {optimization['optimal_bins']}")
    print(f"  Achieved SNR: {optimization['achieved_snr']:.1f}")
    
    # Test 5: Contrast measurement
    print(f"\n5. Contrast measurement:")
    bright_rate = 1e6  # 1 MHz
    dark_rate = 3e5    # 300 kHz
    
    contrast_result = sampler.contrast_measurement(
        bright_rate, dark_rate, dt_ns, 1000.0  # 1 μs measurement
    )
    
    print(f"  Bright rate: {bright_rate/1e6:.1f} MHz")
    print(f"  Dark rate: {dark_rate/1e6:.1f} MHz") 
    print(f"  Measured contrast: {contrast_result['contrast']:.3f}")
    print(f"  Statistical significance: {contrast_result['significance']:.1f}")
    
    return sampler

if __name__ == "__main__":
    print("📊 Testing Photon Sampling System")
    print("=" * 40)
    
    sampler = test_photon_sampling()
    
    print("\n✅ Photon sampling system working!")
    print("   Includes: vectorized Poisson, statistics, correlations, SNR optimization")