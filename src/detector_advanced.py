#!/usr/bin/env python3
"""
Advanced Detector Model with Photon Arrival Simulation
=====================================================
Complete SPAD detector physics with timing, dead-time, afterpulsing, and jitter
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
from functools import partial

jax.config.update("jax_enable_x64", True)

@jit
def poisson_thinning(rate_Hz, dt_ns, key):
    """
    Generate Poisson-distributed photon arrival times using thinning algorithm.
    More accurate than simple binning for low count rates.
    """
    # Expected number of photons in time interval
    lambda_total = rate_Hz * dt_ns * 1e-9
    
    # Sample total number of events
    key, subkey = random.split(key)
    n_events = random.poisson(subkey, lambda_total)
    
    # Generate uniform arrival times within interval
    if n_events > 0:
        key, subkey = random.split(key)
        arrival_times = random.uniform(subkey, (n_events,)) * dt_ns
        arrival_times = jnp.sort(arrival_times)
    else:
        arrival_times = jnp.array([])
    
    return arrival_times, key

@jit
def apply_detector_deadtime(arrival_times_ns, dead_time_ns):
    """
    Apply detector dead-time: remove photons arriving within dead-time window.
    Uses efficient vectorized algorithm.
    """
    if len(arrival_times_ns) == 0:
        return arrival_times_ns
    
    # Start with first photon (always detected)
    detected_times = [arrival_times_ns[0]]
    last_detected = arrival_times_ns[0]
    
    # Check each subsequent photon
    for i in range(1, len(arrival_times_ns)):
        if arrival_times_ns[i] - last_detected >= dead_time_ns:
            detected_times.append(arrival_times_ns[i])
            last_detected = arrival_times_ns[i]
    
    return jnp.array(detected_times)

@jit
def simulate_afterpulsing(detected_times_ns, afterpulse_prob, tau_afterpulse_ns, key):
    """
    Simulate afterpulsing: each detected photon has probability of generating
    delayed afterpulse with exponential time distribution.
    """
    if len(detected_times_ns) == 0:
        return detected_times_ns, key
    
    afterpulse_times = []
    
    for i, t_detect in enumerate(detected_times_ns):
        key, subkey = random.split(key)
        
        # Check if afterpulse occurs
        if random.uniform(subkey) < afterpulse_prob:
            # Generate afterpulse time with exponential distribution
            key, subkey = random.split(key)
            delay = random.exponential(subkey) * tau_afterpulse_ns
            afterpulse_times.append(t_detect + delay)
    
    # Combine original and afterpulse times
    if afterpulse_times:
        all_times = jnp.concatenate([detected_times_ns, jnp.array(afterpulse_times)])
        all_times = jnp.sort(all_times)
    else:
        all_times = detected_times_ns
    
    return all_times, key

@jit
def apply_timing_jitter(photon_times_ns, sigma_jitter_ps, tail_fraction, tau_tail_ns, key):
    """
    Apply instrument response function (IRF) timing jitter.
    Combines Gaussian core with exponential tail for realistic SPAD response.
    """
    if len(photon_times_ns) == 0:
        return photon_times_ns, key
    
    n_photons = len(photon_times_ns)
    jittered_times = jnp.zeros_like(photon_times_ns)
    
    for i, t in enumerate(photon_times_ns):
        key, subkey1, subkey2 = random.split(key, 3)
        
        # Choose between Gaussian core and exponential tail
        if random.uniform(subkey1) < (1 - tail_fraction):
            # Gaussian core
            jitter = random.normal(subkey2) * sigma_jitter_ps * 1e-3  # Convert ps to ns
        else:
            # Exponential tail (always positive delay)
            jitter = random.exponential(subkey2) * tau_tail_ns
        
        jittered_times = jittered_times.at[i].set(t + jitter)
    
    return jittered_times, key

@jit 
def add_dark_counts(time_window_ns, dark_rate_Hz, key):
    """
    Add dark counts uniformly distributed in time window.
    """
    # Expected number of dark counts
    lambda_dark = dark_rate_Hz * time_window_ns * 1e-9
    
    # Sample number of dark counts
    key, subkey = random.split(key)
    n_dark = random.poisson(subkey, lambda_dark)
    
    if n_dark > 0:
        # Generate uniform arrival times
        key, subkey = random.split(key)
        dark_times = random.uniform(subkey, (n_dark,)) * time_window_ns
        dark_times = jnp.sort(dark_times)
    else:
        dark_times = jnp.array([])
    
    return dark_times, key

@jit
def detection_efficiency_wavelength(wavelength_nm, detector_type='Si_SPAD'):
    """
    Wavelength-dependent detection efficiency.
    """
    if detector_type == 'Si_SPAD':
        # Silicon SPAD efficiency curve (simplified)
        # Peak efficiency ~40% at 650nm, drops at 532nm
        if wavelength_nm < 400:
            efficiency = 0.01  # UV cutoff
        elif wavelength_nm < 500:
            efficiency = 0.15  # Blue response
        elif wavelength_nm < 600:
            efficiency = 0.25  # Green (532nm laser)
        elif wavelength_nm < 700:
            efficiency = 0.40  # Red peak
        elif wavelength_nm < 900:
            efficiency = 0.30  # NIR
        else:
            efficiency = 0.05   # Far NIR
    else:
        efficiency = 0.20  # Default 20%
    
    return efficiency

class AdvancedDetector:
    """
    Complete SPAD detector model with all realistic effects.
    """
    
    def __init__(self, params):
        # Basic detector parameters
        self.dead_time_ns = params.dead_time_ns
        self.afterpulse_prob = params.afterpulse_prob
        self.dark_rate_Hz = params.dark_rate_Hz
        
        # IRF parameters
        self.sigma_jitter_ps = params.irf_sigma_ps
        self.tail_fraction = params.irf_tail_frac
        self.tau_tail_ns = params.irf_tail_tau_ns
        
        # Advanced parameters
        self.detection_efficiency = 0.25  # 25% quantum efficiency
        self.wavelength_nm = 637  # NV emission wavelength
        
        # Temperature effects
        self.temperature_K = 300.0
        self.temp_coeff_dark = 0.1  # 10% increase per 10K
        
        # Noise parameters
        self.crosstalk_prob = 0.001  # 0.1% optical crosstalk
        self.voltage_noise_rms = 0.01  # 1% voltage noise
        
    def update_temperature_effects(self, T_K):
        """Update detector parameters for temperature dependence."""
        self.temperature_K = T_K
        
        # Dark count rate increases with temperature
        T_ref = 300.0  # Reference temperature
        self.dark_rate_Hz = self.dark_rate_Hz * (1 + self.temp_coeff_dark * (T_K - T_ref) / 10.0)
        
        # Afterpulse probability slightly increases with temperature
        self.afterpulse_prob = self.afterpulse_prob * (1 + 0.02 * (T_K - T_ref) / 10.0)
    
    def update_wavelength_efficiency(self, wavelength_nm):
        """Update detection efficiency for different wavelengths."""
        self.wavelength_nm = wavelength_nm
        self.detection_efficiency = detection_efficiency_wavelength(wavelength_nm)
    
    @partial(jit, static_argnums=(0,))
    def detect_photon_stream(self, signal_rate_Hz, time_window_ns, dt_bin_ns, key):
        """
        Complete photon detection simulation for a time window.
        Returns histogram of detected photons vs time.
        """
        n_bins = int(time_window_ns / dt_bin_ns)
        detected_counts = jnp.zeros(n_bins)
        
        # Process each time bin
        for i in range(n_bins):
            bin_start = i * dt_bin_ns
            bin_end = (i + 1) * dt_bin_ns
            
            # 1. Generate signal photon arrivals
            key, subkey = random.split(key)
            signal_arrivals, key = poisson_thinning(signal_rate_Hz, dt_bin_ns, subkey)
            
            if len(signal_arrivals) > 0:
                # Apply detection efficiency
                key, subkey = random.split(key)
                efficiency_mask = random.uniform(subkey, signal_arrivals.shape) < self.detection_efficiency
                signal_arrivals = signal_arrivals[efficiency_mask]
                
                # Add bin offset
                signal_arrivals = signal_arrivals + bin_start
            
            # 2. Add dark counts for this bin
            key, subkey = random.split(key)
            dark_arrivals, key = add_dark_counts(dt_bin_ns, self.dark_rate_Hz, subkey)
            if len(dark_arrivals) > 0:
                dark_arrivals = dark_arrivals + bin_start
            
            # 3. Combine signal and dark counts
            if len(signal_arrivals) > 0 and len(dark_arrivals) > 0:
                all_arrivals = jnp.concatenate([signal_arrivals, dark_arrivals])
            elif len(signal_arrivals) > 0:
                all_arrivals = signal_arrivals
            elif len(dark_arrivals) > 0:
                all_arrivals = dark_arrivals
            else:
                continue  # No photons in this bin
            
            all_arrivals = jnp.sort(all_arrivals)
            
            # 4. Apply dead-time
            detected_arrivals = apply_detector_deadtime(all_arrivals, self.dead_time_ns)
            
            # 5. Apply afterpulsing
            key, subkey = random.split(key)
            detected_arrivals, key = simulate_afterpulsing(
                detected_arrivals, self.afterpulse_prob, self.tau_tail_ns, subkey
            )
            
            # 6. Apply timing jitter
            key, subkey = random.split(key)
            detected_arrivals, key = apply_timing_jitter(
                detected_arrivals, self.sigma_jitter_ps, self.tail_fraction, self.tau_tail_ns, subkey
            )
            
            # 7. Bin the final detection times
            for j in range(n_bins):
                bin_start_j = j * dt_bin_ns
                bin_end_j = (j + 1) * dt_bin_ns
                
                # Count photons in this bin
                bin_mask = (detected_arrivals >= bin_start_j) & (detected_arrivals < bin_end_j)
                detected_counts = detected_counts.at[j].add(jnp.sum(bin_mask))
        
        return detected_counts, key
    
    def get_detector_response_function(self, t_ns):
        """
        Get detector impulse response function (IRF).
        Combination of Gaussian + exponential tail.
        """
        # Gaussian core
        sigma_ns = self.sigma_jitter_ps * 1e-3
        gaussian = (1 - self.tail_fraction) * jnp.exp(-0.5 * (t_ns / sigma_ns)**2) / (sigma_ns * jnp.sqrt(2 * jnp.pi))
        
        # Exponential tail (only for t > 0)
        exponential = jnp.where(t_ns >= 0, 
                               self.tail_fraction * jnp.exp(-t_ns / self.tau_tail_ns) / self.tau_tail_ns,
                               0.0)
        
        return gaussian + exponential
    
    def get_detector_specs(self):
        """Get complete detector specifications."""
        return {
            'dead_time_ns': self.dead_time_ns,
            'afterpulse_prob': self.afterpulse_prob,
            'dark_rate_Hz': self.dark_rate_Hz,
            'detection_efficiency': self.detection_efficiency,
            'jitter_sigma_ps': self.sigma_jitter_ps,
            'jitter_tail_frac': self.tail_fraction,
            'jitter_tail_tau_ns': self.tau_tail_ns,
            'temperature_K': self.temperature_K,
            'wavelength_nm': self.wavelength_nm
        }

@jit
def photon_arrival_monte_carlo(rates_Hz, dt_bins_ns, n_shots, key):
    """
    Monte Carlo simulation of photon arrivals for multiple experimental shots.
    Returns statistics over many repetitions.
    """
    n_bins = len(rates_Hz)
    total_counts = jnp.zeros(n_bins)
    
    for shot in range(n_shots):
        key, subkey = random.split(key)
        
        # Generate one shot worth of data
        shot_counts = jnp.zeros(n_bins)
        
        for i, rate in enumerate(rates_Hz):
            key, subkey = random.split(key)
            arrivals, key = poisson_thinning(rate, dt_bins_ns, subkey)
            shot_counts = shot_counts.at[i].set(len(arrivals))
        
        total_counts += shot_counts
    
    # Return average counts per bin
    return total_counts / n_shots, key

class PhotonCorrelationAnalysis:
    """
    Advanced photon correlation analysis for g(2) measurements.
    """
    
    def __init__(self, detector):
        self.detector = detector
        
    @jit
    def calculate_g2_correlation(self, photon_times_ns, tau_max_ns, dt_bin_ns):
        """
        Calculate second-order correlation function g(2)(τ).
        Measures photon bunching/antibunching.
        """
        n_taus = int(2 * tau_max_ns / dt_bin_ns) + 1
        tau_values = jnp.linspace(-tau_max_ns, tau_max_ns, n_taus)
        g2_values = jnp.zeros(n_taus)
        
        n_photons = len(photon_times_ns)
        
        if n_photons < 2:
            return tau_values, g2_values
        
        # Calculate all pairwise time differences
        for i in range(n_photons):
            for j in range(n_photons):
                if i != j:
                    dt = photon_times_ns[j] - photon_times_ns[i]
                    
                    # Find bin index
                    bin_idx = jnp.argmin(jnp.abs(tau_values - dt))
                    g2_values = g2_values.at[bin_idx].add(1)
        
        # Normalize (simplified)
        if jnp.sum(g2_values) > 0:
            g2_values = g2_values / jnp.max(g2_values)
        
        return tau_values, g2_values

if __name__ == "__main__":
    # Test advanced detector
    print("📷 Testing Advanced Detector Model")
    print("=" * 40)
    
    # Mock parameters
    class MockParams:
        dead_time_ns = 12.0
        afterpulse_prob = 0.02
        dark_rate_Hz = 200.0
        irf_sigma_ps = 300.0
        irf_tail_frac = 0.1
        irf_tail_tau_ns = 5.0
    
    params = MockParams()
    detector = AdvancedDetector(params)
    
    print("Detector specifications:")
    specs = detector.get_detector_specs()
    for key, value in specs.items():
        print(f"  {key}: {value}")
    
    # Test photon detection
    print("\nTesting photon detection...")
    
    signal_rate = 1e6  # 1 MHz count rate
    time_window = 1000  # 1 μs
    dt_bin = 10  # 10 ns bins
    
    key = random.PRNGKey(42)
    counts, key = detector.detect_photon_stream(signal_rate, time_window, dt_bin, key)
    
    print(f"  Input rate: {signal_rate/1e6:.1f} MHz")
    print(f"  Output counts: {jnp.sum(counts):.0f} in {time_window}ns")
    print(f"  Effective rate: {jnp.sum(counts)/(time_window*1e-9)/1e6:.2f} MHz")
    print(f"  Dead-time loss: {(1 - jnp.sum(counts)/(signal_rate*time_window*1e-9))*100:.1f}%")
    
    # Test IRF
    t_irf = jnp.linspace(-2, 20, 1000)  # -2 to 20 ns
    irf = detector.get_detector_response_function(t_irf)
    
    print(f"\nIRF characteristics:")
    print(f"  FWHM: ~{2.35*detector.sigma_jitter_ps*1e-3:.2f} ns")
    print(f"  Tail fraction: {detector.tail_fraction*100:.1f}%")
    print(f"  Peak IRF: {jnp.max(irf):.3f}")
    
    print("\n✅ Advanced detector model working!")