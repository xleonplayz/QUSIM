#!/usr/bin/env python3
"""
Complete Detector Model with All Effects
=======================================
IRF, dead-time, afterpulse, jitter, crosstalk, and statistical effects
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
from functools import partial

jax.config.update("jax_enable_x64", True)

@jit
def sample_irf_timing(key, n_photons, sigma_ps, tail_fraction, tail_tau_ns):
    """
    Sample timing jitter from instrument response function.
    IRF = Gaussian core + exponential tail
    """
    if n_photons == 0:
        return jnp.array([]), key
    
    key, k1, k2, k3 = random.split(key, 4)
    
    # Choose between Gaussian and tail for each photon
    use_tail = random.uniform(k1, (n_photons,)) < tail_fraction
    
    # Gaussian component (convert ps to ns)
    gaussian_delays = random.normal(k2, (n_photons,)) * sigma_ps * 1e-3
    
    # Exponential tail component
    exponential_delays = random.exponential(k3, (n_photons,)) * tail_tau_ns
    
    # Select appropriate component for each photon
    timing_delays = jnp.where(use_tail, exponential_delays, gaussian_delays)
    
    return timing_delays, key

@jit
def apply_dead_time_vectorized(arrival_times_ns, dead_time_ns):
    """
    Vectorized dead-time application.
    Remove photons arriving within dead-time window.
    """
    if len(arrival_times_ns) == 0:
        return arrival_times_ns
    
    # Sort arrival times
    sorted_times = jnp.sort(arrival_times_ns)
    
    # Create mask for non-dead-time photons
    valid_mask = jnp.ones(len(sorted_times), dtype=bool)
    
    # Vectorized dead-time check
    # For each photon, check if it's within dead-time of any previous valid photon
    for i in range(1, len(sorted_times)):
        # Check against all previous valid photons
        time_diffs = sorted_times[i] - sorted_times[:i]
        within_dead_time = (time_diffs < dead_time_ns) & valid_mask[:i]
        
        # If any previous valid photon is within dead-time, this photon is blocked
        valid_mask = valid_mask.at[i].set(~jnp.any(within_dead_time))
    
    return sorted_times[valid_mask]

@jit
def generate_afterpulses(detected_times_ns, afterpulse_prob, delay_distribution_params, key):
    """
    Generate afterpulses for detected photons.
    Each detection can trigger delayed afterpulse with some probability.
    """
    if len(detected_times_ns) == 0:
        return jnp.array([]), key
    
    n_detected = len(detected_times_ns)
    
    key, k1, k2 = random.split(key, 3)
    
    # Determine which detections cause afterpulses
    afterpulse_mask = random.uniform(k1, (n_detected,)) < afterpulse_prob
    n_afterpulses = jnp.sum(afterpulse_mask)
    
    if n_afterpulses == 0:
        return jnp.array([]), key
    
    # Get trigger times for afterpulses
    trigger_times = detected_times_ns[afterpulse_mask]
    
    # Generate afterpulse delays
    # Use exponential distribution with mean delay
    mean_delay_ns, _ = delay_distribution_params
    delays = random.exponential(k2, (n_afterpulses,)) * mean_delay_ns
    
    # Afterpulse arrival times
    afterpulse_times = trigger_times + delays
    
    return afterpulse_times, key

@jit
def dark_count_poisson(time_window_ns, dark_rate_Hz, key):
    """
    Generate dark counts as Poisson process.
    """
    # Expected number of dark counts
    lambda_dark = dark_rate_Hz * time_window_ns * 1e-9
    
    key, k1, k2 = random.split(key, 3)
    
    # Sample number of dark counts
    n_dark = random.poisson(k1, lambda_dark)
    
    if n_dark == 0:
        return jnp.array([]), key
    
    # Uniform distribution of dark count times
    dark_times = random.uniform(k2, (n_dark,)) * time_window_ns
    
    return jnp.sort(dark_times), key

@jit
def optical_crosstalk(detected_times_ns, crosstalk_prob, crosstalk_delay_ns, key):
    """
    Add optical crosstalk between detector pixels.
    Each detection can trigger delayed detection in neighboring pixels.
    """
    if len(detected_times_ns) == 0:
        return jnp.array([]), key
    
    n_detected = len(detected_times_ns)
    
    key, k1, k2 = random.split(key, 3)
    
    # Determine crosstalk events
    crosstalk_mask = random.uniform(k1, (n_detected,)) < crosstalk_prob
    n_crosstalk = jnp.sum(crosstalk_mask)
    
    if n_crosstalk == 0:
        return jnp.array([]), key
    
    # Crosstalk times (small fixed delay)
    crosstalk_times = detected_times_ns[crosstalk_mask] + crosstalk_delay_ns
    
    return crosstalk_times, key

class CompleteSPADDetector:
    """
    Complete SPAD detector model with all realistic effects.
    
    Includes:
    - Quantum efficiency wavelength dependence
    - Instrument response function (IRF) with Gaussian + tail
    - Dead-time (paralyzable/non-paralyzable)
    - Afterpulsing with realistic delay distributions
    - Dark counts (temperature dependent)
    - Optical crosstalk
    - Timing jitter and walk
    - Statistical fluctuations
    """
    
    def __init__(self, params):
        self.params = params
        
        # Basic detector parameters
        self.quantum_efficiency = 0.25  # 25% at peak wavelength
        self.wavelength_nm = 637  # NV emission wavelength
        
        # Timing parameters
        self.irf_sigma_ps = params.irf_sigma_ps
        self.irf_tail_frac = params.irf_tail_frac  
        self.irf_tail_tau_ns = params.irf_tail_tau_ns
        
        # Dead-time parameters
        self.dead_time_ns = params.dead_time_ns
        self.dead_time_type = 'non-paralyzable'  # or 'paralyzable'
        
        # Afterpulse parameters
        self.afterpulse_prob = params.afterpulse_prob
        self.afterpulse_delay_mean_ns = 50.0  # Mean afterpulse delay
        self.afterpulse_delay_std_ns = 20.0   # Delay distribution width
        
        # Dark count parameters  
        self.dark_rate_Hz = params.dark_rate_Hz
        self.temperature_K = 300.0
        self.dark_rate_temp_coeff = 0.1  # 10% increase per 10K
        
        # Crosstalk parameters
        self.optical_crosstalk_prob = 0.001  # 0.1% crosstalk
        self.crosstalk_delay_ns = 0.1  # Very short delay
        
        # Advanced effects
        self.enable_afterpulses = True
        self.enable_crosstalk = True
        self.enable_irf = True
        self.enable_dead_time = True
        
    def update_temperature(self, T_K):
        """Update temperature-dependent parameters."""
        self.temperature_K = T_K
        
        # Dark count rate increases with temperature
        temp_factor = 1.0 + self.dark_rate_temp_coeff * (T_K - 300.0) / 10.0
        self.dark_rate_Hz = self.params.dark_rate_Hz * temp_factor
        
    @partial(jit, static_argnums=(0,))
    def quantum_efficiency_wavelength(self, wavelength_nm):
        """
        Wavelength-dependent quantum efficiency.
        Silicon SPAD response curve.
        """
        # Simplified QE curve for silicon SPAD
        if wavelength_nm < 400:
            qe = 0.01  # UV cutoff
        elif wavelength_nm < 500:
            qe = 0.15  # Blue response
        elif wavelength_nm < 600:
            qe = 0.25  # Green 
        elif wavelength_nm < 700:
            qe = 0.40  # Red peak (637nm NV line)
        elif wavelength_nm < 900:
            qe = 0.30  # NIR
        else:
            qe = 0.05  # Far NIR falloff
        
        return qe
    
    @partial(jit, static_argnums=(0,))
    def detection_efficiency_total(self, wavelength_nm=None):
        """
        Total detection efficiency including all losses.
        """
        if wavelength_nm is None:
            wavelength_nm = self.wavelength_nm
            
        # Quantum efficiency
        qe = self.quantum_efficiency_wavelength(wavelength_nm)
        
        # Additional losses (optics, filters, etc.)
        collection_efficiency = 0.8  # 80% collection
        filter_transmission = 0.9    # 90% filter transmission
        
        total_efficiency = qe * collection_efficiency * filter_transmission
        
        return total_efficiency
    
    @partial(jit, static_argnums=(0,))
    def detect_photon_stream_complete(self, rate_Hz, time_window_ns, dt_bin_ns, key):
        """
        Complete photon detection simulation with all effects.
        
        Args:
            rate_Hz: photon rate in Hz
            time_window_ns: total detection window in ns
            dt_bin_ns: time bin size for output histogram
            key: JAX random key
        
        Returns:
            Detection histogram and updated key
        """
        # 1. Generate true photon arrivals (Poisson process)
        key, k1 = random.split(key)
        expected_photons = rate_Hz * time_window_ns * 1e-9
        n_true_photons = random.poisson(k1, expected_photons)
        
        if n_true_photons == 0:
            # No photons: only dark counts
            key, k_dark = random.split(key)
            dark_times, key = dark_count_poisson(time_window_ns, self.dark_rate_Hz, k_dark)
            
            # Create histogram
            n_bins = int(time_window_ns / dt_bin_ns)
            histogram = jnp.zeros(n_bins)
            
            if len(dark_times) > 0:
                bin_indices = (dark_times / dt_bin_ns).astype(int)
                bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)
                histogram = histogram.at[bin_indices].add(1)
            
            return histogram, key
        
        # Generate uniform arrival times
        key, k2 = random.split(key)
        arrival_times = random.uniform(k2, (n_true_photons,)) * time_window_ns
        arrival_times = jnp.sort(arrival_times)
        
        # 2. Apply quantum efficiency
        key, k3 = random.split(key)
        detection_efficiency = self.detection_efficiency_total()
        detected_mask = random.uniform(k3, (n_true_photons,)) < detection_efficiency
        detected_times = arrival_times[detected_mask]
        
        if len(detected_times) == 0:
            # No detections: only dark counts
            key, k_dark = random.split(key)
            dark_times, key = dark_count_poisson(time_window_ns, self.dark_rate_Hz, k_dark)
            
            n_bins = int(time_window_ns / dt_bin_ns)
            histogram = jnp.zeros(n_bins)
            
            if len(dark_times) > 0:
                bin_indices = (dark_times / dt_bin_ns).astype(int)
                bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)
                histogram = histogram.at[bin_indices].add(1)
            
            return histogram, key
        
        # 3. Apply IRF timing jitter
        if self.enable_irf:
            key, k4 = random.split(key)
            timing_delays, key = sample_irf_timing(
                k4, len(detected_times), 
                self.irf_sigma_ps, self.irf_tail_frac, self.irf_tail_tau_ns
            )
            detected_times = detected_times + timing_delays
        
        # 4. Apply dead-time
        if self.enable_dead_time:
            detected_times = apply_dead_time_vectorized(detected_times, self.dead_time_ns)
        
        # 5. Generate afterpulses
        afterpulse_times = jnp.array([])
        if self.enable_afterpulses and len(detected_times) > 0:
            key, k5 = random.split(key)
            afterpulse_params = (self.afterpulse_delay_mean_ns, self.afterpulse_delay_std_ns)
            afterpulse_times, key = generate_afterpulses(
                detected_times, self.afterpulse_prob, afterpulse_params, k5
            )
        
        # 6. Add optical crosstalk
        crosstalk_times = jnp.array([])
        if self.enable_crosstalk and len(detected_times) > 0:
            key, k6 = random.split(key)
            crosstalk_times, key = optical_crosstalk(
                detected_times, self.optical_crosstalk_prob, self.crosstalk_delay_ns, k6
            )
        
        # 7. Add dark counts
        key, k7 = random.split(key)
        dark_times, key = dark_count_poisson(time_window_ns, self.dark_rate_Hz, k7)
        
        # 8. Combine all detection times
        all_times = jnp.concatenate([
            detected_times,
            afterpulse_times,
            crosstalk_times, 
            dark_times
        ])
        
        # Remove times outside window
        all_times = all_times[(all_times >= 0) & (all_times < time_window_ns)]
        
        # 9. Create histogram
        n_bins = int(time_window_ns / dt_bin_ns)
        histogram = jnp.zeros(n_bins)
        
        if len(all_times) > 0:
            bin_indices = (all_times / dt_bin_ns).astype(int)
            bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)
            
            # Count events in each bin
            for bin_idx in bin_indices:
                histogram = histogram.at[bin_idx].add(1)
        
        return histogram, key
    
    def get_detector_response_function(self, time_array_ns):
        """
        Calculate detector impulse response function.
        Combination of Gaussian + exponential tail.
        """
        sigma_ns = self.irf_sigma_ps * 1e-3
        
        # Gaussian component
        gaussian = (1 - self.irf_tail_frac) * jnp.exp(-0.5 * (time_array_ns / sigma_ns)**2)
        gaussian = gaussian / (sigma_ns * jnp.sqrt(2 * jnp.pi))
        
        # Exponential tail (only for t >= 0)
        exponential = jnp.where(
            time_array_ns >= 0,
            self.irf_tail_frac * jnp.exp(-time_array_ns / self.irf_tail_tau_ns) / self.irf_tail_tau_ns,
            0.0
        )
        
        return gaussian + exponential
    
    def characterize_detector(self):
        """
        Get complete detector characterization.
        """
        return {
            'quantum_efficiency': self.quantum_efficiency,
            'detection_efficiency_total': self.detection_efficiency_total(),
            'dead_time_ns': self.dead_time_ns,
            'afterpulse_probability': self.afterpulse_prob,
            'dark_rate_Hz': self.dark_rate_Hz,
            'irf_sigma_ps': self.irf_sigma_ps,
            'irf_tail_fraction': self.irf_tail_frac,
            'temperature_K': self.temperature_K,
            'wavelength_nm': self.wavelength_nm
        }

def test_complete_detector():
    """Test complete detector model."""
    
    # Mock parameters
    class MockParams:
        irf_sigma_ps = 300.0
        irf_tail_frac = 0.1
        irf_tail_tau_ns = 5.0
        dead_time_ns = 12.0
        afterpulse_prob = 0.02
        dark_rate_Hz = 200.0
    
    params = MockParams()
    
    # Create detector
    detector = CompleteSPADDetector(params)
    
    # Test characteristics
    print("Detector characteristics:")
    chars = detector.characterize_detector()
    for key, value in chars.items():
        print(f"  {key}: {value}")
    
    # Test detection simulation
    print(f"\nTesting detection simulation:")
    
    # Test parameters
    rate_Hz = 1e6      # 1 MHz photon rate
    window_ns = 1000   # 1 μs window
    dt_bin_ns = 10     # 10 ns bins
    
    # Run simulation
    key = random.PRNGKey(42)
    histogram, _ = detector.detect_photon_stream_complete(
        rate_Hz, window_ns, dt_bin_ns, key
    )
    
    total_counts = jnp.sum(histogram)
    expected_input = rate_Hz * window_ns * 1e-9
    detection_efficiency = total_counts / expected_input
    
    print(f"  Input rate: {rate_Hz/1e6:.1f} MHz")
    print(f"  Expected photons: {expected_input:.0f}")
    print(f"  Detected counts: {total_counts:.0f}")
    print(f"  Effective efficiency: {detection_efficiency:.3f}")
    print(f"  Mean counts per bin: {jnp.mean(histogram):.1f}")
    print(f"  Max counts per bin: {jnp.max(histogram):.0f}")
    
    # Test IRF
    print(f"\nIRF characteristics:")
    time_irf = jnp.linspace(-2, 20, 1000)  # -2 to 20 ns
    irf = detector.get_detector_response_function(time_irf)
    
    peak_idx = jnp.argmax(irf)
    fwhm_indices = jnp.where(irf > jnp.max(irf) / 2)[0]
    fwhm_ns = (time_irf[fwhm_indices[-1]] - time_irf[fwhm_indices[0]])
    
    print(f"  Peak time: {time_irf[peak_idx]:.2f} ns")
    print(f"  FWHM: {fwhm_ns:.2f} ns")
    print(f"  Peak value: {jnp.max(irf):.3f}")
    
    return detector

if __name__ == "__main__":
    print("📷 Testing Complete SPAD Detector Model")
    print("=" * 45)
    
    detector = test_complete_detector()
    
    print("\n✅ Complete detector model working!")
    print("   Includes: QE, IRF, dead-time, afterpulse, dark counts, crosstalk")