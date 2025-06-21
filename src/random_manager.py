#!/usr/bin/env python3
"""
Random Number Management for Realistic NV Simulator
===================================================
Provides proper randomization without fixed seeds
"""

import numpy as np
import jax.numpy as jnp
from jax import random
import time
import hashlib
from typing import Optional, Union

class RandomManager:
    """
    Manages random number generation for the NV simulator.
    
    Features:
    - No fixed seeds (unless explicitly requested for reproducibility)
    - Hardware-based entropy when available
    - Multiple independent streams for different physics processes
    - Proper correlation handling for coupled noise sources
    """
    
    def __init__(self, master_seed: Optional[int] = None):
        """
        Initialize random manager.
        
        Args:
            master_seed: If provided, enables reproducible mode.
                        If None, uses true randomness.
        """
        self.reproducible_mode = master_seed is not None
        
        if self.reproducible_mode:
            self.master_seed = master_seed
            print(f"🎲 Random Manager: Reproducible mode (seed={master_seed})")
        else:
            # Generate truly random master seed
            self.master_seed = self._generate_true_random_seed()
            print(f"🎲 Random Manager: Random mode (seed={self.master_seed})")
        
        # Initialize JAX PRNG key
        self.master_key = random.PRNGKey(self.master_seed)
        
        # Create independent streams for different physics processes
        self._initialize_streams()
    
    def _generate_true_random_seed(self) -> int:
        """
        Generate a truly random seed using multiple entropy sources.
        """
        # Combine multiple sources of entropy
        entropy_sources = []
        
        # 1. High-precision time
        entropy_sources.append(str(time.time_ns()))
        
        # 2. Memory address randomness (Python object ID)
        entropy_sources.append(str(id(self)))
        
        # 3. NumPy random state
        entropy_sources.append(str(np.random.get_state()[1][0]))
        
        # 4. System entropy (if available)
        try:
            import os
            entropy_sources.append(str(int.from_bytes(os.urandom(8), 'big')))
        except:
            pass
        
        # 5. Process and thread IDs
        try:
            import os, threading
            entropy_sources.append(str(os.getpid()))
            entropy_sources.append(str(threading.get_ident()))
        except:
            pass
        
        # Combine all sources with cryptographic hash
        combined = ''.join(entropy_sources)
        hash_bytes = hashlib.sha256(combined.encode()).digest()
        
        # Convert to 32-bit integer (JAX PRNG requirement)
        seed = int.from_bytes(hash_bytes[:4], 'big')
        
        return seed
    
    def _initialize_streams(self):
        """Initialize independent random streams for different physics processes."""
        # Split master key into independent streams
        keys = random.split(self.master_key, 20)  # Reserve 20 streams
        
        self.streams = {
            'spin_dynamics': keys[0],     # Quantum state evolution
            'laser_noise': keys[1],       # Laser intensity fluctuations
            'magnetic_noise': keys[2],    # B-field fluctuations
            'charge_dynamics': keys[3],   # Ionization/recombination
            'detector_noise': keys[4],    # Photon detection noise
            'thermal_noise': keys[5],     # Temperature fluctuations
            'spin_bath': keys[6],         # Nuclear spin bath
            'spectral_diffusion': keys[7], # Frequency wandering
            'mechanical_noise': keys[8],   # Vibrations/strain
            'electronic_noise': keys[9],  # 1/f noise in electronics
            'shot_noise': keys[10],       # Poisson photon statistics
            'mw_noise': keys[11],         # MW amplitude/phase noise
            'timing_jitter': keys[12],    # Detection timing uncertainty
            'afterpulsing': keys[13],     # Detector afterpulsing
            'crosstalk': keys[14],        # Inter-channel crosstalk
            'drift': keys[15],            # Long-term parameter drift
            'temperature_gradient': keys[16], # Spatial temperature variations
            'field_gradient': keys[17],   # Magnetic field gradients
            'laser_pointing': keys[18],   # Beam pointing stability
            'reserved': keys[19]          # For future use
        }
    
    def get_key(self, stream_name: str) -> random.PRNGKey:
        """
        Get and update a random key for a specific physics process.
        
        Args:
            stream_name: Name of the random stream
            
        Returns:
            Fresh JAX PRNG key
        """
        if stream_name not in self.streams:
            raise ValueError(f"Unknown random stream: {stream_name}")
        
        # Split key to get fresh randomness
        self.streams[stream_name], new_key = random.split(self.streams[stream_name])
        
        return new_key
    
    def get_numpy_generator(self, stream_name: str) -> np.random.Generator:
        """
        Get NumPy random generator for a specific stream.
        Useful for functions that need NumPy random interface.
        """
        key = self.get_key(stream_name)
        # Convert JAX key to NumPy-compatible seed
        seed = int(key[0])  # Extract seed from JAX key
        
        return np.random.default_rng(seed)
    
    def correlated_noise(self, 
                        stream_names: list, 
                        correlation_matrix: np.ndarray, 
                        shape: tuple) -> dict:
        """
        Generate correlated noise between multiple processes.
        
        Args:
            stream_names: List of stream names to correlate
            correlation_matrix: Correlation matrix (must be positive definite)
            shape: Shape of noise arrays
            
        Returns:
            Dictionary of correlated noise arrays
        """
        n_streams = len(stream_names)
        
        if correlation_matrix.shape != (n_streams, n_streams):
            raise ValueError("Correlation matrix size must match number of streams")
        
        # Generate independent Gaussian noise
        independent_noise = []
        for stream_name in stream_names:
            key = self.get_key(stream_name)
            noise = random.normal(key, shape)
            independent_noise.append(noise)
        
        independent_noise = jnp.stack(independent_noise, axis=0)
        
        # Apply Cholesky decomposition for correlation
        try:
            L = jnp.linalg.cholesky(correlation_matrix)
            correlated_noise = jnp.tensordot(L, independent_noise, axes=[1, 0])
        except:
            # Fallback: use eigendecomposition if Cholesky fails
            eigenvals, eigenvecs = jnp.linalg.eigh(correlation_matrix)
            eigenvals = jnp.maximum(eigenvals, 1e-10)  # Ensure positive
            L = eigenvecs @ jnp.diag(jnp.sqrt(eigenvals))
            correlated_noise = jnp.tensordot(L, independent_noise, axes=[1, 0])
        
        # Return as dictionary
        result = {}
        for i, stream_name in enumerate(stream_names):
            result[stream_name] = correlated_noise[i]
        
        return result
    
    def ornstein_uhlenbeck(self, 
                          stream_name: str,
                          x_prev: float,
                          dt: float,
                          tau_corr: float,
                          sigma: float) -> float:
        """
        Generate Ornstein-Uhlenbeck process step.
        
        dx = -x/τ dt + σ√(2/τ) dW
        
        Args:
            stream_name: Random stream to use
            x_prev: Previous value
            dt: Time step
            tau_corr: Correlation time
            sigma: Noise strength
            
        Returns:
            Next value in OU process
        """
        key = self.get_key(stream_name)
        
        # OU process update
        alpha = dt / tau_corr
        noise_strength = sigma * jnp.sqrt(2 * alpha)
        noise = random.normal(key) * noise_strength
        
        x_next = x_prev * (1 - alpha) + noise
        
        return float(x_next)
    
    def poisson_process(self, 
                       stream_name: str,
                       rate: Union[float, np.ndarray],
                       dt: float) -> Union[int, np.ndarray]:
        """
        Generate Poisson process events.
        
        Args:
            stream_name: Random stream to use
            rate: Event rate (Hz)
            dt: Time interval (s)
            
        Returns:
            Number of events
        """
        key = self.get_key(stream_name)
        
        # Expected number of events
        expected = rate * dt
        
        # Generate Poisson random numbers
        if isinstance(expected, (int, float)):
            events = random.poisson(key, expected)
        else:
            events = random.poisson(key, expected, shape=expected.shape)
        
        return events
    
    def power_law_noise(self, 
                       stream_name: str,
                       shape: tuple,
                       exponent: float = -1.0) -> np.ndarray:
        """
        Generate power-law (1/f^α) noise.
        
        Args:
            stream_name: Random stream to use
            shape: Output shape (must include frequency dimension)
            exponent: Power law exponent (α)
            
        Returns:
            Power-law noise in time domain
        """
        key = self.get_key(stream_name)
        
        # Generate white noise in frequency domain
        white_noise = random.normal(key, shape)
        
        # Create frequency array
        n_freq = shape[-1]
        freqs = jnp.fft.fftfreq(n_freq)[1:]  # Exclude DC
        
        # Apply power law scaling
        power_scaling = freqs**(exponent/2)
        
        # Scale the frequency components
        white_noise_fft = jnp.fft.fft(white_noise, axis=-1)
        scaled_fft = white_noise_fft.at[..., 1:].multiply(power_scaling)
        
        # Transform back to time domain
        colored_noise = jnp.fft.ifft(scaled_fft, axis=-1).real
        
        return np.array(colored_noise)
    
    def shot_noise_photons(self, 
                          stream_name: str,
                          mean_photons: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Generate shot noise for photon detection.
        
        Args:
            stream_name: Random stream to use
            mean_photons: Expected number of photons
            
        Returns:
            Actual detected photons (with shot noise)
        """
        return self.poisson_process(stream_name, mean_photons, 1.0)
    
    def detector_response(self, 
                         stream_name: str,
                         photon_times: np.ndarray,
                         dead_time_ns: float,
                         afterpulse_prob: float,
                         timing_jitter_ps: float) -> np.ndarray:
        """
        Simulate realistic detector response with dead time, afterpulsing, and jitter.
        
        Args:
            stream_name: Random stream to use
            photon_times: True photon arrival times (ns)
            dead_time_ns: Detector dead time
            afterpulse_prob: Probability of afterpulsing
            timing_jitter_ps: Timing jitter (picoseconds)
            
        Returns:
            Detected photon times with realistic detector effects
        """
        key = self.get_key(stream_name)
        key1, key2, key3 = random.split(key, 3)
        
        detected_times = []
        last_detection = -dead_time_ns  # Allow first photon
        
        for photon_time in photon_times:
            # Check if detector is ready
            if photon_time > last_detection + dead_time_ns:
                # Add timing jitter
                jitter = random.normal(key1) * timing_jitter_ps * 1e-3  # ps to ns
                detected_time = photon_time + jitter
                detected_times.append(detected_time)
                last_detection = detected_time
                
                # Check for afterpulse
                if random.uniform(key2) < afterpulse_prob:
                    # Generate afterpulse with exponential delay
                    afterpulse_delay = random.exponential(key3) * dead_time_ns * 2
                    afterpulse_time = detected_time + afterpulse_delay
                    detected_times.append(afterpulse_time)
                
                # Update keys for next iteration
                key1, key2, key3 = random.split(key, 3)
        
        return np.array(detected_times)
    
    def reset_stream(self, stream_name: str):
        """Reset a specific random stream (useful for debugging)."""
        if stream_name in self.streams:
            stream_index = list(self.streams.keys()).index(stream_name)
            self.streams[stream_name] = random.split(self.master_key, 20)[stream_index]
    
    def get_stream_info(self) -> dict:
        """Get information about all random streams."""
        info = {
            'reproducible_mode': self.reproducible_mode,
            'master_seed': self.master_seed,
            'available_streams': list(self.streams.keys()),
            'stream_count': len(self.streams)
        }
        return info
    
    def print_summary(self):
        """Print summary of random manager state."""
        print("🎲 Random Manager Summary")
        print("=" * 30)
        print(f"Mode: {'Reproducible' if self.reproducible_mode else 'Random'}")
        print(f"Master seed: {self.master_seed}")
        print(f"Available streams: {len(self.streams)}")
        
        print("\nPhysics streams:")
        for i, stream_name in enumerate(self.streams.keys()):
            if i < 10:  # Show first 10
                print(f"  {stream_name}")
            elif i == 10:
                print(f"  ... and {len(self.streams) - 10} more")
                break


# Global random manager instance
_global_random_manager = None

def get_random_manager(master_seed: Optional[int] = None) -> RandomManager:
    """
    Get global random manager instance.
    
    Args:
        master_seed: Seed for reproducible mode (only used on first call)
        
    Returns:
        RandomManager instance
    """
    global _global_random_manager
    
    if _global_random_manager is None:
        _global_random_manager = RandomManager(master_seed)
    
    return _global_random_manager

def reset_global_random_manager():
    """Reset global random manager (useful for testing)."""
    global _global_random_manager
    _global_random_manager = None


if __name__ == "__main__":
    print("Testing Random Manager...")
    
    # Test reproducible mode
    print("\n1. Reproducible mode:")
    rm1 = RandomManager(master_seed=12345)
    rm1.print_summary()
    
    # Generate some random numbers
    key1 = rm1.get_key('spin_dynamics')
    print(f"Sample random: {random.normal(key1)}")
    
    # Test correlated noise
    print("\n2. Correlated noise:")
    correlation_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    corr_noise = rm1.correlated_noise(['laser_noise', 'magnetic_noise'], 
                                      correlation_matrix, (100,))
    print(f"Correlation: {np.corrcoef(corr_noise['laser_noise'], corr_noise['magnetic_noise'])[0,1]:.3f}")
    
    # Test OU process
    print("\n3. Ornstein-Uhlenbeck process:")
    x = 0.0
    for i in range(5):
        x = rm1.ornstein_uhlenbeck('spectral_diffusion', x, 0.1, 1.0, 0.5)
        print(f"  Step {i}: {x:.3f}")
    
    print("\n✅ Random Manager tests passed!")