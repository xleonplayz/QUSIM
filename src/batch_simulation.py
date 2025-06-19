#!/usr/bin/env python3
"""
Batch Vectorization with JAX vmap
================================
High-performance parallel simulation of multiple experiments
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
from functools import partial
import time

jax.config.update("jax_enable_x64", True)

@jit
def single_nv_evolution(key, tau_ns, params):
    """
    Single NV evolution: MW pulse → readout → counts.
    Optimized for batching with vmap.
    """
    # Simplified physics for fast computation
    # Basic Rabi evolution
    rabi_angle = params['Omega_Rabi_Hz'] * tau_ns * 1e-9 * jnp.pi
    
    # Add noise
    key1, key2, key3 = random.split(key, 3)
    
    # MW noise effects
    amplitude_noise = 1.0 + 0.01 * random.normal(key1)  # 1% amplitude noise
    phase_noise = 0.02 * random.normal(key2)  # Phase noise
    
    # Effective populations
    p_ms0 = jnp.cos((rabi_angle * amplitude_noise + phase_noise) / 2)**2
    p_ms1 = 1 - p_ms0
    
    # Readout rates
    rate_bright = params['Beta_max_Hz'] * params['collection_eff'] * p_ms0
    rate_dark = params['Beta_max_Hz'] * params['collection_eff'] * 0.3 * p_ms1
    rate_total = rate_bright + rate_dark + params['dark_rate_Hz']
    
    # Simple Poisson detection
    expected_counts = rate_total * params['readout_time_ns'] * 1e-9 * params['n_shots']
    
    # Add Poisson noise
    counts = random.poisson(key3, expected_counts)
    
    return {
        'p_ms0': p_ms0,
        'p_ms1': p_ms1,
        'counts': counts,
        'rate_total': rate_total
    }

# Vectorize over multiple seeds
batch_nv_evolution = vmap(single_nv_evolution, in_axes=(0, None, None))

# Vectorize over multiple tau values
batch_tau_sweep = vmap(single_nv_evolution, in_axes=(None, 0, None))

# Vectorize over both seeds and tau values
batch_full_simulation = vmap(vmap(single_nv_evolution, in_axes=(0, None, None)), in_axes=(None, 0, None))

@jit
def parameter_sweep_simulation(keys, param_values, param_name, base_params):
    """
    Vectorized parameter sweep: vary one parameter over many values.
    """
    def single_param_run(key, param_val):
        # Update parameter
        updated_params = base_params.copy()
        updated_params[param_name] = param_val
        
        # Fixed tau for parameter sweep
        tau_ns = 100.0  # π/2 pulse typically
        
        return single_nv_evolution(key, tau_ns, updated_params)
    
    # Vectorize over parameter values
    batch_param_sweep = vmap(single_param_run, in_axes=(0, 0))
    
    return batch_param_sweep(keys, param_values)

@jit 
def noise_characterization_batch(keys, tau_ns, noise_levels):
    """
    Batch simulation for noise characterization.
    Varies noise level and measures impact on fidelity.
    """
    def single_noise_run(key, noise_level):
        params = {
            'Omega_Rabi_Hz': 5e6,
            'Beta_max_Hz': 13e6,
            'collection_eff': 0.15,
            'dark_rate_Hz': 200.0,
            'readout_time_ns': 1000.0,
            'n_shots': 1000,
            'noise_amplitude': noise_level  # Variable noise
        }
        
        # Include noise in evolution
        key1, key2 = random.split(key)
        
        rabi_angle = params['Omega_Rabi_Hz'] * tau_ns * 1e-9 * jnp.pi
        noise_factor = 1.0 + noise_level * random.normal(key1)
        
        p_ms0_ideal = jnp.cos(rabi_angle / 2)**2
        p_ms0_noisy = jnp.cos((rabi_angle * noise_factor) / 2)**2
        
        # Fidelity measure
        fidelity = 1.0 - jnp.abs(p_ms0_ideal - p_ms0_noisy)
        
        # Count statistics
        rate = params['Beta_max_Hz'] * params['collection_eff'] * p_ms0_noisy
        counts = random.poisson(key2, rate * params['readout_time_ns'] * 1e-9 * params['n_shots'])
        
        return {
            'fidelity': fidelity,
            'p_ms0': p_ms0_noisy,
            'counts': counts,
            'noise_level': noise_level
        }
    
    # Vectorize over noise levels and random seeds
    batch_noise = vmap(vmap(single_noise_run, in_axes=(0, None)), in_axes=(None, 0))
    
    return batch_noise(keys, noise_levels)

@jit
def temperature_sweep_batch(keys, temperatures_K, base_params):
    """
    Batch temperature sweep with thermal effects.
    """
    def single_temp_run(key, temp_K):
        # Temperature-dependent parameters
        params = base_params.copy()
        
        # Debye-Waller factor temperature dependence
        DW_T = params['DW0'] * jnp.exp(-0.005 * temp_K / 150.0)
        
        # Collection efficiency with phonon effects
        collection_eff_T = params['collection_eff'] * (DW_T / params['DW0'])
        
        # ISC rates (Arrhenius)
        k_B_eV = 8.617e-5  # Boltzmann constant in eV/K
        gamma_ISC_T = params['gamma_ISC_base'] * jnp.exp(-0.01 / (k_B_eV * temp_K))
        
        # Update parameters
        params['collection_eff'] = collection_eff_T
        params['gamma_ISC'] = gamma_ISC_T
        params['temperature_K'] = temp_K
        
        # Run simulation
        tau_ns = 100.0  # Fixed pulse length
        result = single_nv_evolution(key, tau_ns, params)
        
        # Add temperature-specific results
        result['temperature_K'] = temp_K
        result['collection_eff_T'] = collection_eff_T
        result['DW_factor'] = DW_T
        
        return result
    
    # Vectorize over temperatures
    batch_temp = vmap(single_temp_run, in_axes=(0, 0))
    
    return batch_temp(keys, temperatures_K)

@jit
def pulse_optimization_batch(keys, pulse_params_array):
    """
    Batch pulse optimization: find optimal pulse parameters.
    """
    def evaluate_pulse(key, pulse_params):
        # Extract pulse parameters
        tau_ns = pulse_params[0]
        amplitude_factor = pulse_params[1]
        phase_offset = pulse_params[2]
        
        # Base parameters
        params = {
            'Omega_Rabi_Hz': 5e6 * amplitude_factor,
            'Beta_max_Hz': 13e6,
            'collection_eff': 0.15,
            'dark_rate_Hz': 200.0,
            'readout_time_ns': 1000.0,
            'n_shots': 1000
        }
        
        # Pulse evolution with phase
        rabi_angle = params['Omega_Rabi_Hz'] * tau_ns * 1e-9 * jnp.pi + phase_offset
        p_ms0 = jnp.cos(rabi_angle / 2)**2
        
        # Target: maximize contrast for π/2 pulse (p_ms0 = 0.5)
        target_p_ms0 = 0.5
        fidelity = 1.0 - jnp.abs(p_ms0 - target_p_ms0)
        
        # Simulate readout
        rate = params['Beta_max_Hz'] * params['collection_eff'] * p_ms0
        counts = random.poisson(key, rate * params['readout_time_ns'] * 1e-9 * params['n_shots'])
        
        return {
            'fidelity': fidelity,
            'p_ms0': p_ms0,
            'counts': counts,
            'pulse_params': pulse_params
        }
    
    # Vectorize over pulse parameters
    batch_optimize = vmap(evaluate_pulse, in_axes=(0, 0))
    
    return batch_optimize(keys, pulse_params_array)

class BatchSimulationManager:
    """
    Manager for large-scale batch simulations with performance monitoring.
    """
    
    def __init__(self, n_cores=None):
        self.n_cores = n_cores or jax.device_count()
        self.timing_stats = {}
        
    def run_tau_sweep_batch(self, tau_values, n_seeds=100, base_params=None):
        """
        Run large-scale tau sweep with performance monitoring.
        """
        if base_params is None:
            base_params = {
                'Omega_Rabi_Hz': 5e6,
                'Beta_max_Hz': 13e6,
                'collection_eff': 0.15,
                'dark_rate_Hz': 200.0,
                'readout_time_ns': 1000.0,
                'n_shots': 1000
            }
        
        # Generate random seeds
        keys = random.split(random.PRNGKey(42), n_seeds)
        
        start_time = time.time()
        
        # Batch computation: n_seeds × n_tau_values
        results = batch_full_simulation(keys, tau_values, base_params)
        
        # Block until computation is complete (JAX lazy evaluation)
        jax.block_until_ready(results)
        
        computation_time = time.time() - start_time
        
        # Performance statistics
        n_simulations = len(tau_values) * n_seeds
        throughput = n_simulations / computation_time
        
        self.timing_stats['tau_sweep'] = {
            'n_simulations': n_simulations,
            'computation_time_s': computation_time,
            'throughput_sims_per_s': throughput,
            'n_tau_values': len(tau_values),
            'n_seeds': n_seeds
        }
        
        return results, self.timing_stats['tau_sweep']
    
    def run_parameter_study(self, param_name, param_values, n_seeds=50):
        """
        Run parameter study with batch processing.
        """
        base_params = {
            'Omega_Rabi_Hz': 5e6,
            'Beta_max_Hz': 13e6,
            'collection_eff': 0.15,
            'dark_rate_Hz': 200.0,
            'readout_time_ns': 1000.0,
            'n_shots': 1000
        }
        
        # Generate keys for each parameter value
        keys = random.split(random.PRNGKey(123), len(param_values))
        
        start_time = time.time()
        
        # Run parameter sweep
        results = parameter_sweep_simulation(keys, param_values, param_name, base_params)
        
        jax.block_until_ready(results)
        computation_time = time.time() - start_time
        
        # Statistics
        n_simulations = len(param_values)
        throughput = n_simulations / computation_time
        
        self.timing_stats[f'{param_name}_sweep'] = {
            'n_simulations': n_simulations,
            'computation_time_s': computation_time,
            'throughput_sims_per_s': throughput,
            'parameter': param_name,
            'n_values': len(param_values)
        }
        
        return results, self.timing_stats[f'{param_name}_sweep']
    
    def benchmark_performance(self, n_simulations_list=[100, 1000, 10000]):
        """
        Benchmark simulation performance for different scales.
        """
        benchmark_results = {}
        
        for n_sims in n_simulations_list:
            # Create test parameters
            tau_values = jnp.linspace(0, 200, min(100, n_sims))
            n_seeds = max(1, n_sims // len(tau_values))
            
            start_time = time.time()
            
            # Run batch simulation
            keys = random.split(random.PRNGKey(0), n_seeds)
            base_params = {
                'Omega_Rabi_Hz': 5e6,
                'Beta_max_Hz': 13e6,
                'collection_eff': 0.15,
                'dark_rate_Hz': 200.0,
                'readout_time_ns': 1000.0,
                'n_shots': 1000
            }
            
            results = batch_full_simulation(keys, tau_values, base_params)
            jax.block_until_ready(results)
            
            computation_time = time.time() - start_time
            actual_sims = len(tau_values) * n_seeds
            
            benchmark_results[n_sims] = {
                'actual_simulations': actual_sims,
                'computation_time_s': computation_time,
                'throughput_sims_per_s': actual_sims / computation_time,
                'time_per_simulation_ms': (computation_time / actual_sims) * 1000
            }
        
        return benchmark_results
    
    def get_performance_summary(self):
        """Get summary of all timing statistics."""
        return self.timing_stats

if __name__ == "__main__":
    # Test batch simulation
    print("⚡ Testing JAX Batch Vectorization")
    print("=" * 40)
    
    manager = BatchSimulationManager()
    
    # Test 1: Small tau sweep
    print("Test 1: Tau sweep batch...")
    tau_values = jnp.linspace(0, 200, 21)  # 21 tau values
    n_seeds = 10  # 10 random seeds
    
    results, stats = manager.run_tau_sweep_batch(tau_values, n_seeds)
    
    print(f"  {stats['n_simulations']} simulations in {stats['computation_time_s']:.3f}s")
    print(f"  Throughput: {stats['throughput_sims_per_s']:.0f} sims/s")
    
    # Test 2: Parameter sweep
    print("\nTest 2: Power parameter sweep...")
    power_values = jnp.linspace(0.1, 5.0, 20)  # 20 power values
    
    results2, stats2 = manager.run_parameter_study('Beta_max_Hz', power_values * 1e6)
    
    print(f"  {stats2['n_simulations']} simulations in {stats2['computation_time_s']:.3f}s")
    print(f"  Throughput: {stats2['throughput_sims_per_s']:.0f} sims/s")
    
    # Test 3: Performance benchmark
    print("\nTest 3: Performance benchmark...")
    benchmark = manager.benchmark_performance([100, 1000])
    
    for n_sims, stats in benchmark.items():
        print(f"  {stats['actual_simulations']} sims: {stats['time_per_simulation_ms']:.2f} ms/sim")
    
    print("\n✅ JAX batch vectorization working!")
    print(f"Total performance tests: {len(manager.get_performance_summary())} completed")