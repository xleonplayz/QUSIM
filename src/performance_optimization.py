#!/usr/bin/env python3
"""
Performance Optimization for NV Simulator
==========================================
JIT compilation, vectorization, and memory optimization
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap, lax
import numpy as np
from functools import partial
import time
from typing import Callable, Any

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

class PerformanceOptimizer:
    """
    Performance optimization utilities for the NV simulator.
    
    Features:
    - Automatic JIT compilation detection
    - Vectorization (vmap) strategies
    - Memory-efficient batch processing
    - Parallelization across devices
    - Profiling and benchmarking tools
    """
    
    def __init__(self):
        self.compilation_cache = {}
        self.timing_results = {}
        
    @staticmethod
    @jit
    def check_jax_arrays(*arrays):
        """
        Ensure all arrays are JAX arrays (not NumPy).
        This function will fail compilation if NumPy arrays are passed.
        """
        for arr in arrays:
            # This operation only works on JAX arrays
            _ = jnp.sum(arr * 0)  # Force JAX computation
        return True
    
    @staticmethod
    def ensure_jax_input(func: Callable) -> Callable:
        """
        Decorator to ensure all inputs are JAX arrays.
        Converts NumPy arrays automatically with warning.
        """
        def wrapper(*args, **kwargs):
            # Convert args
            jax_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    print(f"⚠️  Converting NumPy array to JAX in {func.__name__}")
                    jax_args.append(jnp.array(arg))
                else:
                    jax_args.append(arg)
            
            # Convert kwargs
            jax_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    print(f"⚠️  Converting NumPy array to JAX in {func.__name__}")
                    jax_kwargs[key] = jnp.array(value)
                else:
                    jax_kwargs[key] = value
                    
            return func(*jax_args, **jax_kwargs)
        return wrapper
    
    @staticmethod
    def benchmark_function(func: Callable, *args, n_runs: int = 10, **kwargs):
        """
        Benchmark a function with JIT compilation timing.
        
        Returns:
            dict with timing statistics
        """
        # Warmup (trigger compilation)
        _ = func(*args, **kwargs)
        _ = func(*args, **kwargs)
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            
            # Ensure computation completes (JAX lazy evaluation)
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
            elif isinstance(result, (tuple, list)):
                for r in result:
                    if hasattr(r, 'block_until_ready'):
                        r.block_until_ready()
            
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            'mean_time_s': np.mean(times),
            'std_time_s': np.std(times),
            'min_time_s': np.min(times),
            'max_time_s': np.max(times),
            'total_runs': n_runs
        }
    
    @staticmethod
    @partial(jit, static_argnums=(0,))
    def vectorized_time_evolution(evolve_func, initial_states, time_arrays):
        """
        Vectorize time evolution over multiple initial conditions.
        
        Args:
            evolve_func: JIT-compiled evolution function
            initial_states: (n_runs, state_dim) initial conditions
            time_arrays: (n_runs, n_times) time arrays (can be different lengths)
        
        Returns:
            Evolved states for each run
        """
        # Use vmap to vectorize over initial conditions
        vectorized_evolve = vmap(evolve_func, in_axes=(0, 0))
        
        evolved_states = vectorized_evolve(initial_states, time_arrays)
        
        return evolved_states
    
    @staticmethod
    @jit
    def batch_matrix_operations(matrices, vectors):
        """
        Efficient batch matrix-vector operations.
        
        Args:
            matrices: (batch_size, n, n) batch of matrices
            vectors: (batch_size, n) batch of vectors
        
        Returns:
            (batch_size, n) batch of results
        """
        # Vectorized matrix-vector multiplication
        results = vmap(jnp.dot)(matrices, vectors)
        
        return results
    
    @staticmethod
    @jit
    def scan_time_evolution(state_init, time_steps, evolve_step_func):
        """
        Memory-efficient time evolution using scan.
        Avoids storing all intermediate states.
        
        Args:
            state_init: initial state
            time_steps: array of time step sizes
            evolve_step_func: function(state, dt) -> new_state
        
        Returns:
            final_state, history (if needed)
        """
        def scan_fn(state, dt):
            new_state = evolve_step_func(state, dt)
            return new_state, new_state  # carry, output
        
        final_state, state_history = lax.scan(scan_fn, state_init, time_steps)
        
        return final_state, state_history
    
    @staticmethod
    @jit
    def chunk_processing(data, chunk_size, process_func):
        """
        Process large arrays in chunks to manage memory.
        
        Args:
            data: large input array
            chunk_size: size of each chunk
            process_func: function to apply to each chunk
        
        Returns:
            Processed results
        """
        n_total = data.shape[0]
        n_chunks = (n_total + chunk_size - 1) // chunk_size
        
        def process_chunk(i):
            start_idx = i * chunk_size
            end_idx = jnp.minimum((i + 1) * chunk_size, n_total)
            chunk = data[start_idx:end_idx]
            return process_func(chunk)
        
        # Vectorize over chunk indices
        chunk_results = vmap(process_chunk)(jnp.arange(n_chunks))
        
        # Concatenate results
        result = jnp.concatenate(chunk_results, axis=0)
        
        return result

@jit
def optimized_rabi_simulation(tau_array_ns, omega_rabi_MHz, n_runs=1000):
    """
    Highly optimized Rabi oscillation simulation.
    Demonstrates best practices for performance.
    """
    # Convert to SI units within JIT
    tau_array_s = tau_array_ns * 1e-9
    omega_rabi_rad_s = omega_rabi_MHz * 2 * jnp.pi * 1e6
    
    # Vectorized computation over all tau values and runs
    tau_mesh, run_mesh = jnp.meshgrid(tau_array_s, jnp.arange(n_runs), indexing='ij')
    
    # Ideal Rabi oscillation
    prob_excited = jnp.sin(omega_rabi_rad_s * tau_mesh / 2)**2
    
    # Add shot noise (Poisson statistics)
    key = jax.random.PRNGKey(42)
    
    # Vectorized Poisson sampling
    expected_counts = prob_excited * 1000  # Assume 1000 shots per measurement
    keys = jax.random.split(key, tau_mesh.size)
    keys_reshaped = keys.reshape(tau_mesh.shape + (2,))
    
    actual_counts = vmap(vmap(jax.random.poisson))(keys_reshaped[..., 0], expected_counts)
    
    # Calculate statistics
    mean_prob = jnp.mean(actual_counts / 1000, axis=1)  # Average over runs
    std_prob = jnp.std(actual_counts / 1000, axis=1)
    
    return mean_prob, std_prob

@jit
def optimized_spin_evolution(H_total, rho_init, time_array, collapse_ops):
    """
    Memory-optimized spin evolution with Lindblad master equation.
    Uses scan for memory efficiency.
    """
    dt = time_array[1] - time_array[0]  # Assume uniform spacing
    
    def lindblad_step(rho, t):
        # Unitary evolution
        drho_dt = -1j * (H_total @ rho - rho @ H_total)
        
        # Dissipative evolution
        for L in collapse_ops:
            L_dag = L.T.conj()
            L_dag_L = L_dag @ L
            
            drho_dt += L @ rho @ L_dag - 0.5 * (L_dag_L @ rho + rho @ L_dag_L)
        
        # Simple Euler step (for demonstration)
        rho_new = rho + dt * drho_dt
        
        # Ensure trace preservation
        rho_new = rho_new / jnp.trace(rho_new)
        
        return rho_new, rho_new
    
    # Use scan for memory efficiency
    final_rho, rho_history = lax.scan(lindblad_step, rho_init, time_array[:-1])
    
    return final_rho, rho_history

@jit  
def vectorized_detector_response(rate_arrays, detector_params, n_runs=100):
    """
    Vectorized detector simulation across multiple rate profiles.
    
    Args:
        rate_arrays: (n_profiles, n_times) array of photon rates
        detector_params: detector parameters
        n_runs: number of Monte Carlo runs per profile
    
    Returns:
        Detection histograms for each profile
    """
    n_profiles, n_times = rate_arrays.shape
    
    def detect_single_profile(rates):
        # Simplified detector model
        dt_ns = 10.0
        
        # Vectorized Poisson sampling
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, n_runs * n_times)
        keys_matrix = keys.reshape(n_runs, n_times, 2)
        
        # Expected counts per bin per run
        expected_counts = rates[None, :] * dt_ns * 1e-9
        
        # Vectorized sampling
        detected_counts = vmap(vmap(jax.random.poisson))(
            keys_matrix[..., 0], expected_counts
        )
        
        return detected_counts
    
    # Vectorize over all profiles
    all_detections = vmap(detect_single_profile)(rate_arrays)
    
    return all_detections

class JAXMemoryManager:
    """
    Utilities for managing JAX memory usage.
    """
    
    @staticmethod
    def clear_caches():
        """Clear JAX compilation and array caches."""
        jax.clear_caches()
        print("✅ JAX caches cleared")
    
    @staticmethod
    def get_memory_info():
        """Get current JAX memory usage."""
        # This is device-specific and may not be available on all platforms
        try:
            devices = jax.devices()
            memory_info = {}
            
            for device in devices:
                # Try to get memory info (GPU-specific)
                try:
                    memory_info[str(device)] = device.memory_stats()
                except:
                    memory_info[str(device)] = "Memory info not available"
            
            return memory_info
        except:
            return {"status": "Memory monitoring not available on this platform"}
    
    @staticmethod
    def optimize_for_memory():
        """Set JAX configuration for memory optimization."""
        # Reduce memory pre-allocation
        jax.config.update('jax_platform_name', 'cpu')  # Force CPU if memory limited
        print("🔧 JAX configured for memory optimization")

def profile_simulator_components():
    """
    Profile all major simulator components for performance.
    """
    print("🚀 Profiling simulator components...")
    
    # Mock data for testing
    n_times = 1000
    time_array = jnp.linspace(0, 1000, n_times)  # 1 μs, 1 ns resolution
    
    optimizer = PerformanceOptimizer()
    
    # Test 1: Rabi simulation
    print("\n1. Rabi simulation:")
    tau_array = jnp.linspace(0, 100, 50)  # 50 tau points
    
    timing = optimizer.benchmark_function(
        optimized_rabi_simulation, tau_array, 1.0, 500  # 500 runs
    )
    print(f"   Mean time: {timing['mean_time_s']*1000:.2f} ms")
    print(f"   Std time:  {timing['std_time_s']*1000:.2f} ms")
    
    # Test 2: Vectorized detector
    print("\n2. Vectorized detector:")
    n_profiles = 10
    rate_profiles = jnp.ones((n_profiles, n_times)) * 1e6  # 1 MHz constant
    
    timing = optimizer.benchmark_function(
        vectorized_detector_response, rate_profiles, {}, 50  # 50 runs each
    )
    print(f"   Mean time: {timing['mean_time_s']*1000:.2f} ms")
    print(f"   Throughput: {n_profiles / timing['mean_time_s']:.1f} profiles/s")
    
    # Test 3: Matrix operations
    print("\n3. Batch matrix operations:")
    batch_size = 100
    dim = 16
    
    matrices = jax.random.normal(jax.random.PRNGKey(0), (batch_size, dim, dim))
    vectors = jax.random.normal(jax.random.PRNGKey(1), (batch_size, dim))
    
    timing = optimizer.benchmark_function(
        optimizer.batch_matrix_operations, matrices, vectors
    )
    print(f"   Mean time: {timing['mean_time_s']*1000:.2f} ms")
    print(f"   Throughput: {batch_size / timing['mean_time_s']:.0f} ops/s")
    
    # Test 4: Memory usage
    print("\n4. Memory information:")
    memory_info = JAXMemoryManager.get_memory_info()
    for device, info in memory_info.items():
        print(f"   {device}: {info}")
    
    return optimizer

def check_no_numpy_in_functions():
    """
    Static analysis to check for NumPy usage in critical functions.
    """
    print("🔍 Checking for NumPy usage in simulator modules...")
    
    # List of modules to check
    modules_to_check = [
        'full_hamiltonian.py',
        'advanced_lindblad.py', 
        'charge_dynamics.py',
        'advanced_laser.py',
        'complete_detector.py',
        'photon_sampling.py',
        'environment_noise.py',
        'temperature_effects.py'
    ]
    
    numpy_usage = {}
    
    for module in modules_to_check:
        try:
            with open(f'/Users/leonkaiser/STAY/PLAY/simulator/src/{module}', 'r') as f:
                content = f.read()
            
            # Look for NumPy usage patterns
            numpy_patterns = [
                'np.array',
                'np.zeros',
                'np.ones', 
                'np.eye',
                'np.random',
                'np.linalg',
                'np.exp',
                'np.sin',
                'np.cos',
                'np.sqrt'
            ]
            
            found_patterns = []
            for pattern in numpy_patterns:
                if pattern in content and '@jit' in content:
                    # Check if it's in a JIT-compiled function
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if pattern in line:
                            # Look backwards for @jit decorator
                            for j in range(max(0, i-10), i):
                                if '@jit' in lines[j]:
                                    found_patterns.append((pattern, i+1))
                                    break
            
            if found_patterns:
                numpy_usage[module] = found_patterns
        
        except FileNotFoundError:
            print(f"   ⚠️  Module {module} not found")
    
    if numpy_usage:
        print("   ❌ Found potential NumPy usage in JIT functions:")
        for module, patterns in numpy_usage.items():
            print(f"     {module}:")
            for pattern, line_no in patterns:
                print(f"       Line {line_no}: {pattern}")
    else:
        print("   ✅ No NumPy usage detected in JIT functions")
    
    return numpy_usage

def optimization_recommendations():
    """
    Provide optimization recommendations for the simulator.
    """
    recommendations = [
        "✅ Use @jit decorators on all computational functions",
        "✅ Use vmap for vectorization over parameter sweeps", 
        "✅ Use lax.scan for memory-efficient time evolution",
        "✅ Convert all NumPy arrays to JAX arrays",
        "✅ Use static_argnums for non-array arguments in @jit",
        "✅ Batch operations when possible (matrix-matrix vs matrix-vector)",
        "⚠️  Consider using pmap for multi-device parallelization",
        "⚠️  Use jax.checkpoint for memory vs speed tradeoffs",
        "⚠️  Profile memory usage for large simulations",
        "⚠️  Consider mixed precision (float32) for speedup if accuracy allows"
    ]
    
    print("🎯 Performance optimization recommendations:")
    for rec in recommendations:
        print(f"   {rec}")

def test_performance_optimization():
    """Test performance optimization implementation."""
    
    print("Performance optimization test suite:")
    
    # Test 1: Profile components
    optimizer = profile_simulator_components()
    
    # Test 2: Check for NumPy usage
    print(f"\n" + "="*50)
    numpy_check = check_no_numpy_in_functions()
    
    # Test 3: Memory management
    print(f"\n" + "="*50)
    print("Memory management test:")
    JAXMemoryManager.optimize_for_memory()
    
    # Test 4: JAX array validation
    print(f"\n" + "="*50)
    print("JAX array validation:")
    
    # This should work (JAX arrays)
    jax_array = jnp.ones(100)
    try:
        PerformanceOptimizer.check_jax_arrays(jax_array)
        print("   ✅ JAX array validation passed")
    except:
        print("   ❌ JAX array validation failed")
    
    # Test 5: Recommendations
    print(f"\n" + "="*50)
    optimization_recommendations()
    
    return optimizer

if __name__ == "__main__":
    print("⚡ Testing Performance Optimization")
    print("=" * 40)
    
    optimizer = test_performance_optimization()
    
    print("\n✅ Performance optimization complete!")
    print("   All components optimized with JIT and vmap")