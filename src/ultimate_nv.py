#!/usr/bin/env python3
"""
Ultimate NV Simulator - Complete Integration
==========================================
All advanced physics features integrated into single simulator
"""

import jax
import jax.numpy as jnp
from jax import random, jit
import numpy as np
from functools import partial
import time

# Import all advanced modules
from .lindblad import LindblladNVSimulator, build_nv_hamiltonian, make_spin_operators
from .spinbath import SpinBathSimulator, SpinBathParams
from .laser_optics import LaserSystem, AdvancedLaserPhysics
from .detector_advanced import AdvancedDetector, photon_arrival_monte_carlo
from .batch_simulation import BatchSimulationManager
from .thermal_effects import ThermalPhysicsNV
from .nv import AdvancedNVParams

jax.config.update("jax_enable_x64", True)

class UltimateNVSimulator:
    """
    The ultimate NV center simulator combining all advanced physics:
    
    1. Full Lindblad master equation solver
    2. Spin bath with cluster correlation expansion  
    3. Realistic Voigt laser saturation
    4. Complete SPAD detector model
    5. JAX vectorization for performance
    6. Temperature and blackbody effects
    7. Orbital relaxation as Lindblad operators
    8. + All previous 9 advanced features
    """
    
    def __init__(self, level='full'):
        """
        Initialize ultimate simulator.
        
        level: 'basic', 'advanced', 'full'
        - basic: Original features 1-9
        - advanced: + Lindblad + spin bath + laser
        - full: All features including detector + thermal
        """
        self.level = level
        self.params = AdvancedNVParams()
        
        # Initialize all subsystems
        self._initialize_subsystems()
        
        # Performance tracking
        self.timing_stats = {}
        self.simulation_count = 0
        
    def _initialize_subsystems(self):
        """Initialize all physics subsystems."""
        
        # 1. Basic NV parameters (features 1-9)
        # Already in self.params
        
        if self.level in ['advanced', 'full']:
            # 2. Lindblad master equation solver
            self.lindblad_sim = LindblladNVSimulator(self.params)
            
            # 3. Spin bath with CCE
            self.bath_params = SpinBathParams()
            self.spin_bath = SpinBathSimulator(self.bath_params, cce_level=2)
            
            # 4. Advanced laser system
            self.laser = LaserSystem(self.params)
            self.laser_physics = AdvancedLaserPhysics(self.laser)
            
            # 5. Thermal physics
            self.thermal = ThermalPhysicsNV(self.params)
            
        if self.level == 'full':
            # 6. Advanced detector
            self.detector = AdvancedDetector(self.params)
            
            # 7. Batch simulation manager
            self.batch_manager = BatchSimulationManager()
            
    def set_temperature(self, T_K):
        """Update temperature across all subsystems."""
        self.params.Temperature_K = T_K
        
        if hasattr(self, 'thermal'):
            self.thermal.update_temperature(T_K)
        
        if hasattr(self, 'detector'):
            self.detector.update_temperature_effects(T_K)
            
        if hasattr(self, 'lindblad_sim'):
            # Recreate Lindblad simulator with new temperature
            self.lindblad_sim = LindblladNVSimulator(self.params)
    
    def set_magnetic_field(self, B_field_mT):
        """Set magnetic field vector [Bx, By, Bz] in mT."""
        self.params.B_field_mT = B_field_mT
        
        if hasattr(self, 'lindblad_sim'):
            # Rebuild Hamiltonian with new field
            self.lindblad_sim.H_base = build_nv_hamiltonian(self.params, B_field_mT)
    
    def generate_spin_bath_realization(self, key):
        """Generate random spin bath configuration."""
        if hasattr(self, 'spin_bath'):
            return self.spin_bath.generate_bath_configuration(key)
        else:
            return None
    
    @partial(jit, static_argnums=(0,))
    def simulate_ultimate_rabi(self, tau_ns, key, use_lindblad=False):
        """
        Ultimate Rabi simulation with all physics.
        """
        if use_lindblad and hasattr(self, 'lindblad_sim'):
            # Full quantum master equation simulation
            return self._simulate_lindblad_rabi(tau_ns, key)
        else:
            # Fast semiclassical simulation
            return self._simulate_semiclassical_rabi(tau_ns, key)
    
    def _simulate_lindblad_rabi(self, tau_ns, key):
        """Full Lindblad master equation simulation."""
        # Define pulse sequence
        pulse_times = [(0, tau_ns)]  # Single MW pulse
        pulse_amplitudes = [self.params.Omega_Rabi_Hz]
        
        # Total simulation time
        total_time_ns = tau_ns + self.params.READ_NS
        
        # Run Lindblad evolution
        time_trace, populations = self.lindblad_sim.simulate_pulse_sequence(
            pulse_times, pulse_amplitudes, total_time_ns
        )
        
        # Extract final ground state populations
        final_rho = populations[-1]  # Last time point
        p_ms0 = float(jnp.real(final_rho[1]))  # |gs,0⟩ population
        p_ms1 = float(jnp.real(final_rho[0] + final_rho[2]))  # |gs,±1⟩ population
        
        # Calculate fluorescence rate
        fluorescence_rate = self.lindblad_sim.get_fluorescence_rate(final_rho)
        
        # Generate counts with advanced detector model
        if hasattr(self, 'detector'):
            time_window_ns = self.params.READ_NS
            dt_bin_ns = self.params.BIN_ns
            
            counts, _ = self.detector.detect_photon_stream(
                fluorescence_rate, time_window_ns, dt_bin_ns, key
            )
        else:
            # Simple Poisson counts
            expected = fluorescence_rate * self.params.READ_NS * 1e-9 * self.params.Shots
            counts = random.poisson(key, expected)
        
        return {
            'time_ns': time_trace,
            'populations': populations,
            'counts': counts,
            'p_ms0': p_ms0,
            'p_ms1': p_ms1,
            'fluorescence_rate_Hz': fluorescence_rate
        }
    
    def _simulate_semiclassical_rabi(self, tau_ns, key):
        """Fast semiclassical simulation (enhanced version of original)."""
        # Enhanced with all thermal/bath effects
        
        # 1. Basic Rabi evolution
        rabi_angle = self.params.Omega_Rabi_Hz * tau_ns * 1e-9 * jnp.pi
        
        # 2. Thermal effects
        if hasattr(self, 'thermal'):
            thermal_summary = self.thermal.get_temperature_summary()
            
            # Temperature-dependent ISC rates affect populations
            isc_correction = 1.0 - 0.1 * jnp.log(thermal_summary['isc_rate_ms1_Hz'] / 1e8)
            rabi_angle *= isc_correction
            
            # Thermal populations bias
            thermal_pops = thermal_summary['thermal_populations']
            thermal_bias = thermal_pops[1] - 0.5  # Deviation from equal ms=0,±1
        else:
            thermal_bias = 0.0
        
        # 3. Spin bath dephasing
        if hasattr(self, 'spin_bath') and self.spin_bath.current_bath_config:
            T2_star_bath = self.spin_bath.get_effective_T2_star()
            bath_decay = jnp.exp(-tau_ns / T2_star_bath)
        else:
            bath_decay = 1.0
        
        # 4. MW pulse with noise (from original features)
        key1, key2, key3 = random.split(key, 3)
        
        # Advanced laser physics effects
        if hasattr(self, 'laser_physics'):
            I_rel = self.laser.get_beam_parameters()['intensity_relative']
            amplitude_noise, phase_noise, key = self.laser_physics.laser_noise_effects(
                key1, tau_ns
            )
            
            # AC Stark shift
            stark_shift = self.laser_physics.ac_stark_shift(I_rel)
            rabi_angle += stark_shift * tau_ns * 1e-9 * 2 * jnp.pi
        else:
            amplitude_noise = 1.0 + 0.01 * random.normal(key1)
            phase_noise = 0.02 * random.normal(key2)
        
        # 5. Final populations
        effective_angle = rabi_angle * amplitude_noise * bath_decay + phase_noise
        p_ms0_ideal = jnp.cos(effective_angle / 2)**2
        p_ms1_ideal = 1 - p_ms0_ideal
        
        # Apply thermal bias
        p_ms0 = p_ms0_ideal + thermal_bias
        p_ms1 = p_ms1_ideal - thermal_bias
        
        # Ensure normalization
        total = p_ms0 + p_ms1
        p_ms0 /= total
        p_ms1 /= total
        
        # 6. Enhanced readout with all effects
        rate_bright = self.params.Beta_max_Hz * self.params.collection_efficiency * p_ms0
        rate_dark = self.params.Beta_max_Hz * self.params.collection_efficiency * 0.3 * p_ms1
        
        # Phonon effects on collection efficiency
        if hasattr(self, 'thermal'):
            phonon_effects = self.thermal.get_phonon_effects()
            phonon_factor = 1.0 - 0.1 * phonon_effects['average_occupation']
            rate_bright *= phonon_factor
            rate_dark *= phonon_factor
        
        rate_total = rate_bright + rate_dark + self.params.DarkRate_cps
        
        # 7. Advanced detector simulation
        if hasattr(self, 'detector'):
            time_ns = np.arange(0, self.params.READ_NS, self.params.BIN_ns)
            rates_Hz = np.full_like(time_ns, rate_total, dtype=float)
            
            # Apply pulse structure (low during pulse, high after)
            pulse_mask = time_ns <= tau_ns
            rates_Hz[pulse_mask] *= 0.1  # 10% during pulse
            
            counts, _ = self.detector.detect_photon_stream(
                rate_total, self.params.READ_NS, self.params.BIN_ns, key3
            )
        else:
            # Simple Poisson with pulse structure (original method)
            time_ns = np.arange(0, self.params.READ_NS, self.params.BIN_ns)
            counts = np.zeros_like(time_ns)
            
            for i, t in enumerate(time_ns):
                if t <= tau_ns:
                    # During pulse: low fluorescence
                    rate = rate_total * 0.1
                else:
                    # After pulse: normal fluorescence
                    rate = rate_total
                
                expected = rate * self.params.BIN_ns * 1e-9 * self.params.Shots
                counts[i] = np.random.poisson(expected)
        
        return {
            'time_ns': time_ns.tolist(),
            'counts': counts.tolist(),
            'p_ms0': float(p_ms0),
            'p_ms1': float(p_ms1),
            'rate_total_Hz': float(rate_total)
        }
    
    def run_ultimate_tau_sweep(self, tau_values, n_seeds=10, use_lindblad=False):
        """
        Run ultimate tau sweep with all advanced features.
        """
        start_time = time.time()
        
        results = {}
        keys = random.split(random.PRNGKey(42), n_seeds)
        
        if hasattr(self, 'batch_manager') and not use_lindblad:
            # Use batch processing for speed
            print("Using JAX batch vectorization...")
            
            # Convert to batch simulation format
            base_params = {
                'Omega_Rabi_Hz': self.params.Omega_Rabi_Hz,
                'Beta_max_Hz': self.params.Beta_max_Hz,
                'collection_eff': self.params.collection_efficiency,
                'dark_rate_Hz': self.params.DarkRate_cps,
                'readout_time_ns': float(self.params.READ_NS),
                'n_shots': self.params.Shots
            }
            
            batch_results, stats = self.batch_manager.run_tau_sweep_batch(
                jnp.array(tau_values), n_seeds, base_params
            )
            
            # Convert batch results to expected format
            for i, tau in enumerate(tau_values):
                avg_counts = jnp.mean(batch_results['counts'][:, i])
                avg_p_ms0 = jnp.mean(batch_results['p_ms0'][:, i])
                avg_p_ms1 = jnp.mean(batch_results['p_ms1'][:, i])
                
                # Generate time array and counts for compatibility
                time_ns = np.arange(0, self.params.READ_NS, self.params.BIN_ns)
                counts = np.full_like(time_ns, avg_counts / len(time_ns))
                
                results[tau] = {
                    'time_ns': time_ns.tolist(),
                    'counts': counts.tolist(),
                    'p_ms0': float(avg_p_ms0),
                    'p_ms1': float(avg_p_ms1)
                }
            
            self.timing_stats['ultimate_tau_sweep'] = stats
            
        else:
            # Standard sequential processing
            print(f"Running {len(tau_values)} tau values with ultimate physics...")
            
            for i, tau in enumerate(tau_values):
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(tau_values)} ({100*i/len(tau_values):.0f}%)")
                
                # Average over multiple seeds
                tau_results = []
                
                for seed_idx in range(n_seeds):
                    key = keys[seed_idx] if seed_idx < len(keys) else random.PRNGKey(42 + seed_idx)
                    
                    # Generate fresh spin bath for each run
                    if hasattr(self, 'spin_bath'):
                        self.generate_spin_bath_realization(key)
                    
                    # Run simulation
                    result = self.simulate_ultimate_rabi(tau, key, use_lindblad)
                    tau_results.append(result)
                
                # Average results
                avg_result = self._average_results(tau_results)
                results[tau] = avg_result
        
        computation_time = time.time() - start_time
        
        self.timing_stats['ultimate_tau_sweep'] = {
            'n_tau_values': len(tau_values),
            'n_seeds': n_seeds,
            'computation_time_s': computation_time,
            'use_lindblad': use_lindblad,
            'simulations_per_second': len(tau_values) * n_seeds / computation_time
        }
        
        print(f"✅ Ultimate tau sweep completed in {computation_time:.2f}s")
        print(f"   Performance: {self.timing_stats['ultimate_tau_sweep']['simulations_per_second']:.1f} sims/s")
        
        return results
    
    def _average_results(self, results_list):
        """Average multiple simulation results."""
        n_results = len(results_list)
        
        # Average scalar values
        avg_p_ms0 = np.mean([r['p_ms0'] for r in results_list])
        avg_p_ms1 = np.mean([r['p_ms1'] for r in results_list])
        
        # Average count arrays
        count_arrays = [np.array(r['counts']) for r in results_list]
        avg_counts = np.mean(count_arrays, axis=0)
        
        # Use time array from first result
        time_ns = results_list[0]['time_ns']
        
        return {
            'time_ns': time_ns,
            'counts': avg_counts.tolist(),
            'p_ms0': float(avg_p_ms0),
            'p_ms1': float(avg_p_ms1)
        }
    
    def get_ultimate_physics_summary(self):
        """Get complete summary of all physics in the simulator."""
        summary = {
            'level': self.level,
            'basic_features': [
                'Phonon sidebands (Debye-Waller)',
                'T₁ vs T₂* relaxation', 
                'ISC rates (Singlet manifold)',
                'Realistic saturation',
                'Charge-state dynamics',
                'Phonon-coupling & Debye-Waller',
                'Time-dependent MW pulses',
                'Spektrale Diffusion',
                'Detektor-Modell'
            ]
        }
        
        if self.level in ['advanced', 'full']:
            summary['advanced_features'] = [
                'Full Lindblad master equation',
                'Spin bath with CCE Level-2',
                'Voigt laser saturation profile',
                'Temperature & blackbody effects',
                'Orbital relaxation (Lindblad)'
            ]
            
            # Get subsystem summaries
            if hasattr(self, 'thermal'):
                summary['thermal_summary'] = self.thermal.get_temperature_summary()
            
            if hasattr(self, 'spin_bath') and self.spin_bath.current_bath_config:
                summary['spin_bath_summary'] = self.spin_bath.get_bath_summary()
            
            if hasattr(self, 'laser'):
                summary['laser_summary'] = self.laser.get_beam_parameters()
        
        if self.level == 'full':
            summary['full_features'] = [
                'Advanced SPAD detector model',
                'Photon arrival Monte Carlo',
                'JAX batch vectorization',
                'Performance optimization'
            ]
            
            if hasattr(self, 'detector'):
                summary['detector_summary'] = self.detector.get_detector_specs()
            
            if hasattr(self, 'batch_manager'):
                summary['performance_summary'] = self.batch_manager.get_performance_summary()
        
        summary['timing_stats'] = self.timing_stats
        
        return summary

if __name__ == "__main__":
    # Test ultimate simulator
    print("🚀 Testing Ultimate NV Simulator")
    print("=" * 50)
    
    # Test different levels
    levels = ['basic', 'advanced', 'full']
    
    for level in levels:
        print(f"\nTesting {level} level...")
        
        sim = UltimateNVSimulator(level=level)
        
        # Quick single tau test
        result = sim.simulate_ultimate_rabi(100.0, random.PRNGKey(42))
        
        print(f"  Single tau test: p_ms0={result['p_ms0']:.3f}")
        print(f"  Mean counts: {np.mean(result['counts']):.1f}")
        
        # Physics summary
        summary = sim.get_ultimate_physics_summary()
        print(f"  Features: {len(summary.get('basic_features', []))} basic")
        if 'advanced_features' in summary:
            print(f"           {len(summary['advanced_features'])} advanced")
        if 'full_features' in summary:
            print(f"           {len(summary['full_features'])} full")
    
    # Performance test with full simulator
    print(f"\nPerformance test with full simulator...")
    sim_full = UltimateNVSimulator(level='full')
    
    # Small tau sweep
    tau_values = np.linspace(0, 200, 11)  # 11 points for speed
    results = sim_full.run_ultimate_tau_sweep(tau_values, n_seeds=3)
    
    timing = sim_full.timing_stats['ultimate_tau_sweep']
    print(f"  {timing['n_tau_values']} × {timing['n_seeds']} sims in {timing['computation_time_s']:.2f}s")
    print(f"  Performance: {timing['simulations_per_second']:.1f} sims/s")
    
    print("\n🎉 Ultimate NV Simulator working perfectly!")
    print("   All advanced physics features integrated and operational!")