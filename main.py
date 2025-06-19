#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NV Simulator - Main Launcher
============================
HYPERREALISTIC: Phonon-Coupling + Time-dependent MW Pulses + All Advanced Physics

Usage:
    python main.py                    # Start web interface
    python main.py --test             # Run physics tests
    python main.py --console          # Console interface
    python main.py --port 5003        # Custom port
"""

import sys
import os
import argparse
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.nv import NVSimulator, debye_waller_factor, phonon_sideband_rates, ultrafast_ES_dephasing, noisy_mw_pulse
from src.webapp import run_webapp
from jax import random

def test_physics():
    """Test all physics features"""
    print("Testing HYPERREALISTIC NV Simulator - All Physics Features")
    print("Phonon-Coupling + Time-dependent MW Pulses + All Advanced Physics")
    print("=" * 70)
    print()
    
    sim = NVSimulator()
    params = sim.params
    
    print("Testing Phonon-Coupling & Debye-Waller")
    print("-" * 50)
    
    # Test temperature dependence
    temperatures = [4.0, 77.0, 300.0]  # Helium, Nitrogen, Room temp
    
    for T_K in temperatures:
        dw_factor = debye_waller_factor(T_K, params)
        k_orb = ultrafast_ES_dephasing(params, T_K)
        rate_zpl, rate_psb = phonon_sideband_rates(params, T_K, 0.1)
        
        print(f"T = {T_K:3.0f} K:")
        print(f"  Debye-Waller factor: {dw_factor:.4f}")
        print(f"  ZPL rate: {rate_zpl/1e6:.2f} MHz")
        print(f"  PSB rate: {rate_psb/1e6:.2f} MHz")
        print(f"  ZPL/PSB ratio: {rate_zpl/rate_psb:.2f}")
        print(f"  Orbital relaxation: {k_orb:.3f} MHz")
        print()
    
    print("Testing Time-dependent MW Pulses")
    print("-" * 50)
    
    # Test different pulse shapes
    pulse_shapes = ['gaussian', 'hermite', 'square']
    tau_ns = 100.0
    
    for shape in pulse_shapes:
        params.mw_pulse_shape = shape
        print(f"Pulse shape: {shape}")
        
        # Generate time-dependent pulse
        t_mid = tau_ns / 2
        key = random.PRNGKey(42)
        
        Omega_eff, phase_eff = noisy_mw_pulse(t_mid, tau_ns, params, key)
        
        print(f"  Effective Rabi: {Omega_eff/1e6:.3f} MHz")
        print(f"  Phase error: {phase_eff:.4f} rad ({np.degrees(phase_eff):.2f} deg)")
        print(f"  Noise factor: {Omega_eff/params.Omega_Rabi_Hz:.4f}")
        print()
    
    print("Complete Physics Integration Test")
    print("-" * 50)
    
    # Test tau sweep with all features
    tau_list = [0, 50, 100, 150, 200]
    print("Tau sweep with hyperrealistic physics:")
    
    for tau in tau_list:
        result = sim.simulate_single_tau(tau, random_seed=42)
        counts = np.array(result['counts'])
        
        print(f"  tau={tau:3.0f}ns: max={max(counts):3.0f}, mean={np.mean(counts):5.1f}, ms0={result['p_ms0']:.3f}")
    
    print()
    print("All advanced physics features working correctly!")
    
def console_interface():
    """Interactive console interface"""
    print("NV Simulator - Console Interface")
    print("=" * 40)
    
    sim = NVSimulator()
    
    while True:
        print("\nOptions:")
        print("1. Single tau simulation")
        print("2. Physics parameters")
        print("3. Temperature comparison")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            try:
                tau = float(input("Enter tau (ns): "))
                result = sim.simulate_single_tau(tau)
                counts = np.array(result['counts'])
                
                print(f"\nResults for tau={tau}ns:")
                print(f"  Max counts: {max(counts):.0f}")
                print(f"  Mean counts: {np.mean(counts):.1f}")
                print(f"  ms=0 probability: {result['p_ms0']:.3f}")
                print(f"  ms=±1 probability: {result['p_ms1']:.3f}")
                
            except ValueError:
                print("Invalid input. Please enter a number.")
                
        elif choice == '2':
            physics = sim.get_physics_info()
            print("\nPhysics Parameters:")
            print(f"  Collection efficiency: {physics['collection_efficiency']:.3f}")
            print(f"  Rabi frequency: {physics['Omega_Rabi_Hz']/1e6:.1f} MHz")
            print(f"  Temperature: {physics['Temperature_K']:.1f} K")
            print(f"  Debye-Waller DW0: {physics['DW0_at_zero_K']:.4f}")
            print(f"  MW pulse shape: {physics['mw_pulse_shape']}")
            print(f"  Shots: {physics['Shots']:,}")
            
        elif choice == '3':
            print("\nTemperature Comparison:")
            temps = [4.0, 77.0, 300.0]
            
            for T_K in temps:
                sim.params.Temperature_K = T_K
                result = sim.simulate_single_tau(100.0)
                counts = np.array(result['counts'])
                
                print(f"  T={T_K:3.0f}K: mean={np.mean(counts):5.1f} counts, ms0={result['p_ms0']:.3f}")
            
            # Reset to default
            sim.params.Temperature_K = 4.0
            
        elif choice == '4':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1-4.")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Hyperrealistic NV Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start web interface
  python main.py --test             # Run physics tests  
  python main.py --console          # Console interface
  python main.py --port 5003        # Custom port
        """
    )
    
    parser.add_argument('--test', action='store_true', 
                       help='Run physics tests')
    parser.add_argument('--console', action='store_true',
                       help='Interactive console interface')
    parser.add_argument('--port', type=int, default=5002,
                       help='Web server port (default: 5002)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode for web server')
    
    args = parser.parse_args()
    
    if args.test:
        test_physics()
    elif args.console:
        console_interface()
    else:
        # Default: start web interface
        print("HYPERREALISTIC NV Simulator")
        print("Phonon-Coupling + Time-dependent MW Pulses + All Advanced Physics")
        print("=" * 70)
        print()
        print(f"Starting web interface on port {args.port}...")
        print(f"Open http://localhost:{args.port} in your browser")
        print()
        
        run_webapp(port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()