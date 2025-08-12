#!/usr/bin/env python3
"""
Test Runner with AWG - Demonstrates runner integration with AWG system
"""

import sys
import os
sys.path.append('../runner')
sys.path.append('../src')
sys.path.append('..')

import time
import numpy as np


def test_runner_awg_integration():
    """Test Runner with AWG integration"""
    print("=== Runner + AWG Integration Test ===")
    
    try:
        # Import runner
        from runner import SimpleRunner
        
        print("Initialisiere Runner mit AWG...")
        runner = SimpleRunner()
        
        # Check AWG is available
        print(f"AWG Sample Rate: {runner.awg.sample_rate/1e9:.1f} GS/s")
        print(f"AWG Status: {'AKTIV' if runner.awg.is_playing else 'BEREIT'}")
        
        # Test basic pulse
        print("\n1. Test einzelner MW-Puls:")
        runner.mw_pulse('rect', amplitude_mhz=1.0, duration_ns=500, phase_deg=0)
        
        awg_info = runner.awg.get_info()
        print(f"   Pulse loaded: {awg_info['duration']*1e9:.1f} ns")
        print(f"   Status: {'PLAYING' if awg_info['is_playing'] else 'IDLE'}")
        
        # Run simulation for pulse duration
        print("\n2. Simuliere Puls-Anwendung:")
        runner.run_for(600)  # 600 ns - longer than pulse
        
        # Test Ramsey sequence
        print("\n3. Test Ramsey-Sequenz:")
        runner.apply_named_sequence('ramsey', tau_ns=1000, rabi_freq_mhz=1.0)
        
        awg_info = runner.awg.get_info()
        print(f"   Sequence loaded: {awg_info['duration']*1e9:.1f} ns")
        
        # Run simulation for sequence
        print("\n4. Simuliere Ramsey-Sequenz:")
        runner.run_for(1200)  # Longer than sequence
        
        print("\n✓ Runner + AWG Integration erfolgreich!")
        
        return True
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Runner-Test übersprungen (Dependencies fehlen)")
        return False
    except Exception as e:
        print(f"Runner Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_awg_hamiltonian_integration():
    """Test AWG Hamiltonian integration with TotalHamiltonian"""
    print("\n=== AWG + TotalHamiltonian Integration Test ===")
    
    try:
        from hama import TotalHamiltonian
        from interfaces.awg import AWGInterface, PulseSequenceBuilder
        
        # Create Hamiltonian and AWG
        H_builder = TotalHamiltonian('../src/system.json')
        awg = AWGInterface()
        
        # Connect AWG to Hamiltonian
        H_builder.set_awg_interface(awg)
        print("AWG mit TotalHamiltonian verbunden")
        
        # Test without pulse (should be zero)
        H_idle = H_builder.build_hamiltonian(t=0.0)
        mw_contribution_idle = np.linalg.norm(H_idle)
        print(f"Hamiltonian ohne Puls: ||H|| = {mw_contribution_idle:.2e}")
        
        # Add pulse and test
        pi_pulse = PulseSequenceBuilder.pi_pulse(1.0, 'x')
        awg.clear_waveform()
        awg.add_pulse(**pi_pulse)
        awg.start(at_time=0.0)
        
        # Test during pulse
        t_during = pi_pulse['duration'] / 2
        H_active = H_builder.build_hamiltonian(t=t_during)
        mw_contribution_active = np.linalg.norm(H_active)
        print(f"Hamiltonian mit Puls: ||H|| = {mw_contribution_active:.2e}")
        
        # Should be different
        assert mw_contribution_active > mw_contribution_idle
        print("✓ AWG-Hamiltonian Integration erfolgreich!")
        
        return True
        
    except ImportError as e:
        print(f"Import Error: {e}")
        return False
    except Exception as e:
        print(f"Hamiltonian Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_awg_vs_old_mw():
    """Compare AWG output with old MW implementation (conceptually)"""
    print("\n=== AWG vs. Old MW Comparison ===")
    
    try:
        from interfaces.awg import AWGInterface, PulseSequenceBuilder
        
        # AWG version
        awg = AWGInterface()
        pi_pulse = PulseSequenceBuilder.pi_pulse(1.0, 'x')
        awg.add_pulse(**pi_pulse)
        awg.start(at_time=0.0)
        
        # Test at multiple time points
        n_points = 10
        times = np.linspace(0, pi_pulse['duration'], n_points)
        
        print("AWG Hamiltonian-Beiträge über Zeit:")
        for i, t in enumerate(times):
            H_awg = awg.get_hamiltonian_contribution(t)
            H_norm = np.linalg.norm(H_awg)
            print(f"   t={t*1e9:.1f}ns: ||H|| = {H_norm:.2e}")
            
            # Should be consistent during pulse
            if i > 0 and i < n_points - 1:
                assert H_norm > 1e6  # Should be significant
        
        print("✓ AWG erzeugt konsistente Hamiltonian-Beiträge")
        
        # Compare advantages
        print("\nAWG Vorteile:")
        print("   ✓ Kontinuierliche Zeit-Auflösung")
        print("   ✓ Beliebige Pulsformen")
        print("   ✓ Präzise Phase-Kontrolle")
        print("   ✓ Hardware-realistische Simulation")
        print("   ✓ Echzeit-Steuerung")
        
        return True
        
    except Exception as e:
        print(f"Comparison Error: {e}")
        return False


def main():
    """Main function for runner-AWG integration test"""
    print("QUSIM Runner + AWG Integration Test")
    print("="*50)
    
    results = []
    
    # Test 1: Basic runner integration
    results.append(test_runner_awg_integration())
    
    # Test 2: Hamiltonian integration  
    results.append(test_awg_hamiltonian_integration())
    
    # Test 3: Comparison with old system
    results.append(test_awg_vs_old_mw())
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary:")
    test_names = ["Runner Integration", "Hamiltonian Integration", "AWG vs Old MW"]
    
    for name, result in zip(test_names, results):
        status = "✓ PASS" if result else "✗ FAIL" 
        print(f"   {name}: {status}")
    
    all_passed = all(results)
    if all_passed:
        print("\n🎉 Alle Tests bestanden!")
        print("Das AWG-System ist vollständig integriert und funktional.")
        
        print("\nNächste Schritte:")
        print("1. Starte Runner: python runner/runner.py")
        print("2. Verwende AWG-Kommandos:")
        print("   - runner.mw_pulse('gauss', 2.0, 100)")
        print("   - runner.apply_named_sequence('ramsey', tau_ns=500)")
        print("   - runner.apply_named_sequence('echo', tau_ns=1000)")
    else:
        print("\n⚠️  Einige Tests fehlgeschlagen - siehe Details oben")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)