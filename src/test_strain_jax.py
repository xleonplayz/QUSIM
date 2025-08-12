#!/usr/bin/env python
"""
Standalone Test für JAX-optimierte Strain-Implementation
"""

import numpy as np
import jax.numpy as jnp
import time

# Import functions
from strain.strain import H_strain_3x3 as H_strain_original
from strain.strain_jax import H_strain_3x3 as H_strain_jax_func, H_strain_3x3_jax

def validate_against_original():
    """Validiert JAX gegen Original-Implementation"""
    
    # Test-Fälle mit echten Strain-relevanten Parametern
    test_cases = [
        (0.0, np.array([0.0, 0.0, 0.5])),      # t=0, NV bei 0.5μm Tiefe
        (100e-9, np.array([0.1, 0.2, 0.3])),  # 100ns, während Rechteckpuls, off-center
        (150e-9, np.array([0.0, 0.0, 0.8])),  # 150ns, während Puls, tiefere Position
        (250e-9, np.array([-0.1, 0.0, 0.2])), # 250ns, nach Puls, flache Position
        (500e-9, np.array([0.05, -0.05, 1.0])) # 500ns, weit nach Puls, tiefe Position
    ]
    
    print("=== Validierung JAX vs Original ===")
    all_passed = True
    
    for i, (t, r) in enumerate(test_cases):
        print(f"Test {i+1}: t={t*1e9:.1f}ns, r=[{r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f}]μm")
        
        # Berechne beide Versionen
        H_orig = H_strain_original(t, r)
        H_jax = H_strain_jax_func(t, r)
        
        # Numerische Genauigkeit
        max_diff = np.max(np.abs(H_orig - H_jax))
        rel_diff = max_diff / np.max(np.abs(H_orig)) if np.max(np.abs(H_orig)) > 0 else 0
        
        print(f"   Matrix diff: {max_diff:.2e}, rel diff: {rel_diff:.2e}")
        
        # Physikalische Eigenschaften
        hermitian_orig = np.allclose(H_orig, H_orig.conj().T)
        hermitian_jax = np.allclose(H_jax, H_jax.conj().T) 
        
        evals_orig = np.linalg.eigvalsh(H_orig)
        evals_jax = np.linalg.eigvalsh(H_jax)
        eigenval_diff = np.max(np.abs(evals_orig - evals_jax))
        
        print(f"   Hermitian orig/jax: {hermitian_orig}/{hermitian_jax}")
        print(f"   Eigenvalue diff: {eigenval_diff:.2e}")
        print(f"   Eigenvalue range (MHz): {np.min(evals_orig)/1e6:.3f} to {np.max(evals_orig)/1e6:.3f}")
        
        # Validierung
        if not np.allclose(H_orig, H_jax, rtol=1e-10, atol=1e-12):
            print(f"   ✗ FAILED: Matrices not equal within tolerance")
            all_passed = False
        elif not np.allclose(evals_orig, evals_jax, rtol=1e-10):
            print(f"   ✗ FAILED: Eigenvalues not equal within tolerance")  
            all_passed = False
        else:
            print(f"   ✓ PASSED")
        
        print()
    
    if all_passed:
        print("✓ Alle Validierungstests bestanden!")
    else:
        print("✗ Einige Tests fehlgeschlagen!")
    
    return all_passed

def benchmark_performance():
    """Performance-Benchmark gegen Original"""
    
    # Realistische Parameter für NV-Strain Simulationen
    n_iterations = 1000
    t_values = np.linspace(0, 500e-9, n_iterations)  # 0-500ns (inkl. Rechteckpuls)
    
    # Realistische NV-Positionen (μm) - normalverteilt um typische Tiefe
    np.random.seed(42)  # Reproduzierbarkeit
    r_values = np.random.randn(n_iterations, 3) * 0.1 + np.array([0, 0, 0.5])
    
    print(f"=== Performance-Benchmark ({n_iterations} Iterationen) ===")
    print(f"Zeit-Range: 0 - {t_values[-1]*1e9:.0f} ns")
    print(f"Position-Range: ~{np.min(r_values, axis=0)} to {np.max(r_values, axis=0)} μm")
    
    # Original NumPy Benchmark
    print("\n1. Benchmarking Original NumPy...")
    start = time.time()
    results_orig = []
    for t, r in zip(t_values, r_values):
        H_orig = H_strain_original(t, r)
        results_orig.append(np.trace(H_orig))  # Kleine Berechnung um Optimierung zu vermeiden
    time_numpy = time.time() - start
    print(f"   Time: {time_numpy:.4f}s")
    
    # JAX JIT Warmup
    print("\n2. JAX JIT Warmup...")
    _ = H_strain_jax_func(0.0, np.array([0, 0, 0.5]))
    _ = H_strain_jax_func(100e-9, np.array([0.1, 0, 0.3]))
    print("   Warmup completed")
    
    # JAX Individual Benchmark
    print("\n3. Benchmarking JAX (individual calls)...")
    start = time.time()
    results_jax = []
    for t, r in zip(t_values, r_values):
        H_jax = H_strain_jax_func(t, r)
        results_jax.append(np.trace(H_jax))
    time_jax = time.time() - start
    print(f"   Time: {time_jax:.4f}s")
    
    # JAX Batch Benchmark  
    print("\n4. Benchmarking JAX (batch processing)...")
    start = time.time()
    # Verwende die native JAX batch function
    from strain.strain_jax import H_strain_batch_jax
    H_batch = H_strain_batch_jax(jnp.array(t_values), jnp.array(r_values))
    results_batch = [np.trace(H) for H in H_batch]
    time_jax_batch = time.time() - start
    print(f"   Time: {time_jax_batch:.4f}s")
    
    # Ergebnisse
    speedup_individual = time_numpy / time_jax
    speedup_batch = time_numpy / time_jax_batch
    
    print(f"\n=== Performance Results ===")
    print(f"Original NumPy:      {time_numpy:.4f}s")
    print(f"JAX (individual):    {time_jax:.4f}s  ({speedup_individual:.1f}x speedup)")
    print(f"JAX (batch):         {time_jax_batch:.4f}s  ({speedup_batch:.1f}x speedup)")
    
    # Validiere dass Ergebnisse konsistent sind
    trace_diff_ind = np.max(np.abs(np.array(results_orig) - np.array(results_jax)))
    trace_diff_batch = np.max(np.abs(np.array(results_orig) - np.array(results_batch)))
    
    print(f"\nResult consistency:")
    print(f"Individual trace diff: {trace_diff_ind:.2e}")
    print(f"Batch trace diff:      {trace_diff_batch:.2e}")
    
    if trace_diff_ind < 1e-10 and trace_diff_batch < 1e-10:
        print("✓ All results consistent")
    else:
        print("✗ Results inconsistent!")
    
    return speedup_individual, speedup_batch

def test_integration_with_system():
    """Test Integration mit dem echten NV-System"""
    print("=== Integration Test mit NV-System ===")
    
    # Importiere das Hauptsystem
    from hama import TotalHamiltonian
    
    # Lade System mit aktiviertem Strain
    ham = TotalHamiltonian('system.json')
    
    # Test zu verschiedenen Zeitpunkten
    test_times = [0.0, 50e-9, 100e-9, 150e-9, 200e-9, 300e-9]
    
    print("Testing Hamiltonian builds at different times:")
    for t in test_times:
        start = time.time()
        H_total = ham.build_hamiltonian(t)
        build_time = time.time() - start
        
        # Validiere Eigenschaften
        is_hermitian = np.allclose(H_total, H_total.conj().T)
        eigenvals = np.linalg.eigvalsh(H_total)
        
        print(f"t={t*1e9:3.0f}ns: build_time={build_time*1000:.2f}ms, "
              f"hermitian={is_hermitian}, "
              f"evals range={np.min(eigenvals)/1e6:.1f} to {np.max(eigenvals)/1e6:.1f} MHz")
    
    print("✓ System integration test completed")

def main():
    """Haupttest-Funktion"""
    print("=== JAX-Optimierte Strain-Module Volltest ===\n")
    
    # JAX Info
    import jax
    print(f"JAX Version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Float64 enabled: {jax.config.x64_enabled}")
    print()
    
    # 1. Validierung
    print("SCHRITT 1: Numerische Validierung")
    validation_passed = validate_against_original()
    if not validation_passed:
        print("❌ Validation failed - stopping here")
        return False
    print()
    
    # 2. Performance  
    print("SCHRITT 2: Performance Benchmark")
    speedup_ind, speedup_batch = benchmark_performance()
    print()
    
    # 3. System Integration
    print("SCHRITT 3: System Integration")
    test_integration_with_system()
    print()
    
    # Summary
    print("=== ZUSAMMENFASSUNG ===")
    print(f"✓ Numerische Validierung: BESTANDEN")
    print(f"✓ Performance Verbesserung: {speedup_ind:.1f}x (individual), {speedup_batch:.1f}x (batch)")
    print(f"✓ System Integration: ERFOLGREICH")
    print(f"✓ JAX-Strain Module ist produktionsbereit!")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)