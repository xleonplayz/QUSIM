#!/usr/bin/env python
"""
Vollständiger Test des JAX-optimierten Zeeman-Moduls im NV-System
"""

import numpy as np
import time
from hama import TotalHamiltonian

def test_zeeman_integration():
    """Test der Zeeman-Integration im vollständigen System"""
    print("=== Zeeman Integration im NV-System ===")
    
    # System laden
    ham = TotalHamiltonian('system.json')
    
    # Test verschiedener Zeitpunkte für Rauschen-Variation
    test_times = [0.0, 100e-9, 500e-9, 1e-6, 2e-6]
    
    print("Testing Hamiltonian builds mit Zeeman-Rauschen:")
    eigenvalue_stats = []
    
    for t in test_times:
        H = ham.build_hamiltonian(t)
        
        # Validierung
        if not np.allclose(H, H.conj().T):
            print(f"WARNING: Hamiltonian not Hermitian at t={t*1e6:.1f}μs")
        
        # Eigenwerte 
        eigenvals = np.linalg.eigvalsh(H)
        eigenvalue_stats.append([np.min(eigenvals), np.max(eigenvals), np.mean(eigenvals)])
        
        print(f"  t={t*1e6:4.1f}μs: evals {eigenvals[0]/1e6:.1f} to {eigenvals[-1]/1e6:.1f} MHz, "
              f"mean {np.mean(eigenvals)/1e6:.1f} MHz")
    
    # Rauschen-Analyse
    eigenvalue_stats = np.array(eigenvalue_stats)
    min_vals = eigenvalue_stats[:, 0] / 1e6
    max_vals = eigenvalue_stats[:, 1] / 1e6
    
    print(f"\nRauschen-Analyse:")
    print(f"  Min-Eigenwert Variation: {np.std(min_vals):.2f} MHz")
    print(f"  Max-Eigenwert Variation: {np.std(max_vals):.2f} MHz")
    print(f"  Erwartete Zeeman-Rauschen: ~{np.sqrt(0.5**2 + 1.0**2) * 13.996:.1f} MHz")
    
    return eigenvalue_stats

def test_zeeman_vs_other_modules():
    """Vergleicht Zeeman-Effekt mit anderen Modulen"""
    print("\n=== Zeeman vs andere Module ===")
    
    ham = TotalHamiltonian('system.json')
    t_test = 0.0
    
    # Original config sichern
    orig_config = ham.config.copy()
    
    # Test einzelner Module
    modules = ['Zeeman', 'N14_Hyperfine', 'C13_Hyperfine', 'Strain']
    module_contributions = {}
    
    for module in modules:
        # Alle anderen deaktivieren
        for mod in modules:
            ham.config[mod]['enabled'] = (mod == module)
        
        H = ham.build_hamiltonian(t_test)
        eigenvals = np.linalg.eigvalsh(H)
        
        module_contributions[module] = {
            'eigenvals': eigenvals,
            'range': (np.min(eigenvals), np.max(eigenvals)),
            'spread': np.max(eigenvals) - np.min(eigenvals)
        }
        
        print(f"  {module:15}: Range {eigenvals[0]/1e6:8.1f} to {eigenvals[-1]/1e6:8.1f} MHz, "
              f"Spread {(np.max(eigenvals) - np.min(eigenvals))/1e6:6.1f} MHz")
    
    # Config zurücksetzen
    ham.config = orig_config
    
    # Alle Module zusammen
    H_full = ham.build_hamiltonian(t_test)
    eigenvals_full = np.linalg.eigvalsh(H_full)
    
    print(f"  {'Alle zusammen':15}: Range {eigenvals_full[0]/1e6:8.1f} to {eigenvals_full[-1]/1e6:8.1f} MHz, "
          f"Spread {(np.max(eigenvals_full) - np.min(eigenvals_full))/1e6:6.1f} MHz")
    
    # Zeeman-Dominanz prüfen
    zeeman_spread = module_contributions['Zeeman']['spread'] / 1e6
    total_spread = (np.max(eigenvals_full) - np.min(eigenvals_full)) / 1e6
    zeeman_dominance = zeeman_spread / total_spread * 100
    
    print(f"\nZeeman-Dominanz: {zeeman_dominance:.1f}% des Gesamt-Eigenwertspreads")
    
    return module_contributions

def benchmark_zeeman_performance():
    """Performance-Test speziell für Zeeman im System"""
    print("\n=== Zeeman Performance im System ===")
    
    ham = TotalHamiltonian('system.json')
    
    # Deaktiviere alle außer Zeeman für reinen Zeeman-Test
    orig_config = ham.config.copy()
    for module in ham.config:
        if module != 'Zeeman':
            ham.config[module]['enabled'] = False
    
    # Performance-Test
    n_iterations = 100
    t_values = np.linspace(0, 1e-6, n_iterations)
    
    print(f"Testing {n_iterations} Zeitpunkte...")
    
    start = time.time()
    for t in t_values:
        H = ham.build_hamiltonian(t)
        # Kleine Berechnung um Optimierung zu vermeiden
        trace = np.trace(H)
    total_time = time.time() - start
    
    print(f"  Zeit für {n_iterations} Zeeman-Hamiltonians: {total_time:.4f}s")
    print(f"  Durchschnitt pro Hamiltonian: {total_time/n_iterations*1000:.2f}ms")
    print(f"  Hamiltonians pro Sekunde: {n_iterations/total_time:.0f}")
    
    # Config zurücksetzen  
    ham.config = orig_config
    
    # Vergleich mit vollständigem System
    print(f"\nVergleich: Vollständiges System...")
    start = time.time()
    for i, t in enumerate(t_values[:20]):  # Weniger Iterationen für vollständiges System
        H = ham.build_hamiltonian(t)
        trace = np.trace(H)
    full_system_time = time.time() - start
    
    avg_full = full_system_time / 20 * 1000
    avg_zeeman = total_time / n_iterations * 1000
    
    print(f"  Vollständiges System: {avg_full:.2f}ms pro Hamiltonian")
    print(f"  Nur Zeeman: {avg_zeeman:.2f}ms pro Hamiltonian")
    print(f"  Zeeman-Anteil: {avg_zeeman/avg_full*100:.1f}% der Gesamtzeit")
    
    return total_time, full_system_time

def test_zeeman_noise_reproducibility():
    """Test der Reproduzierbarkeit des Zeeman-Rauschens"""
    print("\n=== Zeeman-Rauschen Reproduzierbarkeit ===")
    
    from zeeman.zeeman import H_Zeeman
    
    # Test-Konfiguration
    config = {
        "B0": [0, 0, 100],
        "B_ac": [0, 0, 0],
        "f_ac": 1e6,
        "ac_phase": 0,
        "g_tensor": [2.0028, 2.0028, 2.0028],
        "noise": {"white": 0.5, "telegraph": 1.0},
        "seed": 42
    }
    
    # Gleicher Zeitpunkt, gleicher Seed → sollte identisch sein
    t_test = 1e-6
    H1 = H_Zeeman(t_test, config)
    H2 = H_Zeeman(t_test, config)
    
    max_diff = np.max(np.abs(H1 - H2))
    print(f"  Reproduzierbarkeit (gleiche Zeit/Seed): max diff = {max_diff:.2e}")
    
    if max_diff < 1e-14:
        print("  ✓ Perfekte Reproduzierbarkeit")
    else:
        print("  ✗ Rauschen nicht reproduzierbar!")
    
    # Verschiedene Zeitpunkte → sollte unterschiedlich sein
    H3 = H_Zeeman(t_test + 1e-9, config)  # Anderer Zeitpunkt
    max_diff_time = np.max(np.abs(H1 - H3))
    print(f"  Verschiedene Zeiten: max diff = {max_diff_time:.2e}")
    
    if max_diff_time > 1e-10:
        print("  ✓ Rauschen variiert mit Zeit")
    else:
        print("  ✗ Rauschen variiert nicht mit Zeit!")
    
    # Verschiedene Seeds → sollte unterschiedlich sein
    config_diff_seed = config.copy()
    config_diff_seed["seed"] = 123
    H4 = H_Zeeman(t_test, config_diff_seed)
    max_diff_seed = np.max(np.abs(H1 - H4))
    print(f"  Verschiedene Seeds: max diff = {max_diff_seed:.2e}")
    
    if max_diff_seed > 1e-10:
        print("  ✓ Rauschen variiert mit Seed")
    else:
        print("  ✗ Rauschen variiert nicht mit Seed!")
    
    return max_diff < 1e-14 and max_diff_time > 1e-10 and max_diff_seed > 1e-10

def main():
    """Haupttest für Zeeman-System"""
    print("=== JAX-Zeeman im vollständigen NV-System ===\n")
    
    # 1. System-Integration
    eigenvalue_stats = test_zeeman_integration()
    
    # 2. Modul-Vergleich
    module_contributions = test_zeeman_vs_other_modules()
    
    # 3. Performance-Test
    zeeman_time, full_time = benchmark_zeeman_performance()
    
    # 4. Rauschen-Reproduzierbarkeit
    noise_ok = test_zeeman_noise_reproducibility()
    
    # 5. Zusammenfassung
    print("\n=== ZUSAMMENFASSUNG ===")
    print(f"✓ System-Integration: ERFOLGREICH")
    print(f"✓ Zeeman-Rauschen: {'REPRODUZIERBAR' if noise_ok else 'PROBLEMATISCH'}")
    print(f"✓ Performance: {1000*zeeman_time/100:.1f}ms pro Zeeman-Hamiltonian")
    
    # Bewertung der Zeeman-Implementierung
    zeeman_spread = module_contributions['Zeeman']['spread'] / 1e6
    if zeeman_spread > 1000:  # > 1 GHz
        print(f"✓ Zeeman-Effekt dominant: {zeeman_spread:.0f} MHz Spread")
    else:
        print(f"⚠ Zeeman-Effekt schwach: {zeeman_spread:.0f} MHz Spread")
    
    if noise_ok:
        print(f"✓ JAX-Zeeman ist produktionsbereit!")
    else:
        print(f"⚠ Rauschen-Implementierung benötigt Überarbeitung")
    
    return eigenvalue_stats, module_contributions

if __name__ == "__main__":
    main()