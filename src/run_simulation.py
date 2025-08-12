#!/usr/bin/env python
"""
Beispiel-Script für NV-Zentrum Simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from hama import TotalHamiltonian
from lindblad import LindbladEvolution

# Setup
print("Lade Konfiguration...")
ham = TotalHamiltonian('system.json')

# Hamiltonian bei t=0
H = ham.build_hamiltonian(0.0)
print(f"Hamiltonian-Dimension: {H.shape}")

# Eigenwerte
eigvals = np.linalg.eigvalsh(H)
print(f"\nEigenwerte (MHz):")
print(f"  Min: {np.min(eigvals)/1e6:.3f}")
print(f"  Max: {np.max(eigvals)/1e6:.3f}")
print(f"  Spread: {(np.max(eigvals)-np.min(eigvals))/1e6:.3f}")

# Optional: Zeitentwicklung
run_dynamics = input("\nZeitentwicklung berechnen? (j/n): ")

if run_dynamics.lower() == 'j':
    # Lindblad-Solver
    solver = LindbladEvolution('system.json')
    
    # Anfangszustand: |ms=0, mN=0, mC=-1/2⟩
    rho0 = np.zeros((18, 18), dtype=complex)
    rho0[4, 4] = 1.0  # mittlerer Zustand
    
    # Zeitentwicklung
    print("\nBerechne Zeitentwicklung...")
    t_span = [0, 1e-6]  # 1 Mikrosekunde
    t_eval = np.linspace(0, 1e-6, 100)
    
    t_result, rho_result = solver.evolve(t_span, initial_state_type='ground')
    
    # Populationen extrahieren
    populations = np.real([[rho[i, i] for rho in rho_result] for i in range(18)])
    
    # Plot
    plt.figure(figsize=(10, 6))
    for i in range(min(6, 18)):  # Erste 6 Zustände
        plt.plot(t_result * 1e6, populations[i], label=f'|{i}⟩')
    
    plt.xlabel('Zeit (μs)')
    plt.ylabel('Population')
    plt.title('NV-Zentrum Dynamik')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"\nEndpopulationen:")
    for i in range(min(6, 18)):
        print(f"  |{i}⟩: {populations[i][-1]:.4f}")

print("\n✅ Simulation abgeschlossen!")