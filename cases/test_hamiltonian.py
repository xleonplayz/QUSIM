import numpy as np
import sys
sys.path.append('../src')

from hama import TotalHamiltonian

# Test des Gesamt-Hamiltonians
print("=== Test des 18x18 Gesamt-Hamiltonians ===\n")

# Erstelle Hamiltonian
H_builder = TotalHamiltonian('../src/system.json')
H = H_builder.build_hamiltonian(t=0.0)

print(f"Hamiltonian-Dimension: {H.shape}")
print(f"Hamiltonian ist hermitesch: {np.allclose(H, H.conj().T)}")

# Berechne Eigenwerte
evals = np.linalg.eigvalsh(H)
evals = np.sort(evals)

print("\nEnergie-Eigenwerte (MHz):")
for i in range(min(10, len(evals))):
    print(f"  E_{i}: {evals[i]/1e6:.3f} MHz")

# Energie-Aufspaltungen
print("\nEnergie-Aufspaltungen vom Grundzustand (MHz):")
for i in range(1, min(6, len(evals))):
    dE = evals[i] - evals[0]
    print(f"  ΔE_{i,0}: {dE/1e6:.3f} MHz")

# Zeige welche Terme aktiv sind
print("\n=== Aktive Hamiltonian-Terme ===")
config = H_builder.config
for term in ['ZFS', 'Zeeman', 'N14_Hyperfine', 'N14_Quadrupole', 'C13_Hyperfine']:
    if config.get(term, {}).get('enabled', False):
        print(f"✓ {term}")
    else:
        print(f"✗ {term}")

# Test mit nur ZFS
print("\n=== Test mit nur ZFS ===")
import json
test_config = {
    "ZFS": {
        "enabled": True,
        "D": 2.87e9,
        "Ex": 0.0,
        "Ey": 0.0
    },
    "Zeeman": {"enabled": False},
    "N14_Hyperfine": {"enabled": False},
    "N14_Quadrupole": {"enabled": False},
    "C13_Hyperfine": {"enabled": False}
}

with open('test_zfs_only.json', 'w') as f:
    json.dump(test_config, f)

H_zfs_only = TotalHamiltonian('test_zfs_only.json').build_hamiltonian()
evals_zfs = np.linalg.eigvalsh(H_zfs_only)
evals_zfs = np.sort(evals_zfs)

# ZFS sollte Entartung haben (viele gleiche Eigenwerte)
unique_evals = np.unique(np.round(evals_zfs/1e6, 3))
print(f"Anzahl eindeutiger Eigenwerte: {len(unique_evals)}")
print("Eindeutige Eigenwerte (MHz):")
for E in unique_evals[:5]:
    count = np.sum(np.abs(evals_zfs/1e6 - E) < 0.001)
    print(f"  {E:.3f} MHz (Entartung: {count})")

# Aufräumen
import os
os.remove('test_zfs_only.json')