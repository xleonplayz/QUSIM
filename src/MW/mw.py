import numpy as np

# ----------------------------
# 1) Basis-Operatoren definieren
# ----------------------------
# Spin-1 Matrices (3×3)
Sx = (1/np.sqrt(2)) * np.array([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]])
Sy = (1j/np.sqrt(2)) * np.array([[0, -1,  0],
                                 [1,  0, -1],
                                 [0,  1,  0]])
Sz = np.diag([1, 0, -1])

S_plus  = Sx + 1j*Sy
S_minus = Sx - 1j*Sy

# Kernspin als Identität (3×3), wirkt nicht auf MW
I3 = np.eye(3)

# g/e Manifold
Ig = np.array([[1,0],[0,0]])  # Projektor auf g
Ie = np.array([[0,0],[0,1]])  # Projektor auf e

# Identity über alle Untermatrizen
Id2 = np.eye(2)
Id3 = np.eye(3)

# ----------------------------
# 2) Kronecker-Strukturen
# ----------------------------
# Elektronspin ⊗ Kernspin
def kron_elec_nuc(mat_elec):
    return np.kron(mat_elec, I3)

# g/e ⊗ (Elekt.⊗Kern)
def embed_in_2x9(mat_9x9, manifold='g'):
    if manifold=='g':
        P = np.kron(Ig, mat_9x9)
    else:
        P = np.kron(Ie, mat_9x9)
    return P

# ----------------------------
# 3) MW-Hamiltonian als Funktion
# ----------------------------
def H_MW(t, Omegas, Phis, T):
    """
    Gibt H_MW(t) als 18x18-Matrix zurück.
    Omegas, Phis: Arrays der Länge N (Bin-Amplituden und -Phasen)
    T: Gesamtdauer
    """
    N = len(Omegas)
    dt = T / N
    H = np.zeros((18,18), dtype=complex)
    for i in range(N):
        # Zeitfenster-Indicator: rect-Funktion
        if (i*dt) <= t < ((i+1)*dt):
            A = Omegas[i]
            phi = Phis[i]
            # MW-Term auf Elektronspin alleine
            H_elec = (A/2) * (S_plus * np.exp(-1j*phi) + S_minus * np.exp(+1j*phi))
            # In Elektron⊗Kern einbetten (9×9)…
            H_9 = kron_elec_nuc(H_elec)
            # …und auf g+e manifolds ausdehnen (2×9→18×18)
            H = embed_in_2x9(H_9, 'g') + embed_in_2x9(H_9, 'e')
            break
    return H

# ----------------------------
# 4) Beispielaufruf und Ausgabe
# ----------------------------
if __name__ == "__main__":
    # Pulsdauer
    T = 1e-6  # 1 µs
    # Beispiel-Bins: Rechteck mit 10 Bins
    N = 10
    Omegas = np.ones(N) * 2*np.pi*5e6   # je 5 MHz Rabi in jedem Bin
    Phis   = np.zeros(N)                # keine Phasenmodulation

    # Matrix zur t = T/2
    H18 = H_MW(T/2, Omegas, Phis, T)
    print("Dimension:", H18.shape)
    print(np.round(H18, 3))
