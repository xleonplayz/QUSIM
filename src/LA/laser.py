import numpy as np

# ----------------------------
# 1) Projektoren auf |g> und |e| (2×2)
# ----------------------------
# Basisreihenfolge: [|g>, |e>]
Pg = np.array([[1, 0],
               [0, 0]], dtype=complex)   # |g><g|
Pe = np.array([[0, 0],
               [0, 1]], dtype=complex)   # |e><e|
g_e = np.array([[0, 1],
                [0, 0]], dtype=complex)   # |g><e|
e_g = np.array([[0, 0],
                [1, 0]], dtype=complex)   # |e><g|

# ----------------------------
# 2) Identitäten für Elektron-Spin (3×3) und Kernspin (3×3)
# ----------------------------
I3 = np.eye(3, dtype=complex)

# ----------------------------
# 3) Kronecker-Produkt-Hilfsfunktion
# ----------------------------
def kron(*mats):
    """n-faches Kronecker-Produkt"""
    result = mats[0]
    for M in mats[1:]:
        result = np.kron(result, M)
    return result

# ----------------------------
# 4) Laser-Hamiltonian H_laser(t) als 18×18
# ----------------------------
ħ = 1.054571817e-34  # Plancksches Wirkungsquantum [J·s]

def H_laser(t, Omega_L, omega_L):
    """
    Berechnet den Laser-Hamiltonian H_laser(t) im 18×18-Raum:
    
      H_laser(t)
      = (ħ/2) · Ω_L(t) · [ |g><e| e^{-i ω_L t} + |e><g| e^{+i ω_L t} ]
      eingebettet als (2×2) ⊗ I3 ⊗ I3.

    Args:
        t        : Zeitpunkt in Sekunden
        Omega_L  : Funktion Ω_L(t) oder konstanter Skalar
        omega_L  : Laser-Trägerfrequenz in rad/s
    Returns:
        H18      : 18×18 complex numpy array
    """
    # Bestimme Ω_L(t)
    Ω = Omega_L(t) if callable(Omega_L) else Omega_L

    # 2×2 Laser-Kopplung h2(t)
    h2 = (ħ * Ω / 2) * (
        g_e * np.exp(-1j * omega_L * t) +
        e_g * np.exp(+1j * omega_L * t)
    )

    # Einbettung in den Gesamt-Hamiltonian: (2×2) ⊗ I3 ⊗ I3
    H18 = kron(h2, I3, I3)
    return H18

# ----------------------------
# 5) Beispiel: Ausgabe der kompletten 18×18-Matrix
# ----------------------------
if __name__ == "__main__":
    # Parameter
    omega_L = 2 * np.pi * 4.66e14   # z.B. sichtbares Licht ~640 nm
    # Definition einer Puls-Hüllkurve: 2π·10 MHz Rechteckpuls von 100 ns
    def Omega_L(t):
        return 2 * np.pi * 10e6 if 0 <= t < 100e-9 else 0

    # Zeitpunkt wählen
    t0 = 50e-9  # 50 ns
    # Hamiltonian berechnen
    H = H_laser(t0, Omega_L, omega_L)

    # Ausgabe
    np.set_printoptions(precision=3, suppress=True)
    print("H_laser(t0) shape:", H.shape)
    print("H_laser(t0) =\n", H)
