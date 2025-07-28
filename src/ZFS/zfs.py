import numpy as np
from numpy.linalg import eig
from scipy.linalg import expm

# ---------- 1) Spin-1 Operatoren erzeugen ----------
def spin1_ops():
    """
    Gibt die Spin-1 Matrizen Sx, Sy, Sz zurück
    in der Basis {|+1>, |0>, |-1>}.
    """
    Sx = (1/np.sqrt(2)) * np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=complex)
    Sy = (1/np.sqrt(2)) * np.array([
        [0, -1j, 0],
        [1j,  0, -1j],
        [0,  1j,  0]
    ], dtype=complex)
    Sz = np.array([
        [ 1, 0,  0],
        [ 0, 0,  0],
        [ 0, 0, -1]
    ], dtype=complex)
    return Sx, Sy, Sz

# ---------- 2) Temperatur-Shift und Strain-Terme ----------
def D_of_T(T, D0=2.870e9, a1=-74e3, a2=0.0):
    """
    Temperaturabhängiger Zero-Field-Splitting-Parameter D(T) in Hz.
    D0 bei T=300K, linearer Koeffizient a1, optional quadratischer Term a2.
    """
    dT = T - 300.0
    return D0 + a1*dT + a2*dT**2

def ExEy_from_strain(strain_vec, kx=1.0e6, ky=1.0e6):
    """
    Wandelt die Dehnungskomponenten (εx, εy) in
    Frequenz-Terme Ex und Ey (Hz) um via Faktoren kx, ky.
    """
    epsx, epsy = strain_vec
    return kx * epsx, ky * epsy

# ---------- 3) ZFS-Hamiltonian in 3×3 ----------
def H_ZFS(D, Ex=0.0, Ey=0.0, Sx=None, Sy=None, Sz=None):
    """
    Bildet den Hamiltonoperator
      H = D Sz^2 + Ex (Sx^2 - Sy^2) + Ey (Sx Sy + Sy Sx)
    als 3×3-Matrix ab.
    """
    # Falls Spin-Operatoren nicht übergeben, selbst erzeugen
    if Sx is None:
        Sx, Sy, Sz = spin1_ops()
    Sz2 = Sz @ Sz
    Sx2_minus_Sy2  = Sx @ Sx - Sy @ Sy
    SxSy_plus_SySx = Sx @ Sy + Sy @ Sx
    return D * Sz2 + Ex * Sx2_minus_Sy2 + Ey * SxSy_plus_SySx

# ---------- 4) Parameter-Beispiel und 3×3 Hamiltonian ----------
T = 295.0                                  # Temperatur in K
D  = D_of_T(T)                             # D(T) in Hz
Ex, Ey = ExEy_from_strain(
    (5e-6, 2e-6),                          # Dehnungsvektor εx, εy
    kx=1e9, ky=1e9                         # Umrechnungsfaktoren in Hz
)

# Erzeuge Spin-1-Operatoren und 3×3-Hamiltonian
Sx, Sy, Sz = spin1_ops()
H3 = H_ZFS(D, Ex, Ey, Sx, Sy, Sz)

# ---------- 5) Aufweiten auf 18×18 ----------
# Variante A: Block-Diagonal (H in oberem 3×3-Block)
H18_block = np.zeros((18,18), dtype=complex)
H18_block[:3, :3] = H3

# Variante B: Tensorprodukt mit 6×6-Einheitsoperator
I6       = np.eye(6, dtype=complex)
H18_tensor = np.kron(H3, I6)

# ---------- 6) Ausgabe ----------
np.set_printoptions(precision=3, suppress=True)

print("3×3 Zero-Field-Splitting Hamiltonian H3:\n", H3, "\n")
print("18×18 Block-diagonal Einbettung:\n", H18_block, "\n")
print("18×18 Tensorprodukt Einbettung:\n", H18_tensor)
