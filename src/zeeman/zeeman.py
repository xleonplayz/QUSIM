# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Leon Kaiser QUSIM Quantum Simulator für NV center simulations
#  MSQC Goethe University https://msqc.cgi-host6.rz.uni-frankfurt.de
#  I.kaiser[at]em.uni-frankfurt.de
#

import numpy as np

# ---------- 1) Spin-1 Operatoren ----------
def spin1_ops():
    """Spin-1 Matrizen Sx, Sy, Sz in der Basis {|+1>,|0>,|-1>}."""
    Sx = (1/np.sqrt(2)) * np.array([[0,1,0],
                                    [1,0,1],
                                    [0,1,0]], dtype=complex)
    Sy = (1/np.sqrt(2)) * np.array([[0,-1j,0],
                                    [1j, 0,-1j],
                                    [0, 1j, 0]], dtype=complex)
    Sz =            np.array([[1,0,0],
                              [0,0,0],
                              [0,0,-1]], dtype=complex)
    return Sx, Sy, Sz

# ---------- 2) Magnetfeld-Generator ----------
def generate_B_field(t, config):
    """
    Erzeugt B0 + B_ac*sin(2πf_ac t + phase) + Noise(t).
    Noise-Komponenten: white und telegraph.
    Rückgabe in Gauss als Länge-3-Array.
    """
    # Basis-Feld
    B0      = np.array(config.get("B0",      [0,0,500]), dtype=float)
    # AC-Feld
    Bac     = np.array(config.get("B_ac",    [0,0,0]),   dtype=float)
    f_ac    = config.get("f_ac", 1e6)    # Hz
    phase   = config.get("ac_phase", 0)  
    B_ac    = Bac * np.sin(2*np.pi*f_ac*t + phase)
    # Rauschen
    noise   = np.zeros(3, dtype=float)
    rng     = np.random.default_rng(config.get("seed", None))
    nc      = config.get("noise", {})
    if "white" in nc:
        sigma = nc["white"]
        noise += rng.normal(0, sigma, size=3)
    if "telegraph" in nc:
        amp   = nc["telegraph"]
        # ±amp mit 50:50 Wahrscheinlichkeit
        if rng.integers(0,2)==1:
            noise +=  amp
        else:
            noise -=  amp
    return B0 + B_ac + noise

# ---------- 3) Zeeman-Hamiltonian (3×3) ----------
def H_Zeeman(t, config, Sx=None, Sy=None, Sz=None):
    """
    H = γ_x B_x(t) Sx + γ_y B_y(t) Sy + γ_z B_z(t) Sz
    in Hz als 3×3-Matrix.
    """
    if Sx is None:
        Sx, Sy, Sz = spin1_ops()
    Bx, By, Bz = generate_B_field(t, config)
    g_tensor   = np.array(config.get("g_tensor", [2.0028]*3), dtype=float)
    mu_B       = 13.996e6  # Hz/G
    gamma      = g_tensor * mu_B
    return gamma[0]*Bx*Sx + gamma[1]*By*Sy + gamma[2]*Bz*Sz

# ---------- 4) Beispiel & Aufweitung auf 18×18 ----------
if __name__ == "__main__":
    # Zeitpunkt
    t = 0.0

    # Beispiel-Konfiguration
    config = {
      "B0":       [0,0,500],
      "B_ac":     [0,0,0],
      "f_ac":     1e6,
      "ac_phase": 0,
      "noise":    {"white":0.5, "telegraph":1.0},
      "g_tensor": [2.0028,2.0028,2.0028],
      "seed":     42
    }

    # 3×3 Zeeman-Hamiltonian
    Sx, Sy, Sz = spin1_ops()
    H3 = H_Zeeman(t, config, Sx, Sy, Sz)
    print("3×3 Zeeman-Hamiltonian H3 [Hz]:\n", np.round(H3,2), "\n")

    # Variante A: Block-Diagonal
    H18_block = np.zeros((18,18), dtype=complex)
    H18_block[:3, :3] = H3
    print("18×18 Block-Diagonal Einbettung:\n", np.round(H18_block,2), "\n")

    # Variante B: Tensorprodukt mit I6
    I6        = np.eye(6, dtype=complex)
    H18_tensor= np.kron(H3, I6)
    print("18×18 Tensor-Produkt Einbettung:\n", np.round(H18_tensor,2))
