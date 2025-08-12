# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Leon Kaiser QUSIM Quantum Simulator für NV center simulations
#  MSQC Goethe University https://msqc.cgi-host6.rz.uni-frankfurt.de
#  I.kaiser[at]em.uni-frankfurt.de
#

import numpy as np

# -----------------------------
# 1) Physikalische Parameter
# -----------------------------
# Statischer Strain-Tensor im Principal-Frame (Hz)
eps0_PA = np.array([[ 3.0e6,  1.5e6, 0.0],
                    [ 1.5e6, -3.0e6, 0.0],
                    [ 0.0 ,  0.0 ,  0.0 ]], dtype=float)

# Zeitabhängige Moden (Amplitude A in Hz, Frequenz f in Hz, Phase phi in rad)
modes_PA = [
    {"A": 1.0e5 * np.array([[1,0,0],[0,-1,0],[0,0,0]]), "f": 5.0e6,  "phi": 0.0},
    {"A": 8.0e4 * np.array([[0,1,0],[1,0,0],[0,0,0]]), "f":12.0e6,  "phi": 0.3}
]

# Spatial-Gradient G_{ijα} (Hz/µm)
G_PA = np.zeros((3,3,3))
G_PA[:,:,2] = 1.0e3

# Nichtlineare Suszeptibilität χ_{ij,kl} (Hz/StressUnit) und Stress-Funktion σ(t)
chi_PA = np.zeros((3,3,3,3))
chi_PA[0,0,0,0] = 1e-3

def stress_sigma(t):
    # Rechteckpuls von 0 bis 200 ns
    if 0 <= t <= 200e-9:
        return 1e7 * np.array([[1,0,0],[0,-1,0],[0,0,0]])
    return np.zeros((3,3))

# Euler-Winkel (Z–Y–Z) zur Rotation vom PA-Frame in den Lab-Frame
euler_strain = (0.1, 0.02, -0.05)

# NV-Position (µm) für den Gradient-Anteil
r_vec_um = np.array([0.0, 0.0, 0.5])


# -----------------------------
# 2) Spin-1 Operatoren
# -----------------------------
def spin1_ops():
    Sx = (1/np.sqrt(2))*np.array([[0,1,0],[1,0,1],[0,1,0]], complex)
    Sy = (1/np.sqrt(2))*np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]], complex)
    Sz = np.array([[1,0,0],[0,0,0],[0,0,-1]], complex)
    return Sx, Sy, Sz


# -----------------------------
# 3) Hilfsfunktionen
# -----------------------------
def R_euler(alpha, beta, gamma):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    Rz1 = np.array([[ ca,-sa,0],[ sa, ca,0],[  0,  0,1]])
    Ry  = np.array([[ cb, 0, sb],[  0, 1,  0],[-sb, 0, cb]])
    Rz2 = np.array([[ cg,-sg,0],[ sg, cg,0],[  0,  0,1]])
    return Rz1 @ Ry @ Rz2

def sym_op(A, B):
    return 0.5 * (A @ B + B @ A)


# -----------------------------
# 4) Gesamten Strain-Tensor im Lab-Frame
# -----------------------------
def eps_total_lab(t, r_um):
    eps_PA = eps0_PA.copy()
    for m in modes_PA:
        eps_PA += m["A"] * np.sin(2*np.pi*m["f"]*t + m["phi"])
    eps_PA += np.tensordot(G_PA, r_um, axes=([2],[0]))
    eps_PA += np.tensordot(chi_PA, stress_sigma(t), axes=([2,3],[0,1]))
    R = R_euler(*euler_strain)
    return R @ eps_PA @ R.T


# -----------------------------
# 5) Strain-Hamiltonian (3×3)
# -----------------------------
def H_strain_3x3(t, r_um, S_ops=None):
    if S_ops is None:
        S_ops = spin1_ops()
    eps_lab = eps_total_lab(t, r_um)
    H = np.zeros((3,3), complex)
    for i in range(3):
        for j in range(3):
            H += eps_lab[i,j] * sym_op(S_ops[i], S_ops[j])
    return H


# -----------------------------
# 6) Lift in den 18×18-Raum
# -----------------------------
def kron(*ops):
    M = np.array([[1]], complex)
    for X in ops:
        M = np.kron(M, X)
    return M

def lift(op, idx, dims):
    mats = [np.eye(d, dtype=complex) for d in dims]
    mats[idx] = op
    return kron(*mats)


# -----------------------------
# 7) Komplettes Beispiel
# -----------------------------
if __name__ == "__main__":
    # Zeitpunkt und Position
    t = 100e-9   # 100 ns
    r = r_vec_um

    # 7.1: 3×3 Strain-Hamiltonian
    H3 = H_strain_3x3(t, r)

    # 7.2: Hebe in NV (3) ⊗ Defekt1 (2) ⊗ Defekt2 (3) → 18×18
    dims = [3, 2, 3]
    H18 = lift(H3, idx=0, dims=dims)

    print("Shape of full Hamiltonian:", H18.shape)  # (18, 18)
    print("Eigenwerte [Hz]:", np.sort(np.real(np.linalg.eigvals(H18))))
