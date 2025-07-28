import numpy as np

# -----------------------------
# 1) Physikalische Parameter
# -----------------------------
# Stark-Kopplungskoeffizienten (Hz per V/m)
d_par  = 3.5e-3
d_perp = 17e-3

# Stark-Tensor im Principal-Frame (Hz/(V/m))
D_PA = np.array([[ d_perp,   0.0  , 0.0 ],
                 [  0.0  , -d_perp, 0.0 ],
                 [  0.0  ,   0.0  , d_par ]],
                dtype=float)

# Euler-Winkel (Z–Y–Z) zur Rotation Principal→Lab
euler_D = (0.0, 0.0, 0.0)

# Statisches elektrisches Feld im Lab-Frame (V/m)
E0_lab = np.array([1.0e6, 2.0e6, 0.0])

# Optionale zeitabhängige Feldmoden
E_modes = [
    # {"A": np.array([5e5,0,0]), "f": 8e6, "phi": 0.0},
]

# Rauschfunktion (falls benötigt)
def E_noise(t):
    return np.zeros(3)


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
    return 0.5*(A @ B + B @ A)


# -----------------------------
# 4) Elektrisches Feld E(t)
# -----------------------------
def E_of_t(t, E0, modes=None, noise_fun=None):
    E = E0.copy()
    if modes:
        for m in modes:
            E += m["A"] * np.sin(2*np.pi*m["f"]*t + m.get("phi", 0.0))
    if noise_fun:
        E += noise_fun(t)
    return E


# -----------------------------
# 5) Aufbau des Stark-Tensors im Lab-Frame
# -----------------------------
def build_D_lab(D_PA, euler):
    R = R_euler(*euler)
    return R @ D_PA @ R.T


# -----------------------------
# 6) Stark-Hamiltonian (3×3)
# -----------------------------
def H_stark_3x3(t, params, S_ops=None):
    if S_ops is None:
        S_ops = spin1_ops()
    # Tensor im Lab-Frame
    D_lab = build_D_lab(params["D_PA"], params["euler_D"])
    # Feldvektor zur Zeit t
    E_t = E_of_t(t, params["E0_lab"], params["E_modes"], params["E_noise"])
    # Baue Hamiltonian
    H = np.zeros((3,3), complex)
    for i in range(3):
        for j in range(3):
            H += D_lab[i,j] * E_t[i] * sym_op(S_ops[i], S_ops[j])
    return H


# -----------------------------
# 7) Tensor-Lift in den 18×18-Raum
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
# 8) Beispiel: 18×18 Stark-Hamiltonian
# -----------------------------
if __name__ == "__main__":
    params = {
        "D_PA":    D_PA,
        "euler_D": euler_D,
        "E0_lab":  E0_lab,
        "E_modes": E_modes,
        "E_noise": E_noise
    }

    # Zeitpunkt
    t = 0.0

    # 8.1: Erzeuge 3×3-Hamiltonian
    H3 = H_stark_3x3(t, params)

    # 8.2: Hebe in NV⊗Defekt1⊗Defekt2 (3×2×3=18) auf
    dims = [3, 2, 3]
    H18 = lift(H3, idx=0, dims=dims)

    # Ausgabe
    print("Shape of full Stark Hamiltonian:", H18.shape)  # (18, 18)
    print("Full Stark Hamiltonian:\n", H18)
