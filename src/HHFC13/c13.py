import numpy as np

# ---------- Spin-Operatoren ----------
def spin1_ops():  # S=1
    Sx = (1/np.sqrt(2))*np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=complex)
    Sy = (1/np.sqrt(2))*np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]], dtype=complex)
    Sz =               np.array([[ 1,0,0],[0,0,0],[0,0,-1]], dtype=complex)
    return Sx, Sy, Sz

def spin_half_ops():  # I=1/2
    sx = 0.5*np.array([[0,1],[1,0]], dtype=complex)
    sy = 0.5*np.array([[0,-1j],[1j,0]], dtype=complex)
    sz = 0.5*np.array([[1,0],[0,-1]], dtype=complex)
    return sx, sy, sz

def kron(*ops):
    out = np.array([[1]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out

# ---------- Dipolar-Tensor wie gehabt ----------
def A_tensor_dipolar(r_vec, gamma_e=28.024951e9, gamma_c=10.705e6,
                     mu0=4*np.pi*1e-7, hbar=1.054571817e-34):
    r = np.linalg.norm(r_vec)
    rhat = r_vec / r
    pref = mu0/(4*np.pi) * (gamma_e*gamma_c*hbar**2) / r**3
    T = np.eye(3) - 3*np.outer(rhat, rhat)
    return pref / (6.62607015e-34) * T

# ---------- C13-HF-Term in 18×18 ----------
def H_C13_in_18x18(r_vec, A_iso=0.0):
    S_ops = spin1_ops()          # Elektronen-Operatoren (3×3)
    Ic_ops = spin_half_ops()     # 13C-Operatoren (2×2)

    # Dimensionen: [Electron(3), N(3), 13C(2)]
    dims = [3, 3, 2]

    def lift(op, idx):
        mats = []
        for k, d in enumerate(dims):
            mats.append(op if k == idx else np.eye(d, dtype=complex))
        return kron(*mats)

    # Hyperfeiner Tensor für 13C
    A_dip = A_tensor_dipolar(r_vec)
    A = A_iso * np.eye(3) + A_dip   # 3×3

    # Aufbau der 18×18-Matrix
    H = np.zeros((np.prod(dims),)*2, dtype=complex)
    for a, Sa in enumerate(S_ops):
        S_l = lift(Sa, 0)            # Elektron an pos 0
        for b, Ib in enumerate(Ic_ops):
            I_l = lift(Ib, 2)        # 13C an pos 2
            H += A[a,b] * (S_l @ I_l)
    return H

# ---------- Beispielaufruf und Ausgabe ----------
if __name__ == "__main__":
    r = np.array([0.45e-9, 0.0, 0.0])    # Position des 13C
    H18 = H_C13_in_18x18(r, A_iso=0.0)
    # Ausgabe der kompletten Matrix
    np.set_printoptions(precision=6, suppress=True)
    print("18×18 Hyperfein-Hamiltonian für 13C:\n", H18)
