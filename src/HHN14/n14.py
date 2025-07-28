import numpy as np

# ---------- Physikalische Konstanten ----------
mu0     = 4*np.pi*1e-7               # N·A⁻²
h       = 6.62607015e-34             # J·s
hbar    = 1.054571817e-34            # J·s
gamma_e = 28.024951e9                # Hz/T (Elektron)
gamma_N = 3.0766e7                   # Hz/T (14N)
gamma_c = 10.705e6                   # Hz/T (13C), nur Identity hier

# Beispielparameter (experimentell/DFT)
rho_e0_N   = 1e30    # 1/m³ Elektronendichte am 14N-Kern
A_iso_N    = 2e6     # Hz isotroper Anteil für 14N
g_tensor_e = np.diag([2.0028,2.0028,2.0028])
P_quad     = 5e6     # Hz Quadrupol‐Konstante für 14N
C_pc_N     = 1e6     # Stärke des Pseudo‐Kontakt‐Terms

# ---------- Spin-Operatoren ----------
def spin1_ops():
    Sx = (1/np.sqrt(2))*np.array([[0,1,0],
                                  [1,0,1],
                                  [0,1,0]], dtype=complex)
    Sy = (1/np.sqrt(2))*np.array([[0,-1j,0],
                                  [1j,0,-1j],
                                  [0,1j,0]], dtype=complex)
    Sz =               np.array([[ 1,0,0],
                                  [ 0,0,0],
                                  [ 0,0,-1]], dtype=complex)
    return Sx, Sy, Sz

# ---------- Kopplungstensor-Anteile für 14N ----------
def A_fermi_contact_N(rho_e0):
    return (2*mu0/3 * gamma_e*gamma_N * hbar**2 * rho_e0) / h

def A_dipolar_N(r_vec):
    r = np.linalg.norm(r_vec)
    rhat = r_vec / r
    T0 = np.eye(3, dtype=complex) - 3*np.outer(rhat, rhat)
    pref = mu0/(4*np.pi) * (gamma_e*gamma_N * hbar**2) / r**3
    return (g_tensor_e @ T0 @ g_tensor_e) * (pref/h)

def A_pseudo_contact_N(r_vec, C_pc=C_pc_N):
    r = np.linalg.norm(r_vec)
    cosθ = r_vec[2]/r
    D = np.diag([1,-0.5,-0.5])
    return C_pc * (3*cosθ**2 - 1)/r**3 * D

def A_total_N(r_vec, rho_e0, A_iso):
    A_FC   = A_fermi_contact_N(rho_e0) * np.eye(3, dtype=complex)
    A_isoM = A_iso * np.eye(3, dtype=complex)
    A_dip  = A_dipolar_N(r_vec)
    A_pc   = A_pseudo_contact_N(r_vec)
    return A_FC + A_isoM + A_dip + A_pc

# ---------- Aufbau des 18×18-Hamiltonians für 14N-Hyperfein ----------
def H_N14_realistic_18x18(r_vec, rho_e0, A_iso, P_quad):
    # 1) Operatoren
    S_ops = spin1_ops()        # Elektron S=1 (3×3)
    I_N   = spin1_ops()        # 14N    I=1 (3×3)
    # 13C-Subraum nur Identity, damit 18×18 insgesamt
    dims  = [3, 3, 2]           # e, N, C

    # 2) Lift-Funktion
    def lift(op, idx):
        mats = [np.eye(d, dtype=complex) for d in dims]
        mats[idx] = op
        return mats[0] if len(mats)==1 else np.kron(np.kron(mats[0], mats[1]), mats[2])

    # 3) Hyperfine-Tensor
    A_N = A_total_N(r_vec, rho_e0, A_iso)

    # 4) Hamilton-Aufbau
    H = np.zeros((18,18), dtype=complex)
    # Summe über a,b für 14N-Term
    for a, Sa in enumerate(S_ops):
        S_l = lift(Sa, 0)
        for b, Ib in enumerate(I_N):
            I_l = lift(Ib, 1)
            H += A_N[a,b] * (S_l @ I_l)

    # 5) Quadrupol-Term: P*(I_z^2 - 2/3 I·I) ⊗ I_C
    I_z = I_N[2]
    II = sum(np.eye(3)*0 + op@op for op in I_N) / 1  # I·I = I_x^2+I_y^2+I_z^2
    Q = P_quad * (np.kron(np.kron(np.eye(3), (I_z@I_z - 2/3*II)), np.eye(2)))
    H += Q

    return H

# ---------- Beispielaufruf & Ausgabe ----------
if __name__ == "__main__":
    r = np.array([0.45e-9, 0.0, 0.0])
    H18_N = H_N14_realistic_18x18(r, rho_e0_N, A_iso_N, P_quad)
    np.set_printoptions(precision=6, suppress=True)
    print("Realistischer 18×18 14N-Hyperfein-Hamiltonian:\n", H18_N)
