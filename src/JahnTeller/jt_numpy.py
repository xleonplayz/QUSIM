# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Leon Kaiser QUSIM Quantum Simulator für NV center simulations
#  MSQC Goethe University https://msqc.cgi-host6.rz.uni-frankfurt.de
#  I.kaiser[at]em.uni-frankfurt.de
#

import numpy as np

def spin1_ops():
    """Spin-1 Matrizen"""
    Sx = (1/np.sqrt(2)) * np.array([[0,1,0],[1,0,1],[0,1,0]], complex)
    Sy = (1j/np.sqrt(2))* np.array([[0,-1,0],[1,0,-1],[0,1,0]], complex)
    Sz = np.diag([1,0,-1])
    return Sx, Sy, Sz

def destroy(N):
    """Vernichtungsoperator für N Fock-Zustände"""
    a = np.zeros((N,N), complex)
    for n in range(1,N):
        a[n-1,n] = np.sqrt(n)
    return a

def kron(*ops):
    """n-faches Kronecker-Produkt"""
    M = ops[0]
    for m in ops[1:]:
        M = np.kron(M, m)
    return M

def H_JT_18x18(params=None):
    """
    Jahn-Teller Hamiltonian als 18×18 Matrix
    Spin: 3, a_x: 3, a_y: 2 → 3×3×2 = 18
    """
    # Dimensionen
    N_spin = 3
    N_ax = 3
    N_ay = 2
    
    # Default Parameter
    if params is None:
        params = {
            'omega_x': 1.0 * 2*np.pi,
            'omega_y': 1.2 * 2*np.pi,
            'lambda_x': 0.05,
            'lambda_y': 0.04,
            'alpha_x': 0.01,
            'alpha_y': 0.01
        }
    
    # Extrahiere Parameter
    omega_x = params.get('omega_x', 1.0 * 2*np.pi)
    omega_y = params.get('omega_y', 1.2 * 2*np.pi)
    lambda_x = params.get('lambda_x', 0.05)
    lambda_y = params.get('lambda_y', 0.04)
    alpha_x = params.get('alpha_x', 0.01)
    alpha_y = params.get('alpha_y', 0.01)
    
    # Identitäten
    I_spin = np.eye(N_spin, dtype=complex)
    I_ax = np.eye(N_ax, dtype=complex)
    I_ay = np.eye(N_ay, dtype=complex)
    
    # Boson-Operatoren
    a_x = kron(I_spin, destroy(N_ax), I_ay)
    a_y = kron(I_spin, I_ax, destroy(N_ay))
    a_x_dag = a_x.conj().T
    a_y_dag = a_y.conj().T
    
    # Spin-Operatoren
    Sx, Sy, Sz = spin1_ops()
    Sx2 = Sx @ Sx
    Sy2 = Sy @ Sy
    SxSy = Sx @ Sy + Sy @ Sx
    
    # Erweiterte Spin-Operatoren
    Sx2_full = kron(Sx2, I_ax, I_ay)
    Sy2_full = kron(Sy2, I_ax, I_ay)
    SxSy_full = kron(SxSy, I_ax, I_ay)
    
    # Hamiltonian-Komponenten
    H_mode = omega_x * (a_x_dag @ a_x) + omega_y * (a_y_dag @ a_y)
    H_JT_x = lambda_x * (a_x + a_x_dag) @ (Sx2_full - Sy2_full)
    H_JT_y = lambda_y * (a_y + a_y_dag) @ SxSy_full
    
    # Anharmonische Terme
    Qx = a_x + a_x_dag
    Qy = a_y + a_y_dag
    H_anh_x = alpha_x * (Qx @ Qx @ Qx)
    H_anh_y = alpha_y * (Qy @ Qy @ Qy)
    
    # Gesamt-Hamiltonian
    H_total = H_mode + H_JT_x + H_JT_y + H_anh_x + H_anh_y
    
    return H_total

if __name__ == "__main__":
    # Test
    H = H_JT_18x18()
    print("Shape:", H.shape)
    print("Hermitesch:", np.allclose(H, H.conj().T))
    
    # Eigenwerte
    evals = np.linalg.eigvalsh(H)
    print("Erste 5 Eigenwerte:", evals[:5])