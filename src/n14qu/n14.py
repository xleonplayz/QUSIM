# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Leon Kaiser QUSIM Quantum Simulator für NV center simulations
#  MSQC Goethe University https://msqc.cgi-host6.rz.uni-frankfurt.de
#  I.kaiser[at]em.uni-frankfurt.de
#

import numpy as np

def spin1_ops():
    """Spin-1 Matrizen I_x, I_y, I_z in der Basis {|+1>,|0>,|-1>}."""
    I_x = (1/np.sqrt(2)) * np.array([[0,1,0],
                                     [1,0,1],
                                     [0,1,0]], dtype=complex)
    I_y = (1/np.sqrt(2)) * np.array([[0,-1j,0],
                                     [1j,0,-1j],
                                     [0,1j,0]], dtype=complex)
    I_z =            np.array([[1,0,0],
                               [0,0,0],
                               [0,0,-1]], dtype=complex)
    return I_x, I_y, I_z

def H_Q_N14(P=-5.0e6, eta=0.0):
    """
    Effektiver Quadrupol-Hamiltonian H_Q (3×3) für den 14N-Kern (I=1):
      H_Q = P (I_z^2 - 2/3 I) + (eta*P/3) (I_x^2 - I_y^2)
    """
    I_x, I_y, I_z = spin1_ops()
    # Axial-Teil (eta=0):
    H_axial = P * (I_z @ I_z - (2/3)*np.eye(3, dtype=complex))
    # Transversal-Asymmetrie (entfällt für eta=0, gezeigt hier der Vollterm)
    H_asym  = (eta*P/3) * (I_x @ I_x - I_y @ I_y)
    return H_axial + H_asym

def extend_to_18(H3):
    """
    Hebt einen 3×3-Operator H3 auf 18×18 per Tensorprodukt:
      H18 = I6 ⊗ H3
    """
    I6 = np.eye(6, dtype=complex)
    return np.kron(I6, H3)

if __name__ == "__main__":
    # 1) 3×3-Quadrupol-Hamiltonian
    P, eta = -5.0e6, 0.0
    H3 = H_Q_N14(P=P, eta=eta)
    print("3×3 Quadrupol-Hamiltonian H3 [Hz]:\n", np.round(H3,2), "\n")

    # 2) Aufweitung auf 18×18
    H18 = extend_to_18(H3)
    np.set_printoptions(precision=2, suppress=True)
    print("18×18 Quadrupol-Hamiltonian H18 [Hz]:\n", H18)
