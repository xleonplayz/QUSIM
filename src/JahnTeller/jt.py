# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Leon Kaiser QUSIM Quantum Simulator für NV center simulations
#  MSQC Goethe University https://msqc.cgi-host6.rz.uni-frankfurt.de
#  I.kaiser[at]em.uni-frankfurt.de
#

import numpy as np
from qutip import *

# ==== Ziel: 18x18 Matrix ====
# Spin: 3, a_x: 3, a_y: 2 → 3×3×2 = 18
N_spin = 3
N_ax = 3
N_ay = 2

# ==== Parameter ====
omega_x = 1.0 * 2*np.pi
omega_y = 1.2 * 2*np.pi
lambda_x = 0.05
lambda_y = 0.04
alpha_x = 0.01
alpha_y = 0.01

# ==== Operatoren ====
# Boson-Moden
a_x = tensor(qeye(N_spin), destroy(N_ax), qeye(N_ay))
a_y = tensor(qeye(N_spin), qeye(N_ax), destroy(N_ay))

# Spin-1-Operatoren
Sx = jmat(1, 'x')
Sy = jmat(1, 'y')
Sx2 = Sx @ Sx
Sy2 = Sy @ Sy
SxSy = Sx @ Sy + Sy @ Sx

Sx2_full = tensor(Qobj(Sx2), qeye(N_ax), qeye(N_ay))
Sy2_full = tensor(Qobj(Sy2), qeye(N_ax), qeye(N_ay))
SxSy_full = tensor(Qobj(SxSy), qeye(N_ax), qeye(N_ay))

# ==== Hamiltonian-Komponenten ====
H_mode = omega_x * a_x.dag() * a_x + omega_y * a_y.dag() * a_y
H_JT_x = lambda_x * (a_x + a_x.dag()) * (Sx2_full - Sy2_full)
H_JT_y = lambda_y * (a_y + a_y.dag()) * (SxSy_full)
H_anh_x = alpha_x * (a_x + a_x.dag())**3
H_anh_y = alpha_y * (a_y + a_y.dag())**3

# ==== Gesamt-Hamiltonian (zeitunabhängig) ====
H_total = H_mode + H_JT_x + H_JT_y + H_anh_x + H_anh_y

# ==== Matrix anzeigen (als NumPy) ====
H_matrix = H_total.full()  # ergibt 18x18 komplexe Matrix

# Optional: als reale Matrix ausgeben
np.set_printoptions(precision=3, suppress=True)
print(H_matrix.real)
