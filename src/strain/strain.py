# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Leon Kaiser QUSIM Quantum Simulator für NV center simulations - JAX Optimized
#  MSQC Goethe University https://msqc.cgi-host6.rz.uni-frankfurt.de
#  I.kaiser[at]em.uni-frankfurt.de
#
#  JAX-optimierte Strain-Implementierung (ersetzt Original für bessere Performance)

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import numpy as np

# KRITISCH: Float64 für physikalische Genauigkeit
jax.config.update('jax_enable_x64', True)

# -----------------------------
# 1) Physikalische Parameter
# -----------------------------
# Statischer Strain-Tensor im Principal-Frame (Hz) - JAX Arrays
eps0_PA = jnp.array([[ 3.0e6,  1.5e6, 0.0],
                     [ 1.5e6, -3.0e6, 0.0],
                     [ 0.0 ,  0.0 ,  0.0 ]], dtype=jnp.float64)

# Zeitabhängige Moden - als strukturierte JAX Arrays
modes_amplitudes = jnp.array([
    [[ 1.0e5,  0.0, 0.0],
     [ 0.0, -1.0e5, 0.0], 
     [ 0.0,  0.0,  0.0]],
    [[ 0.0,  8.0e4, 0.0],
     [8.0e4,  0.0,  0.0],
     [ 0.0,  0.0,  0.0]]
], dtype=jnp.float64)

modes_frequencies = jnp.array([5.0e6, 12.0e6], dtype=jnp.float64)
modes_phases = jnp.array([0.0, 0.3], dtype=jnp.float64)

# Spatial-Gradient G_{ijα} (Hz/µm)
G_PA = jnp.zeros((3, 3, 3), dtype=jnp.float64)
G_PA = G_PA.at[:, :, 2].set(1.0e3)

# Nichtlineare Suszeptibilität χ_{ij,kl} (Hz/StressUnit)
chi_PA = jnp.zeros((3, 3, 3, 3), dtype=jnp.float64)
chi_PA = chi_PA.at[0, 0, 0, 0].set(1e-3)

# Euler-Winkel (Z–Y–Z) zur Rotation vom PA-Frame in den Lab-Frame
euler_strain = jnp.array([0.1, 0.02, -0.05], dtype=jnp.float64)

# Rechteckpuls-Parameter
pulse_start = 0.0
pulse_end = 200e-9
pulse_amplitude = 1e7

# NV-Position (µm) für den Gradient-Anteil
r_vec_um = jnp.array([0.0, 0.0, 0.5], dtype=jnp.float64)

# -----------------------------
# 2) JAX-optimierte Spin-1 Operatoren
# -----------------------------
@jit
def spin1_ops():
    """JAX-optimierte Spin-1 Operatoren"""
    Sx = (1/jnp.sqrt(2)) * jnp.array([[0,1,0],[1,0,1],[0,1,0]], dtype=jnp.complex128)
    Sy = (1/jnp.sqrt(2)) * jnp.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]], dtype=jnp.complex128)
    Sz = jnp.array([[1,0,0],[0,0,0],[0,0,-1]], dtype=jnp.complex128)
    return Sx, Sy, Sz

# -----------------------------
# 3) JAX-optimierte Hilfsfunktionen
# -----------------------------
@jit
def R_euler(alpha, beta, gamma):
    """JAX-optimierte Euler-Rotation (Z-Y-Z Konvention)"""
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    cb, sb = jnp.cos(beta),  jnp.sin(beta)
    cg, sg = jnp.cos(gamma), jnp.sin(gamma)
    
    Rz1 = jnp.array([[ ca,-sa, 0],
                     [ sa, ca, 0],
                     [  0,  0, 1]], dtype=jnp.float64)
    Ry  = jnp.array([[ cb,  0, sb],
                     [  0,  1,  0],
                     [-sb,  0, cb]], dtype=jnp.float64)
    Rz2 = jnp.array([[ cg,-sg, 0],
                     [ sg, cg, 0],
                     [  0,  0, 1]], dtype=jnp.float64)
    return Rz1 @ Ry @ Rz2

@jit
def sym_op(A, B):
    """JAX-optimierter symmetrischer Operator"""
    return 0.5 * (A @ B + B @ A)

@jit
def stress_sigma(t):
    """JAX-optimierte Stress-Funktion (Rechteckpuls)"""
    pulse_matrix = jnp.array([[1,0,0],[0,-1,0],[0,0,0]], dtype=jnp.float64)
    # Verwende jnp.where für bedingte Logik (JIT-kompatibel)
    condition = (t >= pulse_start) & (t <= pulse_end)
    return jnp.where(condition, pulse_amplitude * pulse_matrix, 0.0)

# -----------------------------
# 4) JAX-optimierte Kern-Algorithmen
# -----------------------------
@jit
def eps_modes(t, amplitudes, frequencies, phases):
    """Vektorisierte Berechnung aller oszillierenden Moden"""
    # Berechne sin-Terme für alle Moden gleichzeitig
    sin_terms = jnp.sin(2 * jnp.pi * frequencies * t + phases)
    # Erweitere Dimensionen für Broadcasting: (n_modes, 1, 1) * (n_modes, 3, 3)
    sin_expanded = sin_terms[:, jnp.newaxis, jnp.newaxis]
    # Summiere über alle Moden
    return jnp.sum(sin_expanded * amplitudes, axis=0)

@jit
def eps_total_lab(t, r_um):
    """JAX-optimierte Berechnung des gesamten Strain-Tensors im Lab-Frame"""
    # Statischer Term
    eps_PA_temp = eps0_PA.copy()
    
    # Zeitabhängige Moden (vektorisiert)
    eps_PA_temp += eps_modes(t, modes_amplitudes, modes_frequencies, modes_phases)
    
    # Gradienten-Term (optimiert mit einsum)
    eps_PA_temp += jnp.einsum('ijk,k->ij', G_PA, r_um)
    
    # Nichtlinearer Stress-Term (optimiert mit einsum)
    sigma_t = stress_sigma(t)
    eps_PA_temp += jnp.einsum('ijkl,kl->ij', chi_PA, sigma_t)
    
    # Rotation ins Lab-Frame
    R = R_euler(*euler_strain)
    return R @ eps_PA_temp @ R.T

@jit 
def _H_strain_3x3_jax(t, r_um, S_ops=None):
    """Interne JAX-optimierte Version"""
    if S_ops is None:
        S_ops = spin1_ops()
    
    eps_lab = eps_total_lab(t, r_um)
    
    # Direkter Hamiltonian-Aufbau
    H = jnp.zeros((3, 3), dtype=jnp.complex128)
    for i in range(3):
        for j in range(3):
            H += eps_lab[i, j] * sym_op(S_ops[i], S_ops[j])
    
    return H

def H_strain_3x3(t, r_um, S_ops=None):
    """Kompatibilitäts-Wrapper für hama.py"""
    # Konvertiere NumPy-Eingaben zu JAX
    t_jax = jnp.asarray(t, dtype=jnp.float64)
    r_um_jax = jnp.asarray(r_um, dtype=jnp.float64)
    
    if S_ops is None:
        S_ops_jax = None
    else:
        # Konvertiere bereitgestellte S_ops zu JAX
        S_ops_jax = tuple(jnp.asarray(S, dtype=jnp.complex128) for S in S_ops)
    
    # JAX-Berechnung
    H_jax = _H_strain_3x3_jax(t_jax, r_um_jax, S_ops_jax)
    
    # Konvertiere zurück zu NumPy für Kompatibilität
    return np.array(H_jax)

# -----------------------------
# 5) Erweiterte JAX-Funktionen für Performance (optional)
# -----------------------------
@jit
def H_strain_batch(t_array, r_array):
    """Batch-Berechnung für mehrere Zeit-/Positionswerte"""
    def single_calc(t, r):
        return _H_strain_3x3_jax(t, r)
    return vmap(single_calc)(t_array, r_array)

@jit
def H_strain_time_series(t_array, r_um_fixed):
    """Optimiert für Zeitserie bei fester Position"""
    def single_calc(t):
        return _H_strain_3x3_jax(t, r_um_fixed)
    return vmap(single_calc)(t_array)

# -----------------------------
# 6) Tensor-Lift in 18×18-Raum (für Vollständigkeit)
# -----------------------------
def kron(*ops):
    """n-faches Kronecker-Produkt"""
    M = np.array([[1]], dtype=complex)
    for X in ops:
        M = np.kron(M, X)
    return M

def lift(op, idx, dims):
    """Tensor-Lift für 18×18 Erweiterung"""
    mats = [np.eye(d, dtype=complex) for d in dims]
    mats[idx] = op
    return kron(*mats)

# -----------------------------
# 7) Beispiel & Test (wie Original)
# -----------------------------
if __name__ == "__main__":
    # Zeitpunkt und Position (wie Original)
    t = 100e-9   # 100 ns
    r = r_vec_um

    # 3×3 Strain-Hamiltonian (jetzt JAX-optimiert)
    H3 = H_strain_3x3(t, r)
    
    print(f"JAX-optimierte Strain-Implementation:")
    print(f"H3 shape: {H3.shape}")
    print(f"Hermitian: {np.allclose(H3, H3.conj().T)}")
    
    # Eigenwerte
    eigenvals = np.linalg.eigvalsh(H3)
    print(f"Eigenvalues (MHz): {eigenvals/1e6}")

    # 18×18 Lift (NV ⊗ Defekt1 ⊗ Defekt2 → 18×18)
    dims = [3, 2, 3]  # 3×2×3 = 18
    H18 = lift(H3, idx=0, dims=dims)

    print(f"Shape of full Hamiltonian: {H18.shape}")  # (18, 18)
    print(f"Eigenwerte 18x18 [Hz]: {np.sort(np.real(np.linalg.eigvals(H18)))}")