# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Leon Kaiser QUSIM Quantum Simulator für NV center simulations - JAX Optimized
#  MSQC Goethe University https://msqc.cgi-host6.rz.uni-frankfurt.de
#  I.kaiser[at]em.uni-frankfurt.de
#

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

# KRITISCH: Float64 für physikalische Genauigkeit
jax.config.update('jax_enable_x64', True)

# ---------- Physikalische Konstanten (JAX) ----------
mu0 = 4*jnp.pi*1e-7               # N·A⁻²
h = 6.62607015e-34                # J·s
hbar = 1.054571817e-34            # J·s
gamma_e = 28.024951e9             # Hz/T (Elektron)
gamma_N = 3.0766e7                # Hz/T (14N)
gamma_c = 10.705e6                # Hz/T (13C), nur Identity hier

# Beispielparameter (experimentell/DFT)
rho_e0_N = 0.0                    # Setze auf 0 wenn A_iso verwendet wird
A_iso_N = 2e6                     # Hz isotroper Anteil für 14N (beinhaltet Fermi-Kontakt)
g_tensor_e = jnp.diag(jnp.array([2.0028, 2.0028, 2.0028]))
P_quad = 5e6                      # Hz Quadrupol‐Konstante für 14N
C_pc_N = 0.0                      # Pseudo-Kontakt meist vernachlässigbar für 14N in NV

# ---------- Spin-Operatoren (JAX-optimiert) ----------
@jit
def spin1_ops():
    """Spin-1 Operatoren für Elektron und 14N"""
    Sx = (1/jnp.sqrt(2))*jnp.array([[0,1,0],
                                     [1,0,1],
                                     [0,1,0]], dtype=jnp.complex128)
    Sy = (1/jnp.sqrt(2))*jnp.array([[0,-1j,0],
                                     [1j,0,-1j],
                                     [0,1j,0]], dtype=jnp.complex128)
    Sz = jnp.array([[1,0,0],
                    [0,0,0],
                    [0,0,-1]], dtype=jnp.complex128)
    return Sx, Sy, Sz

# ---------- Kopplungstensor-Anteile für 14N (JIT-optimiert) ----------
@jit
def A_fermi_contact_N(rho_e0):
    """Fermi-Kontakt Term"""
    return (2*mu0/3 * gamma_e*gamma_N * hbar**2 * rho_e0) / h

@partial(jit, static_argnums=())
def A_dipolar_N(r_vec):
    """Dipolare Wechselwirkung für 14N"""
    r = jnp.linalg.norm(r_vec)
    rhat = r_vec / r
    T0 = jnp.eye(3, dtype=jnp.complex128) - 3*jnp.outer(rhat, rhat)
    pref = mu0/(4*jnp.pi) * (gamma_e*gamma_N * hbar**2) / r**3
    return (g_tensor_e @ T0 @ g_tensor_e) * (pref/h)

@partial(jit, static_argnums=(1,))
def A_pseudo_contact_N(r_vec, C_pc=C_pc_N):
    """Pseudo-Kontakt Term"""
    r = jnp.linalg.norm(r_vec)
    cosθ = r_vec[2]/r
    D = jnp.diag(jnp.array([1.0, -0.5, -0.5]))
    return C_pc * (3*cosθ**2 - 1)/r**3 * D

@partial(jit, static_argnums=(1, 2))
def A_total_N(r_vec, rho_e0, A_iso):
    """Gesamter Hyperfein-Tensor für 14N
    
    Hinweis: A_iso sollte bereits den Fermi-Kontakt beinhalten.
    Wenn rho_e0 gegeben ist, wird es zusätzlich berechnet (meist nicht erwünscht).
    """
    # Verwende lax.cond für bedingte Ausführung in JAX
    # Aber eigentlich sollte rho_e0 = 0 sein wenn A_iso verwendet wird
    A_FC = jnp.where(
        rho_e0 < 1e28,  # Nur realistische Werte verwenden
        A_fermi_contact_N(rho_e0) * jnp.eye(3, dtype=jnp.complex128),
        jnp.zeros((3, 3), dtype=jnp.complex128)
    )
    
    A_isoM = A_iso * jnp.eye(3, dtype=jnp.complex128)
    A_dip = A_dipolar_N(r_vec)
    A_pc = A_pseudo_contact_N(r_vec)
    return A_FC + A_isoM + A_dip + A_pc

# ---------- 14N Hyperfein + Quadrupol (JIT-optimiert) ----------
@partial(jit, static_argnums=(1, 2, 3))
def _H_N14_realistic_18x18_jax(r_vec, rho_e0, A_iso, P_quad):
    """Interne JAX-Version - Berechnet 14N-Hyperfein + Quadrupol in 18×18"""
    S_ops = spin1_ops()        # Elektron S=1 (3×3)
    I_N = spin1_ops()          # 14N    I=1 (3×3)
    
    # Hyperfine-Tensor für 14N
    A_N = A_total_N(r_vec, rho_e0, A_iso)
    
    # Erst 9×9 Matrix bauen (Elektron × 14N)
    H_9x9 = jnp.zeros((9, 9), dtype=jnp.complex128)
    
    # Hyperfein-Wechselwirkung: Elektron-14N
    for a, Sa in enumerate(S_ops):
        for b, Ib in enumerate(I_N):
            # Direktes Kronecker-Produkt für Elektron ⊗ 14N
            S_N = jnp.kron(Sa, Ib)
            H_9x9 = H_9x9 + A_N[a,b] * S_N
    
    # Quadrupol-Term: P*(I_z^2 - 2/3 I·I) ⊗ I_e
    I_z = I_N[2]  # 14N z-Operator
    I_squared = I_N[0] @ I_N[0] + I_N[1] @ I_N[1] + I_N[2] @ I_N[2]  # I·I
    Q_op = I_z @ I_z - (2.0/3.0) * I_squared  # Quadrupol-Operator
    
    # Quadrupol wirkt nur auf 14N, Elektron bekommt Identity
    Q_9x9 = jnp.kron(jnp.eye(3, dtype=jnp.complex128), Q_op) * P_quad
    H_9x9 = H_9x9 + Q_9x9
    
    # Erweitere auf 18×18 durch Einbettung für C13-Raum
    # Ordnung: ersten 9 für mC=-1/2, nächste 9 für mC=+1/2
    H_18x18 = jnp.zeros((18, 18), dtype=jnp.complex128)
    
    # Füge die 9×9 Matrix zweimal diagonal ein (für jeden C13-Zustand)
    H_18x18 = H_18x18.at[0:9, 0:9].set(H_9x9)      # mC = -1/2
    H_18x18 = H_18x18.at[9:18, 9:18].set(H_9x9)    # mC = +1/2
    
    return H_18x18

# ---------- Hauptfunktion für hama.py Kompatibilität ----------
def H_N14_realistic_18x18(r_vec, rho_e0, A_iso, P_quad):
    """Wrapper-Funktion die NumPy Arrays akzeptiert und zurückgibt.
    Intern wird JAX für Performance verwendet."""
    # Konvertiere NumPy zu JAX
    r_vec_jax = jnp.array(r_vec)
    
    # WICHTIG: Wenn A_iso gegeben ist, setze rho_e0 auf 0
    # um Doppelzählung zu vermeiden
    if A_iso > 0 and rho_e0 > 1e28:
        rho_e0 = 0.0  # Verhindere unrealistische Fermi-Kontakt Berechnung
    
    # Berechne mit JAX
    H_jax = _H_N14_realistic_18x18_jax(r_vec_jax, rho_e0, A_iso, P_quad)
    
    # Konvertiere zurück zu NumPy
    return np.array(H_jax)

# ---------- Physik-Validierung ----------
def validate_n14_physics(H):
    """Validiert physikalische Eigenschaften des 14N-Hamiltonians"""
    # 1. Hermitezität
    is_hermitian = jnp.allclose(H, H.conj().T, rtol=1e-10)
    
    # 2. Reelle Eigenwerte
    eigvals = jnp.linalg.eigvalsh(H)
    all_real = jnp.all(jnp.abs(jnp.imag(eigvals)) < 1e-10)
    
    # 3. Quadrupol-Term sollte spurlos sein
    # (aber Gesamtmatrix nicht, wegen isotropem Term)
    
    return {
        'hermitian': is_hermitian,
        'real_eigenvalues': all_real,
        'eigenvalues': eigvals
    }

def validate_n14_spin_operators():
    """Validiert 14N Spin-Operator Kommutatorrelationen"""
    Sx, Sy, Sz = spin1_ops()
    
    # Kommutatorrelationen [Si, Sj] = i*εijk*Sk für Spin-1
    comm_xy_z = Sx @ Sy - Sy @ Sx - 1j * Sz
    comm_yz_x = Sy @ Sz - Sz @ Sy - 1j * Sx
    comm_zx_y = Sz @ Sx - Sx @ Sz - 1j * Sy
    
    valid_comm = (jnp.allclose(comm_xy_z, 0, atol=1e-10) and
                  jnp.allclose(comm_yz_x, 0, atol=1e-10) and
                  jnp.allclose(comm_zx_y, 0, atol=1e-10))
    
    # Casimir-Operator I²
    I2 = Sx @ Sx + Sy @ Sy + Sz @ Sz
    I2_expected = 2 * jnp.eye(3)  # I(I+1) = 1(1+1) = 2
    valid_casimir = jnp.allclose(I2, I2_expected, atol=1e-10)
    
    return valid_comm and valid_casimir

# ---------- Beispielaufruf & Ausgabe ----------
if __name__ == "__main__":
    print("=== JAX-Optimierte 14N Hyperfein-Simulation ===\n")
    
    # Device-Info
    print(f"JAX Version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Float64 enabled: {jax.config.x64_enabled}\n")
    
    # Physik-Validierung
    print("1. Validiere Spin-Operatoren...")
    if validate_n14_spin_operators():
        print("   ✓ Spin-Operatoren korrekt\n")
    else:
        print("   ✗ FEHLER in Spin-Operatoren!\n")
    
    # Beispiel-Rechnung
    r = jnp.array([0.45e-9, 0.0, 0.0])
    H18_N = H_N14_realistic_18x18(r, rho_e0_N, A_iso_N, P_quad)
    
    print("2. Validiere Hamiltonian...")
    validation = validate_n14_physics(H18_N)
    print(f"   Hermitesch: {validation['hermitian']}")
    print(f"   Reelle Eigenwerte: {validation['real_eigenvalues']}")
    print(f"   Eigenwerte (MHz): {validation['eigenvalues'][:6]/1e6}\n")
    
    # Ausgabe
    np.set_printoptions(precision=6, suppress=True)
    print("3. JAX-Optimierter 18×18 14N-Hyperfein-Hamiltonian:")
    print(H18_N)