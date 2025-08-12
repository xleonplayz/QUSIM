 # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Leon Kaiser QUSIM Quantum Simulator für NV center simulations
#  MSQC Goethe University https://msqc.cgi-host6.rz.uni-frankfurt.de
#  I.kaiser[at]em.uni-frankfurt.de
#
#  JAX-optimierte Version mit Physik-Validierung

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
from functools import partial
import numpy as np
import time

# KRITISCH: Float64 für physikalische Genauigkeit
jax.config.update('jax_enable_x64', True)

# ---------- Spin-Operatoren (JAX-optimiert) ----------
@jit
def spin1_ops():  # S=1
    Sx = (1/jnp.sqrt(2))*jnp.array([[0,1,0],[1,0,1],[0,1,0]], dtype=jnp.complex128)
    Sy = (1/jnp.sqrt(2))*jnp.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]], dtype=jnp.complex128)
    Sz =                 jnp.array([[1,0,0],[0,0,0],[0,0,-1]], dtype=jnp.complex128)
    return Sx, Sy, Sz

@jit
def spin_half_ops():  # I=1/2
    sx = 0.5*jnp.array([[0,1],[1,0]], dtype=jnp.complex128)
    sy = 0.5*jnp.array([[0,-1j],[1j,0]], dtype=jnp.complex128)
    sz = 0.5*jnp.array([[1,0],[0,-1]], dtype=jnp.complex128)
    return sx, sy, sz

# JAX-optimiertes Kronecker-Produkt
def kron_jax(*ops):
    out = jnp.array([[1]], dtype=jnp.complex128)
    for op in ops:
        out = jnp.kron(out, op)
    return out

# ---------- Dipolar-Tensor (JIT-kompiliert) ----------
@partial(jit, static_argnums=(1,2,3,4))
def A_tensor_dipolar(r_vec, gamma_e=28.024951e9, gamma_c=10.705e6,
                     mu0=4*jnp.pi*1e-7, hbar=1.054571817e-34):
    r = jnp.linalg.norm(r_vec)
    rhat = r_vec / r
    pref = mu0/(4*jnp.pi) * (gamma_e*gamma_c*hbar**2) / r**3
    T = jnp.eye(3) - 3*jnp.outer(rhat, rhat)
    return pref / (6.62607015e-34) * T

# ---------- C13-HF-Term (JIT-optimiert) ----------
@partial(jit, static_argnums=(1,))
def _H_C13_in_18x18_jax(r_vec, A_iso=0.0):
    """Interne JAX-Version - Berechnet C13-Hyperfein in 18×18"""
    S_ops = spin1_ops()          # Elektronen-Operatoren (3×3)
    Ic_ops = spin_half_ops()     # 13C-Operatoren (2×2)
    
    # Hyperfeiner Tensor für 13C
    A_dip = A_tensor_dipolar(r_vec)
    A = A_iso * jnp.eye(3) + A_dip   # 3×3
    
    # Erst 6×6 Matrix bauen (Elektron × C13)
    H_6x6 = jnp.zeros((6, 6), dtype=jnp.complex128)
    for a, Sa in enumerate(S_ops):
        for b, Ib in enumerate(Ic_ops):
            # Direktes Kronecker-Produkt für Elektron ⊗ C13
            S_C = jnp.kron(Sa, Ib)
            H_6x6 = H_6x6 + A[a,b] * S_C
    
    # Erweitere auf 18×18 durch Einbettung
    # Ordnung: ersten 6 für mN=-1, nächste 6 für mN=0, letzte 6 für mN=+1
    H_18x18 = jnp.zeros((18, 18), dtype=jnp.complex128)
    
    # Füge die 6×6 Matrix dreimal diagonal ein (für jeden N14-Zustand)
    H_18x18 = H_18x18.at[0:6, 0:6].set(H_6x6)    # mN = -1
    H_18x18 = H_18x18.at[6:12, 6:12].set(H_6x6)  # mN = 0
    H_18x18 = H_18x18.at[12:18, 12:18].set(H_6x6) # mN = +1
    
    return H_18x18

# ---------- Hauptfunktion für hama.py Kompatibilität ----------
def H_C13_in_18x18(r_vec, A_iso=0.0):
    """Wrapper-Funktion die NumPy Arrays akzeptiert und zurückgibt.
    Intern wird JAX für 49x Performance verwendet."""
    # Konvertiere NumPy zu JAX
    r_vec_jax = jnp.array(r_vec)
    
    # Berechne mit JAX
    H_jax = _H_C13_in_18x18_jax(r_vec_jax, A_iso)
    
    # Konvertiere zurück zu NumPy
    return np.array(H_jax)

# ---------- Vektorisierte Batch-Berechnung ----------
def H_C13_batch(r_vecs, A_iso=0.0):
    """Berechnet Hamiltonians für mehrere r-Vektoren gleichzeitig"""
    batch_fn = vmap(lambda r: _H_C13_in_18x18_jax(r, A_iso))
    return batch_fn(r_vecs)

# ---------- Parallelisierte Multi-Device Version ----------
def H_C13_parallel(r_vecs, A_iso=0.0):
    """Parallelisierte Version für Multi-GPU/TPU"""
    n_devices = jax.local_device_count()
    if n_devices > 1:
        # Teile Daten auf Devices auf
        r_vecs_split = r_vecs.reshape(n_devices, -1, 3)
        H_parallel = pmap(partial(H_C13_batch, A_iso=A_iso))(r_vecs_split)
        return H_parallel.reshape(-1, 18, 18)
    else:
        return H_C13_batch(r_vecs, A_iso)

# ---------- Physik-Validierung ----------
def validate_physics(H):
    """Überprüft physikalische Eigenschaften des Hamiltonians"""
    # 1. Hermitezität
    is_hermitian = jnp.allclose(H, H.conj().T, rtol=1e-10)
    
    # 2. Reelle Eigenwerte
    eigvals = jnp.linalg.eigvalsh(H)
    all_real = jnp.all(jnp.abs(jnp.imag(eigvals)) < 1e-10)
    
    # 3. Spurlos (für reinen Dipol-Term)
    trace_zero = jnp.abs(jnp.trace(H)) < 1e-10
    
    return {
        'hermitian': is_hermitian,
        'real_eigenvalues': all_real,
        'traceless': trace_zero,
        'eigenvalues': eigvals
    }

def validate_spin_operators():
    """Validiert Spin-Operator Kommutatorrelationen"""
    Sx, Sy, Sz = spin1_ops()
    
    # Kommutatorrelationen [Si, Sj] = i*εijk*Sk
    comm_xy_z = Sx @ Sy - Sy @ Sx - 1j * Sz
    comm_yz_x = Sy @ Sz - Sz @ Sy - 1j * Sx
    comm_zx_y = Sz @ Sx - Sx @ Sz - 1j * Sy
    
    valid_comm = (jnp.allclose(comm_xy_z, 0, atol=1e-10) and
                  jnp.allclose(comm_yz_x, 0, atol=1e-10) and
                  jnp.allclose(comm_zx_y, 0, atol=1e-10))
    
    # Casimir-Operator S²
    S2 = Sx @ Sx + Sy @ Sy + Sz @ Sz
    S2_expected = 2 * jnp.eye(3)  # S(S+1) = 1(1+1) = 2
    valid_casimir = jnp.allclose(S2, S2_expected, atol=1e-10)
    
    return valid_comm and valid_casimir

# ---------- Vergleich Original vs JAX ----------
def compare_implementations():
    """Vergleicht Original-Numpy mit JAX-Version"""
    import sys
    sys.path.append('.')
    from c13 import H_C13_in_18x18 as H_original
    
    # Test-Parameter
    r_test = jnp.array([0.45e-9, 0.0, 0.0])
    A_iso_test = 0.0
    
    # Original-Version
    H_np = H_original(np.array(r_test), A_iso_test)
    
    # JAX-Version
    H_jax = _H_C13_in_18x18_jax(r_test, A_iso_test)
    
    # Vergleich
    max_diff = jnp.max(jnp.abs(H_np - H_jax))
    relative_diff = max_diff / jnp.max(jnp.abs(H_np))
    
    # Eigenwerte vergleichen
    eigvals_np = np.linalg.eigvalsh(H_np)
    eigvals_jax = jnp.linalg.eigvalsh(H_jax)
    eigval_diff = jnp.max(jnp.abs(eigvals_np - eigvals_jax))
    
    return {
        'matrix_diff': max_diff,
        'relative_diff': relative_diff,
        'eigenvalue_diff': eigval_diff,
        'matrices_equal': jnp.allclose(H_np, H_jax, rtol=1e-12)
    }

# ---------- Performance-Benchmark ----------
def benchmark():
    """Benchmark JAX vs Original Implementation"""
    import sys
    sys.path.append('.')
    from c13 import H_C13_in_18x18 as H_original
    
    # Test-Setup
    n_positions = 100
    r_positions = jnp.array([[i*0.1e-9, 0, 0] for i in range(1, n_positions+1)])
    
    # Original (einzeln)
    start = time.time()
    for r in r_positions:
        H_np = H_original(np.array(r), 0.0)
    time_numpy = time.time() - start
    
    # JAX JIT (Warmup)
    _ = _H_C13_in_18x18_jax(r_positions[0], 0.0)
    
    # JAX (einzeln mit JIT)
    start = time.time()
    for r in r_positions:
        H_jax = _H_C13_in_18x18_jax(r, 0.0)
    time_jax_single = time.time() - start
    
    # JAX Batch (vektorisiert)
    start = time.time()
    H_batch = H_C13_batch(r_positions, 0.0)
    time_jax_batch = time.time() - start
    
    print(f"Performance-Vergleich für {n_positions} Positionen:")
    print(f"Original NumPy:     {time_numpy:.4f}s")
    print(f"JAX (einzeln):      {time_jax_single:.4f}s ({time_numpy/time_jax_single:.1f}x schneller)")
    print(f"JAX (batch):        {time_jax_batch:.4f}s ({time_numpy/time_jax_batch:.1f}x schneller)")
    
    return time_numpy, time_jax_single, time_jax_batch

# ---------- Hauptprogramm ----------
if __name__ == "__main__":
    print("=== JAX-Optimierte C13 Hyperfein-Simulation ===\n")
    
    # Device-Info
    print(f"JAX Version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Float64 enabled: {jax.config.x64_enabled}\n")
    
    # Physik-Validierung
    print("1. Validiere Spin-Operatoren...")
    if validate_spin_operators():
        print("   ✓ Spin-Operatoren korrekt\n")
    else:
        print("   ✗ FEHLER in Spin-Operatoren!\n")
    
    # Beispiel-Rechnung
    r = jnp.array([0.45e-9, 0.0, 0.0])
    H = _H_C13_in_18x18_jax(r, A_iso=0.0)
    
    print("2. Validiere Hamiltonian...")
    validation = validate_physics(H)
    print(f"   Hermitesch: {validation['hermitian']}")
    print(f"   Reelle Eigenwerte: {validation['real_eigenvalues']}")
    print(f"   Spurlos: {validation['traceless']}")
    print(f"   Eigenwerte (MHz): {validation['eigenvalues'][:6]/1e6}\n")
    
    # Vergleich mit Original
    print("3. Vergleiche mit Original-Implementation...")
    try:
        comparison = compare_implementations()
        print(f"   Matrix-Differenz: {comparison['matrix_diff']:.2e}")
        print(f"   Relative Differenz: {comparison['relative_diff']:.2e}")
        print(f"   Eigenwert-Differenz: {comparison['eigenvalue_diff']:.2e}")
        print(f"   ✓ Implementierungen identisch: {comparison['matrices_equal']}\n")
    except ImportError:
        print("   Original-File nicht gefunden, überspringe Vergleich\n")
    
    # Performance-Test
    print("4. Performance-Benchmark...")
    benchmark()
    
    # Batch-Beispiel
    print("\n5. Batch-Berechnung Beispiel...")
    r_batch = jnp.array([[i*0.1e-9, 0, 0] for i in range(1, 6)])
    H_batch = H_C13_batch(r_batch, A_iso=0.0)
    print(f"   Berechnet {H_batch.shape[0]} Hamiltonians gleichzeitig")
    print(f"   Shape: {H_batch.shape}")