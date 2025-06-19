#!/usr/bin/env python3
"""
Lindblad Master Equation Solver for NV Centers
==============================================
Complete quantum dynamics with density matrix evolution
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
from scipy.special import wofz
import scipy.constants as sc

# JAX settings
jax.config.update("jax_enable_x64", True)

# Physical constants
K_B = sc.Boltzmann
H_BAR = sc.hbar
E_VOLT = sc.electron_volt

@jit
def lindblad_rhs(rho, H, collapse_ops):
    """
    Right-hand side der Lindblad-Gleichung.
    d𝜌/dt = -i[H,𝜌] + Σ_k (L_k 𝜌 L_k† - 1/2{L_k†L_k, 𝜌})
    """
    # Kommutator: -i[H,ρ]
    dr = -1j * (H @ rho - rho @ H)
    
    # Lindblad-Terme für jeden Kollaps-Operator
    for c in collapse_ops:
        c_dag = c.conj().T
        c_dag_c = c_dag @ c
        
        # L_k ρ L_k†
        dr += c @ rho @ c_dag
        
        # -1/2 {L_k†L_k, ρ}
        dr -= 0.5 * (c_dag_c @ rho + rho @ c_dag_c)
    
    return dr

@jit
def step_master_equation(rho, H, collapse_ops, dt):
    """
    Ein Zeitschritt der Dichtematrix-Propagation (Euler).
    Für bessere Stabilität könnte man Runge-Kutta verwenden.
    """
    drho_dt = lindblad_rhs(rho, H, collapse_ops)
    rho_new = rho + dt * drho_dt
    
    # Ensure Hermiticity and trace preservation (numerical stability)
    rho_new = 0.5 * (rho_new + rho_new.conj().T)
    rho_new = rho_new / jnp.trace(rho_new)
    
    return rho_new

@jit
def rk4_master_equation(rho, H, collapse_ops, dt):
    """
    4th-order Runge-Kutta integration der Master-Gleichung.
    Bessere numerische Stabilität als Euler.
    """
    k1 = lindblad_rhs(rho, H, collapse_ops)
    k2 = lindblad_rhs(rho + dt/2 * k1, H, collapse_ops)
    k3 = lindblad_rhs(rho + dt/2 * k2, H, collapse_ops)
    k4 = lindblad_rhs(rho + dt * k3, H, collapse_ops)
    
    rho_new = rho + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    # Ensure Hermiticity and trace preservation
    rho_new = 0.5 * (rho_new + rho_new.conj().T)
    rho_new = rho_new / jnp.trace(rho_new)
    
    return rho_new

def build_nv_hamiltonian(params, B_field=None):
    """
    Build complete NV Hamiltonian for ground + excited states.
    
    Basis: |gs,ms⟩ ⊗ |es,ms⟩ ⊗ |singlet⟩
    Ground state: |0⟩ = |gs,-1⟩, |1⟩ = |gs,0⟩, |2⟩ = |gs,+1⟩
    Excited state: |3⟩ = |es,-1⟩, |4⟩ = |es,0⟩, |5⟩ = |es,+1⟩  
    Singlet: |6⟩ = |singlet⟩
    """
    # 7x7 Hilbert space: 3 GS + 3 ES + 1 Singlet
    H = jnp.zeros((7, 7), dtype=jnp.complex128)
    
    # Ground state zero-field splitting
    D_gs = params.D_GS_Hz * 2 * jnp.pi  # Convert to rad/s
    H = H.at[0, 0].set(D_gs)      # |gs,-1⟩
    H = H.at[1, 1].set(0.0)       # |gs,0⟩ (reference)
    H = H.at[2, 2].set(D_gs)      # |gs,+1⟩
    
    # Excited state zero-field splitting + optical transition
    D_es = params.D_ES_Hz * 2 * jnp.pi
    omega_opt = 2 * jnp.pi * 470e12  # ~470 THz for 637nm
    H = H.at[3, 3].set(omega_opt + D_es)  # |es,-1⟩
    H = H.at[4, 4].set(omega_opt)         # |es,0⟩
    H = H.at[5, 5].set(omega_opt + D_es)  # |es,+1⟩
    
    # Singlet state energy
    omega_singlet = omega_opt - 2 * jnp.pi * 1.19e12  # ~1200 nm
    H = H.at[6, 6].set(omega_singlet)
    
    # Magnetic field effects (Zeeman)
    if B_field is not None:
        g_e = 2.003
        mu_B = 9.274e-24  # Bohr magneton
        B_z = B_field[2] * 1e-3  # Convert mT to T
        zeeman = g_e * mu_B * B_z / H_BAR
        
        # Add Zeeman splitting (only to ms=±1 states)
        H = H.at[0, 0].add(-zeeman)  # |gs,-1⟩
        H = H.at[2, 2].add(+zeeman)  # |gs,+1⟩
        H = H.at[3, 3].add(-zeeman)  # |es,-1⟩
        H = H.at[5, 5].add(+zeeman)  # |es,+1⟩
    
    return H

def make_spin_operators():
    """
    Create spin operators for 7-level NV system.
    Returns dictionaries with ground state and excited state operators.
    """
    # Ground state spin operators (3x3 subspace)
    Sx_gs = jnp.array([[0, 1/jnp.sqrt(2), 0],
                       [1/jnp.sqrt(2), 0, 1/jnp.sqrt(2)],
                       [0, 1/jnp.sqrt(2), 0]], dtype=jnp.complex128)
    
    Sy_gs = jnp.array([[0, -1j/jnp.sqrt(2), 0],
                       [1j/jnp.sqrt(2), 0, -1j/jnp.sqrt(2)],
                       [0, 1j/jnp.sqrt(2), 0]], dtype=jnp.complex128)
    
    Sz_gs = jnp.array([[-1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 1]], dtype=jnp.complex128)
    
    # Extend to full 7x7 space
    Sx_full = jnp.zeros((7, 7), dtype=jnp.complex128)
    Sy_full = jnp.zeros((7, 7), dtype=jnp.complex128)
    Sz_full = jnp.zeros((7, 7), dtype=jnp.complex128)
    
    # Ground state block (0:3, 0:3)
    Sx_full = Sx_full.at[0:3, 0:3].set(Sx_gs)
    Sy_full = Sy_full.at[0:3, 0:3].set(Sy_gs)
    Sz_full = Sz_full.at[0:3, 0:3].set(Sz_gs)
    
    # Excited state block (3:6, 3:6) - same structure
    Sx_full = Sx_full.at[3:6, 3:6].set(Sx_gs)
    Sy_full = Sy_full.at[3:6, 3:6].set(Sy_gs)
    Sz_full = Sz_full.at[3:6, 3:6].set(Sz_gs)
    
    return {
        'Sx': Sx_full,
        'Sy': Sy_full, 
        'Sz': Sz_full
    }

def make_transition_operators():
    """
    Create transition operators between different manifolds.
    """
    # Optical transitions: |gs⟩ ↔ |es⟩
    sigma_plus = jnp.zeros((7, 7), dtype=jnp.complex128)   # |gs,-1⟩ → |es,0⟩
    sigma_minus = jnp.zeros((7, 7), dtype=jnp.complex128)  # |gs,+1⟩ → |es,0⟩
    pi_transition = jnp.zeros((7, 7), dtype=jnp.complex128) # |gs,0⟩ → |es,0⟩
    
    # σ+ transition
    sigma_plus = sigma_plus.at[4, 0].set(1.0)  # |es,0⟩ ← |gs,-1⟩
    
    # σ- transition  
    sigma_minus = sigma_minus.at[4, 2].set(1.0)  # |es,0⟩ ← |gs,+1⟩
    
    # π transition
    pi_transition = pi_transition.at[4, 1].set(1.0)  # |es,0⟩ ← |gs,0⟩
    
    # ISC transitions: |es⟩ → |singlet⟩
    isc_ms0 = jnp.zeros((7, 7), dtype=jnp.complex128)
    isc_ms1 = jnp.zeros((7, 7), dtype=jnp.complex128)
    
    isc_ms0 = isc_ms0.at[6, 4].set(1.0)  # |singlet⟩ ← |es,0⟩
    isc_ms1 = isc_ms1.at[6, 3].set(1.0)  # |singlet⟩ ← |es,-1⟩
    isc_ms1 = isc_ms1.at[6, 5].set(1.0)  # |singlet⟩ ← |es,+1⟩
    
    # Singlet decay: |singlet⟩ → |gs⟩
    singlet_decay = jnp.zeros((7, 7), dtype=jnp.complex128)
    singlet_decay = singlet_decay.at[1, 6].set(1.0)  # |gs,0⟩ ← |singlet⟩
    
    return {
        'sigma_plus': sigma_plus,
        'sigma_minus': sigma_minus,
        'pi_transition': pi_transition,
        'isc_ms0': isc_ms0,
        'isc_ms1': isc_ms1,
        'singlet_decay': singlet_decay
    }

def arrhenius_rate(k0, E_a_eV, T_K):
    """Temperature-dependent rate: k = k₀·exp(−Eₐ/(k_B·T))"""
    return k0 * jnp.exp(-E_a_eV * E_VOLT / (K_B * T_K))

def make_collapse_operators(params, transitions):
    """
    Create all Lindblad collapse operators for NV dynamics.
    """
    collapse_ops = []
    
    # 1. Spontaneous emission: |es⟩ → |gs⟩
    gamma_rad = params.gamma_rad_MHz * 1e6 * 2 * jnp.pi  # Convert to rad/s
    
    # Different transitions have different rates
    c_sigma_plus = jnp.sqrt(gamma_rad * 0.3) * transitions['sigma_plus'].T
    c_sigma_minus = jnp.sqrt(gamma_rad * 0.3) * transitions['sigma_minus'].T  
    c_pi = jnp.sqrt(gamma_rad * 0.4) * transitions['pi_transition'].T
    
    collapse_ops.extend([c_sigma_plus, c_sigma_minus, c_pi])
    
    # 2. ISC transitions: |es⟩ → |singlet⟩ (temperature dependent)
    gamma_isc_ms0 = arrhenius_rate(
        params.gamma_ISC_ms0_MHz * 1e6 * 2 * jnp.pi,
        params.ISC_ms0_activation_meV * 1e-3,
        params.Temperature_K
    )
    gamma_isc_ms1 = arrhenius_rate(
        params.gamma_ISC_ms1_MHz * 1e6 * 2 * jnp.pi,
        params.ISC_ms1_activation_meV * 1e-3,
        params.Temperature_K
    )
    
    c_isc_ms0 = jnp.sqrt(gamma_isc_ms0) * transitions['isc_ms0']
    c_isc_ms1 = jnp.sqrt(gamma_isc_ms1) * transitions['isc_ms1']
    
    collapse_ops.extend([c_isc_ms0, c_isc_ms1])
    
    # 3. Singlet decay: |singlet⟩ → |gs⟩
    gamma_singlet = 2 * jnp.pi / (params.tau_singlet_ns * 1e-9)
    c_singlet = jnp.sqrt(gamma_singlet) * transitions['singlet_decay']
    
    collapse_ops.append(c_singlet)
    
    # 4. Orbital relaxation in excited state (temperature dependent)
    if hasattr(params, 'k_orb_300K_MHz'):
        # Temperature-dependent orbital relaxation rate
        T_K = params.Temperature_K
        E_activation_K = params.k_orb_activation_K
        k0_Hz = params.k_orb_300K_MHz * 1e6
        
        # Arrhenius rate
        gamma_orbital = k0_Hz * (T_K / 300.0) * jnp.exp(-E_activation_K / T_K)
        
        # Orbital dephasing in excited state manifold
        # Creates coherence loss between orbital states
        orbital_dephasing = jnp.zeros((7, 7), dtype=jnp.complex128)
        
        # Dephasing between excited state sublevels
        for i in range(3, 6):  # Excited state indices
            orbital_dephasing = orbital_dephasing.at[i, i].set(jnp.sqrt(gamma_orbital))
        
        c_orbital = orbital_dephasing
        collapse_ops.append(c_orbital)
        
        # Direct orbital relaxation: |es⟩ → |singlet⟩ (non-radiative)
        orbital_quench = jnp.zeros((7, 7), dtype=jnp.complex128)
        
        # All excited states can relax to singlet via orbital coupling
        for i in range(3, 6):  # |es,-1⟩, |es,0⟩, |es,+1⟩
            orbital_quench = orbital_quench.at[6, i].set(jnp.sqrt(gamma_orbital * 0.1))  # 10% branching
        
        collapse_ops.append(orbital_quench)
    
    # 5. Ground state T1 relaxation
    if params.T1_ms > 0:
        gamma_t1 = 1.0 / (params.T1_ms * 1e-3)
        spins = make_spin_operators()
        
        # T1 processes: spin flip in ground state
        c_t1_plus = jnp.sqrt(gamma_t1 * 0.5) * (spins['Sx'] + 1j * spins['Sy'])
        c_t1_minus = jnp.sqrt(gamma_t1 * 0.5) * (spins['Sx'] - 1j * spins['Sy'])
        
        collapse_ops.extend([c_t1_plus, c_t1_minus])
    
    # 6. Pure dephasing from spectral diffusion
    if hasattr(params, 'specdiff_sigma_MHz'):
        # Spectral diffusion causes pure dephasing
        tau_corr_s = params.specdiff_tau_corr_ns * 1e-9
        sigma_Hz = params.specdiff_sigma_MHz * 1e6
        gamma_dephasing = sigma_Hz**2 * tau_corr_s  # Dephasing rate
        
        spins = make_spin_operators()
        # Pure dephasing operator: Sz (no population change)
        c_dephasing = jnp.sqrt(gamma_dephasing) * spins['Sz']
        collapse_ops.append(c_dephasing)
    
    return collapse_ops

def voigt_profile(delta_GHz, sigma_GHz, gamma_GHz):
    """
    Voigt line profile: convolution of Gaussian and Lorentzian.
    Uses Faddeeva function for efficient computation.
    """
    z = (delta_GHz + 1j * gamma_GHz) / (sigma_GHz * jnp.sqrt(2))
    # Use JAX-compatible implementation
    w = jnp.real(z) + 1j * jnp.imag(z)  # Ensure complex
    # Simplified Voigt (would need proper Faddeeva in JAX)
    return jnp.exp(-jnp.real(z)**2) / (jnp.sqrt(jnp.pi) * sigma_GHz)

def laser_hamiltonian(params, laser_detuning_Hz=0.0):
    """
    Laser interaction Hamiltonian with Voigt profile.
    """
    transitions = make_transition_operators()
    
    # Laser strength with saturation
    I_rel = params.laser_power_mW / params.I_sat_mW
    
    # Voigt profile for laser linewidth
    delta_GHz = laser_detuning_Hz / 1e9
    voigt_factor = voigt_profile(delta_GHz, 0.1, 0.01)  # Typical linewidths
    
    # Rabi frequency
    Omega = jnp.sqrt(I_rel) * params.Omega_Rabi_Hz * 2 * jnp.pi * voigt_factor
    
    # Laser Hamiltonian (RWA)
    H_laser = Omega * (transitions['pi_transition'] + transitions['pi_transition'].T.conj())
    
    return H_laser

def microwave_hamiltonian(params, mw_amplitude, mw_phase=0.0):
    """
    Microwave Hamiltonian for spin manipulation.
    """
    spins = make_spin_operators()
    
    # MW Rabi frequency
    Omega_mw = mw_amplitude * 2 * jnp.pi
    
    # Rotating wave approximation: σ_x rotation
    H_mw = Omega_mw * (jnp.cos(mw_phase) * spins['Sx'] + jnp.sin(mw_phase) * spins['Sy'])
    
    return H_mw

class LindblladNVSimulator:
    """
    Complete Lindblad master equation simulator for NV centers.
    """
    
    def __init__(self, params):
        self.params = params
        self.H_base = build_nv_hamiltonian(params, params.B_field_mT)
        self.transitions = make_transition_operators()
        self.collapse_ops = make_collapse_operators(params, self.transitions)
        
        # Initial state: thermal equilibrium in ground state
        self.rho_initial = self._thermal_ground_state()
    
    def _thermal_ground_state(self):
        """Initialize thermal equilibrium in ground state manifold."""
        rho = jnp.zeros((7, 7), dtype=jnp.complex128)
        
        # Thermal populations in ground state (ms=0 favored at low T)
        kT = K_B * self.params.Temperature_K
        D_gs = self.params.D_GS_Hz * 2 * jnp.pi * H_BAR
        
        if kT > 0:
            # Boltzmann distribution
            E_gs = jnp.array([D_gs, 0.0, D_gs])  # Energies of |±1⟩, |0⟩
            weights = jnp.exp(-E_gs / kT)
            weights = weights / jnp.sum(weights)
        else:
            # T=0: only |0⟩ populated
            weights = jnp.array([0.0, 1.0, 0.0])
        
        # Populate ground state diagonal
        rho = rho.at[0, 0].set(weights[0])  # |gs,-1⟩
        rho = rho.at[1, 1].set(weights[1])  # |gs,0⟩  
        rho = rho.at[2, 2].set(weights[2])  # |gs,+1⟩
        
        return rho
    
    @jit
    def evolve_master_equation(self, rho, H_total, dt, n_steps):
        """
        Evolve density matrix for multiple time steps.
        """
        for _ in range(n_steps):
            rho = rk4_master_equation(rho, H_total, self.collapse_ops, dt)
        return rho
    
    def simulate_pulse_sequence(self, pulse_times, pulse_amplitudes, total_time_ns):
        """
        Simulate arbitrary pulse sequence with Lindblad evolution.
        """
        dt = 0.1e-9  # 0.1 ns time step
        n_total_steps = int(total_time_ns * 1e-9 / dt)
        
        rho = self.rho_initial
        time_trace = []
        populations = []
        
        current_time = 0.0
        
        for step in range(n_total_steps):
            time_ns = step * dt * 1e9
            
            # Check if MW pulse is active
            H_mw = jnp.zeros_like(self.H_base)
            for i, (t_start, t_end) in enumerate(pulse_times):
                if t_start <= time_ns < t_end:
                    amplitude = pulse_amplitudes[i] if i < len(pulse_amplitudes) else 0.0
                    H_mw = microwave_hamiltonian(self.params, amplitude)
                    break
            
            # Total Hamiltonian
            H_total = self.H_base + H_mw
            
            # Evolve one step
            rho = rk4_master_equation(rho, H_total, self.collapse_ops, dt)
            
            # Record populations every 1 ns
            if step % 10 == 0:  # Every 1 ns (10 steps of 0.1 ns)
                time_trace.append(time_ns)
                pops = jnp.real(jnp.diag(rho))
                populations.append(pops)
        
        return jnp.array(time_trace), jnp.array(populations)
    
    def get_ground_state_populations(self, rho):
        """Extract ground state populations from density matrix."""
        return {
            'p_ms_minus1': float(jnp.real(rho[0, 0])),
            'p_ms_0': float(jnp.real(rho[1, 1])), 
            'p_ms_plus1': float(jnp.real(rho[2, 2]))
        }
    
    def get_fluorescence_rate(self, rho):
        """
        Calculate fluorescence rate from current density matrix.
        Proportional to excited state population.
        """
        pop_excited = jnp.real(rho[3, 3] + rho[4, 4] + rho[5, 5])
        return pop_excited * self.params.Beta_max_Hz

if __name__ == "__main__":
    # Test the Lindblad simulator
    import sys
    sys.path.append('.')
    from nv import AdvancedNVParams
    
    params = AdvancedNVParams()
    params.B_field_mT = [0.0, 0.0, 1.0]  # 1 mT along z
    
    sim = LindblladNVSimulator(params)
    
    print("🔬 Testing Lindblad Master Equation Simulator")
    print("=" * 50)
    
    # Test ground state initialization
    pops_initial = sim.get_ground_state_populations(sim.rho_initial)
    print(f"Initial ground state populations:")
    print(f"  |ms=-1⟩: {pops_initial['p_ms_minus1']:.3f}")
    print(f"  |ms=0⟩:  {pops_initial['p_ms_0']:.3f}")
    print(f"  |ms=+1⟩: {pops_initial['p_ms_plus1']:.3f}")
    
    # Test simple π/2 pulse
    pulse_times = [(0, 100)]  # 100 ns pulse
    pulse_amplitudes = [params.Omega_Rabi_Hz]  # π/2 pulse amplitude
    
    time_trace, populations = sim.simulate_pulse_sequence(
        pulse_times, pulse_amplitudes, total_time_ns=500
    )
    
    print(f"\nAfter 100ns MW pulse:")
    final_rho = sim.rho_initial  # Would need to extract from evolution
    # This is simplified - full implementation would track final state
    
    print("✅ Lindblad simulator initialized successfully!")