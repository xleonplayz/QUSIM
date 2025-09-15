# nv_simulator_v3.py
# ------------------------------------------------------------
# NV Hyper-Realistic Simulator - VERSION 3 (Mit erweitertem AWG-Controller)
# Klassen:
#   1) NVSystem      – Zustandsraum, Operatoren, Hamiltonians, Lindblad-Stacks
#   2) NVKernels     – JAX-Kerne (drho, rk4_step) aus System + Config
#   3) AWGController – STARK ERWEITERT: Vollständiger AWG70000A-ähnlicher Controller
#   4) NVSimulator   – orchestriert Simulation, Noise & Binning
#
# Version 3 Features:
# - Vollständige AWG-Emulation mit Sample-Rate, Quantisierung, Pre-Emphasis
# - Sequencer mit Waveform-Speicher und Repeat-Modi
# - Marker/Trigger-System
# - Multiple Envelope-Typen (Gaussian, Square, Blackman, FlatTop)
# - SCPI-ähnliche Kommandoschnittstelle
# - Hardware-Fidelity-Modelle (Harmonics, Phase Noise, Jitter)

import jax
import time
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Callable
import matplotlib.pyplot as plt

# ------------------------- Utilities -------------------------

def kron(A, B): return jnp.kron(A, B)

def dagger(X):  return X.conj().T

def dissipator(L, rho):
    return L @ rho @ dagger(L) - 0.5 * (dagger(L) @ L @ rho + rho @ dagger(L) @ L)

def lindblad_group(rho, Ls, Ksum, gamma):
    if isinstance(gamma, (float, int)) and gamma == 0.0:
        return jnp.zeros_like(rho)
    term = jnp.einsum('kij,jl,klm->im', Ls, rho, Ls.conj(), optimize=True) \
         - 0.5*(Ksum @ rho + rho @ Ksum)
    return gamma * term

def ou_step(x, dt, tau, sigma, key):
    a = jnp.exp(-dt/max(tau, 1e-18))
    mean = a * x
    var  = (sigma**2) * (1 - a**2)
    x_new = mean + jnp.sqrt(jnp.maximum(var, 0.0)) * jax.random.normal(key)
    return x_new

def proj_from_indices(idxs, dim):
    M = jnp.zeros((dim, dim), dtype=jnp.complex64)
    for i in idxs:
        M = M.at[i, i].set(1.0+0.0j)
    return M

# Hilfs-OU für AWG (lokal, damit AWG unabhängig bleibt)
def _ou_step(x, dt, tau, sigma, key):
    a = jnp.exp(-dt/max(tau, 1e-18))
    mean = a * x
    var  = (sigma**2) * (1 - a**2)
    x_new = mean + jnp.sqrt(jnp.maximum(var, 0.0)) * jax.random.normal(key)
    return x_new

# ------------------------- Spin & Orbital operators -------------------------

def spin1_ops():
    sx = (1/jnp.sqrt(2)) * jnp.array([[0,1,0],[1,0,1],[0,1,0]], dtype=jnp.complex64)
    sy = (1/jnp.sqrt(2)) * jnp.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]], dtype=jnp.complex64)
    sz = jnp.array([[1,0,0],[0,0,0],[0,0,-1]], dtype=jnp.complex64)
    I3 = jnp.eye(3, dtype=jnp.complex64)
    return sx, sy, sz, I3

def spin_half_ops():
    sx = 0.5*jnp.array([[0,1],[1,0]], dtype=jnp.complex64)
    sy = 0.5*jnp.array([[0,-1j],[1j,0]], dtype=jnp.complex64)
    sz = 0.5*jnp.array([[1,0],[0,-1]], dtype=jnp.complex64)
    I2 = jnp.eye(2, dtype=jnp.complex64)
    return sx, sy, sz, I2

def orbital_E_ops():
    tx = jnp.array([[0,1],[1,0]], dtype=jnp.complex64)
    ty = jnp.array([[0,-1j],[1j,0]], dtype=jnp.complex64)
    tz = jnp.array([[1,0],[0,-1]], dtype=jnp.complex64)
    I2 = jnp.eye(2, dtype=jnp.complex64)
    return tx, ty, tz, I2

# ------------------------- Dataclasses -------------------------

@dataclass
class NVParams:
    # Ground
    D0: float = 2.870e9
    E_g: float = 0.0
    # Excited (spin-only part)
    D_e: float = 1.420e9
    E_e: float = 0.0
    gamma_e: float = 28.024951e9
    B: Tuple[float,float,float] = (0.0,0.0,10e-3)
    # Hyperfine (ground)
    A_par_N: float  = -2.16e6
    A_perp_N: float = -2.70e6
    A_par_C: float  = 0.0
    A_perp_C: float = 0.0
    # Hyperfine (excited)
    A_par_N_e: float = -40e6
    A_perp_N_e: float = -25e6
    A_par_C_e: float = 0.0
    A_perp_C_e: float = 0.0
    # Nuclear Zeeman & quadrupole
    gamma_N: float = 3.077e6
    Q_N: float = -5e5
    gamma_C: float = 10.705e6
    # Excited-state fine structure
    lambda_par: float = 5.3e9
    lambda_perp: float = 0.3e9
    Delta_ss: float = 1.42e9
    # Strain/E-field mixing in e (orbital)
    Pi_x: float = 0.0
    Pi_y: float = 0.0
    # Electric field (DC Stark)
    E_field: Tuple[float,float,float] = (0.0,0.0,0.0)
    kE_g_par: float = 0.0
    kE_g_perp: float = 0.0
    kE_e_par: float = 0.0
    kE_e_perp: float = 0.0

@dataclass
class OpticalRates:
    gamma_rad: float = 75e6
    # ISC via s1 (^1E) and s2 (^1A1)
    k_es_to_s1_0: float = 2e6
    k_es_to_s1_pm: float = 15e6
    k_s1_to_s2: float = 7e6
    k_s2_to_g0: float = 7e6
    k_s2_to_gpm: float = 0.7e6
    # Saturable excitation g->e
    k_exc_max: float = 120e6
    I_sat: float     = 1.0
    # Temperature dependencies
    T0: float = 300.0
    Ea_s1s2_eV: float = 0.0
    Ea_s2g0_eV: float = 0.0
    Ea_s2gpm_eV: float = 0.0
    alpha_s1s2: float = 0.0
    alpha_s2g0: float = 0.0
    alpha_s2gpm: float = 0.0
    # ESLAC nuclear pumping
    k_eslac_max: float = 2e6
    B_eslac_T: float = 0.051
    B_bw_T: float = 0.01
    # Orbital relaxation (phonon)
    k_orb_base: float = 50e6
    alpha_orb: float = 1.0

@dataclass
class PLRates:
    r0: float  = 3.0e7
    rpm: float = 2.0e7
    model: str = "spin_brightness"
    measured_at_detector: bool = True
    prompt_amp: float = 0.0
    prompt_tau: float = 40e-9
    r_nv0: float = 1.0e6
    nv0_quench: float = 0.1
    zpl_fraction: float = 0.03
    psb_fraction: float = 0.97
    f_zpl: float = 1.0
    f_psb: float = 1.0

@dataclass
class NoiseParams:
    T2star_g: float = 3e-6
    T2star_e: float = 0.5e-6
    T2star_s: float = 0.1e-6
    T2star_nv0: float = 0.1e-6
    sd_tau: float = 50e-6
    sd_sigma: float = 500e3
    rin_enable: bool = True
    rin_tau: float = 80e-9
    rin_sigma: float = 0.08
    mag_tau: float = 1e-6
    mag_sigma: float = 5e-6
    pink_terms: int = 4
    pink_tau_min: float = 1e-6
    pink_tau_max: float = 1e-2
    dD_dT: float = 74e3
    T0: float = 300.0
    T_drift_tau: float = 5.0
    T_drift_sigma: float = 0.01
    gamma_c_bath_g: float = 0.0
    gamma_c_bath_e: float = 0.0

@dataclass
class DetectorParams:
    eta: float = 0.02
    dark_rate: float = 300.0
    dead_time: float = 60e-9
    model: str = "nonparalyzable"
    afterpulse_p: float = 0.0
    afterpulse_delay: float = 50e-9
    afterpulse_tail_tau: float = 0.0
    afterpulse_mode: str = "mixture"
    afterpulse_tau1: float = 40e-9
    afterpulse_tau2: float = 200e-9
    afterpulse_w1: float = 0.7
    afterpulse_generations: int = 1
    overdispersion_k: float = 0.0
    n_nv: int = 1000
    sample: bool = True
    shots_accum: int = 1500
    jitter_sigma: float = 50e-12
    saturation_level: float = 1e12

@dataclass
class Sequence:
    t_mw: float = 50e-9
    t_readout: float = 500e-9
    oversample_per_ns: int = 18
    bin_ns: float = 10.0
    mw_amp: float = 0.1
    awg_kappa: float = 10e6
    laser_tau: float = 20e-9
    def I_of_t(self, t: float):
        if self.t_mw <= t < self.t_mw + self.t_readout:
            x = (t - self.t_mw) / max(self.laser_tau, 1e-12)
            return float(1.0 - jnp.exp(-x))
        return 0.0

@dataclass
class InitialState:
    p0: float = 1/3
    pp: float = 1/3
    pm: float = 1/3
    n_idx: int = 1
    c_idx: int = 0

@dataclass
class MWParams:
    omega_mw: float = 2.87e9
    phase0: float = 0.0
    Omega0: float = 5e6
    target_ms_from: int = 1
    target_ms_to: int = 2
    target_n_idx: int = 1
    target_c_idx: int = 0
    amp_tau: float = 200e-9
    amp_sigma: float = 0.02
    phi_tau: float = 200e-9
    phi_sigma: float = 0.01

@dataclass
class RelaxParams:
    T1_g0: float = 3e-3
    a_Raman: float = 1e-3
    b_Orbach: float = 1e7
    Delta_Orbach_eV: float = 0.08
    d_B2: float = 1e3
    T1_back_enabled: bool = False
    back_ratio: float = 1e-3

@dataclass
class ChargeParams:
    k1_ion_0: float = 1e5
    k2_ion_0: float = 0.0
    k1_ion_pm: float = 5e5
    k2_ion_pm: float = 0.0
    k1_ion_g: float = 1e3
    k_rec: float = 5e5
    repump_gain: float = 0.0

@dataclass
class OpticalExtras:
    ac_stark_coeff: float = 1e6
    ac_stark_coeff_e: float = 0.5e6
    polarization: str = "linear"
    epsilon_flip: float = 0.0

@dataclass
class SimConfig:
    nv: NVParams = field(default_factory=NVParams)
    opt: OpticalRates = field(default_factory=OpticalRates)
    pl:  PLRates = field(default_factory=PLRates)
    noise: NoiseParams = field(default_factory=NoiseParams)
    det: DetectorParams = field(default_factory=DetectorParams)
    seq: Sequence = field(default_factory=Sequence)
    init: InitialState = field(default_factory=InitialState)
    mw: MWParams = field(default_factory=MWParams)
    relax: RelaxParams = field(default_factory=RelaxParams)
    charge: ChargeParams = field(default_factory=ChargeParams)
    oxt: OpticalExtras = field(default_factory=OpticalExtras)
    seed: int = 7
    progress: bool = True
    progress_bins_step: int = 5
    positivity_clamp: bool = True
    clamp_epsilon: float = 1e-12
    clamp_every: int = 10  # Nur alle N Schritte clampen für Performance

# ============================
# AWGController - vollständige Version
# ============================

@dataclass
class AWGSpecs:
    channels: int = 2                 # AWG70002A: 2 Kanäle; AWG70001A: 1 Kanal
    max_sample_rate: float = 50e9     # 50 GS/s max (1Ch), 25 GS/s bei 2Ch typ.
    dac_bits: int = 10
    rf_max_carrier: float = 20e9      # Direkt-RF bis ~20 GHz (modelliert)
    mem_points_max: int = 16_000_000  # konservativ (anpassbar)
    min_points: int = 2400            # mind. Wellenformlänge
    granularity: int = 1              # Granularität (für Trigger-Modi ggf. 2)
    markers_per_ch: int = 2

@dataclass
class TriggerCfg:
    threshold_V: float = 0.0
    polarity: str = "pos"             # "pos" | "neg"
    holdoff_s: float = 0.0
    min_pulse_s: float = 20e-9

@dataclass
class MarkerCfg:
    delay_ps: float = 0.0
    intra_skew_ps: float = 0.0
    inter_skew_ps: float = 0.0
    jitter_rms_ps: float = 0.4        # statisches Jittermodell

@dataclass
class ClockCfg:
    ref_source: str = "internal10MHz" # "external10MHz"
    ext_ref_freq: float = 10e6
    dac_clock_source: str = "internal" # "external"
    ext_clock_freq: float = 12.5e9
    skew_trim_ps: float = 0.0

@dataclass
class OutputCfg:
    units: str = "Vpp"    # "Vpp" | "dBm"
    level: float = 1.0     # 0.05..1.0 Vpp (modelliert)
    offset_V: float = 0.0
    z_out: float = 50.0
    ac_coupled: bool = False

class AWGController:
    """
    Arbitrary Waveform Generator Controller (AWG70000A-ähnlich)

    Features:
    - Modi: "direct_rf" (ein Kanal) | "iq_baseband" (I/Q auf 2 Kanälen)
    - Sample-Rate/Quantisierung (DAC Bits), Ausgangspegel/Einh. (Vpp/dBm)
    - Marker/Flags, Trigger-Modi & Sequencer (Waveform-Speicher & Steps)
    - Clock/Sync (Ref/Ext/Skew)
    - Pre-Emphasis (FIR), einfache Harmonics/SFDR-Modelle, Phase-Jitter
    - Bestehende NV-Hilfen: Drive-Hamiltonian/Detuning/Bloch-Siegert

    Typische Nutzung:
        awg = AWGController(mw, seq)
        awg.set_mode("direct_rf")
        awg.set_sample_rate(25e9)
        awg.configure_output(1, OutputCfg(units="Vpp", level=0.8))
        ch1, ch2, t = awg.synthesize(t0=0.0, T=seq.t_mw)
        # oder: Waveforms laden & Sequenz definieren
        awg.load_waveform("pi", ch1=ch1, ch2=ch2, markers={"M1":m1})
        awg.define_sequence([{"wfm":"pi", "repeat":1000}])
        Tseq, out = awg.run_sequence()
    """

    # ---------------- Lifecycle ----------------
    def __init__(self, mw_params, seq_params, specs: AWGSpecs = AWGSpecs()):
        self.mw = mw_params
        self.seq = seq_params
        self.specs = specs

        # Betriebsmodus & Sample-Rate
        self.mode: str = "direct_rf"      # "direct_rf" | "iq_baseband"
        self.sample_rate: float = min(25e9, self.specs.max_sample_rate)

        # Rauschzustände
        self.amp_noise_state = 0.0
        self.phase_noise_state = 0.0
        self.pulse_center = 0.5 * seq_params.t_mw
        self.pulse_sigma = max(seq_params.t_mw / 3.0, 1e-18)

        # Hardware-ähnliche Zustände
        self.clock_cfg = ClockCfg()
        self.output_cfg: Dict[int, OutputCfg] = {
            1: OutputCfg(), 2: OutputCfg()
        }
        self.marker_cfg: Dict[Tuple[int,int], MarkerCfg] = {}  # (ch, m) -> cfg
        self.trigger_A = TriggerCfg()
        self.trigger_B = TriggerCfg()

        # Waveform-Speicher & Sequencer
        self.memory: Dict[str, Dict] = {}     # name -> {"ch1","ch2","markers","fs","meta"}
        self.sequence: List[Dict] = []        # [{"wfm":str,"repeat":int}]
        self._run_mode: str = "continuous"    # "continuous"|"triggered"|"triggered_continuous"
        self._armed: bool = False

        # Pre-Emphasis / Fidelity
        self._fir: Optional[np.ndarray] = None
        self._harmonics: Optional[Tuple[float,float]] = None  # (k2_db, k3_db)
        self._rj_rms_s: float = 250e-15
        self._pn_profile: Optional[Callable[[np.ndarray], np.ndarray]] = None  # L(f)

    # ---------------- Grund-Setup ----------------
    def set_mode(self, mode: str):
        assert mode in ("direct_rf", "iq_baseband")
        self.mode = mode

    def set_sample_rate(self, fs_hz: float):
        if fs_hz <= 0 or fs_hz > self.specs.max_sample_rate:
            raise ValueError("Sample-Rate außerhalb Spezifikation.")
        self.sample_rate = fs_hz

    def set_frequency(self, freq_hz: float):
        if freq_hz < 0 or (self.mode == "direct_rf" and freq_hz > self.specs.rf_max_carrier):
            raise ValueError("Trägerfrequenz außerhalb Spezifikation.")
        self.mw.omega_mw = freq_hz

    def set_power(self, rabi_hz: float):
        self.mw.Omega0 = max(0.0, float(rabi_hz))

    def set_phase(self, phase_rad: float):
        self.mw.phase0 = float(phase_rad)

    def configure_output(self, ch: int, cfg: OutputCfg):
        if ch not in (1,2): raise ValueError("Kanal muss 1 oder 2 sein.")
        self.output_cfg[ch] = cfg

    def set_clock(self, cfg: ClockCfg):
        self.clock_cfg = cfg

    def set_sync_role(self, role: str):
        # "master"|"slave" - symbolisch; realer Sync via ClockCfg/Skew
        assert role in ("master", "slave")
        # keine weitere Logik nötig für die Simulation
        return

    # ---------------- Noise / Envelope ----------------
    def reset_noise_states(self):
        self.amp_noise_state = 0.0
        self.phase_noise_state = 0.0

    def update_noise_states(self, dt: float, key):
        key, key_amp, key_phase = jax.random.split(key, 3)
        self.amp_noise_state = _ou_step(self.amp_noise_state, dt, self.mw.amp_tau, self.mw.amp_sigma, key_amp)
        self.phase_noise_state = _ou_step(self.phase_noise_state, dt, self.mw.phi_tau, self.mw.phi_sigma, key_phase)
        return self.amp_noise_state, self.phase_noise_state, key

    def get_envelope(self, t: float, envelope_type: str = "gaussian") -> float:
        if t >= self.seq.t_mw: return 0.0
        et = envelope_type.lower()
        if et == "square":
            return self.seq.mw_amp
        if et == "gaussian":
            return self.seq.mw_amp * jnp.exp(-0.5*((t - self.pulse_center)/self.pulse_sigma)**2)
        if et == "blackman":
            alpha = 0.16
            a0 = (1 - alpha) / 2
            a1 = 0.5
            a2 = alpha / 2
            x = t / max(self.seq.t_mw, 1e-18)
            return self.seq.mw_amp * (a0 - a1*jnp.cos(2*jnp.pi*x) + a2*jnp.cos(4*jnp.pi*x))
        if et == "flattop":  # simple Flat-Top (Blackman-Harris Näherung)
            x = (t - self.pulse_center)/self.pulse_sigma
            win = 0.21557895 - 0.41663158*jnp.cos(np.pi*(x+1)) + 0.277263158*jnp.cos(2*np.pi*(x+1)) - 0.083578947*jnp.cos(3*np.pi*(x+1))
            return self.seq.mw_amp * jnp.clip(win, 0.0, 1.0)
        # Fallback
        return self.seq.mw_amp * jnp.exp(-0.5*((t - self.pulse_center)/self.pulse_sigma)**2)

    def get_mw_parameters(self, t: float, envelope_type: str = "gaussian") -> Tuple[float, float]:
        gate = 1.0 if t < self.seq.t_mw else 0.0
        env = self.get_envelope(t, envelope_type)
        Omega_with_noise = self.mw.Omega0 * jnp.exp(self.amp_noise_state)
        Omega_t = gate * Omega_with_noise * (env / max(self.seq.mw_amp, 1e-18))
        phi_t = self.mw.phase0 + self.phase_noise_state
        return float(Omega_t), float(phi_t)

    def get_iq_components(self, t: float, envelope_type: str = "gaussian") -> Tuple[float, float]:
        Omega_t, phi_t = self.get_mw_parameters(t, envelope_type)
        I = Omega_t * np.cos(phi_t)
        Q = Omega_t * np.sin(phi_t)
        return float(I), float(Q)

    # ---------------- NV-Drive-Hamiltonian & Detuning ----------------
    def get_drive_hamiltonian(self, P_from, P_to, Omega_t: float, phi_t: float):
        """JAX-safe drive Hamiltonian (ohne rotating frame correction)"""
        H_drive = 0.5 * Omega_t * (
            jnp.exp(1j * phi_t) * (P_to @ P_from) +
            jnp.exp(-1j * phi_t) * (P_from @ P_to)
        )
        return H_drive

    def calculate_detuning(self, H_eff, P_from, P_to) -> float:
        """Berechnet Detuning in Hz (DEPRECATED - use kernel version)"""
        E_from = jnp.real(jnp.trace(P_from @ H_eff))
        E_to   = jnp.real(jnp.trace(P_to   @ H_eff))
        delta  = (E_to - E_from) - self.mw.omega_mw  # Hz
        return float(delta)

    def get_detuning_corrections(self, delta: float, Omega_t: float, P_from, P_to):
        """DEPRECATED - wird durch JAX-safe Version in NVKernels ersetzt"""
        H_detuning = 0.5 * delta * (P_to - P_from)
        delta_bs = (Omega_t**2) / (4.0 * jnp.maximum(jnp.abs(delta), 1e-9))
        H_bs = 0.5 * delta_bs * (P_from - P_to)
        return H_detuning + H_bs

    # ---------------- Quantisierung & Fidelity ----------------
    def set_precompensation(self, fir_taps: np.ndarray):
        self._fir = np.array(fir_taps, dtype=float).copy()

    def set_phase_noise_profile(self, Lf: Callable[[np.ndarray], np.ndarray], rj_rms_s: float = 250e-15):
        """
        Lf: Funktion f->L(f) in dBc/Hz (optional, für spätere Erweiterungen).
        rj_rms_s: integrierter Random Jitter (RMS).
        """
        self._pn_profile = Lf
        self._rj_rms_s = float(rj_rms_s)

    def enable_harmonics(self, k2_db: float = -50.0, k3_db: float = -60.0):
        """Einfache 2./3. Oberwellen als Pegel in dBc."""
        self._harmonics = (float(k2_db), float(k3_db))

    def _apply_fir(self, x: np.ndarray) -> np.ndarray:
        if self._fir is None or len(self._fir) == 0:
            return x
        return np.convolve(x, self._fir, mode="same")

    def _apply_harmonics(self, x: np.ndarray, fc: float) -> np.ndarray:
        if self._harmonics is None: return x
        k2_db, k3_db = self._harmonics
        a2 = 10**(k2_db/20.0)
        a3 = 10**(k3_db/20.0)
        t = np.arange(len(x))/self.sample_rate
        # Für direct_rf: harmonics relativ zur Carrier
        # Für I/Q: wir harmonisieren die Basisbandhüllkurve (vereinfachtes Modell)
        base = x
        h2 = a2 * np.sin(2*2*np.pi*fc*t) if self.mode=="direct_rf" else a2 * base**2
        h3 = a3 * np.sin(3*2*np.pi*fc*t) if self.mode=="direct_rf" else a3 * base**3
        return base + h2 + h3

    def _quantize_and_scale(self, x: np.ndarray, ch: int) -> np.ndarray:
        # Pre-Emphasis / FIR
        x = self._apply_fir(x)

        # Normalisieren auf [-1, +1] (interner DAC-Vollausschlag)
        x = np.clip(x, -1.0, 1.0)

        # DAC-Quantisierung (mid-tread)
        levels = 2**self.specs.dac_bits
        q = np.round((x + 1.0) * (levels/2 - 1)) / (levels/2 - 1) - 1.0

        # Einheiten/Output (Vpp/dBm) – wir mappen [-1,1] -> Vpp (linear)
        cfg = self.output_cfg[ch]
        if cfg.units.lower() == "vpp":
            vpp = np.clip(cfg.level, 0.01, 1.5)   # konservatives Fenster
            # -1..+1 -> +/- vpp/2
            q = q * (vpp/2.0) + cfg.offset_V
            if cfg.ac_coupled:
                q = q - np.mean(q)                # einfache AC-Kopplung
        else:
            # dBm -> Spannung über 50 Ω: V_rms = sqrt(P*R), Vpp ~ 2*sqrt(2)*V_rms
            p_w = 1e-3 * (10**(cfg.level/10.0))
            v_rms = np.sqrt(p_w * cfg.z_out)
            vpp = 2*np.sqrt(2)*v_rms
            q = q * (vpp/2.0) + cfg.offset_V
            if cfg.ac_coupled:
                q = q - np.mean(q)

        return q.astype(np.float64, copy=False)

    # ---------------- Marker/Trigger ----------------
    def set_markers(self, ch: int, m: int, cfg: MarkerCfg):
        if ch not in (1,2) or m not in (1,2):
            raise ValueError("Marker: Kanal in {1,2}, Marker in {1,2}.")
        self.marker_cfg[(ch,m)] = cfg

    def set_trigger(self, A: Optional[TriggerCfg] = None, B: Optional[TriggerCfg] = None):
        if A is not None: self.trigger_A = A
        if B is not None: self.trigger_B = B

    def _apply_marker_timing(self, mk: np.ndarray, ch: int, m: int) -> np.ndarray:
        cfg = self.marker_cfg.get((ch,m), MarkerCfg())
        sps = self.sample_rate * 1e-12  # samples per ps
        dS = int(np.round(cfg.delay_ps * sps))
        mk2 = np.roll(mk, dS)
        # einfache Jitter-Approx: kleine binomiale Verwischung
        if cfg.jitter_rms_ps > 0:
            sigma_samp = cfg.jitter_rms_ps * 1e-12 * self.sample_rate
            W = max(3, int(6*sigma_samp))
            xs = np.arange(-W, W+1)
            kern = np.exp(-0.5*(xs/sigma_samp)**2)
            kern /= max(kern.sum(), 1e-12)
            mk2 = np.convolve(mk2.astype(float), kern, mode="same") > 0.5
            mk2 = mk2.astype(np.uint8)
        return mk2

    # ---------------- Waveform Speicher / Sequencer ----------------
    def _total_points(self) -> int:
        return sum(len(v["ch1"]) for v in self.memory.values())

    def load_waveform(self, name: str,
                      ch1: np.ndarray,
                      ch2: Optional[np.ndarray] = None,
                      markers: Optional[Dict[str, np.ndarray]] = None,
                      fs: Optional[float] = None,
                      meta: Optional[Dict] = None):
        """
        Speichert eine Wellenform (inkl. Marker) im internen Speicher.
        - Längen-/Granularitäts-Checks
        - Gesamtspeicher-Budget
        """
        if fs is None: fs = self.sample_rate
        # fs-Konsistenz prüfen
        if abs(fs - self.sample_rate) > 1e-6:
            raise ValueError(f"Waveform fs={fs:.3e} != AWG sample_rate={self.sample_rate:.3e} - bitte anpassen oder resampeln.")
        ch1 = np.asarray(ch1, dtype=float).copy()
        if ch2 is not None: ch2 = np.asarray(ch2, dtype=float).copy()

        N = len(ch1)
        if N < self.specs.min_points:
            raise ValueError("Waveform zu kurz.")
        if N % self.specs.granularity != 0:
            raise ValueError("Waveform verletzt Granularität.")
        if self._total_points() + N > self.specs.mem_points_max:
            raise MemoryError("AWG-Speicher erschöpft.")

        if markers is None: markers = {}
        # Marker (optional) auf Länge bringen
        mk_fixed = {}
        for k,v in markers.items():
            vv = np.asarray(v, dtype=np.uint8)
            if len(vv) != N:
                raise ValueError(f"Marker {k}: Länge != Waveform-Länge.")
            mk_fixed[k] = vv

        self.memory[name] = {
            "ch1": ch1,
            "ch2": ch2,
            "markers": mk_fixed,
            "fs": fs,
            "meta": (meta or {}).copy()
        }

    def define_sequence(self, steps: List[Dict]):
        """
        steps = [
          {"wfm":"pi_pulse", "repeat":1000},
          {"wfm":"readout",  "repeat":1}
        ]
        """
        for s in steps:
            if "wfm" not in s or s["wfm"] not in self.memory:
                raise KeyError(f"Waveform '{s.get('wfm','?')}' nicht im Speicher.")
            rep = int(s.get("repeat", 1))
            if rep <= 0: raise ValueError("repeat muss > 0 sein.")
        self.sequence = [dict(wfm=s["wfm"], repeat=int(s.get("repeat",1))) for s in steps]

    def run_mode(self, mode: str):
        mode = mode.lower()
        table = {
            "continuous": "continuous",
            "triggered": "triggered",
            "triggered_continuous": "triggered_continuous",
            "cont": "continuous",
            "trig": "triggered",
            "tcon": "triggered_continuous",
        }
        if mode not in table:
            raise ValueError("Unbekannter run_mode.")
        self._run_mode = table[mode]

    def arm(self):
        self._armed = True

    def trigger(self):
        if not self._armed and self._run_mode != "continuous":
            # im continuous-Mode läuft immer
            return False
        # Bei Trigger- und Triggered-Continuous lösen wir Sequenzstart aus
        self._armed = False
        return True

    def run_sequence(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Führt die Sequenz einmal (oder endlos im continuous) logisch aus und
        gibt die zusammengefügten Ausgänge zurück.

        Returns:
          T: Zeitachse (s)
          out: {"ch1":..., "ch2":..., "M1":..., "M2":..., ...}
        """
        if not self.sequence:
            raise RuntimeError("Keine Sequenz definiert.")
        # Trigger-Logik
        if self._run_mode in ("triggered", "triggered_continuous"):
            if not self.trigger():
                raise RuntimeError("Nicht getriggert/armed.")

        # Baue Output zusammen
        outs = {"ch1": [], "ch2": []}
        mkeys = set()
        for step in self.sequence:
            wf = self.memory[step["wfm"]]
            reps = step["repeat"]
            for _ in range(reps):
                outs["ch1"].append(wf["ch1"])
                outs["ch2"].append(wf["ch2"] if wf["ch2"] is not None else np.zeros_like(wf["ch1"]))
                mkeys.update(wf["markers"].keys())

        ch1 = np.concatenate(outs["ch1"]) if outs["ch1"] else np.array([])
        ch2 = np.concatenate(outs["ch2"]) if outs["ch2"] else np.array([])

        # Marker zusammenbauen
        out = {"ch1": self._quantize_and_scale(self._apply_harmonics(ch1, self.mw.omega_mw), 1),
               "ch2": self._quantize_and_scale(self._apply_harmonics(ch2, self.mw.omega_mw), 2)}

        # Clock-Skew auf ch2 anwenden
        skew = int(round(self.clock_cfg.skew_trim_ps * 1e-12 * self.sample_rate))
        if skew != 0 and out["ch2"] is not None:
            out["ch2"] = np.roll(out["ch2"], skew)

        for mk in sorted(mkeys):
            segs = []
            for step in self.sequence:
                wf = self.memory[step["wfm"]]
                vec = wf["markers"].get(mk, np.zeros_like(wf["ch1"], dtype=np.uint8))
                segs.extend([vec]*step["repeat"])
            m = np.concatenate(segs).astype(np.uint8)
            # Marker Timing auf Kanal 1, Marker 1/2 abbilden (Namenskonvention "M1","M2","M1B" etc.)
            mname = mk.upper()
            ch = 1 if "B" not in mname else 2
            mid = 1 if "1" in mname else 2
            m_adj = self._apply_marker_timing(m, ch=ch, m=mid)
            out[mname] = m_adj

        # Zeitachse (wir gehen von einheitlicher fs aus)
        fs = self.sample_rate
        N = len(out["ch1"])
        T = np.arange(N)/fs

        return T, out

    # ---------------- On-the-fly Synthese (ohne Speicher) ----------------
    def synthesize(self, t0: float, T: float, envelope_type: str = "gaussian"):
        """
        Erzeuge Samples (ohne in Memory zu speichern) aus der aktuellen MW-Definition.
        Gibt (ch1, ch2, t) zurück – bereits quantisiert + Output-Units.

        direct_rf:
            ch1 = RF-Samples, ch2=None
        iq_baseband:
            ch1 = I, ch2 = Q
        """
        fs = self.sample_rate
        N = int(np.round(T*fs))
        N = max(N, self.specs.min_points)
        if N % self.specs.granularity != 0:
            N = int(np.ceil(N/self.specs.granularity)*self.specs.granularity)

        t = t0 + np.arange(N)/fs

        # „Jitter": kleine zufällige Zeitabweichung (RJ) auf Phase
        if self._rj_rms_s > 0:
            dt_jit = np.random.normal(0.0, self._rj_rms_s, size=N)
        else:
            dt_jit = np.zeros(N)

        if self.mode == "direct_rf":
            # Envelope + Phase
            env = np.array([float(self.get_envelope(tt, envelope_type)) for tt in t], dtype=float)
            phi = 2*np.pi*self.mw.omega_mw*(t + dt_jit) + (self.mw.phase0 + float(self.phase_noise_state))
            x = env * np.cos(phi)
            x = self._apply_harmonics(x, self.mw.omega_mw)
            ch1 = self._quantize_and_scale(x, ch=1)
            return ch1, None, t
        else:
            env = np.array([float(self.get_envelope(tt, envelope_type)) for tt in t], dtype=float)
            ph = self.mw.phase0 + float(self.phase_noise_state)
            I = env * np.cos(ph)
            Q = env * np.sin(ph)
            ch1 = self._quantize_and_scale(I, ch=1)
            ch2 = self._quantize_and_scale(Q, ch=2)
            return ch1, ch2, t

    # ---------------- SCPI Mini-Parser ----------------
    def scpi(self, cmd: str):
        """
        Sehr kleine SCPI-Schicht (nur Kernbefehle):
          :SOUR:FREQ <Hz>
          :SOUR:VOLT <Vpp>         (Kanal 1)
          :SOUR:PHAS <rad>
          :AWG:SRAT <Sa/s>
          :AWG:MODE <DIRECT|IQ>
          :TRIG:MODE <CONT|TRIG|TCON>
          :RUN
          :TRIG
        """
        c = cmd.strip().upper()
        try:
            if c.startswith(":SOUR:FREQ"):
                v = float(c.split()[-1])
                self.set_frequency(v)
                return "OK"
            if c.startswith(":SOUR:VOLT"):
                v = float(c.split()[-1]); cfg = self.output_cfg[1]; cfg.level = v; self.configure_output(1, cfg); return "OK"
            if c.startswith(":SOUR:PHAS"):
                v = float(c.split()[-1]); self.set_phase(v); return "OK"
            if c.startswith(":AWG:SRAT"):
                v = float(c.split()[-1]); self.set_sample_rate(v); return "OK"
            if c.startswith(":AWG:MODE"):
                v = c.split()[-1]
                mode = {"DIRECT":"direct_rf","IQ":"iq_baseband"}[v]
                self.set_mode(mode); return "OK"
            if c.startswith(":TRIG:MODE"):
                v = c.split()[-1]
                self.run_mode({"CONT":"continuous","TRIG":"triggered","TCON":"triggered_continuous"}[v]); return "OK"
            if c.startswith(":RUN"):
                self.arm(); return "OK"
            if c.startswith(":TRIG"):
                self.trigger(); return "OK"
        except Exception as e:
            return f"ERR: {e}"
        return "ERR:UNKNOWN"

# ===========================================================
# NVSystem Klasse (unverändert)
# ===========================================================

class NVSystem:
    """
    Elektron S=1 ⊗ 14N I=1 ⊗ (explizit 13C I=1/2) mit angeregtem orbitalem Dublett.
    Manifolds: g:18, e:36, s1:6, s2:6, nv0:2  -> total 68
    """
    def __init__(self, nv: NVParams):
        self.nv = nv
        # Base ops
        Sx, Sy, Sz, I3s = spin1_ops()
        Nx, Ny, Nz, I3n = spin1_ops()
        Cx, Cy, Cz, I2c = spin_half_ops()
        tx, ty, tz, I2o = orbital_E_ops()

        # g-manifold (S ⊗ N ⊗ C)
        self.Sx_g = kron(kron(Sx, I3n), I2c)
        self.Sy_g = kron(kron(Sy, I3n), I2c)
        self.Sz_g = kron(kron(Sz, I3n), I2c)
        self.Nx_g = kron(kron(I3s, Nx), I2c)
        self.Ny_g = kron(kron(I3s, Ny), I2c)
        self.Nz_g = kron(kron(I3s, Nz), I2c)
        self.Cx_g = kron(kron(I3s, I3n), Cx)
        self.Cy_g = kron(kron(I3s, I3n), Cy)
        self.Cz_g = kron(kron(I3s, I3n), Cz)
        self.I_g  = kron(kron(I3s, I3n), I2c)

        # e-manifold (S ⊗ O ⊗ N ⊗ C)
        self.Sx_e = kron(kron(kron(Sx, I2o), I3n), I2c)
        self.Sy_e = kron(kron(kron(Sy, I2o), I3n), I2c)
        self.Sz_e = kron(kron(kron(Sz, I2o), I3n), I2c)
        self.Ox_e = kron(kron(kron(I3s, tx), I3n), I2c)
        self.Oy_e = kron(kron(kron(I3s, ty), I3n), I2c)
        self.Oz_e = kron(kron(kron(I3s, tz), I3n), I2c)
        self.Nx_e = kron(kron(kron(I3s, I2o), Nx), I2c)
        self.Ny_e = kron(kron(kron(I3s, I2o), Ny), I2c)
        self.Nz_e = kron(kron(kron(I3s, I2o), Nz), I2c)
        self.Cx_e = kron(kron(kron(I3s, I2o), I3n), Cx)
        self.Cy_e = kron(kron(kron(I3s, I2o), I3n), Cy)
        self.Cz_e = kron(kron(kron(I3s, I2o), I3n), Cz)
        self.I_e  = kron(kron(kron(I3s, I2o), I3n), I2c)

        # dimensions/offsets
        self.dim_g = 18; self.dim_e = 36; self.dim_s1 = 6; self.dim_s2 = 6; self.dim_nv0 = 2
        self.dim   = self.dim_g + self.dim_e + self.dim_s1 + self.dim_s2 + self.dim_nv0
        self.off_g = 0; self.off_e = self.dim_g; self.off_s1 = self.dim_g + self.dim_e
        self.off_s2 = self.off_s1 + self.dim_s1; self.off_nv0 = self.off_s2 + self.dim_s2

        # zero-block helper
        def Z(m,n): return jnp.zeros((m,n), dtype=jnp.complex64)
        Zgg = Z(self.dim_g, self.dim_g); Zge = Z(self.dim_g, self.dim_e); Zgs1 = Z(self.dim_g, self.dim_s1); Zgs2 = Z(self.dim_g, self.dim_s2); Zgn0 = Z(self.dim_g, self.dim_nv0)
        Zeg = Z(self.dim_e, self.dim_g); Zee = Z(self.dim_e, self.dim_e); Zes1 = Z(self.dim_e, self.dim_s1); Zes2 = Z(self.dim_e, self.dim_s2); Zen0 = Z(self.dim_e, self.dim_nv0)
        Zs1g = Z(self.dim_s1,self.dim_g); Zs1e=Z(self.dim_s1,self.dim_e); Zs1s1=Z(self.dim_s1,self.dim_s1); Zs1s2=Z(self.dim_s1,self.dim_s2); Zs1n0=Z(self.dim_s1,self.dim_nv0)
        Zs2g = Z(self.dim_s2,self.dim_g); Zs2e=Z(self.dim_s2,self.dim_e); Zs2s1=Z(self.dim_s2,self.dim_s1); Zs2s2=Z(self.dim_s2,self.dim_s2); Zs2n0=Z(self.dim_s2,self.dim_nv0)
        Zn0g = Z(self.dim_nv0,self.dim_g); Zn0e=Z(self.dim_nv0,self.dim_e); Zn0s1=Z(self.dim_nv0,self.dim_s1); Zn0s2=Z(self.dim_nv0,self.dim_s2); Zn0n0=Z(self.dim_nv0,self.dim_nv0)

        # projectors (full)
        self.Pg = jnp.block([[jnp.eye(self.dim_g, dtype=jnp.complex64), Zge, Zgs1, Zgs2, Zgn0],
                             [Zeg, Zee, Zes1, Zes2, Zen0],
                             [Zs1g, Zs1e, jnp.eye(self.dim_s1, dtype=jnp.complex64), Zs1s2, Zs1n0],
                             [Zs2g, Zs2e, Zs2s1, jnp.eye(self.dim_s2, dtype=jnp.complex64), Zs2n0],
                             [Zn0g, Zn0e, Zn0s1, Zn0s2, jnp.eye(self.dim_nv0, dtype=jnp.complex64)]])
        self.Pe = jnp.block([[jnp.zeros((self.dim_g, self.dim_g), dtype=jnp.complex64), Zge, Zgs1, Zgs2, Zgn0],
                             [Zeg, jnp.eye(self.dim_e, dtype=jnp.complex64), Zes1, Zes2, Zen0],
                             [Zs1g, Zs1e, Zs1s1, Zs1s2, Zs1n0],
                             [Zs2g, Zs2e, Zs2s1, Zs2s2, Zs2n0],
                             [Zn0g, Zn0e, Zn0s1, Zn0s2, Zn0n0]])
        self.Ps1 = jnp.block([[jnp.zeros((self.dim_g, self.dim_g), dtype=jnp.complex64), Zge, Zgs1, Zgs2, Zgn0],
                              [Zeg, Zee, Zes1, Zes2, Zen0],
                              [Zs1g, Zs1e, jnp.eye(self.dim_s1, dtype=jnp.complex64), Zs1s2, Zs1n0],
                              [Zs2g, Zs2e, Zs2s1, Zs2s2, Zs2n0],
                              [Zn0g, Zn0e, Zn0s1, Zn0s2, Zn0n0]])
        self.Ps2 = jnp.block([[jnp.zeros((self.dim_g, self.dim_g), dtype=jnp.complex64), Zge, Zgs1, Zgs2, Zgn0],
                              [Zeg, Zee, Zes1, Zes2, Zen0],
                              [Zs1g, Zs1e, Zs1s1, Zs1s2, Zs1n0],
                              [Zs2g, Zs2e, Zs2s1, jnp.eye(self.dim_s2, dtype=jnp.complex64), Zs2n0],
                              [Zn0g, Zn0e, Zn0s1, Zn0s2, Zn0n0]])
        self.Pnv0 = jnp.block([[jnp.zeros((self.dim_g, self.dim_g), dtype=jnp.complex64), Zge, Zgs1, Zgs2, Zgn0],
                               [Zeg, Zee, Zes1, Zes2, Zen0],
                               [Zs1g, Zs1e, Zs1s1, Zs1s2, Zs1n0],
                               [Zs2g, Zs2e, Zs2s1, Zs2s2, Zs2n0],
                               [Zn0g, Zn0e, Zn0s1, Zn0s2, jnp.eye(self.dim_nv0, dtype=jnp.complex64)]])

        # embed helpers for g/e Sz etc.
        def embed_g(M):
            return jnp.block([[M, Zge, Zgs1, Zgs2, Zgn0],
                              [Zeg, Zee, Zes1, Zes2, Zen0],
                              [Zs1g, Zs1e, Zs1s1, Zs1s2, Zs1n0],
                              [Zs2g, Zs2e, Zs2s1, Zs2s2, Zs2n0],
                              [Zn0g, Zn0e, Zn0s1, Zn0s2, Zn0n0]])
        def embed_e(M):
            return jnp.block([[jnp.zeros((self.dim_g, self.dim_g), dtype=jnp.complex64), Zge, Zgs1, Zgs2, Zgn0],
                              [Zeg, M,   Zes1, Zes2, Zen0],
                              [Zs1g, Zs1e, Zs1s1, Zs1s2, Zs1n0],
                              [Zs2g, Zs2e, Zs2s1, Zs2s2, Zs2n0],
                              [Zn0g, Zn0e, Zn0s1, Zn0s2, Zn0n0]])
        self.Sx_full_g = embed_g(self.Sx_g)
        self.Sy_full_g = embed_g(self.Sy_g)
        self.Sz_full_g = embed_g(self.Sz_g)
        self.Sx_full_e = embed_e(self.Sx_e)
        self.Sy_full_e = embed_e(self.Sy_e)
        self.Sz_full_e = embed_e(self.Sz_e)

        # Hamiltonians (statisch)
        self.Hg = self.build_Hg()
        self.He = self.build_He()
        Hs1 = jnp.zeros((self.dim_s1, self.dim_s1), dtype=jnp.complex64)
        Hs2 = jnp.zeros((self.dim_s2, self.dim_s2), dtype=jnp.complex64)
        Hnv0= jnp.zeros((self.dim_nv0, self.dim_nv0), dtype=jnp.complex64)

        self.H_static = jnp.block([
            [self.Hg,
             jnp.zeros((self.dim_g,  self.dim_e),  dtype=jnp.complex64),
             jnp.zeros((self.dim_g,  self.dim_s1), dtype=jnp.complex64),
             jnp.zeros((self.dim_g,  self.dim_s2), dtype=jnp.complex64),
             jnp.zeros((self.dim_g,  self.dim_nv0),dtype=jnp.complex64)],
            [jnp.zeros((self.dim_e,  self.dim_g),  dtype=jnp.complex64),
             self.He,
             jnp.zeros((self.dim_e,  self.dim_s1), dtype=jnp.complex64),
             jnp.zeros((self.dim_e,  self.dim_s2), dtype=jnp.complex64),
             jnp.zeros((self.dim_e,  self.dim_nv0),dtype=jnp.complex64)],
            [jnp.zeros((self.dim_s1,self.dim_g),   dtype=jnp.complex64),
             jnp.zeros((self.dim_s1,self.dim_e),   dtype=jnp.complex64),
             Hs1,
             jnp.zeros((self.dim_s1,self.dim_s2),  dtype=jnp.complex64),
             jnp.zeros((self.dim_s1,self.dim_nv0), dtype=jnp.complex64)],
            [jnp.zeros((self.dim_s2,self.dim_g),   dtype=jnp.complex64),
             jnp.zeros((self.dim_s2,self.dim_e),   dtype=jnp.complex64),
             jnp.zeros((self.dim_s2,self.dim_s1),  dtype=jnp.complex64),
             Hs2,
             jnp.zeros((self.dim_s2,self.dim_nv0), dtype=jnp.complex64)],
            [jnp.zeros((self.dim_nv0,self.dim_g),  dtype=jnp.complex64),
             jnp.zeros((self.dim_nv0,self.dim_e),  dtype=jnp.complex64),
             jnp.zeros((self.dim_nv0,self.dim_s1), dtype=jnp.complex64),
             jnp.zeros((self.dim_nv0,self.dim_s2), dtype=jnp.complex64),
             Hnv0],
        ])

        self.Szz_g_block = jnp.block([
            [
                self.Sz_g @ self.Sz_g,
                jnp.zeros((self.dim_g,  self.dim_e),  dtype=jnp.complex64),
                jnp.zeros((self.dim_g,  self.dim_s1), dtype=jnp.complex64),
                jnp.zeros((self.dim_g,  self.dim_s2), dtype=jnp.complex64),
                jnp.zeros((self.dim_g,  self.dim_nv0),dtype=jnp.complex64),
            ],
            [
                jnp.zeros((self.dim_e,  self.dim_g),  dtype=jnp.complex64),
                jnp.zeros((self.dim_e,  self.dim_e),  dtype=jnp.complex64),
                jnp.zeros((self.dim_e,  self.dim_s1), dtype=jnp.complex64),
                jnp.zeros((self.dim_e,  self.dim_s2), dtype=jnp.complex64),
                jnp.zeros((self.dim_e,  self.dim_nv0),dtype=jnp.complex64),
            ],
            [
                jnp.zeros((self.dim_s1,self.dim_g),   dtype=jnp.complex64),
                jnp.zeros((self.dim_s1,self.dim_e),   dtype=jnp.complex64),
                jnp.zeros((self.dim_s1,self.dim_s1),  dtype=jnp.complex64),
                jnp.zeros((self.dim_s1,self.dim_s2),  dtype=jnp.complex64),
                jnp.zeros((self.dim_s1,self.dim_nv0), dtype=jnp.complex64),
            ],
            [
                jnp.zeros((self.dim_s2,self.dim_g),   dtype=jnp.complex64),
                jnp.zeros((self.dim_s2,self.dim_e),   dtype=jnp.complex64),
                jnp.zeros((self.dim_s2,self.dim_s1),  dtype=jnp.complex64),
                jnp.zeros((self.dim_s2,self.dim_s2),  dtype=jnp.complex64),
                jnp.zeros((self.dim_s2,self.dim_nv0), dtype=jnp.complex64),
            ],
            [
                jnp.zeros((self.dim_nv0,self.dim_g),  dtype=jnp.complex64),
                jnp.zeros((self.dim_nv0,self.dim_e),  dtype=jnp.complex64),
                jnp.zeros((self.dim_nv0,self.dim_s1), dtype=jnp.complex64),
                jnp.zeros((self.dim_nv0,self.dim_s2), dtype=jnp.complex64),
                jnp.zeros((self.dim_nv0,self.dim_nv0),dtype=jnp.complex64),
            ],
        ])

        # ms-Projektoren (g)
        self.Nn, self.Nc = 3, 2
        idx_g_p, idx_g_0, idx_g_m = [], [], []
        for ms in range(3):
            for n in range(self.Nn):
                for c in range(self.Nc):
                    gi = self.idx_g(ms, n, c)
                    if ms == 0: idx_g_p.append(gi)
                    elif ms == 1: idx_g_0.append(gi)
                    else: idx_g_m.append(gi)
        self.Pg_p = proj_from_indices(idx_g_p, self.dim)
        self.Pg_0 = proj_from_indices(idx_g_0, self.dim)
        self.Pg_m = proj_from_indices(idx_g_m, self.dim)

        # Lindblad-Stacks
        self.build_L_operators()

    # --- Hamiltonians ---
    def build_Hg(self):
        nv = self.nv; Bx, By, Bz = nv.B; Ex, Ey, Ez = nv.E_field
        H  = nv.D0*(self.Sz_g @ self.Sz_g) + nv.E_g*((self.Sx_g@self.Sx_g) - (self.Sy_g@self.Sy_g))
        H += nv.gamma_e*(Bx*self.Sx_g + By*self.Sy_g + Bz*self.Sz_g)
        H += nv.A_par_N*(self.Sz_g @ self.Nz_g) + nv.A_perp_N*(self.Sx_g @ self.Nx_g + self.Sy_g @ self.Ny_g)
        if abs(nv.A_par_C) + abs(nv.A_perp_C) > 0:
            H += nv.A_par_C*(self.Sz_g @ self.Cz_g) + nv.A_perp_C*(self.Sx_g @ self.Cx_g + self.Sy_g @ self.Cy_g)
        H += nv.gamma_N*(Bx*self.Nx_g + By*self.Ny_g + Bz*self.Nz_g)
        H += nv.Q_N*(self.Nz_g @ self.Nz_g - (2/3)*jnp.eye(self.dim_g))
        H += nv.gamma_C*(Bx*self.Cx_g + By*self.Cy_g + Bz*self.Cz_g)
        # DC Stark (vector & transverse)
        H += nv.kE_g_par*Ez*(self.Sz_g@self.Sz_g) + nv.kE_g_perp*(Ex*((self.Sx_g@self.Sx_g)-(self.Sy_g@self.Sy_g)) + Ey*(self.Sx_g@self.Sy_g + self.Sy_g@self.Sx_g))
        return H.astype(jnp.complex64)

    def build_He(self):
        nv = self.nv; Bx, By, Bz = nv.B; Ex, Ey, Ez = nv.E_field
        # Spin-only ZFS + Zeeman + hyperfine (e)
        Hs  = nv.D_e*(self.Sz_e @ self.Sz_e) + nv.E_e*((self.Sx_e@self.Sx_e) - (self.Sy_e@self.Sy_e))
        Hs += nv.gamma_e*(Bx*self.Sx_e + By*self.Sy_e + Bz*self.Sz_e)
        Hs += nv.A_par_N_e*(self.Sz_e @ self.Nz_e) + nv.A_perp_N_e*(self.Sx_e @ self.Nx_e + self.Sy_e @ self.Ny_e)
        if abs(nv.A_par_C_e) + abs(nv.A_perp_C_e) > 0:
            Hs += nv.A_par_C_e*(self.Sz_e @ self.Cz_e) + nv.A_perp_C_e*(self.Sx_e @ self.Cx_e + self.Sy_e @ self.Cy_e)
        Hs += nv.gamma_N*(Bx*self.Nx_e + By*self.Ny_e + Bz*self.Nz_e)
        Hs += nv.Q_N*(self.Nz_e @ self.Nz_e - (2/3)*jnp.eye(self.dim_e))
        Hs += nv.gamma_C*(Bx*self.Cx_e + By*self.Cy_e + Bz*self.Cz_e)
        # Electric field in e
        Hs += nv.kE_e_par*Ez*(self.Sz_e@self.Sz_e) + nv.kE_e_perp*(Ex*((self.Sx_e@self.Sx_e)-(self.Sy_e@self.Sy_e)) + Ey*(self.Sx_e@self.Sy_e + self.Sy_e@self.Sx_e))
        # Fine-structure + orbital strain
        Hfs = nv.lambda_par*(self.Sz_e @ self.Oz_e) + nv.lambda_perp*(self.Sx_e @ self.Ox_e + self.Sy_e @ self.Oy_e)
        Hfs += nv.Delta_ss*((self.Sz_e@self.Sz_e) - (2/3)*self.I_e)
        Hfs += nv.Pi_x*self.Ox_e + nv.Pi_y*self.Oy_e
        return (Hs + Hfs).astype(jnp.complex64)

    # --- indices ---
    def idx_g(self, ms_idx, n_idx, c_idx):
        return self.off_g + (ms_idx*3*2 + n_idx*2 + c_idx)
    def idx_e(self, ms_idx, orb_idx, n_idx, c_idx):
        return self.off_e + ( ( (ms_idx*2 + orb_idx)*3 + n_idx)*2 + c_idx )
    def idx_s1(self, n_idx, c_idx):
        return self.off_s1 + (n_idx*2 + c_idx)
    def idx_s2(self, n_idx, c_idx):
        return self.off_s2 + (n_idx*2 + c_idx)
    def idx_nv0(self, b_or_d):
        return self.off_nv0 + (0 if b_or_d==0 else 1)
    def Eij(self, i, j):
        M = jnp.zeros((self.dim, self.dim), dtype=jnp.complex64)
        M = M.at[i, j].set(1.0+0.0j)
        return M

    # --- L-operators (incl. orbital-resolved excitation & ESLAC pumping) ---
    def build_L_operators(self):
        L_exc_ms0_ex, L_exc_ms0_ey = [], []
        L_exc_msp_ex, L_exc_msp_ey = [], []
        L_exc_msm_ex, L_exc_msm_ey = [], []
        L_rad = []
        L_es1_0, L_es1_pm = [], []
        L_s1s2, L_s2g0, L_s2gpm = [], [], []
        L_ion0, L_ionpm, L_iong, L_rec = [], [], [], []
        L_t1pm, L_t1back = [], []
        L_eslac = []
        # orbital phonon relaxation
        L_orb_ex2ey, L_orb_ey2ex = [], []
        for ms in range(3):
            for n in range(3):
                for c in range(2):
                    gi = self.idx_g(ms, n, c)
                    # excitation & radiative & ISC
                    for orb in (0,1):
                        ei = self.idx_e(ms, orb, n, c)
                        if ms == 1:
                            L_exc_ms0_ex.append(self.Eij(self.idx_e(ms,0,n,c), gi))
                            L_exc_ms0_ey.append(self.Eij(self.idx_e(ms,1,n,c), gi))
                        elif ms == 0:
                            L_exc_msp_ex.append(self.Eij(self.idx_e(ms,0,n,c), gi))
                            L_exc_msp_ey.append(self.Eij(self.idx_e(ms,1,n,c), gi))
                        else:
                            L_exc_msm_ex.append(self.Eij(self.idx_e(ms,0,n,c), gi))
                            L_exc_msm_ey.append(self.Eij(self.idx_e(ms,1,n,c), gi))
                        L_rad.append(self.Eij(gi, ei))
                        if ms == 1: L_es1_0.append(self.Eij(self.idx_s1(n,c), ei))
                        else:       L_es1_pm.append(self.Eij(self.idx_s1(n,c), ei))
                    # singlet cascade
                    L_s1s2.append(self.Eij(self.idx_s2(n,c), self.idx_s1(n,c)))
                    L_s2g0.append(self.Eij(self.idx_g(1,n,c), self.idx_s2(n,c)))
                    L_s2gpm.append(self.Eij(self.idx_g(0,n,c), self.idx_s2(n,c)))
                    L_s2gpm.append(self.Eij(self.idx_g(2,n,c), self.idx_s2(n,c)))
                    # charge
                    for orb in (0,1):
                        ei = self.idx_e(ms, orb, n, c)
                        if ms == 1: L_ion0.append(self.Eij(self.idx_nv0(0), ei))
                        else:       L_ionpm.append(self.Eij(self.idx_nv0(0), ei))
                    L_iong.append(self.Eij(self.idx_nv0(0), gi))
                    L_rec.append(self.Eij(self.idx_g(1,n,c), self.idx_nv0(0)))
                    if ms in (0,2): L_t1pm.append(self.Eij(self.idx_g(1,n,c), gi))
                    if ms == 1:
                        L_t1back.append(self.Eij(self.idx_g(0,n,c), gi))
                        L_t1back.append(self.Eij(self.idx_g(2,n,c), gi))
                    # orbital phonon relaxation operators
                    L_orb_ex2ey.append(self.Eij(self.idx_e(ms,1,n,c), self.idx_e(ms,0,n,c)))
                    L_orb_ey2ex.append(self.Eij(self.idx_e(ms,0,n,c), self.idx_e(ms,1,n,c)))
        # ESLAC ΔmI ohne Wrap-around
        for n in range(3):
            for c in range(2):
                for orb in (0,1):
                    if n < 2:
                        L_eslac.append(self.Eij(self.idx_e(1,orb,n+1,c), self.idx_e(0,orb,n,c)))
                    if n > 0:
                        L_eslac.append(self.Eij(self.idx_e(1,orb,n-1,c), self.idx_e(2,orb,n,c)))

        def stack(Ls):
            return jnp.stack(Ls) if len(Ls)>0 else jnp.zeros((0,self.dim,self.dim), dtype=jnp.complex64)
        def Ksum(Ls):
            if Ls.shape[0]==0:
                return jnp.zeros((self.dim,self.dim), dtype=jnp.complex64)
            return jnp.einsum('kji,kjm->im', Ls.conj(), Ls)
        # stacks
        self.L_exc_ms0_ex  = stack(L_exc_ms0_ex);  self.K_exc_ms0_ex  = Ksum(self.L_exc_ms0_ex)
        self.L_exc_ms0_ey  = stack(L_exc_ms0_ey);  self.K_exc_ms0_ey  = Ksum(self.L_exc_ms0_ey)
        self.L_exc_msp_ex  = stack(L_exc_msp_ex);  self.K_exc_msp_ex  = Ksum(self.L_exc_msp_ex)
        self.L_exc_msp_ey  = stack(L_exc_msp_ey);  self.K_exc_msp_ey  = Ksum(self.L_exc_msp_ey)
        self.L_exc_msm_ex  = stack(L_exc_msm_ex);  self.K_exc_msm_ex  = Ksum(self.L_exc_msm_ex)
        self.L_exc_msm_ey  = stack(L_exc_msm_ey);  self.K_exc_msm_ey  = Ksum(self.L_exc_msm_ey)
        self.L_rad         = stack(L_rad);         self.K_rad         = Ksum(self.L_rad)
        self.L_es1_0       = stack(L_es1_0);       self.K_es1_0       = Ksum(self.L_es1_0)
        self.L_es1_pm      = stack(L_es1_pm);      self.K_es1_pm      = Ksum(self.L_es1_pm)
        self.L_s1s2        = stack(L_s1s2);        self.K_s1s2        = Ksum(self.L_s1s2)
        self.L_s2g0        = stack(L_s2g0);        self.K_s2g0        = Ksum(self.L_s2g0)
        self.L_s2gpm       = stack(L_s2gpm);       self.K_s2gpm       = Ksum(self.L_s2gpm)
        self.L_ion0        = stack(L_ion0);        self.K_ion0        = Ksum(self.L_ion0)
        self.L_ionpm       = stack(L_ionpm);       self.K_ionpm       = Ksum(self.L_ionpm)
        self.L_iong        = stack(L_iong);        self.K_iong        = Ksum(self.L_iong)
        self.L_rec         = stack(L_rec);         self.K_rec         = Ksum(self.L_rec)
        self.L_t1pm        = stack(L_t1pm);        self.K_t1pm        = Ksum(self.L_t1pm)
        self.L_t1back      = stack(L_t1back);      self.K_t1back      = Ksum(self.L_t1back)
        self.L_eslac       = stack(L_eslac);       self.K_eslac       = Ksum(self.L_eslac)
        self.L_orb_ex2ey   = stack(L_orb_ex2ey);   self.K_orb_ex2ey   = Ksum(self.L_orb_ex2ey)
        self.L_orb_ey2ex   = stack(L_orb_ey2ex);   self.K_orb_ey2ex   = Ksum(self.L_orb_ey2ex)

# Alias für Rückwärtskompatibilität (falls alter Name verwendet wird)
NV_ges_system = NVSystem

# ===========================================================
# NVKernels Klasse (für neue AWG angepasst)
# ===========================================================

class NVKernels:
    """
    Baut aus NVSystem + SimConfig die JAX-Kerne.
    Version 3: Erweitert für neue AWGController Integration
    """
    def __init__(self, sys: NVSystem, cfg: SimConfig, awg: AWGController):
        self.sys = sys
        self.cfg = cfg
        self.awg = awg
        # prebind stepper
        self.rk4_step = self._make_stepper()

    def _make_stepper(self):
        sys = self.sys
        cfg = self.cfg
        awg = self.awg
        two_pi = 2.0*jnp.pi

        H_static      = jax.device_put(sys.H_static)
        Szz_g_block   = jax.device_put(sys.Szz_g_block)
        Sx_full_g     = jax.device_put(sys.Sx_full_g)
        Sy_full_g     = jax.device_put(sys.Sy_full_g)
        Sz_full_g     = jax.device_put(sys.Sz_full_g)
        Sx_full_e     = jax.device_put(sys.Sx_full_e)
        Sy_full_e     = jax.device_put(sys.Sy_full_e)
        Sz_full_e     = jax.device_put(sys.Sz_full_e)
        Ps1_full      = jax.device_put(sys.Ps1)
        Ps2_full      = jax.device_put(sys.Ps2)
        Pnv0_full     = jax.device_put(sys.Pnv0)

        # L stacks
        L_exc_ms0_ex, K_exc_ms0_ex = jax.device_put(sys.L_exc_ms0_ex), jax.device_put(sys.K_exc_ms0_ex)
        L_exc_ms0_ey, K_exc_ms0_ey = jax.device_put(sys.L_exc_ms0_ey), jax.device_put(sys.K_exc_ms0_ey)
        L_exc_msp_ex, K_exc_msp_ex = jax.device_put(sys.L_exc_msp_ex), jax.device_put(sys.K_exc_msp_ex)
        L_exc_msp_ey, K_exc_msp_ey = jax.device_put(sys.L_exc_msp_ey), jax.device_put(sys.K_exc_msp_ey)
        L_exc_msm_ex, K_exc_msm_ex = jax.device_put(sys.L_exc_msm_ex), jax.device_put(sys.K_exc_msm_ex)
        L_exc_msm_ey, K_exc_msm_ey = jax.device_put(sys.L_exc_msm_ey), jax.device_put(sys.K_exc_msm_ey)
        L_rad,      K_rad    = jax.device_put(sys.L_rad),      jax.device_put(sys.K_rad)
        L_es1_0,    K_es1_0  = jax.device_put(sys.L_es1_0),    jax.device_put(sys.K_es1_0)
        L_es1_pm,   K_es1_pm = jax.device_put(sys.L_es1_pm),   jax.device_put(sys.K_es1_pm)
        L_s1s2,     K_s1s2   = jax.device_put(sys.L_s1s2),     jax.device_put(sys.K_s1s2)
        L_s2g0,     K_s2g0   = jax.device_put(sys.L_s2g0),     jax.device_put(sys.K_s2g0)
        L_s2gpm,    K_s2gpm  = jax.device_put(sys.L_s2gpm),    jax.device_put(sys.K_s2gpm)
        L_ion0,     K_ion0   = jax.device_put(sys.L_ion0),     jax.device_put(sys.K_ion0)
        L_ionpm,    K_ionpm  = jax.device_put(sys.L_ionpm),    jax.device_put(sys.K_ionpm)
        L_iong,     K_iong   = jax.device_put(sys.L_iong),     jax.device_put(sys.K_iong)
        L_rec,      K_rec    = jax.device_put(sys.L_rec),      jax.device_put(sys.K_rec)
        L_t1pm,     K_t1pm   = jax.device_put(sys.L_t1pm),     jax.device_put(sys.K_t1pm)
        L_t1back,   K_t1back = jax.device_put(sys.L_t1back),   jax.device_put(sys.K_t1back)
        L_eslac,    K_eslac  = jax.device_put(sys.L_eslac),    jax.device_put(sys.K_eslac)
        L_orb_ex2ey,K_orb_ex2ey = jax.device_put(sys.L_orb_ex2ey), jax.device_put(sys.K_orb_ex2ey)
        L_orb_ey2ex,K_orb_ey2ex = jax.device_put(sys.L_orb_ey2ex), jax.device_put(sys.K_orb_ey2ex)

        gamma_phi_g = 1.0 / max(cfg.noise.T2star_g, 1e-12)
        gamma_phi_e = 1.0 / max(cfg.noise.T2star_e, 1e-12)
        gamma_phi_s = 1.0 / max(cfg.noise.T2star_s, 1e-12)
        gamma_phi_n = 1.0 / max(cfg.noise.T2star_nv0, 1e-12)

        gamma_cg = cfg.noise.gamma_c_bath_g
        gamma_ce = cfg.noise.gamma_c_bath_e

        # two-level projectors für Detuning (aus AWG)
        P_from = jax.device_put(proj_from_indices([
            sys.idx_g(awg.mw.target_ms_from, awg.mw.target_n_idx, awg.mw.target_c_idx)], sys.dim))
        P_to   = jax.device_put(proj_from_indices([
            sys.idx_g(awg.mw.target_ms_to,   awg.mw.target_n_idx, awg.mw.target_c_idx)], sys.dim))

        # KRITISCH: Übergangsoperatoren für Drive-Hamiltonian
        idx_from = sys.idx_g(awg.mw.target_ms_from, awg.mw.target_n_idx, awg.mw.target_c_idx)
        idx_to   = sys.idx_g(awg.mw.target_ms_to,   awg.mw.target_n_idx, awg.mw.target_c_idx)
        Sig_to_from  = jax.device_put(sys.Eij(idx_to,  idx_from))  # |to><from|
        Sig_from_to  = jax.device_put(sys.Eij(idx_from, idx_to))   # |from><to|

        # JAX-safe Detuning Functions (ersetzt AWG.calculate_detuning/get_detuning_corrections)
        def detuning_from_H(H_eff, P_from, P_to, f_mw_hz):
            E_from = jnp.real(jnp.trace(P_from @ H_eff))
            E_to   = jnp.real(jnp.trace(P_to   @ H_eff))
            return (E_to - E_from) - f_mw_hz  # Hz

        def detuning_corrections(delta_hz, Omega_hz, P_from, P_to):
            H_det = 0.5 * delta_hz * (P_to - P_from)
            delta_bs = (Omega_hz**2) / (4.0 * jnp.maximum(jnp.abs(delta_hz), 1e-9))
            H_bs  = 0.5 * delta_bs * (P_from - P_to)
            return H_det + H_bs

        # Polarisations-Gewichte (normalisiert)
        def pol_weights(polarization: str):
            if polarization == 'sigma+':
                return (1.0, 0.7, 1.3), (1.2, 0.8)
            if polarization == 'sigma-':
                return (1.0, 1.3, 0.7), (0.8, 1.2)
            return (1.0, 1.0, 1.0), (1.0, 1.0)
        (w0, wp, wm), (wo_ex, wo_ey) = pol_weights(cfg.oxt.polarization)
        s_ms  = w0 + wp + wm
        s_orb = wo_ex + wo_ey
        g0, gp, gm = w0/s_ms, wp/s_ms, wm/s_ms
        go_ex, go_ey = wo_ex/s_orb, wo_ey/s_orb

        # T1(T,B) Modell
        eV = 1.602176634e-19
        kB = 1.380649e-23
        def gamma_T1(TK, Bmag):
            g0 = 1.0/max(cfg.relax.T1_g0, 1e-12)
            gR = cfg.relax.a_Raman * (TK**5)
            gO = cfg.relax.b_Orbach * jnp.exp(-(cfg.relax.Delta_Orbach_eV*eV)/(kB*jnp.maximum(TK,1e-6)))
            gB = cfg.relax.d_B2 * (Bmag**2)
            return g0 + gR + gO + gB
        gamma_T1_back_ratio = cfg.relax.back_ratio if cfg.relax.T1_back_enabled else 0.0

        @jax.jit
        def drho(rho, D_shift, dBx, dBy, dBz, TK,
                 Omega_t, phi_t,
                 k_exc, k_rad, k_es1_0, k_es1_pm, k_s1s2, k_s2g0, k_s2gpm, k_orb,
                 k_ion0, k_ionpm, k_iong, k_rec, ac_delta_g, ac_delta_e):
            # dynamisches B
            Bx, By, Bz = cfg.nv.B
            Bx_eff = Bx + dBx; By_eff = By + dBy; Bz_eff = Bz + dBz
            Bmag_eff = jnp.sqrt(Bx_eff*Bx_eff + By_eff*By_eff + Bz_eff*Bz_eff)
            Bpar = Bz_eff

            # Hamiltonian
            H = H_static + D_shift * Szz_g_block

            # MW drive mit korrekten Übergangsoperatoren
            H_drive = 0.5 * Omega_t * (
                jnp.exp(1j * phi_t)  * Sig_to_from  +
                jnp.exp(-1j * phi_t) * Sig_from_to
            )
            H += H_drive

            # AC Stark auf g und e (ohne two_pi, da H bereits in Hz)
            H += ac_delta_g * (Sz_full_g @ Sz_full_g)
            H += ac_delta_e * (Sz_full_e @ Sz_full_e)

            # Detuning berechnen (JAX-safe)
            H_eff = H_static + D_shift*Szz_g_block
            delta = detuning_from_H(H_eff, P_from, P_to, awg.mw.omega_mw)

            # Detuning und Bloch-Siegert Korrekturen (JAX-safe)
            H += detuning_corrections(delta, Omega_t, P_from, P_to)

            # magnetisches Rauschen (g)
            H += cfg.nv.gamma_e*(dBx*Sx_full_g + dBy*Sy_full_g + dBz*Sz_full_g)

            # Unitärteil
            comm = H @ rho - rho @ H
            dr = -1j*two_pi*comm

            # Dephasing: rein + isotrope Bäder
            for A in (Sx_full_g, Sy_full_g, Sz_full_g):
                dr += dissipator(jnp.sqrt(gamma_phi_g/3.0 + gamma_cg/3.0) * A, rho)
            for A in (Sx_full_e, Sy_full_e, Sz_full_e):
                dr += dissipator(jnp.sqrt(gamma_phi_e/3.0 + gamma_ce/3.0) * A, rho)
            dr += dissipator(jnp.sqrt(gamma_phi_s) * Ps1_full,   rho)
            dr += dissipator(jnp.sqrt(gamma_phi_s) * Ps2_full,   rho)
            dr += dissipator(jnp.sqrt(gamma_phi_n) * Pnv0_full,  rho)

            # Optik & Charge mit normalisierten Gewichten
            dr += lindblad_group(rho, L_exc_ms0_ex, K_exc_ms0_ex, g0*go_ex*k_exc)
            dr += lindblad_group(rho, L_exc_ms0_ey, K_exc_ms0_ey, g0*go_ey*k_exc)
            dr += lindblad_group(rho, L_exc_msp_ex, K_exc_msp_ex, gp*go_ex*k_exc)
            dr += lindblad_group(rho, L_exc_msp_ey, K_exc_msp_ey, gp*go_ey*k_exc)
            dr += lindblad_group(rho, L_exc_msm_ex, K_exc_msm_ex, gm*go_ex*k_exc)
            dr += lindblad_group(rho, L_exc_msm_ey, K_exc_msm_ey, gm*go_ey*k_exc)
            dr += lindblad_group(rho, L_rad,     K_rad,     k_rad)
            dr += lindblad_group(rho, L_es1_0,   K_es1_0,   k_es1_0)
            dr += lindblad_group(rho, L_es1_pm,  K_es1_pm,  k_es1_pm)
            dr += lindblad_group(rho, L_s1s2,    K_s1s2,    k_s1s2)
            dr += lindblad_group(rho, L_s2g0,    K_s2g0,    k_s2g0)
            dr += lindblad_group(rho, L_s2gpm,   K_s2gpm,   k_s2gpm)
            dr += lindblad_group(rho, L_orb_ex2ey, K_orb_ex2ey, k_orb)
            dr += lindblad_group(rho, L_orb_ey2ex, K_orb_ey2ex, k_orb)
            dr += lindblad_group(rho, L_ion0,    K_ion0,    k_ion0)
            dr += lindblad_group(rho, L_ionpm,   K_ionpm,   k_ionpm)
            dr += lindblad_group(rho, L_iong,    K_iong,    k_iong)
            dr += lindblad_group(rho, L_rec,     K_rec,     k_rec)
            # ESLAC vs B_parallel und cosθ
            cos_th = jnp.where(Bmag_eff>0, jnp.abs(Bpar)/Bmag_eff, 0.0)
            k_esl = cfg.opt.k_eslac_max * jnp.exp(-((Bpar - cfg.opt.B_eslac_T)/cfg.opt.B_bw_T)**2) * (cos_th**2)
            dr += lindblad_group(rho, L_eslac,   K_eslac,   k_esl)
            # Ground-state T1(T,B)
            gT1 = gamma_T1(TK, Bmag_eff)
            dr += lindblad_group(rho, L_t1pm,   K_t1pm,   gT1)
            dr += lindblad_group(rho, L_t1back, K_t1back, gamma_T1_back_ratio*gT1)
            return dr

        @jax.jit
        def rk4_step(rho, D_shift, dBx, dBy, dBz, TK,
                     Omega_t, phi_t,
                     k_exc, k_rad, k_es1_0, k_es1_pm, k_s1s2, k_s2g0, k_s2gpm, k_orb,
                     k_ion0, k_ionpm, k_iong, k_rec, ac_delta_g, ac_delta_e, dt):
            k1 = drho(rho, D_shift, dBx, dBy, dBz, TK, Omega_t, phi_t,
                      k_exc, k_rad, k_es1_0, k_es1_pm, k_s1s2, k_s2g0, k_s2gpm, k_orb,
                      k_ion0, k_ionpm, k_iong, k_rec, ac_delta_g, ac_delta_e)
            k2 = drho(rho + 0.5*dt*k1, D_shift, dBx, dBy, dBz, TK, Omega_t, phi_t,
                      k_exc, k_rad, k_es1_0, k_es1_pm, k_s1s2, k_s2g0, k_s2gpm, k_orb,
                      k_ion0, k_ionpm, k_iong, k_rec, ac_delta_g, ac_delta_e)
            k3 = drho(rho + 0.5*dt*k2, D_shift, dBx, dBy, dBz, TK, Omega_t, phi_t,
                      k_exc, k_rad, k_es1_0, k_es1_pm, k_s1s2, k_s2g0, k_s2gpm, k_orb,
                      k_ion0, k_ionpm, k_iong, k_rec, ac_delta_g, ac_delta_e)
            k4 = drho(rho + dt*k3,     D_shift, dBx, dBy, dBz, TK, Omega_t, phi_t,
                      k_exc, k_rad, k_es1_0, k_es1_pm, k_s1s2, k_s2g0, k_s2gpm, k_orb,
                      k_ion0, k_ionpm, k_iong, k_rec, ac_delta_g, ac_delta_e)
            rho_new = rho + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            rho_new = 0.5*(rho_new + dagger(rho_new))
            rho_new = rho_new / jnp.trace(rho_new)
            if cfg.positivity_clamp:
                evals, evecs = jnp.linalg.eigh((rho_new + dagger(rho_new))*0.5)
                evals_clamped = jnp.maximum(evals, cfg.clamp_epsilon)
                evals_clamped = evals_clamped / jnp.sum(evals_clamped)
                rho_new = (evecs * evals_clamped) @ dagger(evecs)
            return rho_new

        return rk4_step

# ===========================================================
# NVSimulator Klasse (mit erweitertem AWG)
# ===========================================================

class NVSimulator:
    def __init__(self, cfg: SimConfig, awg_specs: AWGSpecs = AWGSpecs()):
        self.cfg = cfg
        self.sys = NVSystem(cfg.nv)
        self.awg = AWGController(cfg.mw, cfg.seq, awg_specs)  # Erweiterte AWG
        self.kernels = NVKernels(self.sys, cfg, self.awg)
        self.key = jax.random.PRNGKey(cfg.seed)
        self.counter = self._PhotonCounter(cfg.det)

    # --- interner Photonenzähler ---
    class _PhotonCounter:
        def __init__(self, dp: DetectorParams):
            self.dp = dp
        def _dead_time_rate(self, r_true):
            if self.dp.dead_time > 0:
                if self.dp.model == "paralyzable":
                    return r_true * jnp.exp(-r_true * self.dp.dead_time)
                else:
                    return r_true / (1.0 + r_true * self.dp.dead_time)
            return r_true
        def draw_bin_counts(self, mu_true_per_shot, bin_w, key):
            r_true = mu_true_per_shot / bin_w
            r_obs  = self._dead_time_rate(r_true)
            mu_obs_per_shot = r_obs * bin_w + self.dp.dark_rate * bin_w
            mu_total = mu_obs_per_shot * self.dp.shots_accum
            if self.dp.sample:
                if self.dp.overdispersion_k and self.dp.overdispersion_k > 0.0:
                    key, gk, pk = jax.random.split(key, 3)
                    kshape = jnp.float32(self.dp.overdispersion_k)
                    lam_gamma = jax.random.gamma(gk, kshape) * (jnp.maximum(mu_total, 0.0) / kshape)
                    k = float(jax.random.poisson(pk, lam_gamma))
                else:
                    key, sub = jax.random.split(key)
                    k = float(jax.random.poisson(sub, jnp.maximum(mu_total, 0.0)))
            else:
                k = float(mu_total)
            k = min(k, float(self.dp.saturation_level*bin_w))
            return k, key

    # --- gemischter Anfangszustand im g-Manifold ---
    def _mixed_g_state(self):
        rho = jnp.zeros((self.sys.dim, self.sys.dim), dtype=jnp.complex64)
        for p, ms_idx in [(self.cfg.init.p0,1),(self.cfg.init.pp,0),(self.cfg.init.pm,2)]:
            gi = self.sys.idx_g(ms_idx=ms_idx, n_idx=self.cfg.init.n_idx, c_idx=self.cfg.init.c_idx)
            rho = rho.at[gi, gi].add(p)
        return rho

    # --- Hauptsimulation mit erweitertem AWG ---
    def simulate(self, envelope_type: str = "gaussian"):
        cfg = self.cfg
        sys = self.sys
        awg = self.awg
        rk4_step = self.kernels.rk4_step
        key = self.key

        dt = 1e-9 / cfg.seq.oversample_per_ns
        T  = cfg.seq.t_mw + cfg.seq.t_readout
        N  = int(jnp.ceil(T / dt))

        rho = self._mixed_g_state()

        # Noise states (ohne MW-Noise, das ist jetzt im AWG)
        x_sd  = 0.0
        rin_state = 0.0
        M = max(1, int(cfg.noise.pink_terms))
        taus = np.logspace(np.log10(cfg.noise.pink_tau_min), np.log10(cfg.noise.pink_tau_max), M)
        wts  = 1.0/np.sqrt(np.arange(1,M+1))
        dBx_terms = [0.0]*M; dBy_terms=[0.0]*M; dBz_terms=[0.0]*M
        T_state = 0.0

        # AWG noise states reset
        awg.reset_noise_states()

        bin_w = cfg.seq.bin_ns * 1e-9
        nbins = int(np.floor(T / bin_w))
        bins_edges = np.linspace(0.0, nbins*bin_w, nbins+1)
        counts_bins = np.zeros(nbins, dtype=float)
        afterpulse_pending = np.zeros(nbins + 4096, dtype=float)

        t0 = time.time()
        report_every = max(1, int(nbins * cfg.progress_bins_step / 100.0))

        eV = 1.602176634e-19
        kB = 1.380649e-23

        t = 0.0
        next_edge_idx = 1
        lam_int_this_bin = 0.0

        for _ in range(N):
            # Laserintensität & optische Raten
            I = cfg.seq.I_of_t(t)
            sI = I / max(cfg.opt.I_sat, 1e-12)
            k_exc = cfg.opt.k_exc_max * sI / (1.0 + sI)
            k_rad = cfg.opt.gamma_rad
            k_es1_0  = cfg.opt.k_es_to_s1_0
            k_es1_pm = cfg.opt.k_es_to_s1_pm

            # Temperatur-Drift (OU) und temp.-abhängige Raten
            key, sT = jax.random.split(key)
            T_state = ou_step(T_state, dt, cfg.noise.T_drift_tau, cfg.noise.T_drift_sigma, sT)
            T_inst = cfg.noise.T0 + T_state
            def temp_rate(k0, Ea_eV, alpha):
                return jnp.where(Ea_eV>0,
                                 k0 * jnp.exp(-(Ea_eV*eV)/(kB*jnp.maximum(T_inst,1e-6))),
                                 k0 * (1.0 + alpha*(T_inst - cfg.opt.T0)))
            k_s1s2  = temp_rate(cfg.opt.k_s1_to_s2, cfg.opt.Ea_s1s2_eV, cfg.opt.alpha_s1s2)
            k_s2g0  = temp_rate(cfg.opt.k_s2_to_g0, cfg.opt.Ea_s2g0_eV, cfg.opt.alpha_s2g0)
            k_s2gpm = temp_rate(cfg.opt.k_s2_to_gpm, cfg.opt.Ea_s2gpm_eV, cfg.opt.alpha_s2gpm)

            # Orbitalphonon-Relaxation (einfaches T-Scaling)
            k_orb = cfg.opt.k_orb_base * (T_inst/cfg.opt.T0)**cfg.opt.alpha_orb

            # Charge-Dynamik
            k_ion0  = (cfg.charge.k1_ion_0 * sI + cfg.charge.k2_ion_0 * (sI**2))
            k_ionpm = (cfg.charge.k1_ion_pm* sI + cfg.charge.k2_ion_pm* (sI**2))
            k_iong  = cfg.charge.k1_ion_g * sI
            k_rec   = cfg.charge.k_rec * (1.0 + cfg.charge.repump_gain)

            # AC-Stark
            ac_delta_g = cfg.oxt.ac_stark_coeff   * I
            ac_delta_e = cfg.oxt.ac_stark_coeff_e * I

            # OU spektrale Diffusion + D(T)
            key, s1 = jax.random.split(key)
            x_sd = ou_step(x_sd, dt, cfg.noise.sd_tau, cfg.noise.sd_sigma, s1)
            D_shift = (cfg.nv.D0 + x_sd + cfg.noise.dD_dT*(T_inst-cfg.noise.T0)) - cfg.nv.D0

            # magnetisches OU + pink noise (dBx,dBy,dBz)
            dBx = 0.0; dBy = 0.0; dBz = 0.0
            for i in range(M):
                key, kx, ky, kz = jax.random.split(key, 4)
                dBx_terms[i] = ou_step(dBx_terms[i], dt, taus[i], cfg.noise.mag_sigma*wts[i], kx)
                dBy_terms[i] = ou_step(dBy_terms[i], dt, taus[i], cfg.noise.mag_sigma*wts[i], ky)
                dBz_terms[i] = ou_step(dBz_terms[i], dt, taus[i], cfg.noise.mag_sigma*wts[i], kz)
                dBx += dBx_terms[i]; dBy += dBy_terms[i]; dBz += dBz_terms[i]

            # RIN
            if cfg.noise.rin_enable:
                key, s6 = jax.random.split(key)
                rin_state = ou_step(rin_state, dt, cfg.noise.rin_tau, cfg.noise.rin_sigma, s6)

            # ERWEITERT: MW-Parameter vom erweiterten AWG holen
            # Update AWG noise states
            _, _, key = awg.update_noise_states(dt, key)
            # Get MW parameters from AWG (mit konfigurierbarem envelope_type)
            Omega_t, phi_t = awg.get_mw_parameters(t, envelope_type)

            # DGL-Schritt
            TK = T_inst
            rho = rk4_step(rho, D_shift, dBx, dBy, dBz, TK,
                           Omega_t, phi_t,
                           k_exc, k_rad, k_es1_0, k_es1_pm, k_s1s2, k_s2g0, k_s2gpm, k_orb,
                           k_ion0, k_ionpm, k_iong, k_rec, ac_delta_g, ac_delta_e, dt)

            # Photonrate (ZPL/PSB-Filter, optional Prompt, RIN)
            lam_t = 0.0
            if I > 0:
                zpsb = cfg.pl.zpl_fraction*cfg.pl.f_zpl + cfg.pl.psb_fraction*cfg.pl.f_psb
                if cfg.pl.model == "spin_brightness":
                    p0  = float(jnp.real(jnp.trace(sys.Pg_0 @ rho)))
                    pp  = float(jnp.real(jnp.trace(sys.Pg_p @ rho)))
                    pm  = float(jnp.real(jnp.trace(sys.Pg_m @ rho)))
                    ps1 = float(jnp.real(jnp.trace(sys.Ps1  @ rho)))
                    ps2 = float(jnp.real(jnp.trace(sys.Ps2  @ rho)))
                    pnv0= float(jnp.real(jnp.trace(sys.Pnv0 @ rho)))
                    lam_minus = cfg.pl.r0 * p0 + cfg.pl.rpm * (pp + pm)
                    lam_minus *= max(1.0 - (ps1 + ps2), 0.0)
                    lam_per_nv = (1.0 - pnv0)*lam_minus + pnv0*(cfg.pl.r_nv0)
                    base_scale = (cfg.det.n_nv if cfg.pl.measured_at_detector else cfg.det.eta * cfg.det.n_nv)
                    base_rate = base_scale * zpsb * lam_per_nv
                else:
                    Pe_val = float(jnp.real(jnp.trace(sys.Pe @ rho)))
                    pnv0   = float(jnp.real(jnp.trace(sys.Pnv0 @ rho)))
                    base_scale = (cfg.det.n_nv if cfg.pl.measured_at_detector else cfg.det.eta * cfg.det.n_nv)
                    base_rate = base_scale * zpsb * (cfg.opt.gamma_rad * Pe_val * (1.0 - pnv0) + cfg.pl.r_nv0 * pnv0)

                if cfg.pl.prompt_amp > 0.0 and t >= cfg.seq.t_mw:
                    base_rate += cfg.det.n_nv * cfg.pl.prompt_amp * np.exp(-(t - cfg.seq.t_mw)/max(cfg.pl.prompt_tau, 1e-12))
                if cfg.noise.rin_enable:
                    base_rate *= float(jnp.exp(rin_state))
                lam_t = base_rate

            lam_int_this_bin += lam_t * dt

            # Binning & Afterpulsing
            t_next = t + dt
            while next_edge_idx < len(bins_edges) and t_next > bins_edges[next_edge_idx]:
                k_here, key = self.counter.draw_bin_counts(lam_int_this_bin, bin_w, key)
                counts_here_idx = next_edge_idx - 1

                if cfg.det.afterpulse_p > 0.0 and k_here > 0:
                    k_prev = int(np.ceil(k_here))
                    for _gen in range(max(1, int(cfg.det.afterpulse_generations))):
                        if k_prev <= 0:
                            break
                        key, sub1 = jax.random.split(key)
                        k_gen = int(jax.random.binomial(sub1, n=k_prev, p=cfg.det.afterpulse_p))
                        if k_gen <= 0:
                            k_prev = 0
                            break
                        if cfg.det.afterpulse_mode == "mixture":
                            key, s2, s3, s4 = jax.random.split(key, 4)
                            comp = jax.random.bernoulli(s2, cfg.det.afterpulse_w1, (k_gen,))
                            delays1 = jax.random.exponential(s3, (k_gen,)) * cfg.det.afterpulse_tau1
                            delays2 = jax.random.exponential(s4, (k_gen,)) * cfg.det.afterpulse_tau2
                            delays  = jnp.where(comp, delays1, delays2)
                        elif cfg.det.afterpulse_tail_tau > 0.0:
                            key, s2 = jax.random.split(key)
                            delays = jax.random.exponential(s2, (k_gen,)) * cfg.det.afterpulse_tail_tau
                        else:
                            delays = jnp.ones((k_gen,)) * cfg.det.afterpulse_delay
                        for d in np.array(delays):
                            tgt = counts_here_idx + int(np.round(float(d) / bin_w))
                            if tgt < len(afterpulse_pending):
                                afterpulse_pending[tgt] += 1.0
                        k_prev = k_gen

                counts_bins[counts_here_idx] = k_here + afterpulse_pending[counts_here_idx]
                afterpulse_pending[counts_here_idx] = 0.0

                if cfg.progress and ((counts_here_idx+1) % report_every == 0 or counts_here_idx+1 == nbins):
                    done = counts_here_idx + 1
                    pct  = 100.0 * done / nbins if nbins > 0 else 100.0
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1e-9)
                    eta  = (nbins - done) / max(rate, 1e-9)
                    print(f"[{done:>4}/{nbins}] {pct:5.1f}%  ETA {eta:5.2f}s  (elapsed {elapsed:5.2f}s)")

                lam_int_this_bin = 0.0
                t = bins_edges[next_edge_idx]
                next_edge_idx += 1

            t = t_next

        if next_edge_idx <= nbins:
            k_last, key = self.counter.draw_bin_counts(lam_int_this_bin, bin_w, key)
            last_idx = next_edge_idx - 1
            counts_bins[last_idx] = k_last + afterpulse_pending[last_idx]

        # zeitliches Jitter
        if cfg.det.jitter_sigma > 0 and nbins > 3:
            sigma_bins = cfg.det.jitter_sigma / bin_w
            W = int(6*max(1,int(np.ceil(sigma_bins))))
            xs = np.arange(-W, W+1)
            kern = np.exp(-0.5*(xs/sigma_bins)**2)
            kern /= kern.sum()
            counts_bins = np.convolve(counts_bins, kern, mode='same')

        times_ns = 1e9 * 0.5 * (bins_edges[:-1] + bins_edges[1:])

        # PRNG-Status zurückspeichern
        self.key = key
        return times_ns, counts_bins

    # --- Erweiterte AWG-Funktionen ---
    def setup_awg_advanced(self):
        """Setup erweiterte AWG Features"""
        # Pre-Emphasis setzen
        fir_taps = np.array([0.1, 0.8, 0.1])  # Simple Pre-Emphasis
        self.awg.set_precompensation(fir_taps)

        # Harmonics aktivieren
        self.awg.enable_harmonics(k2_db=-50.0, k3_db=-60.0)

        # Phase Noise Profil setzen
        self.awg.set_phase_noise_profile(None, rj_rms_s=250e-15)

        # Output konfigurieren
        self.awg.configure_output(1, OutputCfg(units="Vpp", level=0.8, offset_V=0.0))
        self.awg.configure_output(2, OutputCfg(units="Vpp", level=0.8, offset_V=0.0))

    def demo_waveform_sequence(self):
        """Demonstriert Waveform-Speicher und Sequencer"""
        # Pi-Pulse Waveform erzeugen
        pi_duration = 50e-9
        fs = self.awg.sample_rate
        N_pi = int(pi_duration * fs)
        t_pi = np.arange(N_pi) / fs

        # Gaussian pi-pulse
        sigma = pi_duration / 4
        tc = pi_duration / 2
        env = np.exp(-0.5 * ((t_pi - tc) / sigma)**2)

        if self.awg.mode == "direct_rf":
            # Direct RF: Modulated carrier
            fc = self.awg.mw.omega_mw
            pi_waveform = env * np.cos(2*np.pi*fc*t_pi)
            self.awg.load_waveform("pi_pulse", ch1=pi_waveform)
        else:
            # I/Q Mode
            I = env * np.cos(self.awg.mw.phase0)
            Q = env * np.sin(self.awg.mw.phase0)
            self.awg.load_waveform("pi_pulse", ch1=I, ch2=Q)

        # Readout delay
        delay_duration = 100e-9
        N_delay = int(delay_duration * fs)
        delay_waveform = np.zeros(N_delay)
        self.awg.load_waveform("delay", ch1=delay_waveform, ch2=delay_waveform if self.awg.mode == "iq_baseband" else None)

        # Sequenz definieren
        self.awg.define_sequence([
            {"wfm": "pi_pulse", "repeat": 1},
            {"wfm": "delay", "repeat": 1}
        ])

        # Sequenz ausführen
        self.awg.run_mode("continuous")
        self.awg.arm()
        T_seq, waveforms = self.awg.run_sequence()

        return T_seq, waveforms

# ------------------------- Demo/Anwendungsbeispiel -------------------------

def demo_basic():
    """Exaktes V1 Experiment mit AWG Controller"""
    cfg = SimConfig()

    # Exakte V1 Parameter
    cfg.seq.bin_ns = 1.0
    cfg.seq.oversample_per_ns = 30
    cfg.det.shots_accum = 12000
    cfg.det.model = "nonparalyzable"
    cfg.det.dead_time = 60e-9

    cfg.pl.prompt_amp = 0.2e8
    cfg.pl.prompt_tau = 40e-9

    cfg.det.afterpulse_p = 0.01
    cfg.det.afterpulse_mode = "mixture"
    cfg.det.afterpulse_tau1 = 40e-9
    cfg.det.afterpulse_tau2 = 200e-9
    cfg.det.afterpulse_w1  = 0.7
    cfg.det.afterpulse_generations = 2

    cfg.noise.rin_enable = True
    cfg.noise.rin_tau   = 10e-9
    cfg.noise.rin_sigma = 0.12
    cfg.det.overdispersion_k = 300.0

    cfg.pl.measured_at_detector = True
    cfg.seq.laser_tau = 20e-9

    cfg.mw.Omega0 = 8e6
    cfg.mw.omega_mw = 2.87e9
    cfg.seq.t_mw = 60e-9
    cfg.seq.mw_amp = 1.0

    cfg.nv.lambda_par = 5.3e9
    cfg.nv.lambda_perp= 0.3e9
    cfg.nv.Delta_ss   = 1.4e9
    cfg.nv.Pi_x       = 50e6
    cfg.nv.Pi_y       = 0.0

    cfg.opt.k_eslac_max = 1.5e6

    cfg.relax.T1_g0 = 2e-3
    cfg.relax.a_Raman = 2e-3
    cfg.relax.b_Orbach= 5e7
    cfg.relax.Delta_Orbach_eV = 0.07
    cfg.relax.d_B2 = 2e3

    # AWG mit genügend Sample Rate für 60ns pulse
    awg_specs = AWGSpecs(
        channels=2,
        max_sample_rate=50e9,
        dac_bits=10,
        rf_max_carrier=20e9
    )

    sim = NVSimulator(cfg, awg_specs)
    t_ns, counts = sim.simulate()
    print(f"Total counts (accumulated): {counts.sum():.0f}")

    plt.figure(figsize=(10,4))
    plt.step(t_ns, counts, where="mid")
    plt.xlabel("Zeit (ns)")
    plt.ylabel(f"Counts pro {cfg.seq.bin_ns:.0f} ns")
    plt.title("NV: FS+ESLAC fixes, norm. excitation, orbital relax, AC Stark (g,e), isotropic bath, ZPL/PSB filter")
    plt.tight_layout()
    plt.show()

def demo_advanced_awg():
    """Erweiterte AWG-Demo mit Sequencer und SCPI"""
    print("=== NV Simulator V3 - Erweiterte AWG Demo ===")
    cfg = SimConfig()
    cfg.seq.t_mw = 50e-9
    cfg.seq.t_readout = 200e-9
    cfg.progress = False

    # High-end AWG Spezifikationen
    awg_specs = AWGSpecs(
        channels=2,
        max_sample_rate=50e9,
        dac_bits=12,
        rf_max_carrier=20e9,
        mem_points_max=32_000_000
    )

    sim = NVSimulator(cfg, awg_specs)

    # Erweiterte AWG-Features testen
    print("1. AWG Mode: I/Q Baseband")
    sim.awg.set_mode("iq_baseband")
    sim.awg.set_sample_rate(50e9)

    print("2. SCPI-Kommandos:")
    commands = [
        ":AWG:MODE IQ",
        ":SOUR:FREQ 2870000000",
        ":SOUR:VOLT 0.5",
        ":AWG:SRAT 50000000000",  # 50 GS/s für 50ns pulse @ 2500 points
        ":TRIG:MODE TRIG"
    ]

    for cmd in commands:
        result = sim.awg.scpi(cmd)
        print(f"   {cmd} -> {result}")

    print("\n3. Erweiterte Features aktivieren:")
    sim.setup_awg_advanced()
    print("   - Pre-Emphasis FIR aktiviert")
    print("   - Harmonics: 2nd=-50dBc, 3rd=-60dBc")
    print("   - Phase Noise: RJ=250fs RMS")

    print("\n4. Waveform-Sequencer Demo:")
    T_seq, waveforms = sim.demo_waveform_sequence()
    print(f"   Sequenz-Länge: {len(T_seq)} samples")
    print(f"   Sequenz-Dauer: {T_seq[-1]*1e9:.1f} ns")
    print(f"   Kanäle: {list(waveforms.keys())}")

    print("\n5. On-the-fly Synthese:")
    ch1, ch2, t = sim.awg.synthesize(0.0, cfg.seq.t_mw, envelope_type="gaussian")
    print(f"   Synthesized: {len(t)} samples, {t[-1]*1e9:.1f} ns")

    # I/Q Komponenten anzeigen
    if sim.awg.mode == "iq_baseband" and ch2 is not None:
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(t*1e9, ch1, 'b-', label='I-Kanal')
        plt.ylabel('Amplitude (V)')
        plt.title('AWG I/Q Output')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(t*1e9, ch2, 'r-', label='Q-Kanal')
        plt.ylabel('Amplitude (V)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(t*1e9, np.sqrt(ch1**2 + ch2**2), 'g-', label='Magnitude')
        plt.xlabel('Zeit (ns)')
        plt.ylabel('Magnitude (V)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.show()

    print("==========================================")
    return sim

def demo_full_comparison():
    """Vollständiger Vergleich aller Features"""
    print("=== NV Simulator V3 - Vollständiger Feature-Test ===")

    # Basis-Simulation
    print("\n1. Basis-Simulation:")
    results_basic = demo_basic()

    # Erweiterte AWG
    print("\n2. Erweiterte AWG Features:")
    sim_advanced = demo_advanced_awg()

    # Performance-Vergleich
    print("\n3. Performance-Vergleich:")
    cfg = SimConfig()
    cfg.seq.t_mw = 100e-9
    cfg.seq.t_readout = 500e-9
    cfg.seq.bin_ns = 1.0
    cfg.seq.oversample_per_ns = 50
    cfg.det.shots_accum = 10000
    cfg.progress = False

    sim = NVSimulator(cfg)

    start_time = time.time()
    t_ns, counts = sim.simulate(envelope_type="gaussian")
    sim_time = time.time() - start_time

    print(f"   Simulation Time: {sim_time:.2f}s")
    print(f"   Total Time Points: {len(t_ns)}")
    print(f"   Time Resolution: {cfg.seq.bin_ns} ns")
    print(f"   Total Counts: {counts.sum():.0f}")
    print(f"   Performance: {len(t_ns)/sim_time:.0f} points/s")

    print("\n==========================================")
    print("NV Simulator V3 - Alle Features getestet!")

    return results_basic, sim_advanced

if __name__ == "__main__":
    # Wähle Demo-Modus
    import sys
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "basic":
            demo_basic()
        elif mode == "advanced":
            demo_advanced_awg()
        elif mode == "full":
            demo_full_comparison()
        else:
            print("Usage: python nv_simulator_v3.py [basic|advanced|full]")
    else:
        # Standard: Basis-Demo
        demo_basic()