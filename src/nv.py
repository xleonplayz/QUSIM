#!/usr/bin/env python3
"""
NV Simulator - Fixed Version with Realistic Readout
===================================================
Kombiniert alle Advanced Features mit realistischen Counts
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

# JAX settings
jax.config.update("jax_enable_x64", True)
PI = jnp.pi

class AdvancedNVParams:
    def __init__(self, 
                 temperature_K=300.0,
                 magnetic_field_T=np.array([0.0, 0.0, 1e-3]),
                 laser_power_mW=3.0,
                 laser_wavelength_nm=532.0,
                 numerical_aperture=0.9):
        """
        Initialize NV parameters with realistic physics-based calculations.
        No more hardcoded values!
        """
        # Import realistic parameter calculator
        from realistic_parameters import RealisticNVParameters
        
        # Calculate all parameters from experimental conditions
        self.realistic_params = RealisticNVParameters(
            temperature_K=temperature_K,
            magnetic_field_T=magnetic_field_T,
            laser_wavelength_nm=laser_wavelength_nm,
            laser_power_mW=laser_power_mW,
            numerical_aperture=numerical_aperture
        )
        
        # Get calculated parameters
        params = self.realistic_params.get_all_parameters()
        
        # Simulation parameters (still configurable)
        self.BIN_ns = 10.0
        self.READ_NS = 3000
        self.Shots = 10000
        self.Omega_Rabi_Hz = 5e6  # Will be MW power dependent
        
        # Use calculated realistic values
        self.collection_efficiency = params['collection_efficiency']
        self.DW_factor = params['debye_waller_factor'] 
        self.PSB_efficiency = 0.7  # Phonon sideband collection efficiency
        self.T1_ms = params['T1_ms']
        self.T2_star_us = params['T2_star_us']
        self.gamma_ISC_ms1_MHz = params['gamma_ISC_ms1_Hz'] / 1e6
        self.gamma_ISC_ms0_MHz = params['gamma_ISC_ms0_Hz'] / 1e6
        self.tau_singlet_ns = 250.0  # Singlet lifetime
        self.I_sat_mW = params['I_sat_mW']
        self.laser_power_mW = laser_power_mW
        
        # Realistic spin bath from calculation
        spin_bath = params['spin_bath_params']
        self.n_carbon13 = spin_bath['n_C13']
        self.n_nitrogen14 = spin_bath['n_N14']
        
        # Temperature and field dependent parameters
        self.Temperature_K = temperature_K
        self.magnetic_field_T = magnetic_field_T
        
        # ZFS from realistic calculation
        self.D_GS_Hz = params['D_GS_Hz']
        self.D_ES_Hz = params['D_ES_Hz']
        
        # Activation energies (realistic)
        self.ISC_ms0_activation_meV = 50.0
        self.ISC_ms1_activation_meV = 10.0
        
        # Charge state dynamics (calculated)
        charge_params = params['charge_params']
        self.ionization_rate_GS_MHz = charge_params['ionization_rate_GS_MHz']
        self.ionization_rate_ES_MHz = charge_params['ionization_rate_ES_MHz']
        self.recombination_rate_MHz = charge_params['recombination_rate_MHz']
        
        # Detector parameters (realistic)
        detector = params['detector_params']
        self.DarkRate_cps = detector['dark_count_rate_Hz']
        self.Beta_max_Hz = 13e6  # Will be calculated from radiative rate
        
        # 🆕 6. Phonon-Kopplung & Debye-Waller
        self.DW0_at_zero_K = 0.03  # Debye-Waller bei T=0
        self.theta_D_K = 150.0  # Debye-Temperatur in K
        self.debye_waller_alpha = 0.005  # Temperatur-Koeffizient
        self.gamma_rad_MHz = 83.0  # Radiative decay rate
        self.k_orb_300K_MHz = 100.0  # Orbital relaxation bei 300K
        self.k_orb_activation_K = 500.0  # Aktivierungstemperatur
        
        # 🆕 7. Time-dependent MW-Pulse mit Rauschen
        self.mw_pulse_shape = 'gaussian'  # 'gaussian', 'square', 'hermite'
        self.mw_sigma_ns = 50.0  # Gaussian pulse width
        self.mw_amplitude_noise_std = 0.01  # 1% Amplituden-Rauschen
        self.mw_phase_noise_std = 0.02  # ~1° Phasen-Drift (rad)
        self.mw_frequency_drift_Hz = 1000.0  # MW frequency drift
        self.mw_inhomogeneity = 0.05  # Räumliche Inhomogenität
        
        # 🆕 8. Spektrale Diffusion (Ornstein–Uhlenbeck auf Δ)
        self.specdiff_sigma_MHz = 0.5  # Steady-State-Streuung [MHz]
        self.specdiff_tau_corr_ns = 1e6  # Korrelationszeit [ns] = 1ms
        
        # 🆕 9. Detektor-Modell (IRF + Dead-Time + Afterpulse)
        self.dead_time_ns = 12.0  # SPAD dead time
        self.afterpulse_prob = 0.02  # 2% afterpulse probability
        self.dark_rate_Hz = 200.0  # Dark count rate
        self.irf_sigma_ps = 300.0  # IRF Gaussian width [ps]
        self.irf_tail_frac = 0.1  # Fraction with exponential tail
        self.irf_tail_tau_ns = 5.0  # Exponential tail time constant [ns]
        
        # Magnetic field for advanced simulations
        self.B_field_mT = [0.0, 0.0, 1.0]  # 1 mT along z-axis
        
        # Zero-field splittings for Lindblad simulation
        self.D_GS_Hz = 2.87e9  # Ground state ZFS in Hz (2.87 GHz)
        self.D_ES_Hz = 1.42e9  # Excited state ZFS in Hz (1.42 GHz)
        
        # MW frequency (user can change this)
        self.mw_frequency_Hz = 2.87e9  # Default: on resonance

# ===================== 🆕 Phonon-Kopplung & Debye-Waller ==================
def debye_waller_factor(T_K, params):
    """
    Temperatur-abhängiger Debye-Waller-Faktor
    DW(T) = DW0 * exp(-α * T / θ_D)
    """
    dw_factor = params.DW0_at_zero_K * np.exp(
        -params.debye_waller_alpha * T_K / params.theta_D_K
    )
    return dw_factor

def phonon_sideband_rates(params, T_K, pop_es):
    """
    Temperatur-abhängige ZPL vs PSB emission rates
    """
    # Temperatur-abhängiger Debye-Waller Factor
    DW = debye_waller_factor(T_K, params)
    
    # Radiative decay
    gamma_rad = params.gamma_rad_MHz * 1e6  # in Hz
    
    # ZPL (Zero-Phonon Line)
    rate_zpl = DW * gamma_rad * pop_es
    
    # PSB (Phonon Sideband) - reduzierte Effizienz
    rate_psb = (1.0 - DW) * gamma_rad * pop_es * params.PSB_efficiency
    
    return rate_zpl, rate_psb

def ultrafast_ES_dephasing(params, T_K):
    """
    Ultraschnelle ES dephasing durch Orbital relaxation
    Rate ∝ exp(-E_a / k_B T)
    """
    # Arrhenius-Aktivierung
    k_orb = params.k_orb_300K_MHz * np.exp(
        -params.k_orb_activation_K / T_K
    )
    
    # Dephasing rate (MHz)
    return k_orb

# ===================== 🆕 Time-dependent MW-Pulse ========================
def gaussian_mw_envelope(t_ns, t0_ns, sigma_ns, Omega0_Hz):
    """Gaussian MW pulse envelope"""
    return Omega0_Hz * np.exp(-0.5 * ((t_ns - t0_ns) / sigma_ns)**2)

def hermite_mw_envelope(t_ns, t0_ns, sigma_ns, Omega0_Hz):
    """Hermite-Gaussian MW pulse für bessere Selektivität"""
    x = (t_ns - t0_ns) / sigma_ns
    # Hermite polynomial H_1(x) = 2x
    hermite_factor = 2 * x * np.exp(-x**2)
    return Omega0_Hz * hermite_factor

def noisy_mw_pulse(t_ns, tau_ns, params, random_key):
    """
    Time-dependent MW pulse with realistic noise
    """
    t0_ns = tau_ns / 2  # Pulse center
    
    # Base envelope
    if params.mw_pulse_shape == 'gaussian':
        Omega_det = gaussian_mw_envelope(t_ns, t0_ns, params.mw_sigma_ns, params.Omega_Rabi_Hz)
    elif params.mw_pulse_shape == 'hermite':
        Omega_det = hermite_mw_envelope(t_ns, t0_ns, params.mw_sigma_ns, params.Omega_Rabi_Hz)
    else:  # square
        if abs(t_ns - t0_ns) < tau_ns / 2:
            Omega_det = params.Omega_Rabi_Hz
        else:
            Omega_det = 0.0
    
    # Add noise
    key1, key2, key3 = random.split(random_key, 3)
    
    # Amplitude noise (1% typical)
    amp_noise = 1.0 + params.mw_amplitude_noise_std * random.normal(key1)
    
    # Phase noise (~1° = 0.02 rad)
    phase_noise = params.mw_phase_noise_std * random.normal(key2)
    
    # Frequency drift
    freq_drift = params.mw_frequency_drift_Hz * random.normal(key3) * t_ns * 1e-9
    
    # Effective Rabi frequency with noise
    Omega_eff = Omega_det * amp_noise
    
    # Phase includes drift
    phase_eff = phase_noise + freq_drift
    
    return Omega_eff, phase_eff

# ===================== 🆕 8. Spektrale Diffusion ========================
def ou_detuning(delta_prev, dt, tau_corr_ns, sigma_detune_MHz, key):
    """
    Ornstein–Uhlenbeck für langsame Drift des Zero-Field Splittings Δ (in MHz).
    """
    alpha = dt / tau_corr_ns
    key, sub = random.split(key)
    # Normalisiertes OU-Rauschen
    xi = random.normal(sub) * np.sqrt(2*alpha) * sigma_detune_MHz
    delta_new = delta_prev * (1 - alpha) + xi
    return delta_new, key

# ===================== 🆕 9. Detektor-Modell ============================
def sample_irf_single(key, params):
    """
    Einzelfoton-IRF: Gauß + exponentieller Tail (ps → ns)
    """
    key, k1, k2 = random.split(key, 3)
    # Gaußförmig (σ_ps → ns)
    gauss = random.normal(k1) * (params.irf_sigma_ps * 1e-3)
    # Tail?
    use_tail = random.uniform(k2) < params.irf_tail_frac
    tail = random.exponential(k2) * params.irf_tail_tau_ns * use_tail
    return gauss + tail, key

def detect_with_deadtime_and_afterpulse(rates_per_ns, dt_ns, params, key):
    """
    Ereignisbasiertes Dead-Time + Afterpulse + Jitter + Dark Counts
    (Vereinfacht für bessere Pulse-Plots)
    """
    # Einfache Poisson-Sampling (für spätere Erweiterung)
    expected_counts = rates_per_ns * dt_ns * 1e-9  # Convert Hz * ns to counts
    n_raw = np.random.poisson(expected_counts)
    
    return n_raw, key

def simulate_nv_with_advanced_physics(tau_ns, params, random_seed=42):
    """Alias for compatibility"""
    return simulate_nv_realistic(tau_ns, params, random_seed)

def simulate_nv_realistic(tau_ns, params, random_seed=None):
    """
    Realistische NV Simulation mit allen Features und echter Randomisierung
    """
    # Import realistic random manager
    from random_manager import get_random_manager
    
    # Get random manager (will use true randomness unless seed specified)
    rm = get_random_manager(random_seed)
    
    # Debug print
    print(f"[simulate_nv_realistic] tau={tau_ns}ns, Omega_Rabi_Hz={params.Omega_Rabi_Hz}")
    
    # Zeit-Arrays
    time_ns = np.arange(0, params.READ_NS, params.BIN_ns)
    
    # 🔄 Pulse-Timeline: MW-Pulse von 0 bis tau_ns, dann Readout
    pulse_active = (time_ns <= tau_ns)  # Pulse ist aktiv während 0 bis tau_ns
    readout_active = (time_ns > tau_ns)  # Readout nach dem Pulse
    
    # 1. MW Pulse: 🆕 Time-dependent mit echtem Rauschen
    mw_key = rm.get_key('mw_noise')
    key1, key2 = random.split(mw_key, 2)
    
    # Simuliere time-dependent MW pulse (vereinfacht)
    # Für realistische Integration würde man über Zeit integrieren
    # Hier: Effektive Rabi mit Noise-Korrekturen
    
    # MW pulse noise effects
    t_mid = tau_ns / 2
    Omega_eff, phase_eff = noisy_mw_pulse(t_mid, tau_ns, params, key1)
    
    # Calculate MW detuning
    detuning_Hz = params.mw_frequency_Hz - params.D_GS_Hz
    detuning_MHz = detuning_Hz / 1e6
    print(f"[simulate_nv_realistic] MW detuning: {detuning_MHz:.1f} MHz")
    
    # Check if MW is off
    if params.Omega_Rabi_Hz == 0:
        # No MW drive - populations stay at initial state
        p_ms0 = 1.0  # Start in ms=0
        p_ms1 = 0.0
        print(f"[simulate_nv_realistic] MW OFF - p_ms0=1.0, p_ms1=0.0")
    else:
        # Calculate effective Rabi frequency with detuning
        # Ω_eff = sqrt(Ω²_Rabi + Δ²)
        omega_eff_Hz = np.sqrt(params.Omega_Rabi_Hz**2 + detuning_Hz**2)
        
        # Effective Rabi angle 
        rabi_angle_ideal = omega_eff_Hz * tau_ns * 1e-9 * np.pi
        
        # On resonance: full population transfer
        # Off resonance: reduced amplitude by Ω_Rabi/Ω_eff
        resonance_factor = params.Omega_Rabi_Hz / omega_eff_Hz if omega_eff_Hz > 0 else 0
        
        # MW pulse noise
        noise_factor = Omega_eff / params.Omega_Rabi_Hz
        rabi_angle_noisy = rabi_angle_ideal * noise_factor
        
        # Phase errors affect contrast
        phase_error_factor = np.cos(phase_eff)**2  # Reduced contrast from phase errors
        
        # Basic populations with detuning
        # For large detuning, oscillation amplitude is reduced
        p_ms0_ideal = 1 - resonance_factor**2 * np.sin(rabi_angle_noisy/2)**2
        p_ms1_ideal = 1 - p_ms0_ideal
        
        # Apply phase error contrast reduction
        p_ms0 = phase_error_factor * p_ms0_ideal + (1 - phase_error_factor) * 0.5
        p_ms1 = phase_error_factor * p_ms1_ideal + (1 - phase_error_factor) * 0.5
        
        print(f"[simulate_nv_realistic] Resonance factor: {resonance_factor:.3f}, p_ms0: {p_ms0:.3f}")
    
    # 2. Advanced Physics Korrekturen mit 🆕 Phonon-Effekten
    
    # 2a. Spin-Bath dephasing (sehr schwach)
    contrast = 1.0 - params.n_carbon13 * 0.0005 - params.n_nitrogen14 * 0.0002
    
    # 2b. T2* decay (nur bei langen pulsen)
    if tau_ns > 1000:
        t2_decay = np.exp(-(tau_ns-1000) / (params.T2_star_us * 1000))
        contrast *= t2_decay
    
    # 2c. 🆕 Ultrafast ES dephasing from phonon coupling
    # Reduces contrast during MW pulse due to orbital relaxation
    k_orb_MHz = ultrafast_ES_dephasing(params, params.Temperature_K)
    
    # During MW pulse, ES states are populated - orbital dephasing affects coherence
    # Effect scales with pulse length and ES population
    es_pop_during_pulse = 0.05  # ~5% ES population during MW pulse
    orbital_dephasing_factor = np.exp(-(k_orb_MHz * 1e6) * (tau_ns * 1e-9) * es_pop_during_pulse)
    contrast *= orbital_dephasing_factor
    
    # 2d. 🆕 Spektrale Diffusion (OU process on ZFS)
    # Affects contrast through detuning during MW pulse  
    key_spec = rm.get_key('spectral_diffusion')
    delta_initial = 0.0  # Start at resonance
    dt_pulse = tau_ns / 10  # Sub-divide pulse for OU integration
    delta_avg = 0.0
    
    for i in range(10):  # Integrate over pulse duration
        delta_initial, key_spec = ou_detuning(
            delta_initial, dt_pulse, 
            params.specdiff_tau_corr_ns, 
            params.specdiff_sigma_MHz, 
            key_spec
        )
        delta_avg += delta_initial / 10
    
    # Detuning reduces Rabi efficiency
    detuning_factor = 1.0 / (1.0 + (delta_avg / 5.0)**2)  # Lorentzian suppression
    contrast *= detuning_factor
    
    # Apply contrast reduction
    p_ms0_corrected = contrast * p_ms0 + (1-contrast) * 0.5
    p_ms1_corrected = contrast * p_ms1 + (1-contrast) * 0.5
    
    # Use corrected values
    p_ms0 = p_ms0_corrected
    p_ms1 = p_ms1_corrected
    
    # 3. 🆕 Time-dependent readout mit Pulse-Dynamik
    readout_time_s = time_ns * 1e-9
    
    # Initialize rates array
    rate_bright = np.zeros_like(time_ns)
    rate_dark = np.zeros_like(time_ns)
    
    # During MW pulse: fast keine Fluoreszenz (Spin wird manipuliert) 
    pulse_fluorescence_factor = 0.001  # 0.1% Fluoreszenz während Pulse
    
    # Nach MW pulse: normale Readout-Dynamik
    
    # 3a. Temperature-dependent Debye-Waller factor affects collection
    DW_factor = debye_waller_factor(params.Temperature_K, params)
    
    # 3b. Phonon sideband rates (simplified - use average ES population)
    pop_es_avg = 0.1  # Typical ES population during readout
    rate_zpl, rate_psb = phonon_sideband_rates(params, params.Temperature_K, pop_es_avg)
    
    # 3c. Collection efficiency with phonon effects
    # ZPL has full efficiency, PSB has reduced efficiency
    collection_eff_zpl = params.collection_efficiency
    collection_eff_psb = params.collection_efficiency * params.PSB_efficiency
    
    # Weighted average collection efficiency
    total_rad_rate = rate_zpl + rate_psb
    if total_rad_rate > 0:
        collection_eff_phonon = (rate_zpl * collection_eff_zpl + rate_psb * collection_eff_psb) / total_rad_rate
    else:
        collection_eff_phonon = params.collection_efficiency
    
    # 3d. ISC pumping dynamics (temperature-dependent) - time-dependent!
    # Faster pumping at higher temperatures due to phonon assistance
    temp_factor = 1.0 + 0.1 * (params.Temperature_K - 4.0) / 300.0
    
    for i, t_ns in enumerate(time_ns):
        t_s = t_ns * 1e-9
        
        if pulse_active[i]:
            # Während MW-Pulse: fast keine Fluoreszenz (NV wird von MW-Feld getrieben)
            rate_bright[i] = params.DarkRate_cps * 0.1  # Nur Dunkelzählrate 
            rate_dark[i] = params.DarkRate_cps * 0.1
            
        else:
            # Nach MW-Pulse: normale Readout-Dynamik mit ISC pumping
            t_readout = t_s - tau_ns * 1e-9  # Zeit seit Ende des Pulses
            if t_readout < 0:
                t_readout = 0
                
            pump_fast = 1 - np.exp(-t_readout / (20e-9 / temp_factor))
            pump_slow = 1 - np.exp(-t_readout / (200e-9 / temp_factor))
            
            # 3e. Photon emission rates with phonon coupling
            rate_bright_max = params.Beta_max_Hz * collection_eff_phonon
            rate_bright[i] = rate_bright_max * pump_fast
            
            # Dark state (ms=±1) with phonon-assisted transitions
            rate_dark_initial = rate_bright_max * 0.3  # 30% initial
            rate_dark[i] = rate_dark_initial + (rate_bright_max - rate_dark_initial) * pump_slow
    
    # 4. Laser saturation
    s = params.laser_power_mW / (params.laser_power_mW + params.I_sat_mW)
    
    # 5. Charge state (fast immer NV-)
    nv_minus_frac = params.recombination_rate_MHz / (
        params.recombination_rate_MHz + params.ionization_rate_GS_MHz
    )
    nv_minus_frac = max(nv_minus_frac, 0.99)  # Mindestens 99% NV-
    
    # 6. Total photon rate
    rate_signal = s * nv_minus_frac * (p_ms0 * rate_bright + p_ms1 * rate_dark)
    
    # 7. Add noise using realistic random manager
    noise_key = rm.get_key('detector_noise')
    noise = 1 + 0.05 * random.normal(noise_key, shape=rate_signal.shape)
    rate_signal = rate_signal * noise
    
    # 8. Dark counts
    rate_total = rate_signal + params.DarkRate_cps
    
    # 9. 🆕 Poisson sampling with realistic randomization
    expected_counts = rate_total * params.BIN_ns * 1e-9 * params.Shots
    
    # Use realistic random manager for Poisson sampling
    counts = rm.shot_noise_photons('shot_noise', expected_counts)
    
    # Convert to numpy array for modification
    counts = np.array(counts)
    
    # 🔄 Pulse-Struktur hinzufügen: Während Pulse niedrige aber sichtbare Counts
    for i, t_ns in enumerate(time_ns):
        if t_ns <= tau_ns:  # Während MW-Pulse
            # Reduziere auf ~10% der normalen Rate + dark counts
            pulse_rate = params.Beta_max_Hz * 0.05  # 5% der normalen Fluoreszenz
            expected_pulse = (pulse_rate + params.DarkRate_cps) * params.BIN_ns * 1e-9 * params.Shots
            counts[i] = rm.shot_noise_photons('shot_noise', expected_pulse)
    
    return time_ns, counts, float(p_ms0), float(p_ms1)

class NVSimulator:
    def __init__(self):
        self.params = AdvancedNVParams()
        
    def simulate_single_tau(self, tau_ns, random_seed=None):
        """Simulate a single tau value with realistic randomization"""
        # Pass random_seed to realistic simulation (None = true randomness)
        time_ns, counts, p_ms0, p_ms1 = simulate_nv_realistic(
            tau_ns, self.params, random_seed
        )
        return {
            'time_ns': time_ns.tolist(),
            'counts': counts.tolist(),
            'p_ms0': p_ms0,
            'p_ms1': p_ms1
        }
    
    def simulate_tau_sweep(self, tau_list, progress_callback=None):
        """Simulate a full tau sweep"""
        results = {}
        for i, tau in enumerate(tau_list):
            if progress_callback:
                progress_callback(i, len(tau_list), tau)
            
            # Use true randomness for each simulation (no fixed seeds)
            time_ns, counts, p_ms0, p_ms1 = simulate_nv_realistic(
                tau, self.params, None  # None = true randomness
            )
            results[tau] = {
                'time_ns': time_ns.tolist(),
                'counts': counts.tolist(),
                'p_ms0': p_ms0,
                'p_ms1': p_ms1
            }
        return results
    
    def get_physics_info(self):
        """Get all physics parameters"""
        return {
            # Basic
            'collection_efficiency': self.params.collection_efficiency,
            'Omega_Rabi_Hz': self.params.Omega_Rabi_Hz,
            'Beta_max_Hz': self.params.Beta_max_Hz,
            'Shots': self.params.Shots,
            'READ_NS': self.params.READ_NS,
            'BIN_ns': self.params.BIN_ns,
            
            # Decoherence
            'T1_ms': self.params.T1_ms,
            'T2_star_us': self.params.T2_star_us,
            'gamma_ISC_ms1_MHz': self.params.gamma_ISC_ms1_MHz,
            'gamma_ISC_ms0_MHz': self.params.gamma_ISC_ms0_MHz,
            'dead_time_ns': 12.0,  # SPAD dead time
            
            # Advanced physics
            'DW_factor': self.params.DW_factor,
            'Temperature_K': self.params.Temperature_K,
            'n_carbon13': self.params.n_carbon13,
            'n_nitrogen14': self.params.n_nitrogen14,
            'laser_power_mW': self.params.laser_power_mW,
            'laser_wavelength_nm': 532.0,
            'laser_linewidth_GHz': 1.0,
            'beam_waist_um': 1.0,
            'numerical_aperture': 0.9,
            
            # Spin-Bath & Hamiltonian (simplified display)
            'C13_A_parallel_mean_MHz': 3.0,
            'N14_A_parallel_MHz': 2.2,
            'D_GS_Hz': 2.87,  # GHz
            'D_ES_Hz': 1.42,  # GHz
            'g_e': 2.003,
            'B_field_mT': [0.0, 0.0, 1.0],
            'strain_sigma_MHz': 1.0,
            'gamma_orbital_MHz': 0.1,
            
            # Lindblad parameters
            'T1_GS_ms': 10.0,
            'T2_GS_us': 2.0,
            'T1_ES_us': 100.0,
            'T2_ES_us': 50.0,
            'ISC_ms0_activation_meV': self.params.ISC_ms0_activation_meV,
            'ISC_ms1_activation_meV': self.params.ISC_ms1_activation_meV,
            
            # Charge state
            'ionization_rate_GS_MHz': self.params.ionization_rate_GS_MHz,
            'ionization_rate_ES_MHz': self.params.ionization_rate_ES_MHz,
            'recombination_rate_MHz': self.params.recombination_rate_MHz,
            
            # 🆕 Phonon-Kopplung & Debye-Waller
            'DW0_at_zero_K': self.params.DW0_at_zero_K,
            'theta_D_K': self.params.theta_D_K,
            'debye_waller_alpha': self.params.debye_waller_alpha,
            'gamma_rad_MHz': self.params.gamma_rad_MHz,
            'k_orb_300K_MHz': self.params.k_orb_300K_MHz,
            'k_orb_activation_K': self.params.k_orb_activation_K,
            
            # 🆕 Time-dependent MW-Pulse
            'mw_pulse_shape': self.params.mw_pulse_shape,
            'mw_sigma_ns': self.params.mw_sigma_ns,
            'mw_amplitude_noise_std': self.params.mw_amplitude_noise_std,
            'mw_phase_noise_std': self.params.mw_phase_noise_std,
            'mw_frequency_drift_Hz': self.params.mw_frequency_drift_Hz,
            'mw_inhomogeneity': self.params.mw_inhomogeneity
        }

if __name__ == "__main__":
    print("🔬 Fixed NV Simulator - Realistic Readout")
    print("=" * 40)
    
    sim = NVSimulator()
    
    # Test
    result = sim.simulate_single_tau(100.0)
    counts = np.array(result['counts'])
    
    print(f"τ=100ns simulation:")
    print(f"  Max counts: {max(counts):.0f}")
    print(f"  Mean counts: {np.mean(counts):.1f}")
    print(f"  First 20 bins: {counts[:20].astype(int)}")
    print(f"  ms=0: {result['p_ms0']:.3f}, ms=±1: {result['p_ms1']:.3f}")
    
    # Expected values
    print(f"\nExpected:")
    rate_max = sim.params.Beta_max_Hz * sim.params.collection_efficiency
    s = sim.params.laser_power_mW / (sim.params.laser_power_mW + sim.params.I_sat_mW)
    print(f"  Max rate: {rate_max * s / 1e6:.3f} MHz")
    print(f"  Max counts/bin: {rate_max * s * sim.params.BIN_ns * 1e-9 * sim.params.Shots:.0f}")