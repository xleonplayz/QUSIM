# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Leon Kaiser QUSIM Quantum Simulator für NV center simulations
#  MSQC Goethe University https://msqc.cgi-host6.rz.uni-frankfurt.de
#  I.kaiser[at]em.uni-frankfurt.de
#

import numpy as np
from typing import Dict

class LaserInterface:
    """
    Realistischer Laser für NV-Zentren Simulation
    Mit Linienbreite, Phasenrauschen, Intensitätsfluktuationen und erweiterten Pulsformen
    """
    def __init__(self, config: Dict = None):
        # Standard-Parameter
        self.is_on = False
        self.power_mw = 0.0  # Leistung in mW
        self.wavelength = 637e-9  # NV-Zentrum Resonanz in m
        self.beam_waist = 1e-6  # Strahlradius 1µm
        self.detuning = 0.0  # Frequenzverstimmung in Hz
        
        # Realistische Laser-Parameter
        self.linewidth_hz = 1e6  # Linienbreite in Hz (typisch 1 MHz für Diodenlaser)
        self.intensity_noise_rms = 0.01  # Relative Intensitätsfluktuationen (1%)
        self.phase_noise_enabled = True  # Phasenrauschen aktiviert
        self.current_phase = 0.0  # Akkumulierte Laserphase
        
        # Erweiterte Puls-Parameter
        self.pulse_active = False
        self.pulse_type = 'cw'  # 'cw', 'square', 'gauss', 'sinc', 'blackman'
        self.pulse_start_time = 0.0
        self.pulse_duration = 0.0
        self.pulse_amplitude = 0.0
        self.pulse_parameters = {}  # Zusätzliche Parameter für Pulsformen
        
        # Physikalische Konstanten
        self.h_bar = 1.054571817e-34  # J·s
        self.c = 299792458  # m/s
        self.epsilon_0 = 8.854187817e-12  # F/m
        self.dipole_moment = 1.6e-29  # NV Dipolmoment in C·m
        
        # Matrizen für |g⟩↔|e⟩ Übergang
        self.sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)  # |e⟩⟨g|
        self.sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)  # |g⟩⟨e|
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Identitätsmatrizen für Tensorprodukt
        self.I3 = np.eye(3, dtype=complex)
        
        print("Laser-Interface initialisiert (AUS)")
    
    def turn_on(self, power_mw: float, detuning: float = 0.0, 
               linewidth_hz: float = None, intensity_noise: float = None):
        """
        Schaltet Laser ein mit realistischen Parametern
        
        Args:
            power_mw: Laserleistung in mW
            detuning: Verstimmung von Resonanz in Hz
            linewidth_hz: Laser-Linienbreite in Hz (optional)
            intensity_noise: Relative Intensitätsfluktuationen (optional)
        """
        self.is_on = True
        self.power_mw = power_mw
        self.detuning = detuning
        self.pulse_type = 'cw'  # CW-Betrieb
        self.pulse_active = False
        
        # Optionale Parameter setzen
        if linewidth_hz is not None:
            self.linewidth_hz = linewidth_hz
        if intensity_noise is not None:
            self.intensity_noise_rms = intensity_noise
        
        # Berechne Rabi-Frequenz
        self.rabi_freq = self._calculate_rabi_frequency()
        
        print(f"Laser EINGESCHALTET: {power_mw:.2f}mW, Ω={self.rabi_freq/1e6:.1f}MHz")
        if detuning != 0:
            print(f"  Verstimmung: {detuning/1e6:.1f}MHz")
        if self.linewidth_hz > 0:
            print(f"  Linienbreite: {self.linewidth_hz/1e6:.1f}MHz")
        if self.intensity_noise_rms > 0:
            print(f"  Intensitätsrauschen: {self.intensity_noise_rms*100:.1f}% RMS")
    
    def turn_off(self):
        """Schaltet Laser aus"""
        self.is_on = False
        self.power_mw = 0.0
        self.rabi_freq = 0.0
        print("Laser AUSGESCHALTET")
    
    def _calculate_rabi_frequency(self) -> float:
        """
        Berechnet Rabi-Frequenz aus Laserleistung
        
        Ω = d·E/ℏ mit E = √(2P/(πw²cε₀))
        
        Returns:
            Rabi-Frequenz in rad/s
        """
        if self.power_mw <= 0:
            return 0.0
        
        # Leistung in Watt
        power_w = self.power_mw * 1e-3
        
        # Intensität für Gauss-Strahl
        intensity = 2 * power_w / (np.pi * self.beam_waist**2)
        
        # E-Feld Amplitude
        E_field = np.sqrt(2 * intensity / (self.c * self.epsilon_0))
        
        # Rabi-Frequenz
        omega_rabi = self.dipole_moment * E_field / self.h_bar
        
        return omega_rabi
    
    def get_hamiltonian_contribution(self, t: float, dt: float = 1e-9) -> np.ndarray:
        """
        Berechnet realistischen Laser-Beitrag zum Hamiltonian
        
        H_laser = (ℏ/2) * Ω(t) * (σ₊ * e^(-iφ(t)) + σ₋ * e^(iφ(t))) + ℏ * Δ(t) * σz
        
        Args:
            t: Zeit (für zeitabhängige Pulse)
            dt: Zeitschritt für Rauschentwicklung
            
        Returns:
            18×18 Hamiltonian-Matrix
        """
        if not self.is_on:
            return np.zeros((18, 18), dtype=complex)
        
        # Berechne zeitabhängige Amplitude (Pulsformen)
        current_amplitude = self._get_time_dependent_amplitude(t)
        if current_amplitude == 0:
            return np.zeros((18, 18), dtype=complex)
        
        # Anwende Intensitätsrauschen
        if self.intensity_noise_rms > 0:
            noise_factor = 1 + np.random.normal(0, self.intensity_noise_rms)
            current_amplitude *= max(0, noise_factor)  # Verhindere negative Amplituden
        
        # Berechne zeitabhängige Verstimmung (Linienbreite)
        current_detuning = self._get_time_dependent_detuning(dt)
        
        # Phasenentwicklung für Linienbreite
        current_phase = self._evolve_laser_phase(dt)
        
        # 2×2 Laser-Hamiltonian mit Phasenrauschen
        H_laser_2x2 = (self.h_bar / 2) * (
            current_detuning * 2*np.pi * self.sigma_z +  # Verstimmung mit Rauschen
            current_amplitude * (
                self.sigma_plus * np.exp(-1j * current_phase) + 
                self.sigma_minus * np.exp(1j * current_phase)
            )  # Rabi-Kopplung mit Phase
        )
        
        # Erweitere auf 18×18: (2×2) ⊗ I₃ ⊗ I₃
        H_laser_18x18 = np.kron(np.kron(H_laser_2x2, self.I3), self.I3)
        
        return H_laser_18x18
    
    def _get_time_dependent_amplitude(self, t: float) -> float:
        """
        Berechnet zeitabhängige Laser-Amplitude für verschiedene Pulsformen
        
        Args:
            t: Aktuelle Zeit
            
        Returns:
            Amplitude zu Zeit t
        """
        if self.pulse_type == 'cw':
            return self.rabi_freq
        
        if not self.pulse_active:
            return 0.0
            
        # Relative Zeit innerhalb des Pulses
        t_rel = t - self.pulse_start_time
        
        if t_rel < 0 or t_rel > self.pulse_duration:
            return 0.0
        
        if self.pulse_type == 'square':
            return self.pulse_amplitude
            
        elif self.pulse_type == 'gauss':
            sigma = self.pulse_parameters.get('sigma', self.pulse_duration / 6)
            t_center = self.pulse_duration / 2
            envelope = np.exp(-((t_rel - t_center) / sigma) ** 2 / 2)
            return self.pulse_amplitude * envelope
            
        elif self.pulse_type == 'sinc':
            zero_crossings = self.pulse_parameters.get('zero_crossings', 3)
            t_center = self.pulse_duration / 2
            x = zero_crossings * np.pi * (t_rel - t_center) / (self.pulse_duration / 2)
            envelope = np.sinc(x / np.pi) if x != 0 else 1.0
            return self.pulse_amplitude * envelope
            
        elif self.pulse_type == 'blackman':
            # Blackman-Fenster
            x = t_rel / self.pulse_duration
            envelope = 0.42 - 0.5 * np.cos(2*np.pi*x) + 0.08 * np.cos(4*np.pi*x)
            return self.pulse_amplitude * envelope
            
        else:
            return self.pulse_amplitude  # Fallback
    
    def _get_time_dependent_detuning(self, dt: float) -> float:
        """
        Berechnet zeitabhängige Verstimmung mit Frequenzrauschen
        
        Args:
            dt: Zeitschritt
            
        Returns:
            Aktuelle Verstimmung
        """
        if self.linewidth_hz <= 0:
            return self.detuning
        
        # Frequenzrauschen als Random Walk
        # Frequenz-Diffusion mit Varianz ∝ Linienbreite
        freq_noise_std = np.sqrt(2 * np.pi * self.linewidth_hz * dt)
        freq_noise = np.random.normal(0, freq_noise_std)
        
        return self.detuning + freq_noise
    
    def _evolve_laser_phase(self, dt: float) -> float:
        """
        Entwickelt die Laserphase mit Phasenrauschen
        
        Args:
            dt: Zeitschritt
            
        Returns:
            Aktuelle Phase
        """
        if not self.phase_noise_enabled or self.linewidth_hz <= 0:
            return self.current_phase
        
        # Phasen-Diffusion (Wiener-Prozess)
        # Phasenvariation ∝ √(Linienbreite * dt)
        phase_noise_std = np.sqrt(2 * np.pi * self.linewidth_hz * dt)
        phase_increment = np.random.normal(0, phase_noise_std)
        
        self.current_phase += phase_increment
        
        # Normalisiere Phase auf [0, 2π)
        self.current_phase = self.current_phase % (2 * np.pi)
        
        return self.current_phase
    
    def get_status(self) -> Dict:
        """Gibt erweiterten Laser-Status zurück"""
        status = {
            'is_on': self.is_on,
            'power_mw': self.power_mw,
            'rabi_frequency_mhz': self.rabi_freq / 1e6 if self.is_on else 0.0,
            'detuning_mhz': self.detuning / 1e6,
            'saturation_parameter': self.get_saturation_parameter(),
            
            # Realistische Parameter
            'linewidth_mhz': self.linewidth_hz / 1e6,
            'intensity_noise_percent': self.intensity_noise_rms * 100,
            'phase_noise_enabled': self.phase_noise_enabled,
            'current_phase_rad': self.current_phase,
            'coherence_time_us': self.get_coherence_time() * 1e6 if self.get_coherence_time() < 1e6 else np.inf,
            'coherence_length_m': self.get_coherence_length() if self.get_coherence_length() < 1e6 else np.inf,
            
            # Puls-Information
            'pulse_active': self.pulse_active,
            'pulse_type': self.pulse_type,
            'pulse_duration_ns': self.pulse_duration * 1e9 if self.pulse_active else 0,
            'pulse_amplitude_mhz': self.pulse_amplitude / 1e6 if self.pulse_active else 0
        }
        
        return status
    
    def get_laser_characterization(self) -> Dict:
        """
        Erweiterte Laser-Charakterisierung für Analyse
        
        Returns:
            Dict mit detaillierten Laser-Eigenschaften
        """
        steady_state = self.get_steady_state_populations()
        
        return {
            'wavelength_nm': self.wavelength * 1e9,
            'frequency_thz': self.c / self.wavelength / 1e12,
            'beam_waist_um': self.beam_waist * 1e6,
            'dipole_moment': self.dipole_moment,
            
            # Leistungsabhängige Parameter
            'intensity_w_per_m2': 2 * self.power_mw * 1e-3 / (np.pi * self.beam_waist**2) if self.power_mw > 0 else 0,
            'photon_flux_per_s': self.power_mw * 1e-3 * self.wavelength / (self.h_bar * self.c * 2*np.pi) if self.power_mw > 0 else 0,
            
            # Steady-State Populationen
            'steady_state_ground': steady_state['pop_g'],
            'steady_state_excited': steady_state['pop_e'],
            
            # Rauschen-Charakterisierung
            'relative_intensity_noise_db': 20 * np.log10(self.intensity_noise_rms) if self.intensity_noise_rms > 0 else -np.inf,
            'phase_noise_linewidth_hz': self.linewidth_hz,
            
            # Zeitskalen
            'rabi_period_ns': 2*np.pi / self.rabi_freq * 1e9 if self.is_on and self.rabi_freq > 0 else np.inf
        }
    
    def get_saturation_parameter(self) -> float:
        """
        Berechnet Sättigungsparameter s = Ω²/(2Γ²)
        Γ ≈ 10MHz für NV-Zentren
        """
        if not self.is_on:
            return 0.0
        
        gamma = 1e7 * 2*np.pi  # 10 MHz in rad/s
        s = self.rabi_freq**2 / (2 * gamma**2)
        return s
    
    def get_steady_state_populations(self) -> Dict:
        """
        Berechnet Steady-State Populationen unter CW-Anregung
        
        Returns:
            Dict mit |g⟩ und |e⟩ Populationen
        """
        if not self.is_on:
            return {'pop_g': 1.0, 'pop_e': 0.0}
        
        s = self.get_saturation_parameter()
        
        # Für resonante Anregung (Δ=0)
        if abs(self.detuning) < 1e6:  # < 1 MHz
            pop_e = s / (1 + s)
            pop_g = 1 / (1 + s)
        else:
            # Mit Verstimmung
            gamma = 1e7 * 2*np.pi
            denominator = gamma**2 + 4*self.detuning**2 + 2*self.rabi_freq**2
            pop_e = self.rabi_freq**2 / denominator
            pop_g = 1 - pop_e
        
        return {'pop_g': pop_g, 'pop_e': pop_e}
    
    def apply_pulse(self, pulse_type: str, duration_ns: float, power_mw: float,
                   start_time: float, **kwargs):
        """
        Aktiviert einen zeitabhängigen Laserpuls
        
        Args:
            pulse_type: 'square', 'gauss', 'sinc', 'blackman'
            duration_ns: Pulsdauer in Nanosekunden
            power_mw: Pulsleistung in mW
            start_time: Startzeit des Pulses
            **kwargs: Zusätzliche Parameter für Pulsform
        """
        self.pulse_type = pulse_type
        self.pulse_duration = duration_ns * 1e-9
        self.pulse_start_time = start_time
        self.pulse_active = True
        
        # Temporäre Leistung für Puls-Amplitude-Berechnung
        temp_power = self.power_mw
        self.power_mw = power_mw
        self.pulse_amplitude = self._calculate_rabi_frequency()
        self.power_mw = temp_power  # Restore original power
        
        # Pulsform-spezifische Parameter
        self.pulse_parameters = kwargs
        
        print(f"Laser-Puls aktiviert: {pulse_type}, {duration_ns}ns, {power_mw}mW")
        
    def apply_gauss_pulse(self, duration_ns: float, power_mw: float, 
                         start_time: float, sigma_ns: float = None):
        """
        Gauss-förmiger Laserpuls
        
        Args:
            duration_ns: Pulsdauer (6σ)
            power_mw: Peak-Leistung
            start_time: Startzeit
            sigma_ns: Gauss-Breite (default: duration/6)
        """
        if sigma_ns is None:
            sigma_ns = duration_ns / 6
            
        self.apply_pulse('gauss', duration_ns, power_mw, start_time, 
                        sigma=sigma_ns * 1e-9)
    
    def apply_sinc_pulse(self, duration_ns: float, power_mw: float,
                        start_time: float, zero_crossings: int = 3):
        """
        Sinc-förmiger Laserpuls für Breitband-Anregung
        
        Args:
            duration_ns: Pulsdauer
            power_mw: Peak-Leistung
            start_time: Startzeit
            zero_crossings: Anzahl Nullstellen pro Seite
        """
        self.apply_pulse('sinc', duration_ns, power_mw, start_time,
                        zero_crossings=zero_crossings)
    
    def apply_blackman_pulse(self, duration_ns: float, power_mw: float,
                           start_time: float):
        """
        Blackman-Fenster Puls mit sanften Flanken
        
        Args:
            duration_ns: Pulsdauer
            power_mw: Peak-Leistung
            start_time: Startzeit
        """
        self.apply_pulse('blackman', duration_ns, power_mw, start_time)
    
    def stop_pulse(self):
        """Stoppt aktiven Puls"""
        self.pulse_active = False
        print("Laser-Puls gestoppt")
    
    def pulse_sequence(self, pulse_list: list):
        """
        Definiert eine Sequenz von Laserpulsen
        
        Args:
            pulse_list: Liste von Puls-Dictionaries
                       [{'type': 'gauss', 'duration_ns': 100, 'power_mw': 1.0, 
                         'delay_ns': 0, 'parameters': {...}}, ...]
        """
        sequence_info = []
        
        for i, pulse in enumerate(pulse_list):
            pulse_info = {
                'index': i,
                'type': pulse.get('type', 'square'),
                'duration': pulse.get('duration_ns', 100) * 1e-9,
                'power': pulse.get('power_mw', 1.0),
                'delay': pulse.get('delay_ns', 0) * 1e-9,
                'parameters': pulse.get('parameters', {})
            }
            sequence_info.append(pulse_info)
            
        return sequence_info
    
    def set_laser_parameters(self, linewidth_hz: float = None,
                           intensity_noise: float = None,
                           phase_noise_enabled: bool = None):
        """
        Setzt erweiterte Laser-Parameter
        
        Args:
            linewidth_hz: Laser-Linienbreite in Hz
            intensity_noise: Relative Intensitätsfluktuationen
            phase_noise_enabled: Phasenrauschen aktiviert
        """
        if linewidth_hz is not None:
            self.linewidth_hz = linewidth_hz
        if intensity_noise is not None:
            self.intensity_noise_rms = intensity_noise
        if phase_noise_enabled is not None:
            self.phase_noise_enabled = phase_noise_enabled
            
        print(f"Laser-Parameter aktualisiert:")
        print(f"  Linienbreite: {self.linewidth_hz/1e6:.3f} MHz")
        print(f"  Intensitätsrauschen: {self.intensity_noise_rms*100:.2f}%")
        print(f"  Phasenrauschen: {'EIN' if self.phase_noise_enabled else 'AUS'}")
        
    def get_coherence_time(self) -> float:
        """
        Berechnet Kohärenzzeit basierend auf Linienbreite
        
        Returns:
            Kohärenzzeit in Sekunden
        """
        if self.linewidth_hz <= 0:
            return np.inf
        return 1 / (2 * np.pi * self.linewidth_hz)
    
    def get_coherence_length(self) -> float:
        """
        Berechnet Kohärenzlänge
        
        Returns:
            Kohärenzlänge in Metern
        """
        coherence_time = self.get_coherence_time()
        if coherence_time == np.inf:
            return np.inf
        return self.c * coherence_time