# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Leon Kaiser QUSIM Quantum Simulator für NV center simulations
#  MSQC Goethe University https://msqc.cgi-host6.rz.uni-frankfurt.de
#  I.kaiser[at]em.uni-frankfurt.de
#

import numpy as np
from typing import Dict

class LaserInterface:
    """
    Laser-Interface für NV-Zentren Simulation
    Ermöglicht Ein/Aus-Schaltung während der Simulation
    """
    def __init__(self, config: Dict = None):
        # Standard-Parameter
        self.is_on = False
        self.power_mw = 0.0  # Leistung in mW
        self.wavelength = 637e-9  # NV-Zentrum Resonanz in m
        self.beam_waist = 1e-6  # Strahlradius 1µm
        self.detuning = 0.0  # Frequenzverstimmung in Hz
        
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
    
    def turn_on(self, power_mw: float, detuning: float = 0.0):
        """
        Schaltet Laser ein
        
        Args:
            power_mw: Laserleistung in mW
            detuning: Verstimmung von Resonanz in Hz
        """
        self.is_on = True
        self.power_mw = power_mw
        self.detuning = detuning
        
        # Berechne Rabi-Frequenz
        self.rabi_freq = self._calculate_rabi_frequency()
        
        print(f"Laser EINGESCHALTET: {power_mw:.2f}mW, Ω={self.rabi_freq/1e6:.1f}MHz")
        if detuning != 0:
            print(f"  Verstimmung: {detuning/1e6:.1f}MHz")
    
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
    
    def get_hamiltonian_contribution(self, t: float) -> np.ndarray:
        """
        Berechnet Laser-Beitrag zum Hamiltonian
        
        H_laser = (ℏ/2) * Ω * (σ₊ + σ₋) für Resonanz
        H_laser = (ℏ/2) * [Δσz + Ω(σ₊ + σ₋)] für Verstimmung Δ
        
        Args:
            t: Zeit (für zeitabhängige Pulse)
            
        Returns:
            18×18 Hamiltonian-Matrix
        """
        if not self.is_on:
            return np.zeros((18, 18), dtype=complex)
        
        # 2×2 Laser-Hamiltonian
        H_laser_2x2 = (self.h_bar / 2) * (
            self.detuning * 2*np.pi * self.sigma_z +  # Verstimmung
            self.rabi_freq * (self.sigma_plus + self.sigma_minus)  # Rabi-Kopplung
        )
        
        # Erweitere auf 18×18: (2×2) ⊗ I₃ ⊗ I₃
        H_laser_18x18 = np.kron(np.kron(H_laser_2x2, self.I3), self.I3)
        
        return H_laser_18x18
    
    def get_status(self) -> Dict:
        """Gibt aktuellen Laser-Status zurück"""
        return {
            'is_on': self.is_on,
            'power_mw': self.power_mw,
            'rabi_frequency_mhz': self.rabi_freq / 1e6 if self.is_on else 0.0,
            'detuning_mhz': self.detuning / 1e6,
            'saturation_parameter': self.get_saturation_parameter()
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
    
    def pulse_sequence(self, duration_ns: float, power_mw: float):
        """
        Definiert einen Laserpuls
        
        Args:
            duration_ns: Pulsdauer in Nanosekunden
            power_mw: Pulsleistung in mW
        """
        return {
            'type': 'square_pulse',
            'duration': duration_ns * 1e-9,
            'power': power_mw,
            'start_time': None  # Wird später gesetzt
        }