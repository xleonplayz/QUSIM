import numpy as np
from typing import Dict

class MicrowaveInterface:
    """
    Mikrowellen-Interface für NV-Zentren Simulation
    Ermöglicht Spin-Kontrolle während der Simulation
    Funktioniert analog zum Laser-Interface
    """
    def __init__(self, config: Dict = None):
        # Standard-Parameter
        self.is_on = False
        self.power_dbm = -20.0  # Leistung in dBm
        self.frequency = 2.87e9  # NV Grundzustand-Splitting (~2.87 GHz)
        self.detuning = 0.0  # Frequenzverstimmung in Hz
        self.phase = 0.0  # Mikrowellenphase in rad
        
        # Physikalische Konstanten
        self.h_bar = 1.054571817e-34  # J·s
        self.mu_B = 9.274009994e-24  # Bohr-Magneton in J/T
        self.g_factor = 2.0  # g-Faktor für NV
        
        # Spin-1 Matrizen für |ms=-1,0,+1⟩
        sqrt2 = np.sqrt(2)
        self.Sx = np.array([
            [0, 1/sqrt2, 0],
            [1/sqrt2, 0, 1/sqrt2], 
            [0, 1/sqrt2, 0]
        ], dtype=complex)
        
        self.Sy = np.array([
            [0, -1j/sqrt2, 0],
            [1j/sqrt2, 0, -1j/sqrt2],
            [0, 1j/sqrt2, 0] 
        ], dtype=complex)
        
        self.Sz = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, -1]
        ], dtype=complex)
        
        # Übergangsoperatoren
        self.S_plus = self.Sx + 1j * self.Sy  # |+1⟩⟨0| + |0⟩⟨-1|
        self.S_minus = self.Sx - 1j * self.Sy # |0⟩⟨+1| + |-1⟩⟨0|
        
        # Identitätsmatrizen für Tensorprodukt
        self.I2 = np.eye(2, dtype=complex)  # Elektronischer Zustand
        self.I3 = np.eye(3, dtype=complex)  # Kernspins
        
        print("Mikrowellen-Interface initialisiert (AUS)")
    
    def turn_on(self, power_dbm: float, frequency: float = None, phase: float = 0.0):
        """
        Schaltet Mikrowellen ein
        
        Args:
            power_dbm: MW-Leistung in dBm
            frequency: MW-Frequenz in Hz (None = verwende Standard)
            phase: Phase in rad
        """
        self.is_on = True
        self.power_dbm = power_dbm
        self.phase = phase
        
        if frequency is not None:
            self.frequency = frequency
            self.detuning = frequency - 2.87e9  # Verstimmung von NV-Resonanz
        
        # Berechne Rabi-Frequenz
        self.rabi_freq = self._calculate_rabi_frequency()
        
        print(f"Mikrowellen EINGESCHALTET: {power_dbm:.1f}dBm @ {self.frequency/1e9:.3f}GHz")
        print(f"  Rabi-Frequenz: Ω_MW = {self.rabi_freq/1e6:.2f}MHz")
        if abs(self.detuning) > 1e6:
            print(f"  Verstimmung: Δ = {self.detuning/1e6:.1f}MHz")
        if phase != 0:
            print(f"  Phase: φ = {phase:.2f}rad")
    
    def turn_off(self):
        """Schaltet Mikrowellen aus"""
        self.is_on = False
        self.power_dbm = -100.0  # Sehr niedrig
        self.rabi_freq = 0.0
        print("Mikrowellen AUSGESCHALTET")
    
    def _calculate_rabi_frequency(self) -> float:
        """
        Berechnet MW-Rabi-Frequenz aus Leistung
        
        Typische Werte für NV:
        -20dBm → ~1MHz
        -10dBm → ~3MHz  
        0dBm → ~10MHz
        
        Returns:
            MW-Rabi-Frequenz in rad/s
        """
        if self.power_dbm < -50:
            return 0.0
        
        # Empirische Formel für NV-Zentren
        # Ω_MW [MHz] ≈ 10^((P[dBm] + 20)/20)
        omega_mhz = 10**((self.power_dbm + 20) / 20)
        omega_rads = omega_mhz * 1e6 * 2*np.pi
        
        return omega_rads
    
    def get_hamiltonian_contribution(self, t: float) -> np.ndarray:
        """
        Berechnet MW-Beitrag zum Hamiltonian
        
        H_MW = (ℏ/2) * Ω_MW * [Sx*cos(ωt+φ) + Sy*sin(ωt+φ)]
        
        Für resonante MW (Δ≈0): Rotating Wave Approximation
        H_MW ≈ (ℏ/2) * Ω_MW * [S₊*e^(-iφ) + S₋*e^(iφ)]
        
        Args:
            t: Zeit (für oszillierende MW)
            
        Returns:
            18×18 Hamiltonian-Matrix
        """
        if not self.is_on:
            return np.zeros((18, 18), dtype=complex)
        
        # Resonante Näherung (Rotating Wave)
        if abs(self.detuning) < 10e6:  # < 10 MHz Verstimmung
            H_mw_3x3 = (self.h_bar / 2) * self.rabi_freq * (
                self.S_plus * np.exp(-1j * self.phase) +
                self.S_minus * np.exp(1j * self.phase)
            )
        else:
            # Vollständiger oszillierender Term
            omega_t = 2*np.pi * self.frequency * t + self.phase
            H_mw_3x3 = (self.h_bar / 2) * self.rabi_freq * (
                self.Sx * np.cos(omega_t) + self.Sy * np.sin(omega_t)
            )
            
            # Verstimmung hinzufügen
            H_mw_3x3 += self.h_bar * self.detuning * self.Sz
        
        # Erweitere auf 18×18: I₂ ⊗ (3×3) ⊗ I₃
        H_mw_18x18 = np.kron(np.kron(self.I2, H_mw_3x3), self.I3)
        
        return H_mw_18x18
    
    def get_status(self) -> Dict:
        """Gibt aktuellen MW-Status zurück"""
        return {
            'is_on': self.is_on,
            'power_dbm': self.power_dbm,
            'frequency_ghz': self.frequency / 1e9,
            'rabi_frequency_mhz': self.rabi_freq / 1e6 / 2 / np.pi if self.is_on else 0.0,
            'detuning_mhz': self.detuning / 1e6,
            'phase_deg': self.phase * 180 / np.pi
        }
    
    def set_frequency(self, frequency_ghz: float):
        """
        Setzt MW-Frequenz
        
        Args:
            frequency_ghz: Frequenz in GHz
        """
        self.frequency = frequency_ghz * 1e9
        self.detuning = self.frequency - 2.87e9
        print(f"MW-Frequenz: {frequency_ghz:.3f}GHz (Δ={self.detuning/1e6:.1f}MHz)")
    
    def set_phase(self, phase_deg: float):
        """
        Setzt MW-Phase
        
        Args:
            phase_deg: Phase in Grad
        """
        self.phase = phase_deg * np.pi / 180
        print(f"MW-Phase: {phase_deg:.1f}° ({self.phase:.3f}rad)")
    
    def pi_pulse(self, axis: str = 'x') -> Dict:
        """
        Definiert einen π-Puls
        
        Args:
            axis: Rotationsachse ('x', 'y', oder Winkel in Grad)
            
        Returns:
            Puls-Parameter
        """
        if not self.is_on:
            raise ValueError("Mikrowellen müssen eingeschaltet sein!")
        
        # π-Puls Dauer: τ_π = π/Ω_MW
        duration_ns = np.pi / self.rabi_freq * 1e9
        
        # Phase für Achse
        if axis == 'x':
            phase = 0.0
        elif axis == 'y':
            phase = np.pi / 2
        else:
            # Numerischer Winkel in Grad
            phase = float(axis) * np.pi / 180
        
        return {
            'type': 'pi_pulse',
            'duration_ns': duration_ns,
            'power_dbm': self.power_dbm,
            'phase': phase,
            'axis': axis
        }
    
    def pi_half_pulse(self, axis: str = 'x') -> Dict:
        """
        Definiert einen π/2-Puls
        
        Args:
            axis: Rotationsachse ('x', 'y', oder Winkel in Grad)
            
        Returns:
            Puls-Parameter  
        """
        pulse = self.pi_pulse(axis)
        pulse['type'] = 'pi_half_pulse'
        pulse['duration_ns'] /= 2
        return pulse
    
    def ramsey_sequence(self, free_evolution_ns: float) -> list:
        """
        Definiert Ramsey-Sequenz: π/2 - τ - π/2
        
        Args:
            free_evolution_ns: Freie Evolution zwischen Pulsen
            
        Returns:
            Liste von Puls-Parametern
        """
        return [
            self.pi_half_pulse('x'),
            {'type': 'free_evolution', 'duration_ns': free_evolution_ns},
            self.pi_half_pulse('y')  # Zweiter Puls um 90° phasenverschoben
        ]
    
    def rabi_sequence(self, max_duration_ns: float, steps: int = 50) -> list:
        """
        Definiert Rabi-Oszillations-Sequenz
        
        Args:
            max_duration_ns: Maximale Pulsdauer
            steps: Anzahl Schritte
            
        Returns:
            Liste von Puls-Dauern
        """
        durations = np.linspace(0, max_duration_ns, steps)
        return [{'type': 'variable_pulse', 'duration_ns': dur, 'power_dbm': self.power_dbm} 
                for dur in durations]