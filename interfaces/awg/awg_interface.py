"""
AWG Interface - Core arbitrary waveform generator for microwave control
"""

import numpy as np
from typing import Optional, Tuple, List
import warnings


class AWGInterface:
    """
    Arbitrary Waveform Generator for precise microwave pulse control.
    Generates time-dependent Hamiltonians for NV center spin manipulation.
    """
    
    def __init__(self, sample_rate: float = 1e9):
        """
        Initialize AWG with specified sample rate.
        
        Args:
            sample_rate: Sampling rate in Hz (default 1 GS/s)
        """
        # Waveform storage
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.waveform_amplitude = np.array([])  # Rabi frequency Ω(t) in rad/s
        self.waveform_phase = np.array([])      # Phase φ(t) in radians
        self.waveform_detuning = np.array([])   # Detuning Δ(t) in rad/s
        self.duration = 0.0
        
        # Playback control
        self.is_playing = False
        self.start_time = 0.0
        self.current_time = 0.0
        self.loop_mode = False
        self.loop_count = 0
        
        # Physical parameters
        self.nv_frequency = 2.87e9  # NV center resonance in Hz
        self.base_frequency = 2.87e9  # MW source frequency in Hz
        self.power_dbm = -20  # Default power in dBm
        
        # Spin operators (3x3 for S=1)
        self._init_spin_operators()
        
        # Pre-computed Hamiltonian cache
        self.use_cache = False
        self.H_cache = []
        
    def _init_spin_operators(self):
        """Initialize spin-1 operators."""
        # Spin-1 matrices in the {|+1⟩, |0⟩, |-1⟩} basis
        self.Sx = np.array([
            [0, 1/np.sqrt(2), 0],
            [1/np.sqrt(2), 0, 1/np.sqrt(2)],
            [0, 1/np.sqrt(2), 0]
        ], dtype=complex)
        
        self.Sy = np.array([
            [0, -1j/np.sqrt(2), 0],
            [1j/np.sqrt(2), 0, -1j/np.sqrt(2)],
            [0, 1j/np.sqrt(2), 0]
        ], dtype=complex)
        
        self.Sz = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, -1]
        ], dtype=complex)
        
        # Raising and lowering operators
        self.S_plus = self.Sx + 1j * self.Sy
        self.S_minus = self.Sx - 1j * self.Sy
        
    def clear_waveform(self):
        """Clear all waveform data."""
        self.waveform_amplitude = np.array([])
        self.waveform_phase = np.array([])
        self.waveform_detuning = np.array([])
        self.duration = 0.0
        self.H_cache = []
        self.is_playing = False
        
    def load_waveform(self, amplitude: np.ndarray, phase: np.ndarray, 
                     duration: float, detuning: Optional[np.ndarray] = None):
        """
        Load arbitrary waveform data.
        
        Args:
            amplitude: Array of Rabi frequencies Ω(t) in rad/s
            phase: Array of phases φ(t) in radians
            duration: Total waveform duration in seconds
            detuning: Optional array of detunings Δ(t) in rad/s
        """
        n_samples = int(duration * self.sample_rate)
        
        # Resample if necessary
        if len(amplitude) != n_samples:
            amplitude = np.interp(
                np.linspace(0, duration, n_samples),
                np.linspace(0, duration, len(amplitude)),
                amplitude
            )
        if len(phase) != n_samples:
            phase = np.interp(
                np.linspace(0, duration, n_samples),
                np.linspace(0, duration, len(phase)),
                phase
            )
            
        self.waveform_amplitude = amplitude
        self.waveform_phase = phase
        
        if detuning is not None:
            if len(detuning) != n_samples:
                detuning = np.interp(
                    np.linspace(0, duration, n_samples),
                    np.linspace(0, duration, len(detuning)),
                    detuning
                )
            self.waveform_detuning = detuning
        else:
            self.waveform_detuning = np.zeros(n_samples)
            
        self.duration = duration
        
    def add_pulse(self, shape: str, amplitude: float, duration: float, 
                  phase: float = 0.0, **kwargs):
        """
        Add a pulse to the waveform.
        
        Args:
            shape: Pulse shape ('rect', 'gauss', 'sinc', 'blackman', 'hermite')
            amplitude: Peak Rabi frequency in rad/s
            duration: Pulse duration in seconds
            phase: Pulse phase in radians
            **kwargs: Additional parameters for specific pulse shapes
        """
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples)
        
        # Generate pulse envelope
        if shape == 'rect':
            envelope = np.ones(n_samples)
            
        elif shape == 'gauss':
            sigma = kwargs.get('sigma', duration / 6)
            t0 = duration / 2
            envelope = np.exp(-((t - t0) / sigma) ** 2)
            
        elif shape == 'sinc':
            zero_crossings = kwargs.get('zero_crossings', 3)
            t0 = duration / 2
            x = zero_crossings * np.pi * (t - t0) / (duration / 2)
            envelope = np.sinc(x / np.pi)
            
        elif shape == 'blackman':
            envelope = np.blackman(n_samples)
            
        elif shape == 'hermite':
            order = kwargs.get('order', 0)
            sigma = kwargs.get('sigma', duration / 6)
            t0 = duration / 2
            z = (t - t0) / sigma
            if order == 0:
                envelope = np.exp(-z**2 / 2)
            elif order == 1:
                envelope = z * np.exp(-z**2 / 2)
            elif order == 2:
                envelope = (z**2 - 1) * np.exp(-z**2 / 2)
            else:
                raise ValueError(f"Hermite order {order} not implemented")
                
        else:
            raise ValueError(f"Unknown pulse shape: {shape}")
            
        # Apply amplitude and phase
        pulse_amplitude = amplitude * envelope
        pulse_phase = np.ones(n_samples) * phase
        pulse_detuning = np.zeros(n_samples)
        
        # Append to existing waveform
        self.waveform_amplitude = np.concatenate([self.waveform_amplitude, pulse_amplitude])
        self.waveform_phase = np.concatenate([self.waveform_phase, pulse_phase])
        self.waveform_detuning = np.concatenate([self.waveform_detuning, pulse_detuning])
        self.duration += duration
        
    def add_delay(self, duration: float):
        """
        Add a delay (zero amplitude) to the waveform.
        
        Args:
            duration: Delay duration in seconds
        """
        n_samples = int(duration * self.sample_rate)
        self.waveform_amplitude = np.concatenate([
            self.waveform_amplitude, 
            np.zeros(n_samples)
        ])
        self.waveform_phase = np.concatenate([
            self.waveform_phase, 
            np.zeros(n_samples)
        ])
        self.waveform_detuning = np.concatenate([
            self.waveform_detuning, 
            np.zeros(n_samples)
        ])
        self.duration += duration
        
    def start(self, at_time: float = 0.0):
        """
        Start waveform playback.
        
        Args:
            at_time: Simulation time to start playback
        """
        if len(self.waveform_amplitude) == 0:
            warnings.warn("No waveform loaded, cannot start playback")
            return
            
        self.is_playing = True
        self.start_time = at_time
        self.loop_count = 0
        
    def stop(self):
        """Stop waveform playback."""
        self.is_playing = False
        
    def get_hamiltonian_contribution(self, t: float) -> np.ndarray:
        """
        Get the MW Hamiltonian contribution at time t.
        
        Args:
            t: Current simulation time in seconds
            
        Returns:
            18x18 complex Hamiltonian matrix
        """
        # Return zero if not playing
        if not self.is_playing:
            return np.zeros((18, 18), dtype=complex)
            
        # Calculate relative time
        t_rel = t - self.start_time
        
        # Handle loop mode
        if self.loop_mode and t_rel >= self.duration:
            self.loop_count = int(t_rel / self.duration)
            t_rel = t_rel % self.duration
            
        # Return zero if outside waveform duration
        if t_rel < 0 or (t_rel >= self.duration and not self.loop_mode):
            return np.zeros((18, 18), dtype=complex)
            
        # Get waveform values at current time
        idx = int(t_rel * self.sample_rate)
        if idx >= len(self.waveform_amplitude):
            idx = len(self.waveform_amplitude) - 1
            
        Omega = self.waveform_amplitude[idx]
        phi = self.waveform_phase[idx]
        Delta = self.waveform_detuning[idx]
        
        # Build 3x3 MW Hamiltonian
        H_mw_3x3 = self._build_mw_hamiltonian_3x3(Omega, phi, Delta)
        
        # Expand to 18x18
        H_mw_18x18 = self._expand_to_18x18(H_mw_3x3)
        
        return H_mw_18x18
        
    def _build_mw_hamiltonian_3x3(self, Omega: float, phi: float, 
                                   Delta: float) -> np.ndarray:
        """
        Build 3x3 MW Hamiltonian in rotating frame.
        
        Args:
            Omega: Rabi frequency in rad/s
            phi: Phase in radians
            Delta: Detuning in rad/s
            
        Returns:
            3x3 complex Hamiltonian matrix
        """
        # Rotating wave approximation
        # H = (ℏ/2) * Ω * (S+ e^(-iφ) + S- e^(iφ)) + ℏ * Δ * Sz
        
        # Note: We work in units where ℏ = 1
        H_mw = (Omega / 2) * (
            self.S_plus * np.exp(-1j * phi) + 
            self.S_minus * np.exp(1j * phi)
        )
        
        # Add detuning term
        H_mw += Delta * self.Sz
        
        return H_mw
        
    def _expand_to_18x18(self, H_3x3: np.ndarray) -> np.ndarray:
        """
        Expand 3x3 electron spin Hamiltonian to full 18x18 space.
        
        The 18-dimensional space is:
        {|g,ms,mI⟩} ⊗ {|e,ms,mI⟩}
        where ms ∈ {+1,0,-1} and mI ∈ {+1,0,-1}
        
        Args:
            H_3x3: 3x3 electron spin Hamiltonian
            
        Returns:
            18x18 Hamiltonian in full space
        """
        # Identity for nuclear spin (3x3)
        I_nuc = np.eye(3, dtype=complex)
        
        # Build 9x9 for ground state manifold
        H_9x9 = np.kron(H_3x3, I_nuc)
        
        # Embed in 18x18 (ground and excited state manifolds)
        H_18x18 = np.zeros((18, 18), dtype=complex)
        H_18x18[:9, :9] = H_9x9  # Only acts on ground state
        
        return H_18x18
        
    def get_info(self) -> dict:
        """
        Get current AWG status and parameters.
        
        Returns:
            Dictionary with AWG information
        """
        return {
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'n_samples': len(self.waveform_amplitude),
            'is_playing': self.is_playing,
            'start_time': self.start_time,
            'loop_mode': self.loop_mode,
            'loop_count': self.loop_count,
            'power_dbm': self.power_dbm,
            'base_frequency': self.base_frequency
        }
        
    def set_power(self, power_dbm: float):
        """
        Set output power (affects Rabi frequency scaling).
        
        Args:
            power_dbm: Power in dBm
        """
        self.power_dbm = power_dbm
        
    def set_frequency(self, frequency: float):
        """
        Set base MW frequency.
        
        Args:
            frequency: Frequency in Hz
        """
        self.base_frequency = frequency
        # Calculate detuning from NV resonance
        base_detuning = 2 * np.pi * (frequency - self.nv_frequency)
        # This would be added to any programmed detuning
        
    def enable_loop(self, enable: bool = True):
        """Enable or disable loop mode."""
        self.loop_mode = enable