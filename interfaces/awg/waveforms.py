"""
Waveform Library - Pre-defined pulse shapes and modulation patterns
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from scipy import special


class WaveformLibrary:
    """
    Library of pre-defined waveform shapes for MW pulse generation.
    All methods return (amplitude, phase) arrays for specified duration.
    """
    
    @staticmethod
    def rectangular(amplitude: float, duration: float, sample_rate: float = 1e9, 
                   phase: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectangular (square) pulse.
        
        Args:
            amplitude: Peak Rabi frequency in rad/s
            duration: Pulse duration in seconds
            sample_rate: Sampling rate in Hz
            phase: Phase in radians
            
        Returns:
            (amplitude_array, phase_array)
        """
        n_samples = int(duration * sample_rate)
        amp_array = np.full(n_samples, amplitude)
        phase_array = np.full(n_samples, phase)
        return amp_array, phase_array
        
    @staticmethod
    def gaussian(amplitude: float, duration: float, sample_rate: float = 1e9,
                phase: float = 0.0, sigma: Optional[float] = None,
                truncation: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gaussian pulse with specified width.
        
        Args:
            amplitude: Peak Rabi frequency in rad/s
            duration: Pulse duration in seconds
            sample_rate: Sampling rate in Hz
            phase: Phase in radians
            sigma: Gaussian width (default: duration/6 for ~4σ truncation)
            truncation: Truncation factor in units of σ
            
        Returns:
            (amplitude_array, phase_array)
        """
        if sigma is None:
            sigma = duration / (2 * truncation)
            
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples)
        t0 = duration / 2
        
        envelope = np.exp(-((t - t0) / sigma) ** 2 / 2)
        amp_array = amplitude * envelope
        phase_array = np.full(n_samples, phase)
        
        return amp_array, phase_array
        
    @staticmethod
    def sinc(amplitude: float, duration: float, sample_rate: float = 1e9,
            phase: float = 0.0, zero_crossings: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sinc pulse for broadband excitation.
        
        Args:
            amplitude: Peak Rabi frequency in rad/s
            duration: Pulse duration in seconds
            sample_rate: Sampling rate in Hz
            phase: Phase in radians
            zero_crossings: Number of zero crossings on each side
            
        Returns:
            (amplitude_array, phase_array)
        """
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples)
        t0 = duration / 2
        
        # Normalize time to [-1, 1] range
        x = zero_crossings * np.pi * (t - t0) / (duration / 2)
        
        # Sinc function (numpy's sinc is sin(πx)/(πx))
        envelope = np.sinc(x / np.pi)
        amp_array = amplitude * envelope
        phase_array = np.full(n_samples, phase)
        
        return amp_array, phase_array
        
    @staticmethod
    def blackman(amplitude: float, duration: float, sample_rate: float = 1e9,
                phase: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blackman window pulse for smooth turn-on/off.
        
        Args:
            amplitude: Peak Rabi frequency in rad/s
            duration: Pulse duration in seconds
            sample_rate: Sampling rate in Hz
            phase: Phase in radians
            
        Returns:
            (amplitude_array, phase_array)
        """
        n_samples = int(duration * sample_rate)
        envelope = np.blackman(n_samples)
        amp_array = amplitude * envelope
        phase_array = np.full(n_samples, phase)
        
        return amp_array, phase_array
        
    @staticmethod
    def hermite(amplitude: float, duration: float, sample_rate: float = 1e9,
               phase: float = 0.0, order: int = 0, 
               sigma: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hermite-Gaussian pulse shapes.
        
        Args:
            amplitude: Peak Rabi frequency in rad/s
            duration: Pulse duration in seconds
            sample_rate: Sampling rate in Hz
            phase: Phase in radians
            order: Hermite polynomial order (0, 1, 2, ...)
            sigma: Gaussian width (default: duration/6)
            
        Returns:
            (amplitude_array, phase_array)
        """
        if sigma is None:
            sigma = duration / 6
            
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples)
        t0 = duration / 2
        z = (t - t0) / sigma
        
        # Hermite polynomials
        if order == 0:
            hermite = np.ones_like(z)
        elif order == 1:
            hermite = 2 * z
        elif order == 2:
            hermite = 4 * z**2 - 2
        elif order == 3:
            hermite = 8 * z**3 - 12 * z
        else:
            # Use scipy for higher orders
            hermite = special.hermite(order)(z)
            
        # Gaussian envelope
        gaussian = np.exp(-z**2 / 2)
        
        envelope = hermite * gaussian
        # Normalize to unit peak
        envelope = envelope / np.max(np.abs(envelope))
        
        amp_array = amplitude * envelope
        phase_array = np.full(n_samples, phase)
        
        return amp_array, phase_array
        
    @staticmethod
    def chirped_gaussian(amplitude: float, duration: float, sample_rate: float = 1e9,
                        phase: float = 0.0, chirp_rate: float = 0.0,
                        sigma: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Frequency-chirped Gaussian pulse.
        
        Args:
            amplitude: Peak Rabi frequency in rad/s
            duration: Pulse duration in seconds
            sample_rate: Sampling rate in Hz
            phase: Initial phase in radians
            chirp_rate: Linear frequency chirp rate in rad/s²
            sigma: Gaussian width (default: duration/6)
            
        Returns:
            (amplitude_array, phase_array)
        """
        if sigma is None:
            sigma = duration / 6
            
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples)
        t0 = duration / 2
        
        # Gaussian envelope
        envelope = np.exp(-((t - t0) / sigma) ** 2 / 2)
        amp_array = amplitude * envelope
        
        # Chirped phase
        phase_array = phase + chirp_rate * (t - t0)**2 / 2
        
        return amp_array, phase_array
        
    @staticmethod
    def composite_pulse(pulses: list, sample_rate: float = 1e9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine multiple pulse segments into composite waveform.
        
        Args:
            pulses: List of pulse dictionaries with keys:
                   'shape', 'amplitude', 'duration', 'phase', and shape-specific params
            sample_rate: Sampling rate in Hz
            
        Returns:
            (amplitude_array, phase_array)
        """
        total_amplitude = np.array([])
        total_phase = np.array([])
        
        for pulse in pulses:
            shape = pulse['shape']
            
            if shape == 'rect':
                amp, phase = WaveformLibrary.rectangular(
                    pulse['amplitude'], pulse['duration'], sample_rate, 
                    pulse.get('phase', 0.0)
                )
            elif shape == 'gauss':
                amp, phase = WaveformLibrary.gaussian(
                    pulse['amplitude'], pulse['duration'], sample_rate,
                    pulse.get('phase', 0.0), pulse.get('sigma')
                )
            elif shape == 'sinc':
                amp, phase = WaveformLibrary.sinc(
                    pulse['amplitude'], pulse['duration'], sample_rate,
                    pulse.get('phase', 0.0), pulse.get('zero_crossings', 3)
                )
            elif shape == 'blackman':
                amp, phase = WaveformLibrary.blackman(
                    pulse['amplitude'], pulse['duration'], sample_rate,
                    pulse.get('phase', 0.0)
                )
            elif shape == 'hermite':
                amp, phase = WaveformLibrary.hermite(
                    pulse['amplitude'], pulse['duration'], sample_rate,
                    pulse.get('phase', 0.0), pulse.get('order', 0),
                    pulse.get('sigma')
                )
            elif shape == 'delay':
                n_samples = int(pulse['duration'] * sample_rate)
                amp = np.zeros(n_samples)
                phase = np.zeros(n_samples)
            else:
                raise ValueError(f"Unknown pulse shape: {shape}")
                
            total_amplitude = np.concatenate([total_amplitude, amp])
            total_phase = np.concatenate([total_phase, phase])
            
        return total_amplitude, total_phase
        
    @staticmethod
    def am_modulated(carrier_amplitude: float, duration: float, 
                    modulation_frequency: float, modulation_depth: float = 1.0,
                    sample_rate: float = 1e9, phase: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Amplitude-modulated pulse.
        
        Args:
            carrier_amplitude: Carrier Rabi frequency in rad/s
            duration: Pulse duration in seconds
            modulation_frequency: Modulation frequency in Hz
            modulation_depth: Modulation depth (0 to 1)
            sample_rate: Sampling rate in Hz
            phase: Phase in radians
            
        Returns:
            (amplitude_array, phase_array)
        """
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples)
        
        # AM modulation: A(t) = A0 * (1 + m * cos(2πft))
        modulation = 1 + modulation_depth * np.cos(2 * np.pi * modulation_frequency * t)
        amp_array = carrier_amplitude * modulation
        phase_array = np.full(n_samples, phase)
        
        return amp_array, phase_array
        
    @staticmethod
    def fm_modulated(amplitude: float, duration: float, carrier_frequency: float,
                    modulation_frequency: float, frequency_deviation: float,
                    sample_rate: float = 1e9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Frequency-modulated pulse.
        
        Args:
            amplitude: Constant Rabi frequency in rad/s
            duration: Pulse duration in seconds
            carrier_frequency: Carrier frequency in Hz (usually 0 for baseband)
            modulation_frequency: Modulation frequency in Hz
            frequency_deviation: Peak frequency deviation in Hz
            sample_rate: Sampling rate in Hz
            
        Returns:
            (amplitude_array, phase_array)
        """
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples)
        
        # FM: φ(t) = 2π*fc*t + (Δf/fm)*sin(2π*fm*t)
        carrier_phase = 2 * np.pi * carrier_frequency * t
        modulation_phase = (frequency_deviation / modulation_frequency) * \
                          np.sin(2 * np.pi * modulation_frequency * t)
        
        amp_array = np.full(n_samples, amplitude)
        phase_array = carrier_phase + modulation_phase
        
        return amp_array, phase_array
        
    @staticmethod
    def tanh_pulse(amplitude: float, duration: float, sample_rate: float = 1e9,
                  phase: float = 0.0, rise_time: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hyperbolic tangent pulse with smooth edges.
        
        Args:
            amplitude: Peak Rabi frequency in rad/s
            duration: Pulse duration in seconds
            sample_rate: Sampling rate in Hz
            phase: Phase in radians
            rise_time: 10-90% rise time (default: duration/10)
            
        Returns:
            (amplitude_array, phase_array)
        """
        if rise_time is None:
            rise_time = duration / 10
            
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples)
        
        # Tanh parameters for desired rise time
        # 10-90% rise time ≈ 2.2 * τ where τ = rise_time/2.2
        tau = rise_time / 2.2
        
        # Smooth turn-on and turn-off
        t_on = duration * 0.1
        t_off = duration * 0.9
        
        envelope = 0.5 * (
            np.tanh((t - t_on) / tau) - np.tanh((t - t_off) / tau)
        )
        
        # Normalize to unit peak
        envelope = envelope / np.max(envelope)
        
        amp_array = amplitude * envelope
        phase_array = np.full(n_samples, phase)
        
        return amp_array, phase_array
        
    @staticmethod
    def drag_pulse(amplitude: float, duration: float, sample_rate: float = 1e9,
                  phase: float = 0.0, sigma: Optional[float] = None,
                  alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        DRAG (Derivative Removal by Adiabatic Gating) pulse.
        Reduces gate errors due to AC Stark shift.
        
        Args:
            amplitude: Peak Rabi frequency in rad/s
            duration: Pulse duration in seconds
            sample_rate: Sampling rate in Hz
            phase: Phase in radians
            sigma: Gaussian width (default: duration/6)
            alpha: DRAG parameter (typically 0.5)
            
        Returns:
            (amplitude_array, phase_array)
        """
        if sigma is None:
            sigma = duration / 6
            
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples)
        t0 = duration / 2
        dt = 1 / sample_rate
        
        # Gaussian envelope
        gauss = np.exp(-((t - t0) / sigma) ** 2 / 2)
        
        # Derivative of Gaussian
        gauss_deriv = -(t - t0) / sigma**2 * gauss
        
        # I and Q components
        I_component = gauss
        Q_component = alpha * gauss_deriv / amplitude  # Normalized by amplitude
        
        # Combine into amplitude and phase
        amp_envelope = np.sqrt(I_component**2 + (alpha * gauss_deriv / amplitude)**2)
        amp_array = amplitude * amp_envelope
        
        # Phase modulation from Q component
        phase_mod = np.arctan2(Q_component, I_component)
        phase_array = phase + phase_mod
        
        return amp_array, phase_array