"""
Pulse Sequence Builder - High-level sequences for NV center experiments
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .waveforms import WaveformLibrary


class PulseSequenceBuilder:
    """
    Builder class for creating standard NV center pulse sequences.
    All sequences return lists of pulse dictionaries compatible with AWGInterface.
    """
    
    # Standard NV parameters
    DEFAULT_RABI_FREQ_MHZ = 1.0  # 1 MHz Rabi frequency
    DEFAULT_DETUNING_MHZ = 0.0   # On-resonance
    
    @classmethod
    def pi_pulse(cls, rabi_freq_mhz: float = None, axis: str = 'x',
                 shape: str = 'rect', **kwargs) -> Dict[str, Any]:
        """
        Create a π pulse for complete spin flip.
        
        Args:
            rabi_freq_mhz: Rabi frequency in MHz
            axis: Rotation axis ('x', 'y', '-x', '-y')
            shape: Pulse shape ('rect', 'gauss', 'blackman', etc.)
            **kwargs: Additional parameters for pulse shape
            
        Returns:
            Pulse dictionary
        """
        if rabi_freq_mhz is None:
            rabi_freq_mhz = cls.DEFAULT_RABI_FREQ_MHZ
            
        # π rotation requires Ω*t = π
        duration = np.pi / (2 * np.pi * rabi_freq_mhz * 1e6)  # seconds
        amplitude = 2 * np.pi * rabi_freq_mhz * 1e6  # rad/s
        
        # Phase for different axes
        phase_map = {'x': 0, 'y': np.pi/2, '-x': np.pi, '-y': 3*np.pi/2}
        phase = phase_map.get(axis, 0)
        
        return {
            'shape': shape,
            'amplitude': amplitude,
            'duration': duration,
            'phase': phase,
            **kwargs
        }
        
    @classmethod
    def pi_half_pulse(cls, rabi_freq_mhz: float = None, axis: str = 'x',
                     shape: str = 'rect', **kwargs) -> Dict[str, Any]:
        """
        Create a π/2 pulse for coherent superposition.
        
        Args:
            rabi_freq_mhz: Rabi frequency in MHz
            axis: Rotation axis ('x', 'y', '-x', '-y')
            shape: Pulse shape ('rect', 'gauss', 'blackman', etc.)
            **kwargs: Additional parameters for pulse shape
            
        Returns:
            Pulse dictionary
        """
        if rabi_freq_mhz is None:
            rabi_freq_mhz = cls.DEFAULT_RABI_FREQ_MHZ
            
        # π/2 rotation requires Ω*t = π/2
        duration = np.pi / (4 * np.pi * rabi_freq_mhz * 1e6)  # seconds
        amplitude = 2 * np.pi * rabi_freq_mhz * 1e6  # rad/s
        
        # Phase for different axes
        phase_map = {'x': 0, 'y': np.pi/2, '-x': np.pi, '-y': 3*np.pi/2}
        phase = phase_map.get(axis, 0)
        
        return {
            'shape': shape,
            'amplitude': amplitude,
            'duration': duration,
            'phase': phase,
            **kwargs
        }
        
    @classmethod
    def delay(cls, duration_ns: float) -> Dict[str, Any]:
        """
        Create a delay (no pulse).
        
        Args:
            duration_ns: Delay duration in nanoseconds
            
        Returns:
            Delay pulse dictionary
        """
        return {
            'shape': 'delay',
            'amplitude': 0.0,
            'duration': duration_ns * 1e-9,
            'phase': 0.0
        }
        
    @classmethod
    def rabi_sequence(cls, max_duration_ns: float, steps: int = 50,
                     rabi_freq_mhz: float = None, axis: str = 'x',
                     shape: str = 'rect') -> List[List[Dict[str, Any]]]:
        """
        Create a Rabi oscillation measurement sequence.
        
        Args:
            max_duration_ns: Maximum pulse duration in nanoseconds
            steps: Number of duration steps
            rabi_freq_mhz: Rabi frequency in MHz
            axis: Pulse axis
            shape: Pulse shape
            
        Returns:
            List of sequences, each containing one pulse of increasing duration
        """
        if rabi_freq_mhz is None:
            rabi_freq_mhz = cls.DEFAULT_RABI_FREQ_MHZ
            
        amplitude = 2 * np.pi * rabi_freq_mhz * 1e6  # rad/s
        durations = np.linspace(0, max_duration_ns * 1e-9, steps)
        
        phase_map = {'x': 0, 'y': np.pi/2, '-x': np.pi, '-y': 3*np.pi/2}
        phase = phase_map.get(axis, 0)
        
        sequences = []
        for duration in durations:
            if duration > 0:
                pulse = {
                    'shape': shape,
                    'amplitude': amplitude,
                    'duration': duration,
                    'phase': phase
                }
                sequences.append([pulse])
            else:
                sequences.append([])  # No pulse for zero duration
                
        return sequences
        
    @classmethod
    def ramsey_sequence(cls, tau_ns: float, rabi_freq_mhz: float = None,
                       final_phase: float = 0.0, shape: str = 'rect') -> List[Dict[str, Any]]:
        """
        Create a Ramsey interference sequence: π/2 - τ - π/2.
        
        Args:
            tau_ns: Free evolution time in nanoseconds
            rabi_freq_mhz: Rabi frequency in MHz
            final_phase: Phase of second π/2 pulse in radians
            shape: Pulse shape
            
        Returns:
            List of pulse dictionaries
        """
        pi_half_1 = cls.pi_half_pulse(rabi_freq_mhz, 'x', shape)
        delay_pulse = cls.delay(tau_ns)
        pi_half_2 = cls.pi_half_pulse(rabi_freq_mhz, 'x', shape)
        pi_half_2['phase'] = final_phase
        
        return [pi_half_1, delay_pulse, pi_half_2]
        
    @classmethod
    def ramsey_scan(cls, tau_max_ns: float, steps: int = 50,
                   rabi_freq_mhz: float = None, shape: str = 'rect') -> List[List[Dict[str, Any]]]:
        """
        Create a Ramsey scan with varying free evolution times.
        
        Args:
            tau_max_ns: Maximum free evolution time in nanoseconds
            steps: Number of tau steps
            rabi_freq_mhz: Rabi frequency in MHz
            shape: Pulse shape
            
        Returns:
            List of Ramsey sequences with different tau values
        """
        tau_values = np.linspace(0, tau_max_ns, steps)
        
        sequences = []
        for tau in tau_values:
            sequence = cls.ramsey_sequence(tau, rabi_freq_mhz, 0.0, shape)
            sequences.append(sequence)
            
        return sequences
        
    @classmethod
    def echo_sequence(cls, tau_ns: float, rabi_freq_mhz: float = None,
                     shape: str = 'rect') -> List[Dict[str, Any]]:
        """
        Create a spin echo sequence: π/2 - τ - π - τ - π/2.
        
        Args:
            tau_ns: Half of the echo time in nanoseconds
            rabi_freq_mhz: Rabi frequency in MHz
            shape: Pulse shape
            
        Returns:
            List of pulse dictionaries
        """
        pi_half = cls.pi_half_pulse(rabi_freq_mhz, 'x', shape)
        delay_pulse = cls.delay(tau_ns)
        pi_pulse = cls.pi_pulse(rabi_freq_mhz, 'y', shape)  # π pulse on y-axis
        
        return [pi_half, delay_pulse, pi_pulse, delay_pulse, pi_half]
        
    @classmethod
    def cpmg_sequence(cls, tau_ns: float, n_pulses: int, rabi_freq_mhz: float = None,
                     shape: str = 'rect') -> List[Dict[str, Any]]:
        """
        Create a CPMG (Carr-Purcell-Meiboom-Gill) sequence.
        
        Args:
            tau_ns: Inter-pulse delay in nanoseconds
            n_pulses: Number of π pulses
            rabi_freq_mhz: Rabi frequency in MHz
            shape: Pulse shape
            
        Returns:
            List of pulse dictionaries
        """
        pi_half_x = cls.pi_half_pulse(rabi_freq_mhz, 'x', shape)
        pi_y = cls.pi_pulse(rabi_freq_mhz, 'y', shape)
        delay_pulse = cls.delay(tau_ns)
        
        # Start with π/2 on x
        sequence = [pi_half_x]
        
        # Add τ - π_y - τ blocks
        for i in range(n_pulses):
            sequence.extend([delay_pulse, pi_y, delay_pulse])
            
        # Final π/2 on x
        sequence.append(pi_half_x)
        
        return sequence
        
    @classmethod
    def dd_sequence(cls, sequence_type: str, tau_ns: float, n_pulses: int = 1,
                   rabi_freq_mhz: float = None, shape: str = 'rect') -> List[Dict[str, Any]]:
        """
        Create dynamical decoupling sequences.
        
        Args:
            sequence_type: Type of sequence ('XY4', 'XY8', 'KDD', etc.)
            tau_ns: Basic delay time in nanoseconds
            n_pulses: Number of sequence repetitions
            rabi_freq_mhz: Rabi frequency in MHz
            shape: Pulse shape
            
        Returns:
            List of pulse dictionaries
        """
        delay_pulse = cls.delay(tau_ns)
        
        if sequence_type.upper() == 'XY4':
            # XY4: π/2 - [τ - πX - τ - πY - τ - πX - τ - πY - τ] - π/2
            pi_half_x = cls.pi_half_pulse(rabi_freq_mhz, 'x', shape)
            pi_x = cls.pi_pulse(rabi_freq_mhz, 'x', shape)
            pi_y = cls.pi_pulse(rabi_freq_mhz, 'y', shape)
            
            sequence = [pi_half_x]
            
            for _ in range(n_pulses):
                xy4_block = [
                    delay_pulse, pi_x, delay_pulse, pi_y,
                    delay_pulse, pi_x, delay_pulse, pi_y, delay_pulse
                ]
                sequence.extend(xy4_block)
                
            sequence.append(pi_half_x)
            
        elif sequence_type.upper() == 'XY8':
            # XY8: Extended XY sequence
            pi_half_x = cls.pi_half_pulse(rabi_freq_mhz, 'x', shape)
            pi_x = cls.pi_pulse(rabi_freq_mhz, 'x', shape)
            pi_y = cls.pi_pulse(rabi_freq_mhz, 'y', shape)
            pi_neg_x = cls.pi_pulse(rabi_freq_mhz, '-x', shape)
            pi_neg_y = cls.pi_pulse(rabi_freq_mhz, '-y', shape)
            
            sequence = [pi_half_x]
            
            for _ in range(n_pulses):
                xy8_block = [
                    delay_pulse, pi_x, delay_pulse, pi_y, delay_pulse,
                    pi_x, delay_pulse, pi_y, delay_pulse, pi_neg_y,
                    delay_pulse, pi_neg_x, delay_pulse, pi_neg_y,
                    delay_pulse, pi_neg_x, delay_pulse
                ]
                sequence.extend(xy8_block)
                
            sequence.append(pi_half_x)
            
        else:
            raise ValueError(f"Unknown DD sequence type: {sequence_type}")
            
        return sequence
        
    @classmethod
    def composite_pi_pulse(cls, sequence_type: str, rabi_freq_mhz: float = None,
                          shape: str = 'rect') -> List[Dict[str, Any]]:
        """
        Create composite π pulses for robust excitation.
        
        Args:
            sequence_type: Type of composite pulse ('BB1', 'CORPSE', 'SK1')
            rabi_freq_mhz: Rabi frequency in MHz
            shape: Pulse shape
            
        Returns:
            List of pulse dictionaries forming composite π pulse
        """
        if sequence_type.upper() == 'BB1':
            # BB1: π_φ1 - π_φ2 - π_φ3 with specific phases
            phi1 = np.pi
            phi2 = 0
            phi3 = np.pi
            
            pulse1 = cls.pi_pulse(rabi_freq_mhz, 'x', shape)
            pulse2 = cls.pi_pulse(rabi_freq_mhz, 'x', shape)
            pulse3 = cls.pi_pulse(rabi_freq_mhz, 'x', shape)
            
            pulse1['phase'] = phi1
            pulse2['phase'] = phi2  
            pulse3['phase'] = phi3
            
            return [pulse1, pulse2, pulse3]
            
        elif sequence_type.upper() == 'CORPSE':
            # CORPSE: compensating for off-resonance with a pulse sequence
            # θ_1 = π + 2*arcsin(Δ/(2*Ω))
            # θ_2 = 2*π
            # θ_3 = π - 2*arcsin(Δ/(2*Ω))
            # Assuming small detuning, approximate as π - π - π
            
            pulse1 = cls.pi_pulse(rabi_freq_mhz, 'x', shape)
            pulse2 = cls.pi_pulse(rabi_freq_mhz, '-x', shape)
            pulse3 = cls.pi_pulse(rabi_freq_mhz, 'x', shape)
            
            return [pulse1, pulse2, pulse3]
            
        else:
            raise ValueError(f"Unknown composite pulse type: {sequence_type}")
            
    @classmethod
    def sensing_sequence(cls, sensing_type: str, tau_ns: float, 
                        n_sensing_pulses: int = 1, rabi_freq_mhz: float = None,
                        shape: str = 'rect') -> List[Dict[str, Any]]:
        """
        Create sensing sequences for AC magnetometry.
        
        Args:
            sensing_type: Type of sensing ('AC', 'DC', 'double_quantum')
            tau_ns: Sensing time in nanoseconds
            n_sensing_pulses: Number of sensing periods
            rabi_freq_mhz: Rabi frequency in MHz
            shape: Pulse shape
            
        Returns:
            List of pulse dictionaries
        """
        if sensing_type.upper() == 'AC':
            # Simple AC sensing: π/2 - τ - π/2
            return cls.ramsey_sequence(tau_ns, rabi_freq_mhz, 0.0, shape)
            
        elif sensing_type.upper() == 'DC':
            # DC sensing with echo: π/2 - τ - π - τ - π/2
            return cls.echo_sequence(tau_ns/2, rabi_freq_mhz, shape)
            
        elif sensing_type.upper() == 'DOUBLE_QUANTUM':
            # Double quantum coherence sequence
            # π/2 - τ/2 - π - τ/2 - π/2
            pi_half_x = cls.pi_half_pulse(rabi_freq_mhz, 'x', shape)
            pi_y = cls.pi_pulse(rabi_freq_mhz, 'y', shape)
            delay_pulse = cls.delay(tau_ns/2)
            
            return [pi_half_x, delay_pulse, pi_y, delay_pulse, pi_half_x]
            
        else:
            raise ValueError(f"Unknown sensing type: {sensing_type}")
            
    @classmethod
    def calibration_sequence(cls, calibration_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Create calibration sequences for system characterization.
        
        Args:
            calibration_type: Type of calibration ('rabi', 'ramsey', 't1', 't2', 't2echo')
            **kwargs: Additional parameters specific to calibration type
            
        Returns:
            List of pulse dictionaries or list of sequences
        """
        if calibration_type.lower() == 'rabi':
            max_duration_ns = kwargs.get('max_duration_ns', 1000)
            steps = kwargs.get('steps', 50)
            return cls.rabi_sequence(max_duration_ns, steps)
            
        elif calibration_type.lower() == 'ramsey':
            tau_max_ns = kwargs.get('tau_max_ns', 2000)
            steps = kwargs.get('steps', 50)
            return cls.ramsey_scan(tau_max_ns, steps)
            
        elif calibration_type.lower() == 't2echo':
            tau_max_ns = kwargs.get('tau_max_ns', 10000)
            steps = kwargs.get('steps', 50)
            tau_values = np.linspace(100, tau_max_ns, steps)
            
            sequences = []
            for tau in tau_values:
                sequence = cls.echo_sequence(tau/2)
                sequences.append(sequence)
            return sequences
            
        else:
            raise ValueError(f"Unknown calibration type: {calibration_type}")
            
    @classmethod
    def custom_rotation(cls, theta: float, phi: float, rabi_freq_mhz: float = None,
                       shape: str = 'rect', **kwargs) -> Dict[str, Any]:
        """
        Create custom rotation pulse with arbitrary angle and phase.
        
        Args:
            theta: Rotation angle in radians
            phi: Rotation phase in radians
            rabi_freq_mhz: Rabi frequency in MHz
            shape: Pulse shape
            **kwargs: Additional pulse parameters
            
        Returns:
            Pulse dictionary
        """
        if rabi_freq_mhz is None:
            rabi_freq_mhz = cls.DEFAULT_RABI_FREQ_MHZ
            
        duration = theta / (2 * np.pi * rabi_freq_mhz * 1e6)
        amplitude = 2 * np.pi * rabi_freq_mhz * 1e6
        
        return {
            'shape': shape,
            'amplitude': amplitude,
            'duration': duration,
            'phase': phi,
            **kwargs
        }