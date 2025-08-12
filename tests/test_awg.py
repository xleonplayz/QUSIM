#!/usr/bin/env python3
"""
AWG Interface Tests - Comprehensive testing suite for AWG functionality
"""

import sys
import os
sys.path.append('..')
sys.path.append('../src')

import numpy as np
from interfaces.awg import AWGInterface, WaveformLibrary, PulseSequenceBuilder


class TestAWGInterface:
    """Test basic AWG Interface functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.awg = AWGInterface(sample_rate=1e9)
        
    def test_awg_initialization(self):
        """Test AWG proper initialization"""
        assert self.awg.sample_rate == 1e9
        assert self.awg.dt == 1e-9
        assert len(self.awg.waveform_amplitude) == 0
        assert len(self.awg.waveform_phase) == 0
        assert not self.awg.is_playing
        
    def test_clear_waveform(self):
        """Test waveform clearing"""
        # Add some data first
        self.awg.add_pulse('rect', 1e6, 100e-9, 0.0)
        assert len(self.awg.waveform_amplitude) > 0
        
        # Clear and check
        self.awg.clear_waveform()
        assert len(self.awg.waveform_amplitude) == 0
        assert len(self.awg.waveform_phase) == 0
        assert self.awg.duration == 0.0
        assert not self.awg.is_playing
        
    def test_add_rectangular_pulse(self):
        """Test adding rectangular pulse"""
        amplitude = 2 * np.pi * 1e6  # 1 MHz Rabi frequency
        duration = 100e-9  # 100 ns
        phase = np.pi / 4  # 45 degrees
        
        self.awg.add_pulse('rect', amplitude, duration, phase)
        
        assert self.awg.duration == duration
        expected_samples = int(duration * self.awg.sample_rate)
        assert len(self.awg.waveform_amplitude) == expected_samples
        assert len(self.awg.waveform_phase) == expected_samples
        
        # Check values
        assert np.all(self.awg.waveform_amplitude == amplitude)
        assert np.all(self.awg.waveform_phase == phase)
        
    def test_add_gaussian_pulse(self):
        """Test adding Gaussian pulse"""
        amplitude = 2 * np.pi * 1e6
        duration = 100e-9
        
        self.awg.add_pulse('gauss', amplitude, duration, sigma=duration/6)
        
        assert self.awg.duration == duration
        expected_samples = int(duration * self.awg.sample_rate)
        assert len(self.awg.waveform_amplitude) == expected_samples
        
        # Check Gaussian shape
        max_amplitude = np.max(self.awg.waveform_amplitude)
        assert np.isclose(max_amplitude, amplitude, rtol=0.01)
        
    def test_add_delay(self):
        """Test adding delay"""
        delay_duration = 50e-9
        
        self.awg.add_delay(delay_duration)
        
        assert self.awg.duration == delay_duration
        expected_samples = int(delay_duration * self.awg.sample_rate)
        assert len(self.awg.waveform_amplitude) == expected_samples
        assert np.all(self.awg.waveform_amplitude == 0)
        
    def test_multiple_pulses(self):
        """Test adding multiple pulses in sequence"""
        pulse1_duration = 50e-9
        delay_duration = 100e-9
        pulse2_duration = 75e-9
        
        self.awg.add_pulse('rect', 1e6, pulse1_duration)
        self.awg.add_delay(delay_duration)
        self.awg.add_pulse('gauss', 2e6, pulse2_duration)
        
        total_duration = pulse1_duration + delay_duration + pulse2_duration
        assert np.isclose(self.awg.duration, total_duration)
        
    def test_start_stop(self):
        """Test AWG start/stop functionality"""
        self.awg.add_pulse('rect', 1e6, 100e-9)
        
        # Initially not playing
        assert not self.awg.is_playing
        
        # Start playback
        start_time = 1e-6  # 1 μs
        self.awg.start(at_time=start_time)
        assert self.awg.is_playing
        assert self.awg.start_time == start_time
        
        # Stop playback
        self.awg.stop()
        assert not self.awg.is_playing
        
    def test_hamiltonian_contribution_idle(self):
        """Test Hamiltonian contribution when AWG is idle"""
        H = self.awg.get_hamiltonian_contribution(0.0)
        
        assert H.shape == (18, 18)
        assert np.allclose(H, np.zeros((18, 18)))
        
    def test_hamiltonian_contribution_active(self):
        """Test Hamiltonian contribution during pulse"""
        amplitude = 2 * np.pi * 1e6
        duration = 100e-9
        
        self.awg.add_pulse('rect', amplitude, duration)
        self.awg.start(at_time=0.0)
        
        # During pulse
        t_middle = duration / 2
        H = self.awg.get_hamiltonian_contribution(t_middle)
        
        assert H.shape == (18, 18)
        assert not np.allclose(H, np.zeros((18, 18)))  # Should be non-zero
        
        # After pulse
        t_after = duration + 1e-9
        H_after = self.awg.get_hamiltonian_contribution(t_after)
        assert np.allclose(H_after, np.zeros((18, 18)))  # Should be zero
        
    def test_spin_operators(self):
        """Test that spin operators are properly initialized"""
        assert self.awg.Sx.shape == (3, 3)
        assert self.awg.Sy.shape == (3, 3)
        assert self.awg.Sz.shape == (3, 3)
        
        # Test commutation relations
        comm_xy = self.awg.Sx @ self.awg.Sy - self.awg.Sy @ self.awg.Sx
        expected_comm = 1j * self.awg.Sz
        assert np.allclose(comm_xy, expected_comm, atol=1e-10)
        
    def test_info(self):
        """Test get_info method"""
        info = self.awg.get_info()
        
        assert 'sample_rate' in info
        assert 'duration' in info
        assert 'n_samples' in info
        assert 'is_playing' in info
        assert info['sample_rate'] == 1e9
        

class TestWaveformLibrary:
    """Test Waveform Library functions"""
    
    def test_rectangular(self):
        """Test rectangular pulse generation"""
        amplitude = 1e6
        duration = 100e-9
        sample_rate = 1e9
        phase = np.pi/4
        
        amp, ph = WaveformLibrary.rectangular(amplitude, duration, sample_rate, phase)
        
        expected_samples = int(duration * sample_rate)
        assert len(amp) == expected_samples
        assert len(ph) == expected_samples
        assert np.all(amp == amplitude)
        assert np.all(ph == phase)
        
    def test_gaussian(self):
        """Test Gaussian pulse generation"""
        amplitude = 1e6
        duration = 100e-9
        sample_rate = 1e9
        
        amp, ph = WaveformLibrary.gaussian(amplitude, duration, sample_rate)
        
        expected_samples = int(duration * sample_rate)
        assert len(amp) == expected_samples
        assert len(ph) == expected_samples
        
        # Should peak in the middle
        max_idx = np.argmax(amp)
        expected_max_idx = expected_samples // 2
        assert abs(max_idx - expected_max_idx) < 5  # Within 5 samples
        
        # Maximum should be close to amplitude
        assert np.isclose(np.max(amp), amplitude, rtol=0.01)
        
    def test_sinc(self):
        """Test sinc pulse generation"""
        amplitude = 1e6
        duration = 100e-9
        sample_rate = 1e9
        
        amp, ph = WaveformLibrary.sinc(amplitude, duration, sample_rate)
        
        expected_samples = int(duration * sample_rate)
        assert len(amp) == expected_samples
        assert len(ph) == expected_samples
        
    def test_composite_pulse(self):
        """Test composite pulse creation"""
        pulses = [
            {'shape': 'rect', 'amplitude': 1e6, 'duration': 50e-9, 'phase': 0},
            {'shape': 'delay', 'duration': 25e-9},
            {'shape': 'gauss', 'amplitude': 2e6, 'duration': 50e-9, 'phase': np.pi/2}
        ]
        
        amp, ph = WaveformLibrary.composite_pulse(pulses, sample_rate=1e9)
        
        total_duration = 50e-9 + 25e-9 + 50e-9
        expected_samples = int(total_duration * 1e9)
        assert len(amp) == expected_samples
        assert len(ph) == expected_samples
        

class TestPulseSequenceBuilder:
    """Test Pulse Sequence Builder"""
    
    def test_pi_pulse(self):
        """Test π pulse generation"""
        rabi_freq_mhz = 1.0
        pulse = PulseSequenceBuilder.pi_pulse(rabi_freq_mhz, 'x')
        
        assert pulse['shape'] == 'rect'
        expected_duration = np.pi / (2 * np.pi * rabi_freq_mhz * 1e6)
        assert np.isclose(pulse['duration'], expected_duration)
        assert pulse['phase'] == 0  # x-axis
        
    def test_pi_half_pulse(self):
        """Test π/2 pulse generation"""
        rabi_freq_mhz = 1.0
        pulse = PulseSequenceBuilder.pi_half_pulse(rabi_freq_mhz, 'y')
        
        expected_duration = np.pi / (4 * np.pi * rabi_freq_mhz * 1e6)
        assert np.isclose(pulse['duration'], expected_duration)
        assert pulse['phase'] == np.pi/2  # y-axis
        
    def test_delay(self):
        """Test delay generation"""
        duration_ns = 500
        delay = PulseSequenceBuilder.delay(duration_ns)
        
        assert delay['shape'] == 'delay'
        assert delay['amplitude'] == 0.0
        assert delay['duration'] == duration_ns * 1e-9
        
    def test_ramsey_sequence(self):
        """Test Ramsey sequence generation"""
        tau_ns = 500
        rabi_freq_mhz = 1.0
        
        sequence = PulseSequenceBuilder.ramsey_sequence(tau_ns, rabi_freq_mhz)
        
        assert len(sequence) == 3
        assert sequence[0]['shape'] == 'rect'  # First π/2
        assert sequence[1]['shape'] == 'delay'  # Free evolution
        assert sequence[2]['shape'] == 'rect'  # Second π/2
        
        # Check delay duration
        assert sequence[1]['duration'] == tau_ns * 1e-9
        
    def test_echo_sequence(self):
        """Test echo sequence generation"""
        tau_ns = 1000
        rabi_freq_mhz = 1.0
        
        sequence = PulseSequenceBuilder.echo_sequence(tau_ns, rabi_freq_mhz)
        
        assert len(sequence) == 5
        # π/2 - τ - π - τ - π/2 structure
        pulse_types = [s['shape'] for s in sequence]
        expected_types = ['rect', 'delay', 'rect', 'delay', 'rect']
        assert pulse_types == expected_types
        
        # Check π pulse is on y-axis (different from π/2 pulses)
        assert sequence[2]['phase'] == np.pi/2  # π pulse on y
        
    def test_rabi_sequence(self):
        """Test Rabi oscillation sequence"""
        max_duration_ns = 1000
        steps = 10
        
        sequences = PulseSequenceBuilder.rabi_sequence(max_duration_ns, steps)
        
        assert len(sequences) == steps
        
        # Check duration progression
        for i, seq in enumerate(sequences[1:], 1):  # Skip first (empty)
            if seq:  # Non-empty sequence
                expected_duration = i * max_duration_ns * 1e-9 / (steps - 1)
                assert np.isclose(seq[0]['duration'], expected_duration, rtol=0.1)
                
    def test_custom_rotation(self):
        """Test custom rotation pulse"""
        theta = np.pi / 3  # 60 degrees
        phi = np.pi / 4   # 45 degrees phase
        rabi_freq_mhz = 2.0
        
        pulse = PulseSequenceBuilder.custom_rotation(theta, phi, rabi_freq_mhz)
        
        expected_duration = theta / (2 * np.pi * rabi_freq_mhz * 1e6)
        assert np.isclose(pulse['duration'], expected_duration)
        assert pulse['phase'] == phi


class TestAWGIntegration:
    """Integration tests for AWG system"""
    
    def test_pi_pulse_rotation(self):
        """Test that π pulse produces correct rotation"""
        awg = AWGInterface(sample_rate=1e9)
        
        # Create π pulse
        rabi_freq_mhz = 1.0
        pulse = PulseSequenceBuilder.pi_pulse(rabi_freq_mhz, 'x')
        
        # Load into AWG
        awg.clear_waveform()
        awg.add_pulse(**pulse)
        awg.start(at_time=0.0)
        
        # Check Hamiltonian during pulse
        t_middle = pulse['duration'] / 2
        H = awg.get_hamiltonian_contribution(t_middle)
        
        # Should be non-zero and have correct structure
        assert H.shape == (18, 18)
        assert not np.allclose(H, np.zeros((18, 18)))
        
        # Should only affect ground state manifold (first 9x9 block)
        assert np.allclose(H[9:, 9:], np.zeros((9, 9)))  # Excited state block should be zero
        assert np.allclose(H[:9, 9:], np.zeros((9, 9)))  # Cross terms should be zero
        assert np.allclose(H[9:, :9], np.zeros((9, 9)))  # Cross terms should be zero
        
    def test_sequence_timing(self):
        """Test sequence timing is correct"""
        awg = AWGInterface(sample_rate=1e9)
        
        # Create Ramsey sequence
        tau_ns = 500
        sequence = PulseSequenceBuilder.ramsey_sequence(tau_ns, 1.0)
        
        # Load sequence
        awg.clear_waveform()
        for pulse in sequence:
            if pulse['shape'] == 'delay':
                awg.add_delay(pulse['duration'])
            else:
                awg.add_pulse(**pulse)
                
        # Calculate expected total duration
        total_expected = sum(s['duration'] for s in sequence)
        assert np.isclose(awg.duration, total_expected)
        
        # Test playback timing
        awg.start(at_time=0.0)
        
        # Should be active during sequence
        t_during = total_expected / 2
        H_during = awg.get_hamiltonian_contribution(t_during)
        
        # May be zero during delay, but at least check shape
        assert H_during.shape == (18, 18)
        
        # Should be zero after sequence
        t_after = total_expected + 1e-9
        H_after = awg.get_hamiltonian_contribution(t_after)
        assert np.allclose(H_after, np.zeros((18, 18)))


def run_tests():
    """Run all AWG tests"""
    print("Running AWG Interface Tests...")
    
    # Test AWG Interface
    print("\\n1. Testing AWG Interface...")
    test_awg = TestAWGInterface()
    
    test_methods = [method for method in dir(test_awg) if method.startswith('test_')]
    for method_name in test_methods:
        test_awg.setup_method()
        method = getattr(test_awg, method_name)
        try:
            method()
            print(f"  ✓ {method_name}")
        except Exception as e:
            print(f"  ✗ {method_name}: {e}")
            
    # Test Waveform Library
    print("\\n2. Testing Waveform Library...")
    test_waveform = TestWaveformLibrary()
    
    test_methods = [method for method in dir(test_waveform) if method.startswith('test_')]
    for method_name in test_methods:
        method = getattr(test_waveform, method_name)
        try:
            method()
            print(f"  ✓ {method_name}")
        except Exception as e:
            print(f"  ✗ {method_name}: {e}")
            
    # Test Pulse Sequence Builder
    print("\\n3. Testing Pulse Sequence Builder...")
    test_sequence = TestPulseSequenceBuilder()
    
    test_methods = [method for method in dir(test_sequence) if method.startswith('test_')]
    for method_name in test_methods:
        method = getattr(test_sequence, method_name)
        try:
            method()
            print(f"  ✓ {method_name}")
        except Exception as e:
            print(f"  ✗ {method_name}: {e}")
            
    # Test Integration
    print("\\n4. Testing AWG Integration...")
    test_integration = TestAWGIntegration()
    
    test_methods = [method for method in dir(test_integration) if method.startswith('test_')]
    for method_name in test_methods:
        method = getattr(test_integration, method_name)
        try:
            method()
            print(f"  ✓ {method_name}")
        except Exception as e:
            print(f"  ✗ {method_name}: {e}")
            
    print("\\nAWG Tests completed!")


if __name__ == "__main__":
    run_tests()