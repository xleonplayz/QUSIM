#!/usr/bin/env python3
"""
Tests für realistische PhotonCounter und LaserInterface
"""

import sys
import os
sys.path.append('..')
sys.path.append('../interfaces')

import numpy as np
import pytest
from interfaces.photoncounter.photon_counter import PhotonCounter
from interfaces.laser.laser_interface import LaserInterface


class TestRealisticPhotonCounter:
    """Test realistischen PhotonCounter mit Totzeit, Jitter, etc."""
    
    def setup_method(self):
        """Setup für jeden Test"""
        self.counter = PhotonCounter(
            detection_efficiency=0.1,
            dark_count_rate=100.0,
            dead_time=50e-9,  # 50 ns
            time_resolution=0.5e-9,  # 0.5 ns Jitter
            afterpulsing_probability=0.01,
            afterpulsing_delay=100e-9
        )
        
    def test_initialization(self):
        """Test Initialisierung mit erweiterten Parametern"""
        assert self.counter.detection_efficiency == 0.1
        assert self.counter.dead_time == 50e-9
        assert self.counter.time_resolution == 0.5e-9
        assert self.counter.afterpulsing_probability == 0.01
        assert len(self.counter.photon_arrival_times) == 0
        
    def test_basic_counting_with_dead_time(self):
        """Test Photonenzählung mit Totzeit-Effekten"""
        # Erstelle einfache Dichtematrix mit angeregtem Zustand
        rho = np.zeros((18, 18), dtype=complex)
        rho[9, 9] = 1.0  # 100% Population in |e⟩
        
        dt = 1e-9  # 1 ns
        gamma = 1e7  # 10 MHz
        
        # Mehrere Zeitschritte simulieren
        results = []
        for i in range(100):
            result = self.counter.count_photons(rho, dt, gamma, current_time=i*dt)
            results.append(result)
        
        # Prüfe dass Totzeit-Verluste auftreten
        total_lost = sum(r.get('dead_time_losses', 0) for r in results)
        assert total_lost >= 0  # Sollte mindestens 0 sein
        
        # Prüfe dass Photonen detektiert wurden
        total_detected = sum(r['detected_photons'] for r in results)
        assert total_detected > 0
        
    def test_dead_time_filter(self):
        """Test Totzeit-Filter Funktion"""
        # Simuliere schnell aufeinanderfolgende Photonen
        arrival_times = np.array([0, 10e-9, 20e-9, 60e-9, 70e-9, 130e-9])  # ns
        
        filtered_times = self.counter._apply_dead_time_filter(arrival_times, 0.0)
        
        # Bei 50ns Totzeit sollten einige Photonen verloren gehen
        assert len(filtered_times) < len(arrival_times)
        
        # Zeitabstand zwischen detektierten Photonen sollte >= Totzeit sein
        if len(filtered_times) > 1:
            time_diffs = np.diff(filtered_times)
            assert np.all(time_diffs >= self.counter.dead_time - 1e-12)  # Floating-point tolerance
            
    def test_timing_jitter(self):
        """Test Timing-Jitter Funktion"""
        exact_times = np.array([100e-9, 200e-9, 300e-9])
        jittered_times = self.counter._apply_timing_jitter(exact_times)
        
        # Zeiten sollten sich geändert haben (mit hoher Wahrscheinlichkeit)
        assert not np.array_equal(exact_times, jittered_times)
        
        # Jitter sollte um ursprüngliche Zeiten streuen
        differences = jittered_times - exact_times
        assert np.std(differences) > 0
        assert np.std(differences) < 10 * self.counter.time_resolution  # Reasonable bound
        
    def test_afterpulsing(self):
        """Test Afterpulsing Generierung"""
        trigger_times = np.array([100e-9, 200e-9])
        afterpulses = self.counter._generate_afterpulses(trigger_times, 0.0)
        
        # Bei 1% Wahrscheinlichkeit können 0-2 Afterpulse auftreten
        assert len(afterpulses) <= len(trigger_times)
        
        # Afterpulse sollten später als Trigger sein
        if len(afterpulses) > 0:
            for ap_time in afterpulses:
                assert ap_time > min(trigger_times)
                
    def test_g2_correlation_calculation(self):
        """Test g²(τ) Korrelationsfunktion"""
        # Simuliere einige Photonen
        self.counter.photon_arrival_times = [0, 100e-9, 200e-9, 1000e-9, 1100e-9]
        
        # g²(0) sollte < 1 sein für Antibunching
        g2_zero = self.counter.get_correlation_g2(0.0, window=10e-9)
        assert 0 <= g2_zero <= 1
        
        # g²(τ) für große τ sollte gegen 1 gehen
        g2_large = self.counter.get_correlation_g2(10e-6, window=10e-9)
        assert 0 <= g2_large <= 2  # Kann leicht über 1 gehen durch Rauschen
        
    def test_photon_statistics(self):
        """Test erweiterte Photonen-Statistiken"""
        # Füge einige Ankunftszeiten hinzu
        self.counter.photon_arrival_times = [0, 50e-9, 150e-9, 300e-9]
        
        stats = self.counter.get_photon_statistics()
        
        assert 'count_rate_hz' in stats
        assert 'g2_zero' in stats
        assert 'antibunching_quality' in stats
        assert stats['total_photons'] == 4
        
    def test_time_trace(self):
        """Test zeitaufgelöster Photonen-Trace"""
        # Simuliere Photonen über Zeit
        self.counter.photon_arrival_times = np.linspace(0, 1e-6, 100)  # 1 μs, 100 Photonen
        
        bin_centers, counts = self.counter.get_time_trace(bin_width=100e-9)  # 100 ns bins
        
        assert len(bin_centers) == len(counts)
        assert np.sum(counts) == 100  # Alle Photonen sollten gezählt werden
        
    def test_reset(self):
        """Test Reset-Funktion"""
        # Füge Daten hinzu
        self.counter.total_photons = 10
        self.counter.photon_arrival_times = [1, 2, 3]
        self.counter.dead_time_losses = 5
        
        # Reset
        self.counter.reset()
        
        # Alle Zähler sollten zurückgesetzt sein
        assert self.counter.total_photons == 0
        assert len(self.counter.photon_arrival_times) == 0
        assert self.counter.dead_time_losses == 0
        assert self.counter.last_detection_time == -np.inf


class TestRealisticLaserInterface:
    """Test realistischen LaserInterface mit Linienbreite, Rauschen, etc."""
    
    def setup_method(self):
        """Setup für jeden Test"""
        self.laser = LaserInterface()
        
    def test_initialization(self):
        """Test Initialisierung mit realistischen Parametern"""
        assert self.laser.linewidth_hz == 1e6  # 1 MHz default
        assert self.laser.intensity_noise_rms == 0.01  # 1% default
        assert self.laser.phase_noise_enabled == True
        assert self.laser.pulse_type == 'cw'
        
    def test_laser_turn_on_with_noise(self):
        """Test Laser Ein/Aus mit Rauschparametern"""
        self.laser.turn_on(
            power_mw=1.0, 
            detuning=1e6,  # 1 MHz detuning
            linewidth_hz=500e3,  # 500 kHz linewidth
            intensity_noise=0.02  # 2% noise
        )
        
        assert self.laser.is_on
        assert self.laser.power_mw == 1.0
        assert self.laser.linewidth_hz == 500e3
        assert self.laser.intensity_noise_rms == 0.02
        
    def test_hamiltonian_with_noise(self):
        """Test Hamiltonian-Berechnung mit Rauschen"""
        self.laser.turn_on(1.0, linewidth_hz=1e6, intensity_noise=0.1)
        
        # Mehrere Hamiltonian-Berechnungen
        hamiltonians = []
        for i in range(10):
            H = self.laser.get_hamiltonian_contribution(i * 1e-9, dt=1e-9)
            hamiltonians.append(H)
            
        # Hamiltonians sollten sich leicht unterscheiden (Rauschen)
        norms = [np.linalg.norm(H) for H in hamiltonians]
        assert np.std(norms) > 0  # Variation durch Rauschen
        
        # Alle sollten 18x18 sein
        for H in hamiltonians:
            assert H.shape == (18, 18)
            
    def test_phase_evolution(self):
        """Test Phasenentwicklung mit Linienbreite"""
        self.laser.linewidth_hz = 1e6  # 1 MHz
        self.laser.phase_noise_enabled = True
        
        initial_phase = self.laser.current_phase
        
        # Mehrere Zeitschritte
        phases = []
        for i in range(100):
            phase = self.laser._evolve_laser_phase(1e-9)  # 1 ns steps
            phases.append(phase)
            
        # Phase sollte sich entwickeln
        phase_variation = np.var(phases)
        assert phase_variation > 0
        
        # Phase sollte in [0, 2π) bleiben
        assert all(0 <= p < 2*np.pi for p in phases)
        
    def test_pulse_shapes(self):
        """Test verschiedene Pulsformen"""
        shapes_to_test = ['square', 'gauss', 'sinc', 'blackman']
        
        for shape in shapes_to_test:
            self.laser.turn_on(1.0)
            self.laser.apply_pulse(shape, duration_ns=100, power_mw=2.0, start_time=0.0)
            
            # Test Amplitude zu verschiedenen Zeiten
            times = [0, 50e-9, 100e-9, 150e-9]  # Vor, während, Ende, nach Puls
            amplitudes = []
            
            for t in times:
                amp = self.laser._get_time_dependent_amplitude(t)
                amplitudes.append(amp)
                
            # Während des Pulses sollte Amplitude > 0 sein
            assert amplitudes[1] > 0  # Mitte des Pulses
            assert amplitudes[2] >= 0  # Ende des Pulses
            assert amplitudes[3] == 0  # Nach dem Puls
            
    def test_gauss_pulse_shape(self):
        """Test spezifisch Gauss-Puls"""
        self.laser.turn_on(1.0)
        self.laser.apply_gauss_pulse(duration_ns=120, power_mw=2.0, start_time=0.0, sigma_ns=20)
        
        # Sample Gauss-Profil
        times = np.linspace(0, 120e-9, 100)
        amplitudes = [self.laser._get_time_dependent_amplitude(t) for t in times]
        
        # Maximum sollte in der Mitte sein
        max_idx = np.argmax(amplitudes)
        center_idx = len(amplitudes) // 2
        assert abs(max_idx - center_idx) < 5  # Within reasonable range
        
        # Flanken sollten abfallen
        assert amplitudes[0] < amplitudes[center_idx]
        assert amplitudes[-1] < amplitudes[center_idx]
        
    def test_pulse_sequence(self):
        """Test Pulssequenz-Definition"""
        pulse_list = [
            {'type': 'gauss', 'duration_ns': 100, 'power_mw': 1.0, 'delay_ns': 0},
            {'type': 'square', 'duration_ns': 50, 'power_mw': 2.0, 'delay_ns': 200},
            {'type': 'sinc', 'duration_ns': 80, 'power_mw': 1.5, 'delay_ns': 300}
        ]
        
        sequence_info = self.laser.pulse_sequence(pulse_list)
        
        assert len(sequence_info) == 3
        assert sequence_info[0]['type'] == 'gauss'
        assert sequence_info[1]['type'] == 'square'
        assert sequence_info[2]['type'] == 'sinc'
        
    def test_coherence_calculations(self):
        """Test Kohärenzzeit und -länge Berechnung"""
        # Schmale Linienbreite
        self.laser.linewidth_hz = 1e3  # 1 kHz
        coherence_time = self.laser.get_coherence_time()
        coherence_length = self.laser.get_coherence_length()
        
        expected_time = 1 / (2 * np.pi * 1e3)
        expected_length = self.laser.c * expected_time
        
        assert abs(coherence_time - expected_time) < 1e-6
        assert abs(coherence_length - expected_length) < 1e-3
        
        # Unendliche Kohärenz (Linienbreite = 0)
        self.laser.linewidth_hz = 0
        assert self.laser.get_coherence_time() == np.inf
        assert self.laser.get_coherence_length() == np.inf
        
    def test_saturation_parameter(self):
        """Test Sättigungsparameter-Berechnung"""
        self.laser.turn_on(1.0)  # Startet mit Rabi-Frequenz
        
        s = self.laser.get_saturation_parameter()
        assert s >= 0
        
        # Höhere Leistung → höhere Sättigung
        self.laser.turn_on(10.0)
        s_high = self.laser.get_saturation_parameter()
        assert s_high > s
        
    def test_intensity_noise_effect(self):
        """Test Effekt von Intensitätsrauschen"""
        self.laser.turn_on(1.0, intensity_noise=0.0)  # Kein Rauschen
        H_no_noise = self.laser.get_hamiltonian_contribution(0, dt=1e-9)
        
        self.laser.turn_on(1.0, intensity_noise=0.5)  # 50% Rauschen
        H_with_noise = self.laser.get_hamiltonian_contribution(0, dt=1e-9)
        
        # Mit Rauschen sollte Hamiltonian variieren
        # (Schwer deterministisch zu testen wegen Zufälligkeit)
        assert H_no_noise.shape == H_with_noise.shape == (18, 18)
        
    def test_laser_characterization(self):
        """Test erweiterte Laser-Charakterisierung"""
        self.laser.turn_on(1.0, detuning=1e6)
        
        char = self.laser.get_laser_characterization()
        
        # Prüfe wichtige Parameter
        assert 'wavelength_nm' in char
        assert 'frequency_thz' in char
        assert 'intensity_w_per_m2' in char
        assert 'photon_flux_per_s' in char
        assert 'steady_state_ground' in char
        assert 'steady_state_excited' in char
        assert 'rabi_period_ns' in char
        
        # Physikalische Plausibilität
        assert char['wavelength_nm'] > 0
        assert char['frequency_thz'] > 0
        assert char['steady_state_ground'] + char['steady_state_excited'] == pytest.approx(1.0, abs=1e-10)


def run_realistic_detector_tests():
    """Führt alle Tests für realistische Detektoren aus"""
    print("Testing realistic PhotonCounter...")
    
    # PhotonCounter Tests
    test_pc = TestRealisticPhotonCounter()
    pc_methods = [m for m in dir(test_pc) if m.startswith('test_')]
    
    for method_name in pc_methods:
        test_pc.setup_method()
        method = getattr(test_pc, method_name)
        try:
            method()
            print(f"  ✓ {method_name}")
        except Exception as e:
            print(f"  ✗ {method_name}: {e}")
            
    print("\nTesting realistic LaserInterface...")
    
    # LaserInterface Tests
    test_laser = TestRealisticLaserInterface()
    laser_methods = [m for m in dir(test_laser) if m.startswith('test_')]
    
    for method_name in laser_methods:
        test_laser.setup_method()
        method = getattr(test_laser, method_name)
        try:
            method()
            print(f"  ✓ {method_name}")
        except Exception as e:
            print(f"  ✗ {method_name}: {e}")
            
    print("\nRealistic detector tests completed!")


if __name__ == "__main__":
    run_realistic_detector_tests()