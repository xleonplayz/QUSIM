#!/usr/bin/env python3
"""
AWG Demo - Demonstration der kompletten AWG-Funktionalität im QUSIM
"""

import sys
import os
sys.path.append('../src')
sys.path.append('../runner')
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from interfaces.awg import AWGInterface, WaveformLibrary, PulseSequenceBuilder


def demo_basic_awg():
    """Demonstriert AWG-Grundfunktionen"""
    print("=== AWG Grundfunktionen Demo ===")
    
    # AWG initialisieren
    awg = AWGInterface(sample_rate=1e9)
    print(f"AWG initialisiert: {awg.sample_rate/1e9:.1f} GS/s")
    
    # Rechteckpuls hinzufügen
    print("\n1. Rechteckpuls (π-Puls):")
    rabi_freq_mhz = 1.0
    pi_pulse = PulseSequenceBuilder.pi_pulse(rabi_freq_mhz, 'x')
    print(f"   Amplitude: {pi_pulse['amplitude']/(2*np.pi*1e6):.1f} MHz")
    print(f"   Dauer: {pi_pulse['duration']*1e9:.1f} ns")
    print(f"   Phase: {np.degrees(pi_pulse['phase']):.1f}°")
    
    awg.clear_waveform()
    awg.add_pulse(**pi_pulse)
    
    # Gauss-Puls hinzufügen
    print("\n2. Gauss-Puls:")
    awg.add_delay(100e-9)  # 100 ns Pause
    awg.add_pulse('gauss', 2*np.pi*2e6, 100e-9, np.pi/2, sigma=30e-9)
    print("   Gauss-Puls (2 MHz, 100 ns, 90°) hinzugefügt")
    
    # Waveform-Info
    info = awg.get_info()
    print(f"\nGesamte Waveform:")
    print(f"   Dauer: {info['duration']*1e9:.1f} ns")
    print(f"   Samples: {info['n_samples']}")
    
    # Playback starten
    awg.start(at_time=0.0)
    print(f"   Status: {'AKTIV' if info['is_playing'] else 'IDLE'}")
    
    return awg


def demo_pulse_sequences():
    """Demonstriert vordefinierte Pulssequenzen"""
    print("\n\n=== Pulssequenzen Demo ===")
    
    awg = AWGInterface()
    
    # Ramsey-Sequenz
    print("\n1. Ramsey-Sequenz:")
    tau_ns = 500
    ramsey_seq = PulseSequenceBuilder.ramsey_sequence(tau_ns, rabi_freq_mhz=1.0)
    
    print(f"   Sequenz mit {len(ramsey_seq)} Elementen:")
    for i, pulse in enumerate(ramsey_seq):
        if pulse['shape'] == 'delay':
            print(f"   {i+1}. Delay: {pulse['duration']*1e9:.1f} ns")
        else:
            print(f"   {i+1}. {pulse['shape']}: {pulse['amplitude']/(2*np.pi*1e6):.1f} MHz, "
                  f"{pulse['duration']*1e9:.1f} ns, {np.degrees(pulse['phase']):.1f}°")
    
    # Lade Sequenz ins AWG
    awg.clear_waveform()
    for pulse in ramsey_seq:
        if pulse['shape'] == 'delay':
            awg.add_delay(pulse['duration'])
        else:
            awg.add_pulse(**pulse)
    
    total_duration = awg.duration
    print(f"   Gesamtdauer: {total_duration*1e9:.1f} ns")
    
    # Echo-Sequenz
    print("\n2. Echo-Sequenz:")
    echo_seq = PulseSequenceBuilder.echo_sequence(tau_ns=1000, rabi_freq_mhz=1.0)
    print(f"   Echo-Sequenz mit {len(echo_seq)} Elementen")
    
    # CPMG-Sequenz
    print("\n3. CPMG-Sequenz:")
    cpmg_seq = PulseSequenceBuilder.cpmg_sequence(tau_ns=200, n_pulses=4, rabi_freq_mhz=1.0)
    print(f"   CPMG-Sequenz mit {len(cpmg_seq)} Elementen")
    
    return awg


def demo_waveform_shapes():
    """Demonstriert verschiedene Pulsformen"""
    print("\n\n=== Pulsformen Demo ===")
    
    sample_rate = 1e9
    duration = 100e-9
    amplitude = 2*np.pi*1e6
    
    shapes = ['rect', 'gauss', 'sinc', 'blackman', 'hermite']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, shape in enumerate(shapes):
        if shape == 'hermite':
            amp, phase = WaveformLibrary.hermite(amplitude, duration, sample_rate, order=1)
        else:
            if shape == 'rect':
                amp, phase = WaveformLibrary.rectangular(amplitude, duration, sample_rate)
            elif shape == 'gauss':
                amp, phase = WaveformLibrary.gaussian(amplitude, duration, sample_rate)
            elif shape == 'sinc':
                amp, phase = WaveformLibrary.sinc(amplitude, duration, sample_rate)
            elif shape == 'blackman':
                amp, phase = WaveformLibrary.blackman(amplitude, duration, sample_rate)
        
        t = np.linspace(0, duration*1e9, len(amp))
        
        axes[i].plot(t, amp/(2*np.pi*1e6), 'b-', linewidth=2)
        axes[i].set_title(f'{shape.upper()} Pulse')
        axes[i].set_xlabel('Time (ns)')
        axes[i].set_ylabel('Amplitude (MHz)')
        axes[i].grid(True, alpha=0.3)
        
        print(f"{shape}: Peak amplitude {np.max(amp)/(2*np.pi*1e6):.2f} MHz")
    
    # Remove empty subplot
    axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('awg_waveforms.png', dpi=150, bbox_inches='tight')
    print("\nWaveform-Plot gespeichert als 'awg_waveforms.png'")
    
    return fig


def demo_hamiltonian_calculation():
    """Demonstriert Hamiltonian-Berechnung"""
    print("\n\n=== Hamiltonian Demo ===")
    
    awg = AWGInterface()
    
    # π-Puls laden
    pi_pulse = PulseSequenceBuilder.pi_pulse(rabi_freq_mhz=1.0, axis='x')
    awg.add_pulse(**pi_pulse)
    awg.start(at_time=0.0)
    
    # Hamiltonian zu verschiedenen Zeiten
    times = [0, pi_pulse['duration']/2, pi_pulse['duration'], pi_pulse['duration'] + 1e-9]
    time_labels = ['Start', 'Mitte', 'Ende', 'Nach Puls']
    
    print("Hamiltonian-Beiträge:")
    for t, label in zip(times, time_labels):
        H = awg.get_hamiltonian_contribution(t)
        H_norm = np.linalg.norm(H)
        
        print(f"   {label} (t={t*1e9:.1f}ns): ||H|| = {H_norm:.2e}")
        
        # Prüfe Struktur
        if H_norm > 0:
            # Sollte nur Ground-State Manifold beeinflussen
            ground_block_norm = np.linalg.norm(H[:9, :9])
            excited_block_norm = np.linalg.norm(H[9:, 9:])
            cross_norm = np.linalg.norm(H[:9, 9:]) + np.linalg.norm(H[9:, :9])
            
            print(f"     Ground state: {ground_block_norm:.2e}")
            print(f"     Excited state: {excited_block_norm:.2e}")  
            print(f"     Cross terms: {cross_norm:.2e}")
    
    return awg


def demo_advanced_sequences():
    """Demonstriert erweiterte Sequenzen"""
    print("\n\n=== Erweiterte Sequenzen Demo ===")
    
    # Composite Pulse
    print("\n1. Composite π-Puls (BB1):")
    bb1_seq = PulseSequenceBuilder.composite_pi_pulse('BB1', rabi_freq_mhz=1.0)
    print(f"   BB1 mit {len(bb1_seq)} Pulsen")
    for i, pulse in enumerate(bb1_seq):
        print(f"   Puls {i+1}: Phase {np.degrees(pulse['phase']):.1f}°")
    
    # Dynamical Decoupling
    print("\n2. Dynamical Decoupling (XY4):")
    xy4_seq = PulseSequenceBuilder.dd_sequence('XY4', tau_ns=100, n_pulses=2)
    print(f"   XY4 mit {len(xy4_seq)} Elementen")
    
    # DRAG Pulse
    print("\n3. DRAG Puls:")
    awg = AWGInterface()
    drag_amp, drag_phase = WaveformLibrary.drag_pulse(
        amplitude=2*np.pi*1e6, 
        duration=100e-9, 
        sample_rate=1e9,
        alpha=0.5
    )
    
    print(f"   DRAG Puls: {len(drag_amp)} samples")
    print(f"   Peak amplitude: {np.max(drag_amp)/(2*np.pi*1e6):.2f} MHz")
    print(f"   Phase modulation range: {np.degrees(np.ptp(drag_phase)):.1f}°")
    
    return awg


def demo_runner_integration():
    """Zeigt Integration mit dem Runner (konzeptuell)"""
    print("\n\n=== Runner-Integration Demo ===")
    
    print("Runner-Methoden für AWG-Steuerung:")
    print("1. runner.mw_pulse('rect', amplitude_mhz=1.0, duration_ns=500)")
    print("2. runner.apply_named_sequence('ramsey', tau_ns=1000)")
    print("3. runner.apply_named_sequence('pi_pulse', rabi_freq_mhz=2.0)")
    print("4. runner.awg_stop()")
    
    # Beispiel für Sequenz-Definitionen
    example_sequences = {
        'rabi_calibration': "Rabi-Oszillation für Kalibrierung",
        'ramsey_fringe': "Ramsey-Interferenz für T2*-Messung", 
        'echo_decay': "Echo-Sequenz für T2-Messung",
        'sensing_ac': "AC-Magnetometrie Sequenz",
        'dd_sequence': "Dynamical Decoupling für lange Kohärenz"
    }
    
    print("\nVerfügbare Sequenz-Templates:")
    for name, desc in example_sequences.items():
        print(f"   {name}: {desc}")
    
    print("\nAWG-Vorteile gegenüber alter MW-Implementation:")
    print("   ✓ Kontinuierliche Waveforms (keine binned Steps)")
    print("   ✓ Beliebige Pulsformen (Gauss, DRAG, Composite)")
    print("   ✓ Präzise Zeitsteuerung (1 ns Auflösung)")
    print("   ✓ Komplexe Sequenzen (Ramsey, CPMG, DD)")
    print("   ✓ Hardware-nahe Simulation (Sample-Rate)")
    print("   ✓ Echzeit-Steuerung während Simulation")


def main():
    """Hauptfunktion für AWG-Demo"""
    print("QUSIM AWG (Arbitrary Waveform Generator) - Vollständige Demonstration")
    print("="*70)
    
    try:
        # Grundfunktionen
        awg1 = demo_basic_awg()
        
        # Pulssequenzen
        awg2 = demo_pulse_sequences()
        
        # Verschiedene Waveforms
        try:
            fig = demo_waveform_shapes()
            plt.show()  # Optional - nur wenn matplotlib verfügbar
        except ImportError:
            print("matplotlib nicht verfügbar - Plots übersprungen")
        
        # Hamiltonian-Berechnung
        awg3 = demo_hamiltonian_calculation()
        
        # Erweiterte Sequenzen
        awg4 = demo_advanced_sequences()
        
        # Runner-Integration
        demo_runner_integration()
        
        print("\n" + "="*70)
        print("AWG-Demo erfolgreich abgeschlossen!")
        print("Das AWG-System ist vollständig funktional und bereit für NV-Simulationen.")
        print("\nNächste Schritte:")
        print("1. Runner mit: python runner.py starten")
        print("2. AWG-Pulse mit: runner.mw_pulse() anwenden")  
        print("3. Sequenzen mit: runner.apply_named_sequence() ausführen")
        
    except Exception as e:
        print(f"\nFehler während Demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()