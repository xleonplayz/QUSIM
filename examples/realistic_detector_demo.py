#!/usr/bin/env python3
"""
Realistic Detector & Laser Demo - Showcase verbesserter PhotonCounter und LaserInterface
"""

import sys
import os
sys.path.append('..')
sys.path.append('../interfaces')

import numpy as np
import matplotlib.pyplot as plt
from interfaces.photoncounter.photon_counter import PhotonCounter
from interfaces.laser.laser_interface import LaserInterface


def demo_photon_counter_dead_time():
    """Demonstriert Totzeit-Effekte im PhotonCounter"""
    print("=== PhotonCounter Totzeit-Demo ===")
    
    # Verschiedene Totzeiten vergleichen
    dead_times = [0, 25e-9, 50e-9, 100e-9]  # 0, 25, 50, 100 ns
    
    results = {}
    
    for dead_time in dead_times:
        counter = PhotonCounter(
            detection_efficiency=0.1,
            dark_count_rate=100.0,
            dead_time=dead_time,
            time_resolution=0.5e-9
        )
        
        # Simuliere hohe Photonrate (gesättigtes NV-Zentrum)
        rho = np.zeros((18, 18), dtype=complex)
        rho[9, 9] = 0.5  # 50% Population in |e⟩
        
        dt = 1e-9  # 1 ns Zeitschritte
        gamma = 2e7  # 20 MHz Emissionsrate
        
        total_detected = 0
        total_lost = 0
        
        for i in range(1000):  # 1 μs Simulation
            result = counter.count_photons(rho, dt, gamma, current_time=i*dt)
            total_detected += result['detected_photons']
            total_lost += result.get('dead_time_losses', 0)
        
        efficiency = total_detected / (total_detected + total_lost) if (total_detected + total_lost) > 0 else 0
        
        results[dead_time] = {
            'detected': total_detected,
            'lost': total_lost,
            'efficiency': efficiency
        }
        
        print(f"Totzeit {dead_time*1e9:.0f}ns: {total_detected} detektiert, "
              f"{total_lost} verloren, Effizienz: {efficiency:.2%}")
    
    return results


def demo_g2_correlation():
    """Demonstriert g²(τ) Korrelationsmessungen"""
    print("\n=== g²(τ) Korrelations-Demo ===")
    
    counter = PhotonCounter(
        detection_efficiency=0.1,
        dead_time=50e-9,
        time_resolution=0.5e-9,
        afterpulsing_probability=0.005
    )
    
    # Simuliere Einzelphoton-Emission (Antibunching)
    # Einfaches Modell: Exponentiell verteilte Inter-Photon-Zeiten
    n_photons = 1000
    mean_inter_time = 100e-9  # 100 ns mittlerer Abstand
    
    # Generiere Ankunftszeiten
    inter_times = np.random.exponential(mean_inter_time, n_photons)
    arrival_times = np.cumsum(inter_times)
    
    # Anwende Detektoreffekte
    detected_times = []
    last_detection = -np.inf
    
    for t in arrival_times:
        # Detektionseffizienz
        if np.random.random() < counter.detection_efficiency:
            # Totzeit-Check
            if t - last_detection >= counter.dead_time:
                # Timing-Jitter
                jittered_time = t + np.random.normal(0, counter.time_resolution)
                detected_times.append(max(0, jittered_time))
                last_detection = jittered_time
                
                # Afterpulsing
                if np.random.random() < counter.afterpulsing_probability:
                    afterpulse_time = jittered_time + np.random.exponential(100e-9)
                    if afterpulse_time - jittered_time >= counter.dead_time:
                        detected_times.append(afterpulse_time)
    
    counter.photon_arrival_times = sorted(detected_times)
    
    # Berechne g²(τ) Kurve
    tau_max = 500e-9  # 500 ns
    tau_array, g2_array = counter.calculate_g2_curve(tau_max, n_points=50)
    
    print(f"Detektierte Photonen: {len(detected_times)}")
    print(f"g²(0) = {counter.get_correlation_g2(0):.3f}")
    print(f"g²(τ=100ns) = {counter.get_correlation_g2(100e-9):.3f}")
    
    # Statistiken
    stats = counter.get_photon_statistics()
    print(f"Antibunching-Qualität: {stats['antibunching_quality']:.3f}")
    
    return tau_array, g2_array, stats


def demo_laser_linewidth_effects():
    """Demonstriert Laser-Linienbreite-Effekte"""
    print("\n=== Laser-Linienbreite Demo ===")
    
    linewidths = [1e3, 100e3, 1e6, 10e6]  # 1 kHz bis 10 MHz
    
    results = {}
    
    for linewidth in linewidths:
        laser = LaserInterface()
        laser.turn_on(
            power_mw=1.0,
            linewidth_hz=linewidth,
            intensity_noise=0.0  # Nur Linienbreite testen
        )
        
        # Simuliere Hamiltonian über Zeit
        times = np.linspace(0, 1e-6, 1000)  # 1 μs, 1000 Punkte
        phases = []
        detunings = []
        
        for t in times:
            # Phase und Detuning entwickeln
            phase = laser._evolve_laser_phase(1e-9)
            detuning = laser._get_time_dependent_detuning(1e-9)
            
            phases.append(phase)
            detunings.append(detuning)
        
        # Kohärenzzeit
        coherence_time = laser.get_coherence_time()
        coherence_length = laser.get_coherence_length()
        
        # Phasen-Diffusion
        phase_variance = np.var(phases)
        detuning_variance = np.var(detunings)
        
        results[linewidth] = {
            'coherence_time_us': coherence_time * 1e6,
            'coherence_length_m': coherence_length,
            'phase_variance': phase_variance,
            'detuning_variance': detuning_variance
        }
        
        print(f"Linienbreite {linewidth/1e6:.3f}MHz:")
        print(f"  Kohärenzzeit: {coherence_time*1e6:.1f} μs")
        print(f"  Kohärenzlänge: {coherence_length:.1f} m")
        print(f"  Phasen-Varianz: {phase_variance:.3f}")
    
    return results


def demo_pulse_shapes():
    """Demonstriert verschiedene Laser-Pulsformen"""
    print("\n=== Laser-Pulsformen Demo ===")
    
    laser = LaserInterface()
    laser.turn_on(1.0)
    
    pulse_types = ['square', 'gauss', 'sinc', 'blackman']
    duration_ns = 100
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, pulse_type in enumerate(pulse_types):
        laser.apply_pulse(pulse_type, duration_ns, power_mw=2.0, start_time=0.0)
        
        # Sample Amplitude über Zeit
        times = np.linspace(0, duration_ns * 1e-9, 200)
        amplitudes = [laser._get_time_dependent_amplitude(t) for t in times]
        
        # Plot
        axes[i].plot(times * 1e9, np.array(amplitudes) / 1e6, 'b-', linewidth=2)
        axes[i].set_title(f'{pulse_type.upper()} Pulse')
        axes[i].set_xlabel('Time (ns)')
        axes[i].set_ylabel('Rabi Frequency (MHz)')
        axes[i].grid(True, alpha=0.3)
        
        # Charakteristika
        peak_amp = max(amplitudes) / 1e6
        pulse_energy = np.trapz(amplitudes, times)
        
        print(f"{pulse_type}: Peak {peak_amp:.1f} MHz, Energy {pulse_energy:.2e}")
        
        laser.stop_pulse()
    
    plt.tight_layout()
    plt.savefig('laser_pulse_shapes.png', dpi=150, bbox_inches='tight')
    print("Pulsformen gespeichert als 'laser_pulse_shapes.png'")
    
    return fig


def demo_intensity_noise():
    """Demonstriert Laser-Intensitätsrauschen"""
    print("\n=== Laser-Intensitätsrauschen Demo ===")
    
    noise_levels = [0.0, 0.01, 0.05, 0.1]  # 0%, 1%, 5%, 10%
    
    for noise_level in noise_levels:
        laser = LaserInterface()
        laser.turn_on(
            power_mw=1.0,
            linewidth_hz=0,  # Kein Phasenrauschen
            intensity_noise=noise_level
        )
        
        # Sammle Hamiltonian-Normen über Zeit
        times = np.linspace(0, 1e-6, 1000)
        hamiltonian_norms = []
        
        for t in times:
            H = laser.get_hamiltonian_contribution(t, dt=1e-9)
            hamiltonian_norms.append(np.linalg.norm(H))
        
        # Statistiken
        mean_norm = np.mean(hamiltonian_norms)
        std_norm = np.std(hamiltonian_norms)
        relative_noise = std_norm / mean_norm if mean_norm > 0 else 0
        
        print(f"Intensitätsrauschen {noise_level*100:.1f}%:")
        print(f"  Mittlere Norm: {mean_norm:.2e}")
        print(f"  Relatives Rauschen: {relative_noise:.3f}")
        print(f"  Signal-zu-Rauschen: {1/relative_noise:.1f} dB" if relative_noise > 0 else "  SNR: ∞")
    
    return None


def demo_saturation_measurement():
    """Simuliert Sättigungsmessung mit realistischem Laser"""
    print("\n=== Sättigungs-Spektroskopie Demo ===")
    
    laser = LaserInterface()
    
    # Verschiedene Laserleistungen
    powers_mw = np.logspace(-2, 1, 20)  # 0.01 bis 10 mW
    
    saturation_params = []
    steady_state_excited = []
    
    for power in powers_mw:
        laser.turn_on(power, linewidth_hz=100e3, intensity_noise=0.02)
        
        # Sättigungsparameter und Steady-State
        s = laser.get_saturation_parameter()
        populations = laser.get_steady_state_populations()
        
        saturation_params.append(s)
        steady_state_excited.append(populations['pop_e'])
        
        print(f"P={power:.2f}mW: s={s:.3f}, ⟨e⟩={populations['pop_e']:.3f}")
    
    # Erwartete Sättigungskurve: pop_e = s/(1+s)
    theoretical_pop_e = np.array(saturation_params) / (1 + np.array(saturation_params))
    
    print(f"\nVergleich Theorie vs. Simulation:")
    print(f"Mittlere Abweichung: {np.mean(np.abs(steady_state_excited - theoretical_pop_e)):.4f}")
    
    return powers_mw, saturation_params, steady_state_excited


def demo_realistic_experiment():
    """Vollständiges realistisches Experiment: Photolumineszenz mit Laser-Rauschen"""
    print("\n=== Realistisches PL-Experiment ===")
    
    # Setup
    laser = LaserInterface()
    counter = PhotonCounter(
        detection_efficiency=0.05,  # 5% (realistisch für Si-APD)
        dark_count_rate=500.0,      # 500 Hz Dunkelzählrate
        dead_time=25e-9,            # 25 ns Totzeit
        time_resolution=1e-9,       # 1 ns Timing-Jitter
        afterpulsing_probability=0.01
    )
    
    # Laser einschalten mit realistischen Parametern
    laser.turn_on(
        power_mw=0.5,               # 0.5 mW
        linewidth_hz=1e6,           # 1 MHz Linienbreite
        intensity_noise=0.03        # 3% Intensitätsrauschen
    )
    
    # Simuliere NV-Zentrum Anregung
    gamma_emission = 12e6  # 12 MHz spontane Emission
    dt = 1e-9             # 1 ns Zeitschritte
    simulation_time = 10e-6  # 10 μs
    
    results = []
    photon_times = []
    
    print("Simuliere Photolumineszenz...")
    
    for i in range(int(simulation_time / dt)):
        current_time = i * dt
        
        # Vereinfachte NV-Dynamik: angenommen 30% angeregte Population
        rho = np.zeros((18, 18), dtype=complex)
        rho[9, 9] = 0.3  # 30% in |e⟩
        
        # Zähle Photonen mit allen realistischen Effekten
        result = counter.count_photons(rho, dt, gamma_emission, current_time)
        results.append(result)
        
        # Sammle Photonen-Zeiten
        if 'photon_times' in result and result['photon_times']:
            photon_times.extend(result['photon_times'])
    
    # Analyse
    total_photons = sum(r['detected_photons'] for r in results)
    total_lost_to_dead_time = sum(r.get('dead_time_losses', 0) for r in results)
    
    print(f"Simulation abgeschlossen:")
    print(f"  Gesamtzeit: {simulation_time*1e6:.0f} μs")
    print(f"  Detektierte Photonen: {total_photons}")
    print(f"  Verluste durch Totzeit: {total_lost_to_dead_time}")
    print(f"  Zählrate: {total_photons/(simulation_time*1e-6):.0f} Hz")
    
    # g²(0) Messung
    counter.photon_arrival_times = photon_times
    g2_zero = counter.get_correlation_g2(0, window=5e-9)
    
    print(f"  g²(0) = {g2_zero:.3f} (Antibunching)")
    
    # Detaillierte Statistiken
    stats = counter.get_statistics()
    print(f"  Fano-Faktor: {stats['fano_factor']:.2f}")
    print(f"  Afterpulse: {stats.get('afterpulses', 0)}")
    
    # Laser-Charakterisierung
    laser_char = laser.get_laser_characterization()
    print(f"  Laser-SNR: {-laser_char['relative_intensity_noise_db']:.1f} dB")
    print(f"  Kohärenzzeit: {laser_char.get('coherence_time_us', 0):.1f} μs")
    
    return {
        'total_photons': total_photons,
        'count_rate_hz': total_photons/(simulation_time*1e-6),
        'g2_zero': g2_zero,
        'dead_time_losses': total_lost_to_dead_time,
        'fano_factor': stats['fano_factor'],
        'laser_char': laser_char
    }


def main():
    """Hauptfunktion für Demo-Experimente"""
    print("QUSIM Realistic Detector & Laser Demo")
    print("="*50)
    
    try:
        # 1. Totzeit-Effekte
        dead_time_results = demo_photon_counter_dead_time()
        
        # 2. g²(τ) Korrelationen
        tau_array, g2_array, g2_stats = demo_g2_correlation()
        
        # 3. Laser-Linienbreite
        linewidth_results = demo_laser_linewidth_effects()
        
        # 4. Pulsformen
        try:
            pulse_fig = demo_pulse_shapes()
        except ImportError:
            print("matplotlib nicht verfügbar - Pulsformen-Plots übersprungen")
        
        # 5. Intensitätsrauschen
        demo_intensity_noise()
        
        # 6. Sättigungsmessung
        powers, sat_params, pop_excited = demo_saturation_measurement()
        
        # 7. Vollständiges Experiment
        experiment_results = demo_realistic_experiment()
        
        print("\n" + "="*50)
        print("🎉 Alle Demos erfolgreich abgeschlossen!")
        
        print("\n📊 Zusammenfassung der Verbesserungen:")
        print("✓ PhotonCounter mit Totzeit, Jitter, Afterpulsing")
        print("✓ Realistische g²(τ) Korrelationsmessungen")
        print("✓ Laser mit Linienbreite und Phasenrauschen")
        print("✓ Intensitätsfluktuationen")
        print("✓ Erweiterte Pulsformen (Gauss, Sinc, Blackman)")
        print("✓ Physikalisch korrekte Sättigungsspektroskopie")
        print("✓ Vollständig integriertes realistisches Experiment")
        
        print("\n🔬 Realistische Simulationsparameter:")
        print(f"   Detektoreffizienz: 5% (typisch für Si-APD)")
        print(f"   Totzeit: 25 ns")
        print(f"   Laser-Linienbreite: 1 MHz")
        print(f"   Intensitätsrauschen: 3%")
        print(f"   g²(0): {experiment_results['g2_zero']:.3f}")
        
    except Exception as e:
        print(f"\nFehler während Demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()