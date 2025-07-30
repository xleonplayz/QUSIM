#!/usr/bin/env python3
"""
Test-Script: Simulator + Laser + Photonenzähler für 300ns

Führt folgende Sequenz aus:
1. Startet Simulator
2. Schaltet Laser ein (1mW)
3. Lässt alles 300ns laufen
4. Speichert Photonenzählungen in photon_counts.txt
"""

import sys
import os
sys.path.append('../src')
sys.path.append('../interfaces')
sys.path.append('../runner')

import json
import numpy as np
import time
from datetime import datetime
from lindblad import LindbladEvolution
from photoncounter import PhotonCounter
from laser import LaserInterface
from scipy.integrate import solve_ivp

class LaserPhotonTest:
    def __init__(self):
        # Lade Konfiguration
        with open('../runner/config.json', 'r') as f:
            runner_config = json.load(f)
        
        self.steps_per_ns = runner_config['steps_per_ns']
        self.dt = 1e-9 / self.steps_per_ns  # Zeitschritt
        
        # Initialisiere Komponenten
        print("Initialisiere Komponenten...")
        self.lindblad = LindbladEvolution('../src/system.json')
        self.rho = self.lindblad.initial_state('ground')
        
        # Photonenzähler
        self.photon_counter = PhotonCounter(
            detection_efficiency=0.1,
            dark_count_rate=100.0
        )
        
        # Laser Interface
        self.laser = LaserInterface()
        self.lindblad.H_builder.set_laser_interface(self.laser)
        
        # Hole spontane Emissionsrate
        with open('../src/system.json', 'r') as f:
            system_config = json.load(f)
        self.gamma_emission = system_config.get('lindblad', {}).get('spontaneous_emission', {}).get('gamma', 1e7)
        
        # Datensammlung
        self.data_log = []
        
    def evolve_one_step(self, rho, dt):
        """Ein einzelner Zeitschritt"""
        rho_vec = rho.flatten()
        
        sol = solve_ivp(
            self.lindblad.lindblad_rhs,
            [0, dt],
            rho_vec,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        rho_new = sol.y[:, -1].reshape((18, 18))
        return rho_new
    
    def run_test(self, duration_ns=300, laser_power_mw=1.0):
        """
        Führt Test durch
        
        Args:
            duration_ns: Dauer in Nanosekunden
            laser_power_mw: Laserleistung in mW
        """
        print(f"\nStarte Test für {duration_ns}ns mit {laser_power_mw}mW Laser")
        print("="*60)
        
        # Laser einschalten
        self.laser.turn_on(laser_power_mw)
        
        # Simulationsparameter
        total_steps = int(duration_ns * self.steps_per_ns)
        sim_time = 0.0
        
        # Fortschrittsanzeige
        progress_interval = total_steps // 10  # 10 Updates
        
        print(f"Simuliere {total_steps} Schritte...")
        start_real_time = time.time()
        
        for step in range(total_steps):
            # Evolution
            self.rho = self.evolve_one_step(self.rho, self.dt)
            sim_time += self.dt
            
            # Photonenzählung
            photon_stats = self.photon_counter.count_photons(
                self.rho, self.dt, self.gamma_emission
            )
            
            # Observablen berechnen
            obs = self.lindblad.observables([self.rho])
            
            # Daten sammeln
            self.data_log.append({
                'time_ns': sim_time * 1e9,
                'pop_g': obs['population_g'][0],
                'pop_e': obs['population_e'][0],
                'emission_rate_mhz': photon_stats['emission_rate'] / 1e6,
                'photons_detected': photon_stats['detected_photons'],
                'total_photons': photon_stats['total_count']
            })
            
            # Fortschritt
            if step % progress_interval == 0:
                progress = (step / total_steps) * 100
                current_ns = sim_time * 1e9
                print(f"  {progress:3.0f}% - {current_ns:3.0f}ns - Photonen: {photon_stats['total_count']}")
        
        # Laser ausschalten
        self.laser.turn_off()
        
        # Abschluss
        elapsed_time = time.time() - start_real_time
        print(f"\nSimulation abgeschlossen in {elapsed_time:.1f}s")
        
        # Statistiken
        final_stats = self.photon_counter.get_statistics()
        print(f"\nPhotonenzähler-Statistiken:")
        print(f"  Gesamtphotonen detektiert: {final_stats['total_photons']}")
        print(f"  Mittlere Emissionsrate: {final_stats['average_rate']/1e6:.2f} MHz")
        print(f"  Mittlere Zählrate: {final_stats['average_count_rate']:.1f} Hz")
        print(f"  Fano-Faktor: {final_stats['fano_factor']:.3f}")
        
        # Laser-Statistiken
        print(f"\nLaser-Parameter:")
        print(f"  Leistung: {laser_power_mw} mW")
        print(f"  Rabi-Frequenz: {self.laser.rabi_freq/1e6:.1f} MHz")
        print(f"  Sättigungsparameter: {self.laser.get_saturation_parameter():.3f}")
        
        # Speichere Daten
        self.save_results()
        
    def save_results(self):
        """Speichert Ergebnisse in Datei"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photon_counts_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            # Header
            f.write("# Laser-Photonenzähler Test\n")
            f.write(f"# Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Simulationsdauer: {self.data_log[-1]['time_ns']:.1f}ns\n")
            f.write(f"# Laserleistung: {self.laser.power_mw} mW\n")
            f.write(f"# Detektionseffizienz: {self.photon_counter.detection_efficiency}\n")
            f.write(f"# Dunkelzählrate: {self.photon_counter.dark_count_rate} Hz\n")
            f.write("#\n")
            f.write("# Zeit[ns]  |g⟩    |e⟩    EmRate[MHz]  Photonen  Total\n")
            f.write("#" + "-"*60 + "\n")
            
            # Daten (alle 1ns ausgeben)
            for i, data in enumerate(self.data_log):
                if i % self.steps_per_ns == 0:  # Jede Nanosekunde
                    f.write(f"{data['time_ns']:8.1f}  "
                           f"{data['pop_g']:6.4f} "
                           f"{data['pop_e']:6.4f} "
                           f"{data['emission_rate_mhz']:11.2f}  "
                           f"{data['photons_detected']:8d}  "
                           f"{data['total_photons']:7d}\n")
        
        print(f"\nErgebnisse gespeichert in: {filename}")
        
        # Zusätzlich: Zusammenfassung
        summary_file = "photon_counts_summary.txt"
        with open(summary_file, 'a') as f:
            f.write(f"{timestamp}, "
                   f"{self.laser.power_mw:.2f}mW, "
                   f"{self.data_log[-1]['time_ns']:.0f}ns, "
                   f"{self.data_log[-1]['total_photons']} photons, "
                   f"{self.photon_counter.get_statistics()['average_rate']/1e6:.2f}MHz\n")

if __name__ == "__main__":
    # Erstelle und führe Test aus
    test = LaserPhotonTest()
    
    # Parameter aus Kommandozeile oder Standard
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 300  # ns
    power = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0   # mW
    
    test.run_test(duration_ns=duration, laser_power_mw=power)