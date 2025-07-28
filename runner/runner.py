import sys
import os
sys.path.append('../src')
sys.path.append('../interfaces')

import json
import numpy as np
from lindblad import LindbladEvolution
from scipy.integrate import solve_ivp
from photoncounter import PhotonCounter

class SimpleRunner:
    def __init__(self):
        # Lade Runner-Config
        with open('config.json', 'r') as f:
            runner_config = json.load(f)
        
        self.steps_per_ns = runner_config['steps_per_ns']
        self.dt = 1e-9 / self.steps_per_ns  # z.B. 0.1ns bei 10 steps/ns
        
        # Lindblad System initialisieren
        self.lindblad = LindbladEvolution('../src/system.json')
        self.rho = self.lindblad.initial_state('ground')
        
        # Photonenzähler initialisieren
        self.photon_counter = PhotonCounter(
            detection_efficiency=0.1,  # 10% Detektionseffizienz
            dark_count_rate=100.0      # 100 Hz Dunkelzählrate
        )
        
        # Hole spontane Emissionsrate aus Config
        with open('../src/system.json', 'r') as f:
            system_config = json.load(f)
        self.gamma_emission = system_config.get('lindblad', {}).get('spontaneous_emission', {}).get('gamma', 1e7)
        
        # Simulation State
        self.sim_time = 0.0
        self.step_count = 0
        
        print(f"Runner gestartet: {self.steps_per_ns} steps/ns, dt={self.dt*1e9:.2f}ns")
        print(f"Photonenzähler aktiv: η={self.photon_counter.detection_efficiency}, DCR={self.photon_counter.dark_count_rate} Hz")
        
    def evolve_one_step(self, rho, dt):
        """Ein einzelner Lindblad-Zeitschritt"""
        rho_vec = rho.flatten()
        
        # Löse für einen winzigen Zeitschritt
        sol = solve_ivp(
            self.lindblad.lindblad_rhs,
            [0, dt],
            rho_vec,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Letzter Zustand
        rho_new = sol.y[:, -1].reshape((18, 18))
        return rho_new
    
    def get_observables(self):
        """Berechne wichtige Observablen"""
        obs = self.lindblad.observables([self.rho])
        
        # Photonenzählung
        photon_stats = self.photon_counter.count_photons(
            self.rho, self.dt, self.gamma_emission
        )
        
        # Formatiere als String
        pop_g = obs['population_g'][0]
        pop_e = obs['population_e'][0]
        spin_z = obs['spin_z'][0]
        coherence = obs['coherence'][0]
        purity = obs['purity'][0]
        
        return (f"|g⟩={pop_g:.3f} |e⟩={pop_e:.3f} "
                f"⟨Sz⟩={spin_z:.3f} Coh={coherence:.3f} Pur={purity:.3f} "
                f"γ={photon_stats['emission_rate']/1e6:.1f}MHz Photons={photon_stats['total_count']}")
    
    def run_forever(self):
        """Hauptschleife: Läuft für immer"""
        print("Starte Simulation... (Strg+C zum Beenden)")
        
        # Öffne Output-Datei
        with open('output.txt', 'w') as f:
            f.write("# Simulation Output\n")
            f.write("# Format: [ns]: |g⟩ |e⟩ ⟨Sz⟩ Coherence Purity EmissionRate[MHz] TotalPhotons\n")
            f.write(f"# Photon Counter: Detection efficiency={self.photon_counter.detection_efficiency}, Dark count rate={self.photon_counter.dark_count_rate} Hz\n")
            f.flush()
            
            try:
                while True:
                    # Ein Mikro-Evolutionsschritt
                    self.rho = self.evolve_one_step(self.rho, self.dt)
                    self.step_count += 1
                    self.sim_time += self.dt
                    
                    # Nach N Schritten = 1ns vergangen → Output
                    if self.step_count % self.steps_per_ns == 0:
                        ns_time = int(self.sim_time * 1e9)
                        obs_str = self.get_observables()
                        
                        # In Datei schreiben
                        f.write(f"{ns_time}ns: {obs_str}\n")
                        f.flush()
                        
                        # Terminal-Status (gelegentlich)
                        if ns_time % 100 == 0:  # Alle 100ns
                            print(f"Simulationszeit: {ns_time}ns")
                            
            except KeyboardInterrupt:
                print(f"\nSimulation gestoppt bei {self.sim_time*1e9:.1f}ns")
                print(f"Total steps: {self.step_count}")
                
                # Photonenzähler-Statistiken
                stats = self.photon_counter.get_statistics()
                print(f"\nPhotonenzähler-Statistiken:")
                print(f"  Gesamtphotonen: {stats['total_photons']}")
                print(f"  Mittlere Emissionsrate: {stats['average_rate']/1e6:.2f} MHz")
                print(f"  Mittlere Zählrate: {stats['average_count_rate']:.1f} Hz")
                print(f"  Fano-Faktor: {stats['fano_factor']:.3f}")

if __name__ == "__main__":
    runner = SimpleRunner()
    runner.run_forever()