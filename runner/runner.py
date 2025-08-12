# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Leon Kaiser QUSIM Quantum Simulator für NV center simulations
#  MSQC Goethe University https://msqc.cgi-host6.rz.uni-frankfurt.de
#  I.kaiser[at]em.uni-frankfurt.de
#

import sys
import os
sys.path.append('../src')
sys.path.append('../interfaces')

import json
import numpy as np
from lindblad import LindbladEvolution
from scipy.integrate import solve_ivp
from photoncounter import PhotonCounter
from laser import LaserInterface
from interfaces.awg import AWGInterface, PulseSequenceBuilder, WaveformLibrary

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
        
        # Laser-Interface initialisieren
        self.laser = LaserInterface()
        
        # AWG-Interface initialisieren
        self.awg = AWGInterface(sample_rate=1e9)  # 1 GS/s
        
        # Verbinde Laser und AWG mit Hamiltonian
        self.lindblad.H_builder.set_laser_interface(self.laser)
        self.lindblad.H_builder.set_awg_interface(self.awg)
        
        # Hole spontane Emissionsrate aus Config
        with open('../src/system.json', 'r') as f:
            system_config = json.load(f)
        self.gamma_emission = system_config.get('lindblad', {}).get('spontaneous_emission', {}).get('gamma', 1e7)
        
        # Simulation State
        self.sim_time = 0.0
        self.step_count = 0
        
        print(f"Runner gestartet: {self.steps_per_ns} steps/ns, dt={self.dt*1e9:.2f}ns")
        print(f"Photonenzähler aktiv: η={self.photon_counter.detection_efficiency}, DCR={self.photon_counter.dark_count_rate} Hz")
        print("Laser-Interface bereit (verwende laser_on()/laser_off())")
        print("AWG-Interface bereit (verwende mw_pulse()/apply_sequence())")
        
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
        
        # Laser-Status
        laser_status = "ON" if self.laser.is_on else "OFF"
        laser_power = f"{self.laser.power_mw:.1f}mW" if self.laser.is_on else "0mW"
        
        # AWG-Status
        awg_info = self.awg.get_info()
        awg_status = "PLAYING" if awg_info['is_playing'] else "IDLE"
        awg_duration = f"{awg_info['duration']*1e9:.1f}ns" if awg_info['duration'] > 0 else "0ns"
        
        return (f"|g⟩={pop_g:.3f} |e⟩={pop_e:.3f} "
                f"⟨Sz⟩={spin_z:.3f} Coh={coherence:.3f} Pur={purity:.3f} "
                f"γ={photon_stats['emission_rate']/1e6:.1f}MHz Photons={photon_stats['total_count']} "
                f"Laser={laser_status}({laser_power}) AWG={awg_status}({awg_duration})")
    
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
                    
                    # Update AWG time tracking
                    self.awg.current_time = self.sim_time
                    
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
                
                # Laser-Statistiken
                laser_stats = self.laser.get_status()
                print(f"\nLaser-Status:")
                print(f"  Status: {'EIN' if laser_stats['is_on'] else 'AUS'}")
                print(f"  Leistung: {laser_stats['power_mw']:.2f} mW")
                print(f"  Rabi-Frequenz: {laser_stats['rabi_frequency_mhz']:.1f} MHz")
                
                # AWG-Statistiken
                awg_info = self.awg.get_info()
                print(f"\nAWG-Status:")
                print(f"  Status: {'AKTIV' if awg_info['is_playing'] else 'IDLE'}")
                print(f"  Sample Rate: {awg_info['sample_rate']/1e9:.1f} GS/s")
                print(f"  Waveform Duration: {awg_info['duration']*1e9:.1f} ns")
                print(f"  Samples: {awg_info['n_samples']}")
                if awg_info['loop_mode']:
                    print(f"  Loop Count: {awg_info['loop_count']}")
    
    def laser_on(self, power_mw: float, detuning_mhz: float = 0.0):
        """Schaltet Laser ein (kann während Simulation aufgerufen werden)"""
        self.laser.turn_on(power_mw, detuning_mhz * 1e6)
    
    def laser_off(self):
        """Schaltet Laser aus (kann während Simulation aufgerufen werden)"""
        self.laser.turn_off()
        
    def mw_pulse(self, shape: str = 'rect', amplitude_mhz: float = 1.0, 
                 duration_ns: float = 100, phase_deg: float = 0.0, **kwargs):
        """Führt einzelnen MW-Puls aus (kann während Simulation aufgerufen werden)"""
        amplitude_rad = amplitude_mhz * 2 * np.pi * 1e6  # MHz zu rad/s
        duration_s = duration_ns * 1e-9  # ns zu s
        phase_rad = np.radians(phase_deg)  # Grad zu Radiant
        
        self.awg.clear_waveform()
        self.awg.add_pulse(shape, amplitude_rad, duration_s, phase_rad, **kwargs)
        self.awg.start(at_time=self.sim_time)
        
    def apply_sequence(self, sequence):
        """Führt Pulssequenz aus (kann während Simulation aufgerufen werden)"""
        self.awg.clear_waveform()
        for pulse in sequence:
            if pulse['shape'] == 'delay':
                self.awg.add_delay(pulse['duration'])
            else:
                self.awg.add_pulse(**pulse)
        self.awg.start(at_time=self.sim_time)
        
    def apply_named_sequence(self, sequence_name: str, **kwargs):
        """Führt vordefinierte Sequenz aus"""
        if sequence_name == 'pi_pulse':
            rabi_freq_mhz = kwargs.get('rabi_freq_mhz', 1.0)
            axis = kwargs.get('axis', 'x')
            shape = kwargs.get('shape', 'rect')
            pulse = PulseSequenceBuilder.pi_pulse(rabi_freq_mhz, axis, shape)
            self.apply_sequence([pulse])
            
        elif sequence_name == 'pi_half_pulse':
            rabi_freq_mhz = kwargs.get('rabi_freq_mhz', 1.0)
            axis = kwargs.get('axis', 'x')
            shape = kwargs.get('shape', 'rect')
            pulse = PulseSequenceBuilder.pi_half_pulse(rabi_freq_mhz, axis, shape)
            self.apply_sequence([pulse])
            
        elif sequence_name == 'ramsey':
            tau_ns = kwargs.get('tau_ns', 500)
            rabi_freq_mhz = kwargs.get('rabi_freq_mhz', 1.0)
            final_phase = kwargs.get('final_phase', 0.0)
            shape = kwargs.get('shape', 'rect')
            sequence = PulseSequenceBuilder.ramsey_sequence(tau_ns, rabi_freq_mhz, final_phase, shape)
            self.apply_sequence(sequence)
            
        elif sequence_name == 'echo':
            tau_ns = kwargs.get('tau_ns', 1000)
            rabi_freq_mhz = kwargs.get('rabi_freq_mhz', 1.0)
            shape = kwargs.get('shape', 'rect')
            sequence = PulseSequenceBuilder.echo_sequence(tau_ns, rabi_freq_mhz, shape)
            self.apply_sequence(sequence)
            
        elif sequence_name == 'rabi_scan':
            max_duration_ns = kwargs.get('max_duration_ns', 1000)
            steps = kwargs.get('steps', 50)
            rabi_freq_mhz = kwargs.get('rabi_freq_mhz', 1.0)
            sequences = PulseSequenceBuilder.rabi_sequence(max_duration_ns, steps, rabi_freq_mhz)
            # For scan, we would need to run multiple experiments
            print(f"Rabi scan würde {len(sequences)} Sequenzen ausführen")
            
        else:
            raise ValueError(f"Unbekannte Sequenz: {sequence_name}")
            
    def awg_stop(self):
        """Stoppt AWG-Ausgabe"""
        self.awg.stop()
        
    def set_awg_sample_rate(self, sample_rate: float):
        """Setzt AWG-Sample-Rate"""
        self.awg.sample_rate = sample_rate
        self.awg.dt = 1.0 / sample_rate
        
    def run_for(self, duration_ns: float):
        """Läuft für bestimmte Zeit"""
        target_time = self.sim_time + duration_ns * 1e-9
        
        print(f"Simuliere für {duration_ns} ns...")
        
        with open('run_output.txt', 'w') as f:
            f.write(f"# Run for {duration_ns} ns\n")
            f.write("# Format: [ns]: observables\n")
            
            while self.sim_time < target_time:
                self.rho = self.evolve_one_step(self.rho, self.dt)
                self.step_count += 1
                self.sim_time += self.dt
                
                # Output every ns
                if self.step_count % self.steps_per_ns == 0:
                    ns_time = int(self.sim_time * 1e9)
                    obs_str = self.get_observables()
                    f.write(f"{ns_time}ns: {obs_str}\n")
                    
                # Progress update
                if self.step_count % (self.steps_per_ns * 100) == 0:
                    progress = (self.sim_time - (target_time - duration_ns * 1e-9)) / (duration_ns * 1e-9) * 100
                    print(f"Progress: {progress:.1f}%")
        
        print(f"Simulation abgeschlossen bei {self.sim_time*1e9:.1f}ns")
        print(f"Final state: {self.get_observables()}")

if __name__ == "__main__":
    runner = SimpleRunner()
    runner.run_forever()