"""
NV Simulator Integration für Mock Server
Integriert nv_simulator_v3.py in den FastAPI mock_server
"""

import os
import sys
import configparser
import numpy as np
from typing import List, Dict, Optional

# Import NV Simulator V3 from local hardware directory
try:
    from .nv_simulator_v3 import SimConfig, NVSimulator, AWGSpecs, OutputCfg
    NV_SIMULATOR_AVAILABLE = True
    print("NV Simulator V3 imported successfully")
except ImportError as e:
    print(f"WARNING: nv_simulator_v3.py not found - using dummy data: {e}")
    NV_SIMULATOR_AVAILABLE = False

class NVSimulatorBackend:
    """
    Backend-Klasse die den NV Simulator v3 für den mock_server wraps.
    Lädt Konfiguration aus system.conf und stellt FastAPI-kompatible Methoden bereit.
    """

    def __init__(self):
        self.config = self._load_system_config()
        self.sim_config = None
        self.simulator = None
        self.last_measurement = None
        self._setup_simulator()

    def _load_system_config(self) -> configparser.ConfigParser:
        """Lädt die system.conf Konfigurationsdatei"""
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), 'system.conf')

        if os.path.exists(config_path):
            config.read(config_path)
            print(f"Loaded NV system configuration from {config_path}")
        else:
            print(f"WARNING: system.conf not found at {config_path}")
            # Fallback zu Default-Werten
            self._create_default_config(config)

        return config

    def _parse_float(self, value_str: str) -> float:
        """Parse float value, stripping comments"""
        if isinstance(value_str, str):
            # Remove comments (everything after #)
            clean_value = value_str.split('#')[0].strip().strip('"\'')
            return float(clean_value)
        else:
            return float(value_str)

    def _parse_int(self, value_str: str) -> int:
        """Parse int value, stripping comments"""
        if isinstance(value_str, str):
            clean_value = value_str.split('#')[0].strip().strip('"\'')
            return int(clean_value)
        else:
            return int(value_str)

    def _parse_bool(self, value_str: str) -> bool:
        """Parse bool value, stripping comments"""
        if isinstance(value_str, str):
            clean_value = value_str.split('#')[0].strip().strip('"\'').lower()
            return clean_value in ('true', '1', 'yes', 'on')
        else:
            return bool(value_str)

    def _parse_string(self, value_str: str) -> str:
        """Parse string value, stripping comments"""
        if isinstance(value_str, str):
            clean_value = value_str.split('#')[0].strip().strip('"\'')
            return clean_value
        else:
            return str(value_str)

    def _create_default_config(self, config: configparser.ConfigParser):
        """Erstellt Default-Konfiguration falls system.conf fehlt"""
        config['nv_system'] = {
            'lambda_par': '5.3e9',
            'lambda_perp': '0.3e9',
            'Delta_ss': '1.4e9',
            'Pi_x': '50e6',
            'Pi_y': '0.0'
        }
        config['detector'] = {
            'model': 'nonparalyzable',
            'dead_time': '60e-9',
            'afterpulse_p': '0.01'
        }
        config['awg'] = {
            'max_sample_rate': '50e9',
            'dac_bits': '10',
            'channels': '2'
        }

    def _setup_simulator(self):
        """Initialisiert den NV Simulator mit Konfiguration aus system.conf"""
        if not NV_SIMULATOR_AVAILABLE:
            return

        self.sim_config = SimConfig()

        # NV System Parameter aus Config
        if 'nv_system' in self.config:
            nv = self.config['nv_system']
            self.sim_config.nv.lambda_par = self._parse_float(nv.get('lambda_par', '5.3e9'))
            self.sim_config.nv.lambda_perp = self._parse_float(nv.get('lambda_perp', '0.3e9'))
            self.sim_config.nv.Delta_ss = self._parse_float(nv.get('Delta_ss', '1.4e9'))
            self.sim_config.nv.Pi_x = self._parse_float(nv.get('Pi_x', '50e6'))
            self.sim_config.nv.Pi_y = self._parse_float(nv.get('Pi_y', '0.0'))

        # PL Parameter
        if 'photoluminescence' in self.config:
            pl = self.config['photoluminescence']
            self.sim_config.pl.prompt_amp = self._parse_float(pl.get('prompt_amp', '0.2e8'))
            self.sim_config.pl.prompt_tau = self._parse_float(pl.get('prompt_tau', '40e-9'))
            self.sim_config.pl.measured_at_detector = self._parse_bool(pl.get('measured_at_detector', 'true'))
            self.sim_config.seq.laser_tau = self._parse_float(pl.get('laser_tau', '20e-9'))

        # Detector Parameter
        if 'detector' in self.config:
            det = self.config['detector']
            self.sim_config.det.model = self._parse_string(det.get('model', 'nonparalyzable'))
            self.sim_config.det.dead_time = self._parse_float(det.get('dead_time', '60e-9'))
            self.sim_config.det.afterpulse_p = self._parse_float(det.get('afterpulse_p', '0.01'))
            self.sim_config.det.afterpulse_mode = self._parse_string(det.get('afterpulse_mode', 'mixture'))
            self.sim_config.det.afterpulse_tau1 = self._parse_float(det.get('afterpulse_tau1', '40e-9'))
            self.sim_config.det.afterpulse_tau2 = self._parse_float(det.get('afterpulse_tau2', '200e-9'))
            self.sim_config.det.afterpulse_w1 = self._parse_float(det.get('afterpulse_w1', '0.7'))
            self.sim_config.det.afterpulse_generations = self._parse_int(det.get('afterpulse_generations', '2'))
            self.sim_config.det.overdispersion_k = self._parse_float(det.get('overdispersion_k', '300.0'))

        # Noise Parameter
        if 'noise' in self.config:
            noise = self.config['noise']
            self.sim_config.noise.rin_enable = self._parse_bool(noise.get('rin_enable', 'true'))
            self.sim_config.noise.rin_tau = self._parse_float(noise.get('rin_tau', '10e-9'))
            self.sim_config.noise.rin_sigma = self._parse_float(noise.get('rin_sigma', '0.12'))

        # Simulation Parameter
        if 'simulation' in self.config:
            sim = self.config['simulation']
            self.sim_config.seq.oversample_per_ns = self._parse_int(sim.get('oversample_per_ns', '30'))
            self.sim_config.positivity_clamp = self._parse_bool(sim.get('positivity_clamp', 'true'))
            self.sim_config.clamp_epsilon = self._parse_float(sim.get('clamp_epsilon', '1e-15'))
            self.sim_config.seed = self._parse_int(sim.get('seed', '42'))

        # AWG Specs
        if 'awg' in self.config:
            awg = self.config['awg']
            self.awg_specs = AWGSpecs(
                channels=self._parse_int(awg.get('channels', '2')),
                max_sample_rate=self._parse_float(awg.get('max_sample_rate', '50e9')),
                dac_bits=self._parse_int(awg.get('dac_bits', '10')),
                rf_max_carrier=self._parse_float(awg.get('rf_max_carrier', '20e9'))
            )
        else:
            # Default AWG specs
            self.awg_specs = AWGSpecs(
                channels=2,
                max_sample_rate=50e9,
                dac_bits=10,
                rf_max_carrier=20e9
            )

        print("NV Simulator backend initialized with system.conf parameters")

    def configure_measurement(self, bin_width_s: float, record_length_s: float,
                            number_of_gates: int = 1, mw_params: Optional[Dict] = None):
        """
        Konfiguriert das Measurement basierend auf FastAPI Parametern
        """
        if not NV_SIMULATOR_AVAILABLE:
            return

        # Setze bin width und measurement Parameter
        self.sim_config.seq.bin_ns = bin_width_s * 1e9  # Convert s to ns

        # Setze Anzahl accumulated shots basierend auf gates (begrenzt für Performance)
        self.sim_config.det.shots_accum = min(1000, max(100, number_of_gates * 50))

        # Berechne readout time basierend auf record_length
        self.sim_config.seq.t_readout = record_length_s

        # Reduziere oversampling für bessere Performance in Tests
        if record_length_s < 200e-9:  # Für kurze Messungen
            self.sim_config.seq.oversample_per_ns = 10  # Reduced from 30

        # MW Parameter falls vorhanden
        if mw_params:
            if 'frequency' in mw_params:
                self.sim_config.mw.omega_mw = mw_params['frequency']  # Hz
            if 'power' in mw_params:
                self.sim_config.mw.Omega0 = mw_params['power']  # Hz
            if 'duration' in mw_params:
                self.sim_config.seq.t_mw = mw_params['duration']  # s
            if 'amplitude' in mw_params:
                self.sim_config.seq.mw_amp = mw_params['amplitude']  # 0-1
        else:
            # Default MW parameters für realistische Simulation
            self.sim_config.mw.Omega0 = 8e6        # 8 MHz Rabi frequency
            self.sim_config.mw.omega_mw = 2.87e9   # 2.87 GHz frequency
            self.sim_config.seq.t_mw = 60e-9       # 60 ns pulse
            self.sim_config.seq.mw_amp = 1.0       # full amplitude

        # Erstelle Simulator
        try:
            self.simulator = NVSimulator(self.sim_config, self.awg_specs)
            print(f"NV Simulator configured: bin={bin_width_s*1e9:.1f}ns, record={record_length_s*1e9:.1f}ns")
        except Exception as e:
            print(f"Error creating NV simulator: {e}")
            self.simulator = None

    def run_measurement(self) -> List[int]:
        """
        Führt eine Messung mit dem NV Simulator durch
        Gibt Photon counts zurück die kompatibel mit FastAPI sind
        """
        if not NV_SIMULATOR_AVAILABLE or self.simulator is None:
            return self._generate_dummy_data()

        try:
            # Führe Simulation durch
            t_ns, counts = self.simulator.simulate()

            # Speichere letztes Measurement für Debugging
            self.last_measurement = {
                'time_ns': t_ns,
                'counts': counts,
                'total_counts': counts.sum(),
                'config': {
                    'bin_ns': self.sim_config.seq.bin_ns,
                    'mw_freq_ghz': self.sim_config.mw.omega_mw / 1e9,
                    'mw_power_mhz': self.sim_config.mw.Omega0 / 1e6,
                    'shots': self.sim_config.det.shots_accum
                }
            }

            # Konvertiere zu int List für API compatibility
            counts_list = [int(c) for c in counts]

            print(f"NV Simulation completed: {len(counts_list)} bins, {sum(counts_list)} total counts")
            return counts_list

        except Exception as e:
            print(f"Error in NV simulation: {e}")
            return self._generate_dummy_data()

    def _generate_dummy_data(self) -> List[int]:
        """Fallback dummy data wenn NV Simulator nicht verfügbar"""
        # Lade ursprüngliche dummy daten als fallback
        try:
            dummy_file = os.path.join(os.path.dirname(__file__), 'FastComTec_demo_timetrace.asc')
            data = np.loadtxt(dummy_file, dtype='int64').tolist()
            print(f"Using dummy data fallback: {len(data)} points")
            return data
        except:
            # Noch simpler fallback
            dummy_data = np.random.poisson(1.5, 1000).tolist()
            print("Using generated dummy data fallback")
            return dummy_data

    def get_measurement_info(self) -> Dict:
        """Gibt Info über letzte Messung zurück (für Debugging)"""
        if self.last_measurement:
            return self.last_measurement
        else:
            return {"error": "No measurement performed yet"}

    def update_mw_parameters(self, frequency_hz: Optional[float] = None,
                           power_hz: Optional[float] = None,
                           duration_s: Optional[float] = None,
                           amplitude: Optional[float] = None):
        """Update MW parameters dynamisch"""
        if not self.sim_config:
            return

        if frequency_hz is not None:
            self.sim_config.mw.omega_mw = frequency_hz
        if power_hz is not None:
            self.sim_config.mw.Omega0 = power_hz
        if duration_s is not None:
            self.sim_config.seq.t_mw = duration_s
        if amplitude is not None:
            self.sim_config.seq.mw_amp = amplitude

# Global instance für den mock_server
nv_backend = NVSimulatorBackend()