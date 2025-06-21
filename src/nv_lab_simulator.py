#!/usr/bin/env python3
"""
NV Lab Simulator - Clean API Design
===================================
Simulates laboratory equipment for NV center experiments.
"""

import json
import time
import uuid
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from threading import Lock, Thread
import os

class SimulatorState(Enum):
    IDLE = "idle"
    LASER_ON = "laser_on"
    MW_CONFIGURED = "mw_configured"
    COUNTING = "counting"
    EXPERIMENT_RUNNING = "experiment_running"

@dataclass
class LaserConfig:
    status: str = "off"
    power_mW: float = 0.0
    wavelength_nm: float = 532.0
    actual_power_mW: float = 0.0

@dataclass
class MWConfig:
    configured: bool = False
    frequency_Hz: float = 2.87e9
    power_dBm: float = -10.0
    pulse_length_ns: float = 100.0

@dataclass
class CounterConfig:
    status: str = "idle"
    integration_time_ms: float = 1000.0
    bins: int = 100
    bin_width_ns: float = 10.0
    counts: List[int] = None
    total_counts: int = 0
    start_time: float = 0.0

@dataclass
class ExperimentStatus:
    experiment_id: str = ""
    type: str = ""
    status: str = "idle"
    progress: float = 0.0
    data: Dict = None

class NVLabSimulator:
    """Main simulator class that manages all laboratory equipment"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.state = SimulatorState.IDLE
        self.lock = Lock()
        
        # Equipment states
        self.laser = LaserConfig()
        self.mw = MWConfig()
        self.counter = CounterConfig()
        self.experiment = ExperimentStatus()
        
        # Load configuration
        self.config = self.load_config()
        
        # Background threads
        self._counting_thread = None
        self._experiment_thread = None
        
    def load_config(self) -> Dict:
        """Load simulator configuration from JSON file"""
        default_config = {
            "nv_center": {
                "zero_field_splitting_Hz": 2.87e9,
                "excited_state_lifetime_ns": 12.0,
                "spin_relaxation_time_ms": 5.0,
                "spin_coherence_time_us": 6.0
            },
            "environment": {
                "temperature_K": 4.0,
                "magnetic_field_mT": [0.0, 0.0, 1.0],
                "strain_MHz": 0.0
            },
            "detection": {
                "collection_efficiency": 0.15,
                "detector_efficiency": 0.7,
                "dark_counts_Hz": 200.0,
                "bin_width_ns": 10.0
            },
            "spin_bath": {
                "carbon13_concentration": 0.011,
                "nitrogen_density_ppm": 1.0
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            else:
                return default_config
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            return default_config
    
    def save_config(self, filename: Optional[str] = None) -> bool:
        """Save current configuration to JSON file"""
        try:
            config_path = filename or self.config_file
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def update_config(self, updates: Dict) -> List[str]:
        """Update configuration parameters"""
        updated_keys = []
        
        def update_nested(target, source, prefix=""):
            for key, value in source.items():
                current_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    update_nested(target[key], value, current_key)
                else:
                    if key in target:
                        target[key] = value
                        updated_keys.append(current_key)
        
        update_nested(self.config, updates)
        return updated_keys
    
    # ===== LASER CONTROL =====
    
    def laser_on(self, power_mW: float, wavelength_nm: float = 532.0) -> Dict:
        """Turn laser on with specified power"""
        with self.lock:
            # Simulate power fluctuations (±2%)
            actual_power = power_mW * (1 + np.random.normal(0, 0.02))
            
            self.laser.status = "on"
            self.laser.power_mW = power_mW
            self.laser.wavelength_nm = wavelength_nm
            self.laser.actual_power_mW = actual_power
            
            if self.state == SimulatorState.IDLE:
                self.state = SimulatorState.LASER_ON
            
            return {
                "status": "on",
                "actual_power_mW": round(actual_power, 3)
            }
    
    def laser_off(self) -> Dict:
        """Turn laser off"""
        with self.lock:
            self.laser.status = "off"
            self.laser.power_mW = 0.0
            self.laser.actual_power_mW = 0.0
            
            if self.state == SimulatorState.LASER_ON:
                self.state = SimulatorState.IDLE
            
            return {"status": "off"}
    
    def laser_status(self) -> Dict:
        """Get laser status"""
        return {
            "status": self.laser.status,
            "power_mW": self.laser.power_mW,
            "wavelength_nm": self.laser.wavelength_nm,
            "actual_power_mW": self.laser.actual_power_mW
        }
    
    # ===== MW CONTROL =====
    
    def mw_configure(self, frequency_Hz: float, power_dBm: float, pulse_length_ns: float) -> Dict:
        """Configure MW generator"""
        with self.lock:
            self.mw.configured = True
            self.mw.frequency_Hz = frequency_Hz
            self.mw.power_dBm = power_dBm
            self.mw.pulse_length_ns = pulse_length_ns
            
            if self.state in [SimulatorState.IDLE, SimulatorState.LASER_ON]:
                self.state = SimulatorState.MW_CONFIGURED
            
            return {
                "success": True,
                "configured": {
                    "frequency_Hz": frequency_Hz,
                    "power_dBm": power_dBm,
                    "pulse_length_ns": pulse_length_ns
                }
            }
    
    def mw_pulse(self, number_of_pulses: int = 1, repetition_rate_Hz: float = 1000.0) -> Dict:
        """Send MW pulses"""
        if not self.mw.configured:
            return {"success": False, "error": "MW not configured"}
        
        # Simulate pulse execution time
        pulse_duration = number_of_pulses / repetition_rate_Hz
        time.sleep(min(pulse_duration, 0.1))  # Cap at 100ms for simulation
        
        return {
            "success": True,
            "pulses_sent": number_of_pulses
        }
    
    def mw_status(self) -> Dict:
        """Get MW status"""
        return {
            "configured": self.mw.configured,
            "frequency_Hz": self.mw.frequency_Hz,
            "power_dBm": self.mw.power_dBm,
            "pulse_length_ns": self.mw.pulse_length_ns
        }
    
    # ===== PHOTON COUNTER =====
    
    def _simulate_photon_counts(self) -> List[int]:
        """Simulate photon counting based on current system state"""
        # Calculate expected count rate based on laser and MW state
        base_rate_Hz = self.config["detection"]["dark_counts_Hz"]
        
        if self.laser.status == "on":
            # Laser-induced fluorescence
            max_rate_Hz = 50000  # 50 kHz typical for NV centers
            laser_efficiency = min(self.laser.actual_power_mW / 5.0, 1.0)  # Saturate at 5mW
            collection_eff = self.config["detection"]["collection_efficiency"]
            detector_eff = self.config["detection"]["detector_efficiency"]
            
            # Calculate fluorescence rate
            fluorescence_rate = max_rate_Hz * laser_efficiency * collection_eff * detector_eff
            
            # Add MW effects if configured
            if self.mw.configured:
                # Calculate detuning effect
                nv_frequency = self.config["nv_center"]["zero_field_splitting_Hz"]
                detuning_Hz = abs(self.mw.frequency_Hz - nv_frequency)
                detuning_MHz = detuning_Hz / 1e6
                
                # Resonance causes population transfer -> reduced fluorescence
                if detuning_MHz < 10:  # Within 10 MHz
                    # On resonance: significant reduction
                    resonance_factor = np.exp(-detuning_MHz / 5.0)  # Lorentzian-like
                    mw_power_factor = min(10**(self.mw.power_dBm / 10.0) / 1e-3, 1.0)  # Normalize to 1mW
                    fluorescence_reduction = 0.3 * resonance_factor * mw_power_factor
                    fluorescence_rate *= (1 - fluorescence_reduction)
            
            base_rate_Hz += fluorescence_rate
        
        # Generate counts for each bin
        counts_per_bin = base_rate_Hz * (self.counter.bin_width_ns * 1e-9)
        counts = np.random.poisson(counts_per_bin, self.counter.bins).tolist()
        
        return counts
    
    def _counting_worker(self):
        """Background worker for photon counting"""
        start_time = time.time()
        
        while (time.time() - start_time) < (self.counter.integration_time_ms / 1000.0):
            if self.counter.status != "counting":
                break
            time.sleep(0.01)  # Check every 10ms
        
        if self.counter.status == "counting":
            # Generate final counts
            counts = self._simulate_photon_counts()
            
            with self.lock:
                self.counter.counts = counts
                self.counter.total_counts = sum(counts)
                self.counter.status = "ready"
                
                if self.state == SimulatorState.COUNTING:
                    if self.mw.configured:
                        self.state = SimulatorState.MW_CONFIGURED
                    elif self.laser.status == "on":
                        self.state = SimulatorState.LASER_ON
                    else:
                        self.state = SimulatorState.IDLE
    
    def counter_start(self, integration_time_ms: float, bins: int) -> Dict:
        """Start photon counting"""
        if self.counter.status == "counting":
            return {"success": False, "error": "Counter already running"}
        
        with self.lock:
            self.counter.status = "counting"
            self.counter.integration_time_ms = integration_time_ms
            self.counter.bins = bins
            self.counter.bin_width_ns = (integration_time_ms * 1e6) / bins  # Convert to ns
            self.counter.counts = None
            self.counter.total_counts = 0
            self.counter.start_time = time.time()
            
            self.state = SimulatorState.COUNTING
        
        # Start background counting
        self._counting_thread = Thread(target=self._counting_worker)
        self._counting_thread.start()
        
        return {
            "status": "counting",
            "expected_duration_ms": integration_time_ms
        }
    
    def counter_read(self) -> Dict:
        """Read photon counter results"""
        if self.counter.status == "counting":
            elapsed_ms = (time.time() - self.counter.start_time) * 1000
            progress = min(elapsed_ms / self.counter.integration_time_ms * 100, 100)
            
            return {
                "status": "counting",
                "progress": round(progress, 1),
                "elapsed_ms": round(elapsed_ms, 1)
            }
        
        elif self.counter.status == "ready":
            return {
                "status": "ready",
                "counts": self.counter.counts,
                "total_counts": self.counter.total_counts,
                "integration_time_ms": self.counter.integration_time_ms,
                "bin_width_ns": self.counter.bin_width_ns
            }
        
        else:
            return {
                "status": "idle",
                "message": "Counter not started"
            }
    
    def counter_stop(self) -> Dict:
        """Stop photon counting"""
        with self.lock:
            self.counter.status = "stopped"
            
            if self.state == SimulatorState.COUNTING:
                if self.mw.configured:
                    self.state = SimulatorState.MW_CONFIGURED
                elif self.laser.status == "on":
                    self.state = SimulatorState.LASER_ON
                else:
                    self.state = SimulatorState.IDLE
        
        return {"status": "stopped"}
    
    # ===== SYSTEM STATUS =====
    
    def get_system_status(self) -> Dict:
        """Get complete system status"""
        return {
            "state": self.state.value,
            "laser": self.laser_status(),
            "mw": self.mw_status(),
            "counter": {
                "status": self.counter.status,
                "integration_time_ms": self.counter.integration_time_ms,
                "bins": self.counter.bins
            },
            "experiment": {
                "status": self.experiment.status,
                "type": self.experiment.type,
                "progress": self.experiment.progress
            }
        }