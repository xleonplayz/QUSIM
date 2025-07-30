#!/usr/bin/env python3
"""
Beispiel: Laser Ein/Aus während der Simulation

Verwendung:
1. Terminal 1: python laser_example.py
2. Terminal 2: 
   - echo "laser_on 1.0" > laser_control.txt  # 1mW einschalten
   - echo "laser_off" > laser_control.txt     # ausschalten
"""

import sys
import os
sys.path.append('src')
sys.path.append('interfaces')

import json
import time
import threading
from runner.runner import SimpleRunner

class LaserControlExample:
    def __init__(self):
        self.runner = SimpleRunner()
        self.control_file = "laser_control.txt"
        
        # Starte Control-Thread
        self.control_thread = threading.Thread(target=self.monitor_control_file, daemon=True)
        self.control_thread.start()
        
    def monitor_control_file(self):
        """Überwacht Steuerungsdatei für Laser-Befehle"""
        last_mtime = 0
        
        while True:
            try:
                if os.path.exists(self.control_file):
                    mtime = os.path.getmtime(self.control_file)
                    if mtime > last_mtime:
                        last_mtime = mtime
                        self.process_control_command()
                time.sleep(0.1)  # Check alle 100ms
            except Exception as e:
                print(f"Control file error: {e}")
                time.sleep(1)
    
    def process_control_command(self):
        """Verarbeitet Befehle aus der Steuerungsdatei"""
        try:
            with open(self.control_file, 'r') as f:
                command = f.read().strip()
            
            if command.startswith("laser_on"):
                parts = command.split()
                power = float(parts[1]) if len(parts) > 1 else 1.0
                detuning = float(parts[2]) if len(parts) > 2 else 0.0
                self.runner.laser_on(power, detuning)
                
            elif command == "laser_off":
                self.runner.laser_off()
                
            elif command.startswith("status"):
                stats = self.runner.laser.get_status()
                print(f"\nAktueller Laser-Status:")
                print(f"  Ein/Aus: {stats['is_on']}")
                print(f"  Leistung: {stats['power_mw']:.2f} mW")
                print(f"  Rabi-Frequenz: {stats['rabi_frequency_mhz']:.1f} MHz")
                print(f"  Sättigungsparameter: {stats['saturation_parameter']:.3f}")
                
        except Exception as e:
            print(f"Command error: {e}")
    
    def run_with_scheduled_laser(self):
        """Beispiel mit automatischer Laser-Sequenz"""
        print("Starte Simulation mit Laser-Sequenz...")
        print("  0-100ns: Laser AUS")
        print("  100-200ns: Laser EIN (0.5mW)")
        print("  200-300ns: Laser AUS")
        print("  300-400ns: Laser EIN (2.0mW)")
        print("  >400ns: Laser AUS")
        
        # Starte Runner in separatem Thread
        runner_thread = threading.Thread(target=self.runner.run_forever, daemon=True)
        runner_thread.start()
        
        # Laser-Sequenz
        try:
            time.sleep(1)  # Warte bis Runner läuft
            
            print("\n[100ns] Laser EIN: 0.5mW")
            self.runner.laser_on(0.5)
            time.sleep(2)
            
            print("\n[200ns] Laser AUS")
            self.runner.laser_off()
            time.sleep(2)
            
            print("\n[300ns] Laser EIN: 2.0mW")
            self.runner.laser_on(2.0)
            time.sleep(2)
            
            print("\n[400ns] Laser AUS")
            self.runner.laser_off()
            
            # Warte auf Strg+C
            runner_thread.join()
            
        except KeyboardInterrupt:
            print("\nLaser-Beispiel beendet")

if __name__ == "__main__":
    example = LaserControlExample()
    
    if len(sys.argv) > 1 and sys.argv[1] == "auto":
        # Automatische Sequenz
        example.run_with_scheduled_laser()
    else:
        # Manuelle Steuerung
        print("Laser-Steuerungs-Beispiel gestartet")
        print("Befehle in 'laser_control.txt' schreiben:")
        print("  laser_on 1.0     # 1mW einschalten")
        print("  laser_on 2.0 10  # 2mW, 10MHz verstimmt")
        print("  laser_off        # ausschalten")
        print("  status           # Status anzeigen")
        
        example.runner.run_forever()