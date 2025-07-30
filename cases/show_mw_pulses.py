#!/usr/bin/env python3
"""
Zeigt MW-Puls Parameter und Formen
"""

import sys
import os
sys.path.append('../interfaces')

import numpy as np
import matplotlib.pyplot as plt
from microwave.microwave_interface import MicrowaveInterface

def show_pulse_parameters():
    """Zeigt Puls-Parameter für verschiedene Leistungen"""
    
    mw = MicrowaveInterface()
    
    print("=== MW-PULS PARAMETER ===")
    print(f"{'Leistung [dBm]':>15} {'Rabi [MHz]':>12} {'π-Puls [ns]':>12} {'π/2-Puls [ns]':>14}")
    print("-" * 60)
    
    powers = [-30, -20, -15, -10, -5, 0, +5, +10]
    
    for power in powers:
        mw.turn_on(power)
        status = mw.get_status()
        
        rabi_mhz = status['rabi_frequency_mhz']
        
        if rabi_mhz > 0:
            pi_pulse_ns = 1000 / (2 * rabi_mhz)  # π = 1/(2f) in ns
            pi_half_ns = pi_pulse_ns / 2
            
            print(f"{power:>15.1f} {rabi_mhz:>12.2f} {pi_pulse_ns:>12.1f} {pi_half_ns:>14.1f}")
        else:
            print(f"{power:>15.1f} {'< 0.01':>12} {'> 50000':>12} {'> 25000':>14}")
        
        mw.turn_off()

def show_pulse_shapes():
    """Zeigt MW-Puls Formen"""
    
    mw = MicrowaveInterface()
    mw.turn_on(-10.0)  # -10dBm → ~3MHz
    
    # Zeit-Array
    t = np.linspace(0, 200e-9, 1000)  # 0-200ns
    
    # Puls-Parameter
    pi_pulse = mw.pi_pulse('x')
    pi_duration = pi_pulse['duration_ns'] * 1e-9
    rabi_freq = mw.rabi_freq
    
    print(f"\n=== MW-PULS FORMEN ===")
    print(f"Leistung: -10dBm")
    print(f"Rabi-Frequenz: {rabi_freq/1e6/2/np.pi:.2f} MHz")
    print(f"π-Puls Dauer: {pi_duration*1e9:.1f}ns")
    
    # Verschiedene Puls-Formen
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Rechteckpuls (Standard)
    square_pulse = np.where((t >= 50e-9) & (t <= 50e-9 + pi_duration), 1.0, 0.0)
    axes[0,0].plot(t*1e9, square_pulse * rabi_freq/1e6/2/np.pi, 'b-', linewidth=2)
    axes[0,0].set_title('Rechteck π-Puls')
    axes[0,0].set_xlabel('Zeit [ns]')
    axes[0,0].set_ylabel('Rabi-Frequenz [MHz]')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Gauss-Puls
    sigma = pi_duration / 6  # 6σ Breite
    t_center = 100e-9
    gauss_pulse = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    # Normiere für π-Rotation: ∫ Ω dt = π
    gauss_pulse = gauss_pulse * np.pi / (np.trapz(gauss_pulse, t))
    axes[0,1].plot(t*1e9, gauss_pulse/1e6/2/np.pi, 'r-', linewidth=2)
    axes[0,1].set_title('Gauss π-Puls')
    axes[0,1].set_xlabel('Zeit [ns]')
    axes[0,1].set_ylabel('Rabi-Frequenz [MHz]')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Hamiltonian-Komponenten
    H_x = 0.5 * rabi_freq * np.cos(0)  # Phase = 0 → x-Achse
    H_y = 0.5 * rabi_freq * np.sin(0)
    
    t_pulse = t[(t >= 50e-9) & (t <= 50e-9 + pi_duration)]
    H_x_pulse = np.full_like(t_pulse, H_x)
    H_y_pulse = np.full_like(t_pulse, H_y)
    
    axes[1,0].plot(t_pulse*1e9, H_x_pulse/1e6/2/np.pi, 'b-', label='Hₓ', linewidth=2)
    axes[1,0].plot(t_pulse*1e9, H_y_pulse/1e6/2/np.pi, 'r-', label='Hᵧ', linewidth=2)
    axes[1,0].set_title('Hamiltonian-Komponenten (x-Puls)')
    axes[1,0].set_xlabel('Zeit [ns]')
    axes[1,0].set_ylabel('H-Feld [MHz]')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Phase-modulated Pulse (y-Puls)
    y_pulse = mw.pi_pulse('y')
    H_x_y = 0.5 * rabi_freq * np.cos(np.pi/2)  # Phase = π/2 → y-Achse
    H_y_y = 0.5 * rabi_freq * np.sin(np.pi/2)
    
    H_x_y_pulse = np.full_like(t_pulse, H_x_y)
    H_y_y_pulse = np.full_like(t_pulse, H_y_y)
    
    axes[1,1].plot(t_pulse*1e9, H_x_y_pulse/1e6/2/np.pi, 'b-', label='Hₓ', linewidth=2)
    axes[1,1].plot(t_pulse*1e9, H_y_y_pulse/1e6/2/np.pi, 'r-', label='Hᵧ', linewidth=2)
    axes[1,1].set_title('Hamiltonian-Komponenten (y-Puls)')
    axes[1,1].set_xlabel('Zeit [ns]')
    axes[1,1].set_ylabel('H-Feld [MHz]')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mw_pulse_shapes.png', dpi=150, bbox_inches='tight')
    print("Puls-Formen gespeichert: mw_pulse_shapes.png")
    
    plt.show()

def show_bloch_evolution():
    """Zeigt Bloch-Sphäre Evolution während Pulsen"""
    
    print(f"\n=== BLOCH-SPHÄRE EVOLUTION ===")
    
    # Spin-1/2 Näherung für Visualisierung
    # |ms=0⟩ → |↓⟩, |ms=±1⟩ → |↑⟩
    
    mw = MicrowaveInterface()
    mw.turn_on(-10.0)
    
    # π-Puls Parameter
    pi_pulse = mw.pi_pulse('x')
    duration_ns = pi_pulse['duration_ns']
    rabi_mhz = mw.rabi_freq / 1e6 / 2 / np.pi
    
    print(f"π-Puls (x-Achse):")
    print(f"  Dauer: {duration_ns:.1f}ns")
    print(f"  Rabi: {rabi_mhz:.2f}MHz")
    print(f"  Rotation: |ms=0⟩ → |ms=±1⟩")
    print(f"  Bloch: |↓⟩ → |↑⟩ um x-Achse")
    
    print(f"\nπ/2-Puls (x-Achse):")
    pi_half = mw.pi_half_pulse('x')
    print(f"  Dauer: {pi_half['duration_ns']:.1f}ns")
    print(f"  Rotation: |ms=0⟩ → (|ms=0⟩ + i|ms=±1⟩)/√2")
    print(f"  Bloch: |↓⟩ → (|↓⟩ + i|↑⟩)/√2")
    
    print(f"\nπ/2-Puls (y-Achse):")
    pi_half_y = mw.pi_half_pulse('y')
    print(f"  Dauer: {pi_half_y['duration_ns']:.1f}ns")
    print(f"  Rotation: |ms=0⟩ → (|ms=0⟩ + |ms=±1⟩)/√2")
    print(f"  Bloch: |↓⟩ → (|↓⟩ + |↑⟩)/√2")

def show_sequences():
    """Zeigt komplette Puls-Sequenzen"""
    
    print(f"\n=== PULS-SEQUENZEN ===")
    
    mw = MicrowaveInterface()
    mw.turn_on(-10.0)
    
    # Ramsey-Sequenz
    ramsey = mw.ramsey_sequence(free_evolution_ns=1000)
    
    print("Ramsey-Sequenz: π/2(x) - τ - π/2(y)")
    for i, step in enumerate(ramsey):
        if step['type'] == 'pi_half_pulse':
            print(f"  {i+1}. π/2-Puls: {step['duration_ns']:.1f}ns, Achse: {step['axis']}")
        elif step['type'] == 'free_evolution':
            print(f"  {i+1}. Freie Evolution: {step['duration_ns']:.0f}ns")
    
    # Rabi-Sequenz
    print(f"\nRabi-Oszillation:")
    rabi_pulses = mw.rabi_sequence(max_duration_ns=200, steps=10)
    
    for i, pulse in enumerate(rabi_pulses[:5]):  # Nur erste 5 zeigen
        print(f"  Puls {i+1}: {pulse['duration_ns']:.1f}ns @ {pulse['power_dbm']}dBm")
    print(f"  ... ({len(rabi_pulses)} Pulse total)")

def main():
    show_pulse_parameters()
    show_bloch_evolution()
    show_sequences()
    
    # Optional: Grafiken
    import matplotlib
    try:
        show_pulse_shapes()
    except:
        print("\nGrafiken nicht verfügbar (matplotlib fehlt)")

if __name__ == "__main__":
    main()