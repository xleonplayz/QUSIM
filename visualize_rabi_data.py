#!/usr/bin/env python3
"""
Visualize Rabi oscillation data from PODMR measurement
Each line = one pulse, each number = photon emission count
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data file
data_file = Path("sample/20250527-0002-49_Qubit_XQ1i_PODMR_amp_0.012V_laser_pulses.dat")

print(f"Loading data from: {data_file}")

# Read all lines from the file
with open(data_file, 'r') as f:
    lines = f.readlines()

print(f"Total number of pulses: {len(lines)}")

# Parse each line to get photon counts per pulse
photon_counts_per_pulse = []
pulse_lengths = []

for i, line in enumerate(lines):
    # Split the line by whitespace and convert to integers
    try:
        counts = [int(x) for x in line.strip().split() if x]
        total_counts = sum(counts)
        photon_counts_per_pulse.append(total_counts)
        pulse_lengths.append(len(counts))
    except ValueError:
        print(f"Warning: Could not parse line {i+1}")
        continue

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Rabi Oscillation Data Analysis\n20250527-0002-49_Qubit_XQ1i_PODMR_amp_0.012V', fontsize=14)

# 1. Total photon counts per pulse
ax1 = axes[0, 0]
pulse_numbers = np.arange(len(photon_counts_per_pulse))
ax1.plot(pulse_numbers, photon_counts_per_pulse, 'b.-', markersize=3)
ax1.set_xlabel('Pulse Number')
ax1.set_ylabel('Total Photon Count')
ax1.set_title('Total Photon Emissions per Pulse')
ax1.grid(True, alpha=0.3)

# 2. Histogram of photon counts
ax2 = axes[0, 1]
ax2.hist(photon_counts_per_pulse, bins=50, color='green', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Total Photon Count per Pulse')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Photon Counts')
ax2.grid(True, alpha=0.3)

# 3. Moving average to see oscillation pattern
ax3 = axes[1, 0]
window_size = 10
if len(photon_counts_per_pulse) > window_size:
    moving_avg = np.convolve(photon_counts_per_pulse, 
                             np.ones(window_size)/window_size, 
                             mode='valid')
    ax3.plot(pulse_numbers[:len(moving_avg)], moving_avg, 'r-', linewidth=2)
    ax3.plot(pulse_numbers, photon_counts_per_pulse, 'b.', alpha=0.3, markersize=2)
    ax3.set_xlabel('Pulse Number')
    ax3.set_ylabel('Photon Count')
    ax3.set_title(f'Moving Average (window={window_size}) vs Raw Data')
    ax3.legend(['Moving Average', 'Raw Data'])
else:
    ax3.plot(pulse_numbers, photon_counts_per_pulse, 'b.-')
    ax3.set_xlabel('Pulse Number')
    ax3.set_ylabel('Photon Count')
    ax3.set_title('Photon Count per Pulse')
ax3.grid(True, alpha=0.3)

# 4. FFT to find oscillation frequency
ax4 = axes[1, 1]
if len(photon_counts_per_pulse) > 10:
    # Remove DC component and apply FFT
    counts_centered = photon_counts_per_pulse - np.mean(photon_counts_per_pulse)
    fft_result = np.fft.fft(counts_centered)
    frequencies = np.fft.fftfreq(len(counts_centered))
    
    # Plot only positive frequencies
    pos_mask = frequencies > 0
    ax4.plot(frequencies[pos_mask], np.abs(fft_result[pos_mask]), 'purple')
    ax4.set_xlabel('Frequency (1/pulse)')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('FFT of Photon Count Signal')
    ax4.set_xlim(0, 0.5)  # Nyquist frequency
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Not enough data for FFT', 
             transform=ax4.transAxes, ha='center', va='center')

plt.tight_layout()

# Print statistics
print("\nStatistics:")
print(f"Total pulses: {len(photon_counts_per_pulse)}")
print(f"Mean photons per pulse: {np.mean(photon_counts_per_pulse):.2f}")
print(f"Std dev: {np.std(photon_counts_per_pulse):.2f}")
print(f"Min photons: {min(photon_counts_per_pulse)}")
print(f"Max photons: {max(photon_counts_per_pulse)}")

# Show first few lines to understand data structure
print("\nFirst 5 pulses (raw data):")
for i in range(min(5, len(lines))):
    print(f"Pulse {i+1}: {lines[i].strip()[:100]}{'...' if len(lines[i]) > 100 else ''}")

plt.show()