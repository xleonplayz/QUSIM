# NEWSIM - NV Center Quantum Simulator

Leon Kaiser, MSQC Goethe University  
I.kaiser[at]em.uni-frankfurt.de

## Overview

NEWSIM is a quantum simulator for nitrogen-vacancy (NV) centers in diamond. It solves the Lindblad master equation for open quantum systems with realistic NV physics including zero-field splitting, hyperfine coupling, strain, and environmental interactions.

## Core Features

- **18×18 Hilbert space**: Electronic spin + N14 nuclear spin + C13 bath
- **Lindblad dynamics**: Spontaneous emission, dephasing, relaxation
- **Control interfaces**: Laser excitation and microwave spin control
- **Photon counting**: Realistic detection with efficiency and dark counts
- **Physical effects**: Zeeman, strain, Stark shift, Jahn-Teller

## Usage

Basic simulation:
```bash
cd runner
python3 runner.py
```

Laser-photon experiment:
```bash
cd cases
python3 laser_photon_test.py 1000 1.0  # 1000ns, 1mW laser
```

Microwave control:
```bash
python3 mw_laser_test.py rabi  # Rabi oscillations
```

## Structure

- `src/`: Core physics modules (Hamiltonian, Lindblad, observables)
- `interfaces/`: Hardware control (laser, microwave, photon counter)
- `runner/`: Main simulation engine
- `cases/`: Example experiments and tests

## Requirements

Python 3.7+, NumPy, SciPy