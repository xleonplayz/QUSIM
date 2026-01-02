<p align="center">
  <img src="image.png" alt="QUSIM Banner" width="100%">
</p>

<p align="center">
  <a href="https://github.com/LeonKaiser/QUSIM/actions/workflows/tests.yml">
    <img src="https://github.com/LeonKaiser/QUSIM/actions/workflows/tests.yml/badge.svg" alt="Tests">
  </a>
  <a href="https://github.com/LeonKaiser/QUSIM/releases">
    <img src="https://img.shields.io/badge/version-0.1.0-blue" alt="Version">
  </a>
  <a href="https://qusim.readthedocs.io">
    <img src="https://img.shields.io/badge/docs-readthedocs-green" alt="Documentation">
  </a>
  <a href="docs/QUSIM_Documentation.pdf">
    <img src="https://img.shields.io/badge/PDF-Download-red" alt="PDF Documentation">
  </a>
</p>

---

# QUSIM - Quantum Simulator for NV Centers

A physics-based simulation framework for Nitrogen-Vacancy (NV) centers in diamond. QUSIM solves the full Lindblad master equation for realistic spin dynamics, optical readout, and microwave control.

## Features

- **Full Lindblad Dynamics**: 18×18 density matrix evolution (|g/e⟩ ⊗ |ms⟩ ⊗ |mI⟩)
- **Time-dependent Hamiltonians**: Pulsed laser and microwave control
- **Modular Architecture**: Plug-and-play Hamiltonian terms (ZFS, Zeeman, Hyperfine, Strain, Stark, ...)
- **Realistic Photon Counting**: Based on excited state population with Poisson statistics
- **Qudi Integration**: Drop-in hardware module for [Qudi](https://github.com/Ulm-IQO/qudi)

## Quick Start

```python
from sim import HamiltonianBuilder
from sim.hamiltonian.terms import ZFS, MicrowaveDrive, OpticalCoupling
from sim.dynamics import LindbladSolver
from sim.states import ground_state

# Build Hamiltonian
H = HamiltonianBuilder()
H.add(ZFS(D=2.87))                              # Zero-field splitting
H.add(MicrowaveDrive(omega=10, phase=0))        # 10 MHz Rabi frequency
H.add(OpticalCoupling(omega=50))                # Laser excitation

# Setup Lindblad solver with dissipation
solver = LindbladSolver(H)
# ... add dissipators ...

# Run simulation
rho0 = ground_state(ms=0, mI=0)
result = solver.evolve(rho0, t_span=(0, 1e-6), n_steps=100)
```

## Documentation

Full documentation available at [qusim.readthedocs.io](https://qusim.readthedocs.io) or download the [PDF](docs/QUSIM_Documentation.pdf).

## Installation

```bash
git clone https://github.com/LeonKaiser/QUSIM.git
cd QUSIM
pip install -e .
```

## Project Structure

```
QUSIM/
├── sim/                    # Core simulation library
│   ├── core/               # Operators, constants
│   ├── hamiltonian/        # Hamiltonian builder & terms
│   ├── dynamics/           # Lindblad solver, dissipation
│   └── states/             # Density matrices, projectors
├── integrations/           # Qudi integration
├── experiments/            # Example experiments
└── docs/                   # Documentation (Sphinx)
```

## Implemented Hamiltonian Terms

| Term | Description |
|------|-------------|
| `ZFS` | Zero-field splitting (D, E) |
| `Zeeman` | Magnetic field coupling |
| `HyperfineN14` | N14 nuclear hyperfine + quadrupole |
| `HyperfineC13` | C13 dipolar coupling |
| `MicrowaveDrive` | Time-dependent MW control |
| `OpticalCoupling` | Laser excitation |
| `Strain` | Crystal strain |
| `Stark` | Electric field (DC/AC) |

---

<p align="center">
  <sub>
    Developed at the <a href="https://msqc.cgi-host6.rz.uni-frankfurt.de/">Magnetometry and Spin Quantum Computation Group (MSQC)</a><br>
    Goethe University Frankfurt, Germany
  </sub>
</p>

<p align="center">
  <b>Author:</b> Leon Kaiser<br>
  <a href="mailto:l.kaiser@em.uni-frankfurt.de">l.kaiser@em.uni-frankfurt.de</a>
</p>

---

<p align="center">
  <sub>
    This software is provided for scientific and educational purposes.<br>
    Free to use, modify, and distribute with attribution.
  </sub>
</p>
