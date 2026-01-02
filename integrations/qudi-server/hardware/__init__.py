# -*- coding: utf-8 -*-
"""
Hardware simulators for the QUSIM mock server.

Modules:
- laser_simulator: Laser power and shutter control
- fastcounter_simulator: Photon counting and time traces
- pulser_simulator: AWG/Pulser waveform control
- nv_simulator_v3: Realistic NV center physics (JAX)
- nv_simulator_integration: Integration layer
"""

from .laser_simulator import router as laser_router
from .fastcounter_simulator import router as fastcounter_router
from .pulser_simulator import router as pulser_router

__all__ = ['laser_router', 'fastcounter_router', 'pulser_router']
