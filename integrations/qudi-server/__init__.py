# -*- coding: utf-8 -*-
"""
QUSIM Mock Server - Standalone NV Center Hardware Simulator

This package provides a FastAPI-based mock server that simulates
NV center hardware for testing and development.

Components:
- Laser Simulator (532nm excitation)
- Fast Counter / Photon Counter
- Pulser / AWG Controller
- NV Physics Backend (JAX-based)

Start with:
    uvicorn main:app --reload --ws-max-size 1073741824
"""

__version__ = "1.0.0"
