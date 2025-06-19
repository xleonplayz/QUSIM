"""
NV Simulator Package
===================
🆕 HYPERREALISTIC: Phonon-Coupling + Time-dependent MW Pulses + All Advanced Physics
"""

from .nv import NVSimulator, AdvancedNVParams
from .webapp import run_webapp

__version__ = "2.0.0"
__author__ = "NV Simulator Team"

__all__ = ['NVSimulator', 'AdvancedNVParams', 'run_webapp']