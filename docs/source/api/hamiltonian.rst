===================
Hamiltonian Module
===================

The Hamiltonian module provides a modular builder pattern for constructing
NV center Hamiltonians from individual physical terms.

Hamiltonian Builder
===================

.. automodule:: sim.hamiltonian.builder
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

.. code-block:: python

   from sim import HamiltonianBuilder
   from sim.hamiltonian.terms import ZFS, Zeeman, HyperfineN14

   # Create builder
   H = HamiltonianBuilder()

   # Add terms
   H.add(ZFS(D=2.87))           # Zero-field splitting
   H.add(Zeeman(B=10))          # 10 mT magnetic field
   H.add(HyperfineN14())        # N14 hyperfine structure

   # Build 18×18 matrix at time t
   H_matrix = H.build(t=0)

   # Check if time-dependent
   print(H.is_time_dependent)   # False for static terms

   # Get eigenvalues
   eigvals = H.eigenvalues_ghz()

Hamiltonian Terms
=================

All Hamiltonian terms inherit from ``HamiltonianTerm`` and implement
a ``build(t)`` method returning an 18×18 Hermitian matrix.

Zero-Field Splitting
--------------------

.. automodule:: sim.hamiltonian.terms.zfs
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from sim.hamiltonian.terms import ZFS

   # Standard NV center
   zfs = ZFS(D=2.87)  # GHz

   # With strain splitting
   zfs = ZFS(D=2.87, E=0.005)  # 5 MHz E-term

Zeeman Effect
-------------

.. automodule:: sim.hamiltonian.terms.zeeman
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from sim.hamiltonian.terms import Zeeman

   # Field along NV axis
   zeeman = Zeeman(B=10)  # 10 mT

   # Arbitrary field direction
   zeeman = Zeeman(B=[5, 0, 10])  # mT

   # Get splitting
   print(zeeman.splitting_mhz())  # ~560 MHz

N14 Hyperfine
-------------

.. automodule:: sim.hamiltonian.terms.hyperfine_n14
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from sim.hamiltonian.terms import HyperfineN14

   # Default literature values
   hf = HyperfineN14()

   # Custom parameters
   hf = HyperfineN14(
       A_parallel=-2.14,  # MHz
       A_perp=-2.70,      # MHz
       P_quad=-5.0        # MHz
   )

C13 Hyperfine
-------------

.. automodule:: sim.hamiltonian.terms.hyperfine_c13
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from sim.hamiltonian.terms import HyperfineC13

   # C13 at specific position (in Angstroms)
   hf_c13 = HyperfineC13(position=[1.54, 0, 0])

Microwave Drive
---------------

.. automodule:: sim.hamiltonian.terms.mw_drive
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from sim.hamiltonian.terms import MicrowaveDrive

   # Constant drive (10 MHz Rabi frequency)
   mw = MicrowaveDrive(omega=10, detuning=0, phase=0)

   # Time-dependent Rabi frequency
   def rabi(t):
       return 10 if 0 < t < 50e-9 else 0

   mw = MicrowaveDrive(omega=rabi)

Optical Drive
-------------

.. automodule:: sim.hamiltonian.terms.optical
   :members:
   :undoc-members:
   :show-inheritance:

Stark Effect
------------

.. automodule:: sim.hamiltonian.terms.stark
   :members:
   :undoc-members:
   :show-inheritance:

Strain
------

.. automodule:: sim.hamiltonian.terms.strain
   :members:
   :undoc-members:
   :show-inheritance:

Jahn-Teller
-----------

.. automodule:: sim.hamiltonian.terms.jahn_teller
   :members:
   :undoc-members:
   :show-inheritance:

Creating Custom Terms
=====================

To create a custom Hamiltonian term:

.. code-block:: python

   from sim.hamiltonian.terms.base import HamiltonianTerm
   import numpy as np

   class CustomTerm(HamiltonianTerm):
       def __init__(self, parameter, name=None):
           super().__init__(name=name)
           self.parameter = parameter

       @property
       def is_time_dependent(self):
           return False

       def build(self, t=0.0):
           # Return 18×18 Hermitian matrix
           H = np.zeros((18, 18), dtype=np.complex128)
           # ... fill matrix ...
           return H
