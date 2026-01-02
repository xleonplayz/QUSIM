=============
Core Module
=============

The core module provides fundamental constants and operators for NV center simulation.

Physical Constants
==================

.. automodule:: sim.core.constants
   :members:
   :undoc-members:
   :show-inheritance:

Unit Conversions
----------------

The library uses practical laboratory units for the API:

- Frequencies: GHz, MHz, kHz
- Magnetic fields: mT
- Lengths: μm, nm

Internally, all values are converted to **rad/s** (with ℏ=1).

.. code-block:: python

   from sim.core.constants import GHZ, MHZ, MT

   D = 2.87 * GHZ    # ZFS in rad/s
   B = 10 * MT       # 10 mT in Tesla
   A = 2.14 * MHZ    # Hyperfine in rad/s

Available Constants
-------------------

**Unit Conversions:**

- ``GHZ``: 1 GHz in rad/s (≈ 6.28×10⁹)
- ``MHZ``: 1 MHz in rad/s (≈ 6.28×10⁶)
- ``KHZ``: 1 kHz in rad/s
- ``MT``: 1 mT in Tesla (10⁻³)

**Fundamental Constants:**

- ``HBAR``: Reduced Planck constant (1.055×10⁻³⁴ J·s)
- ``MU_B``: Bohr magneton (9.274×10⁻²⁴ J/T)
- ``KB``: Boltzmann constant (1.381×10⁻²³ J/K)

**NV-Specific:**

- ``D_GS``: Ground state ZFS (2.87 GHz)
- ``D_ES``: Excited state ZFS (1.42 GHz)
- ``GAMMA_E``: Electron gyromagnetic ratio
- ``A_PARALLEL_N14``, ``A_PERP_N14``: N14 hyperfine couplings
- ``P_N14``: N14 quadrupole parameter

Spin Operators
==============

.. automodule:: sim.core.operators
   :members:
   :undoc-members:
   :show-inheritance:

Spin-1 Operators
----------------

The ``spin1_operators()`` function returns a container with all standard spin-1 operators:

.. code-block:: python

   from sim.core.operators import spin1_operators

   ops = spin1_operators()
   Sx = ops.Sx  # 3×3 spin-x matrix
   Sy = ops.Sy  # 3×3 spin-y matrix
   Sz = ops.Sz  # 3×3 spin-z matrix
   Sp = ops.Sp  # Raising operator S+
   Sm = ops.Sm  # Lowering operator S-

**Properties:**

- Basis: {|+1⟩, |0⟩, |-1⟩}
- Sz is diagonal with eigenvalues +1, 0, -1
- [Sx, Sy] = iSz (and cyclic)
- S² = 2𝟙 (for S=1)

Extending to 18×18
------------------

Operators can be extended to the full 18-dimensional Hilbert space:

.. code-block:: python

   from sim.core.operators import extend_to_18x18, extend_electron_only

   # Acts on both g and e manifolds
   H_18 = extend_to_18x18(H_3x3)

   # Acts only on ground state
   H_g = extend_electron_only(H_3x3, manifold="ground")

   # Acts only on excited state
   H_e = extend_electron_only(H_3x3, manifold="excited")

Nuclear Operators
=================

.. automodule:: sim.core.nuclear_operators
   :members:
   :undoc-members:
   :show-inheritance:

The nuclear operators module provides spin-1/2 operators for C13 and utilities
for calculating dipolar hyperfine tensors.
