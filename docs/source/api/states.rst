==============
States Module
==============

The states module provides functions for creating initial states, density
matrices, and measurement projectors.

Initial States
==============

.. automodule:: sim.states.initial_states
   :members:
   :undoc-members:
   :show-inheritance:

Common States
-------------

.. code-block:: python

   from sim.states import (
       ground_state,
       excited_state,
       thermal_state,
       superposition
   )

   # Ground state |g, ms=0, mI=0⟩
   rho0 = ground_state()

   # Excited state |e, ms=0, mI=0⟩
   rho_e = excited_state()

   # Thermal state at temperature T
   rho_th = thermal_state(T=300)  # Kelvin

   # Superposition state
   rho_sup = superposition(
       states=[(0, 0, 0), (0, 1, 0)],  # (g/e, ms, mI)
       weights=[1, 1]
   )

Spin Eigenstates
----------------

Create states with specific spin projections:

.. code-block:: python

   from sim.states import spin_state

   # |g, ms=+1, mI=0⟩
   rho = spin_state(ms=+1, mI=0, manifold="ground")

   # |e, ms=-1, mI=+1⟩
   rho = spin_state(ms=-1, mI=+1, manifold="excited")

Density Matrix
==============

.. automodule:: sim.states.density_matrix
   :members:
   :undoc-members:
   :show-inheritance:

The ``DensityMatrix`` class provides utilities for working with quantum states:

.. code-block:: python

   from sim.states import DensityMatrix

   # Create from array
   dm = DensityMatrix(rho_array)

   # Properties
   print(dm.trace())       # Should be 1
   print(dm.purity())      # Tr(ρ²)
   print(dm.is_valid())    # Check positivity

   # Partial traces
   rho_spin = dm.partial_trace_nuclear()    # 6×6
   rho_nuclear = dm.partial_trace_spin()    # 6×6
   rho_electron = dm.partial_trace_manifold()  # 9×9

Projectors
==========

.. automodule:: sim.states.projectors
   :members:
   :undoc-members:
   :show-inheritance:

Spin Projectors
---------------

.. code-block:: python

   from sim.states import projector_ms, projector_mI

   # Project onto ms=0 (ground state)
   P_ms0 = projector_ms(0, manifold="ground")

   # Project onto ms=±1
   P_ms1 = projector_ms(+1, manifold="ground")
   P_msm1 = projector_ms(-1, manifold="ground")

   # Nuclear spin projectors
   P_mI0 = projector_mI(0)

   # Measure population
   pop = np.trace(P_ms0 @ rho).real

Electronic Manifold Projectors
------------------------------

.. code-block:: python

   from sim.states import projector_ground, projector_excited

   # Ground state manifold (indices 0-8)
   P_g = projector_ground()

   # Excited state manifold (indices 9-17)
   P_e = projector_excited()

   # Excited state population
   pop_excited = np.trace(P_e @ rho).real

Combined Projectors
-------------------

.. code-block:: python

   from sim.states import projector_state

   # Project onto specific state |g, ms=0, mI=0⟩
   P = projector_state(ge="ground", ms=0, mI=0)

   # Population in that state
   pop = np.trace(P @ rho).real

State Tomography
================

Reconstruct the density matrix from measurements:

.. code-block:: python

   from sim.states import tomography_operators

   # Get set of measurement operators
   operators = tomography_operators()

   # Each operator's expectation gives one data point
   expectations = [np.trace(O @ rho).real for O in operators]

   # Reconstruct (maximum likelihood)
   rho_reconstructed = reconstruct_state(expectations)

Visualization
=============

Tools for visualizing quantum states:

.. code-block:: python

   from sim.states import plot_populations, plot_bloch

   # Bar plot of spin populations
   plot_populations(rho)

   # Bloch sphere representation
   plot_bloch(rho)

   # Density matrix heatmap
   import matplotlib.pyplot as plt
   plt.imshow(np.abs(rho))
   plt.colorbar()
