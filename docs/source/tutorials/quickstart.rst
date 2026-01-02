==========
Quickstart
==========

This guide walks through the basic usage of QUSIM.

Basic Simulation
================

1. **Import the necessary modules:**

.. code-block:: python

   from sim import HamiltonianBuilder, LindbladSolver
   from sim.hamiltonian.terms import ZFS, Zeeman, HyperfineN14
   from sim.states import ground_state, projector_ms

2. **Build the Hamiltonian:**

.. code-block:: python

   H = HamiltonianBuilder()
   H.add(ZFS(D=2.87))           # Zero-field splitting (2.87 GHz)
   H.add(Zeeman(B=10))          # Magnetic field (10 mT)
   H.add(HyperfineN14())        # N14 hyperfine structure

   # Check the Hamiltonian
   print(f"Time-dependent: {H.is_time_dependent}")
   print(f"Number of terms: {len(H.terms)}")

3. **Setup dissipation:**

.. code-block:: python

   solver = LindbladSolver(H)
   solver.add_t1_relaxation(gamma=1e3)    # T1 = 1 ms
   solver.add_t2_dephasing(gamma=1e6)     # T2* = 1 μs

4. **Prepare initial state and evolve:**

.. code-block:: python

   rho0 = ground_state()  # |g, ms=0, mI=0⟩

   result = solver.evolve(
       rho0,
       t_span=(0, 1e-6),    # 0 to 1 μs
       n_steps=100
   )

5. **Analyze results:**

.. code-block:: python

   import matplotlib.pyplot as plt

   # Population in ms=0
   pop_ms0 = result.population(projector_ms(0))

   plt.plot(result.times * 1e6, pop_ms0)
   plt.xlabel("Time (μs)")
   plt.ylabel("Population ms=0")
   plt.show()

Rabi Oscillations
=================

Simulate driven spin dynamics:

.. code-block:: python

   from sim.hamiltonian.terms import MicrowaveDrive
   import numpy as np

   # Resonant microwave drive
   H = HamiltonianBuilder()
   H.add(ZFS(D=2.87))
   H.add(MicrowaveDrive(omega=10))  # 10 MHz Rabi frequency

   solver = LindbladSolver(H)
   solver.add_t2_dephasing(gamma=1e5)  # T2* = 10 μs

   rho0 = ground_state()
   result = solver.evolve(rho0, t_span=(0, 500e-9), n_steps=200)

   # Plot oscillations
   pop_ms1 = result.population(projector_ms(+1))

   plt.plot(result.times * 1e9, pop_ms1)
   plt.xlabel("Time (ns)")
   plt.ylabel("Population ms=+1")
   plt.title("Rabi Oscillations")
   plt.show()

Eigenvalue Analysis
===================

Analyze energy levels:

.. code-block:: python

   H = HamiltonianBuilder()
   H.add(ZFS(D=2.87))
   H.add(Zeeman(B=10))
   H.add(HyperfineN14())

   # Get eigenvalues
   eigvals = H.eigenvalues_mhz()
   print("Energy levels (MHz):")
   for i, E in enumerate(sorted(eigvals)):
       print(f"  |{i}>: {E:.2f} MHz")

Saving Results
==============

.. code-block:: python

   import numpy as np

   # Save evolution data
   np.savez(
       "simulation_results.npz",
       times=result.times,
       populations=result.population(projector_ms(0)),
       purity=result.purity()
   )

   # Load later
   data = np.load("simulation_results.npz")
   times = data["times"]
   pops = data["populations"]
