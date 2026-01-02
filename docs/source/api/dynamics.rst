=================
Dynamics Module
=================

The dynamics module provides solvers for open quantum system evolution
using the Lindblad master equation.

Lindblad Solver
===============

.. automodule:: sim.dynamics.lindblad
   :members:
   :undoc-members:
   :show-inheritance:

Basic Usage
-----------

.. code-block:: python

   from sim import HamiltonianBuilder, LindbladSolver
   from sim.hamiltonian.terms import ZFS, Zeeman
   from sim.states import ground_state

   # Build Hamiltonian
   H = HamiltonianBuilder()
   H.add(ZFS(D=2.87))
   H.add(Zeeman(B=10))

   # Create solver
   solver = LindbladSolver(H)

   # Add dissipation channels
   solver.add_t1_relaxation(gamma=1e3)    # T1 = 1 ms
   solver.add_t2_dephasing(gamma=1e6)     # T2* = 1 μs
   solver.add_optical_decay(gamma=8e7)    # τ = 12 ns

   # Initial state
   rho0 = ground_state()

   # Evolve
   result = solver.evolve(
       rho0,
       t_span=(0, 1e-6),
       n_steps=100,
       method="RK45"
   )

Evolution Result
----------------

The ``evolve()`` method returns an ``EvolutionResult`` object:

.. code-block:: python

   # Access time points
   times = result.times  # Array of time values

   # Access density matrices
   rho_t = result.rho_t  # List of 18×18 matrices

   # Get final state
   rho_final = result.final_state()

   # Compute expectation values
   from sim.core.operators import Sz
   exp_Sz = result.expectation(Sz)

   # Compute populations
   from sim.states import projector_ms
   pop_ms0 = result.population(projector_ms(0))

   # Compute purity
   purity = result.purity()

Steady State
------------

Find the asymptotic steady state:

.. code-block:: python

   rho_ss = solver.steady_state(t_max=1e-3, tol=1e-10)

Dissipation Operators
=====================

.. automodule:: sim.dynamics.dissipation
   :members:
   :undoc-members:
   :show-inheritance:

T₁ Relaxation
-------------

Spin-lattice relaxation toward thermal equilibrium:

.. code-block:: python

   solver.add_t1_relaxation(gamma=1e3, manifold="ground")

Parameters:

- ``gamma``: Relaxation rate 1/T₁ in Hz
- ``manifold``: "ground", "excited", or "both"

T₂* Dephasing
-------------

Pure dephasing destroying coherences:

.. code-block:: python

   solver.add_t2_dephasing(gamma=1e6, manifold="ground")

Optical Decay
-------------

Spontaneous emission from excited state:

.. code-block:: python

   solver.add_optical_decay(gamma=8e7)  # τ ≈ 12 ns

Custom Dissipators
------------------

Add any Lindblad operator:

.. code-block:: python

   import numpy as np

   # Create 18×18 Lindblad operator
   L = np.zeros((18, 18), dtype=np.complex128)
   L[0, 9] = np.sqrt(gamma)  # |g,+1,+1⟩ ← |e,+1,+1⟩

   solver.add_dissipator(L)

Solver Options
==============

Integration Methods
-------------------

Available methods (via scipy.integrate.solve_ivp):

- ``"RK45"``: Default, 4th-order Runge-Kutta (adaptive)
- ``"RK23"``: 2nd-order Runge-Kutta
- ``"DOP853"``: 8th-order Dormand-Prince
- ``"Radau"``: Implicit Runge-Kutta (for stiff problems)
- ``"BDF"``: Backward differentiation (for stiff problems)

Tolerances
----------

.. code-block:: python

   result = solver.evolve(
       rho0,
       t_span=(0, 1e-6),
       rtol=1e-8,   # Relative tolerance
       atol=1e-10   # Absolute tolerance
   )

For faster but less accurate simulations, increase tolerances.

Time-Dependent Hamiltonians
===========================

The solver handles time-dependent Hamiltonians automatically:

.. code-block:: python

   from sim.hamiltonian.terms import MicrowaveDrive

   # Pulsed microwave
   def pulse(t):
       if 0 < t < 50e-9:  # 50 ns π-pulse
           return 10  # 10 MHz Rabi
       return 0

   H = HamiltonianBuilder()
   H.add(ZFS(D=2.87))
   H.add(MicrowaveDrive(omega=pulse))

   solver = LindbladSolver(H)
   # The Hamiltonian is rebuilt at each integration step
