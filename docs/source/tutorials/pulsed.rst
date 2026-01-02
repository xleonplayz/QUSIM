=======================
Pulsed Experiments
=======================

This tutorial covers common pulsed measurement sequences.

Rabi Oscillations
=================

Measure the Rabi frequency by varying pulse duration:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from sim import HamiltonianBuilder, LindbladSolver
   from sim.hamiltonian.terms import ZFS, MicrowaveDrive
   from sim.states import ground_state, projector_ms

   # Rabi frequency
   omega_rabi = 10  # MHz

   durations = np.linspace(0, 200e-9, 100)  # 0 to 200 ns
   populations = []

   for t_pulse in durations:
       # Time-dependent MW
       def mw_amplitude(t):
           return omega_rabi if t < t_pulse else 0

       H = HamiltonianBuilder()
       H.add(ZFS(D=2.87))
       H.add(MicrowaveDrive(omega=mw_amplitude))

       solver = LindbladSolver(H)
       solver.add_t2_dephasing(gamma=1e5)

       rho0 = ground_state()
       result = solver.evolve(rho0, (0, t_pulse + 1e-9), n_steps=50)

       pop = np.trace(projector_ms(+1) @ result.final_state()).real
       populations.append(pop)

   plt.figure(figsize=(10, 5))
   plt.plot(durations * 1e9, populations, 'b-')
   plt.xlabel("Pulse Duration (ns)")
   plt.ylabel("Population ms=+1")
   plt.title(f"Rabi Oscillations (Ω = {omega_rabi} MHz)")

   # Expected: cos²(Ωt/2) with period T = 2π/Ω = 100 ns
   t_theory = durations * 1e9
   pop_theory = np.sin(np.pi * omega_rabi * durations * 1e6)**2
   plt.plot(t_theory, pop_theory, 'r--', label='Theory')
   plt.legend()
   plt.show()

Ramsey Interferometry
=====================

Measure T₂* and detuning:

.. code-block:: python

   from sim.pulses.sequences import ramsey_sequence

   tau_values = np.linspace(0, 5e-6, 100)  # 0 to 5 μs
   signal = []

   # Small detuning to see oscillations
   detuning = 1  # MHz

   for tau in tau_values:
       H = HamiltonianBuilder()
       H.add(ZFS(D=2.87))
       H.add(MicrowaveDrive(omega=20, detuning=detuning))  # Fast π/2

       solver = LindbladSolver(H)
       solver.add_t2_dephasing(gamma=0.5e6)  # T2* = 2 μs

       rho = ground_state()

       # First π/2
       t_pi2 = 12.5e-9  # 12.5 ns for 20 MHz Rabi
       result = solver.evolve(rho, (0, t_pi2), n_steps=10)
       rho = result.final_state()

       # Free evolution (no MW)
       H_free = HamiltonianBuilder()
       H_free.add(ZFS(D=2.87))
       solver_free = LindbladSolver(H_free)
       solver_free.add_t2_dephasing(gamma=0.5e6)
       result = solver_free.evolve(rho, (0, tau), n_steps=20)
       rho = result.final_state()

       # Second π/2
       result = solver.evolve(rho, (0, t_pi2), n_steps=10)
       rho = result.final_state()

       pop = np.trace(projector_ms(0) @ rho).real
       signal.append(pop)

   plt.figure(figsize=(10, 5))
   plt.plot(tau_values * 1e6, signal, 'b-')
   plt.xlabel("Free Evolution Time τ (μs)")
   plt.ylabel("Population ms=0")
   plt.title("Ramsey Fringes")

   # Fit: A * exp(-τ/T2*) * cos(2π*Δ*τ) + offset
   plt.show()

Spin Echo (Hahn Echo)
=====================

Measure T₂ by refocusing static noise:

.. code-block:: python

   tau_values = np.linspace(0, 500e-6, 50)  # 0 to 500 μs
   echo_signal = []

   for tau in tau_values:
       H = HamiltonianBuilder()
       H.add(ZFS(D=2.87))
       H.add(MicrowaveDrive(omega=20))

       solver = LindbladSolver(H)
       # Pure dephasing (refocusable)
       solver.add_t2_dephasing(gamma=1e6)
       # T1 relaxation (not refocusable)
       solver.add_t1_relaxation(gamma=100)

       rho = ground_state()
       t_pi2 = 12.5e-9
       t_pi = 25e-9

       # π/2
       result = solver.evolve(rho, (0, t_pi2), n_steps=5)
       rho = result.final_state()

       # τ/2 free evolution
       H_free = HamiltonianBuilder()
       H_free.add(ZFS(D=2.87))
       solver_free = LindbladSolver(H_free)
       solver_free.add_t2_dephasing(gamma=1e6)
       solver_free.add_t1_relaxation(gamma=100)
       result = solver_free.evolve(rho, (0, tau/2), n_steps=10)
       rho = result.final_state()

       # π refocusing pulse
       result = solver.evolve(rho, (0, t_pi), n_steps=5)
       rho = result.final_state()

       # τ/2 free evolution
       result = solver_free.evolve(rho, (0, tau/2), n_steps=10)
       rho = result.final_state()

       # π/2 readout
       result = solver.evolve(rho, (0, t_pi2), n_steps=5)
       rho = result.final_state()

       pop = np.trace(projector_ms(0) @ rho).real
       echo_signal.append(pop)

   plt.figure(figsize=(10, 5))
   plt.plot(tau_values * 1e6, echo_signal, 'b-o')
   plt.xlabel("Total Evolution Time τ (μs)")
   plt.ylabel("Echo Amplitude")
   plt.title("Spin Echo Decay")
   plt.show()

CPMG Dynamical Decoupling
=========================

Extend coherence with multiple refocusing pulses:

.. code-block:: python

   n_pulses_list = [1, 2, 4, 8, 16, 32]
   tau_fixed = 100e-6  # Total evolution time

   coherence_vs_n = []

   for n in n_pulses_list:
       tau_segment = tau_fixed / (2 * n)

       # ... similar pulse sequence with n π-pulses ...
       # (Implementation similar to spin echo but with loop)

       # Placeholder for demonstration
       # In reality, more pulses = better coherence preservation
       T2_eff = 200e-6 * np.sqrt(n)  # Approximate scaling
       coherence = np.exp(-tau_fixed / T2_eff)
       coherence_vs_n.append(coherence)

   plt.figure(figsize=(8, 5))
   plt.semilogy(n_pulses_list, coherence_vs_n, 'bo-')
   plt.xlabel("Number of π-pulses")
   plt.ylabel("Coherence")
   plt.title("CPMG: Coherence vs Number of Pulses")
   plt.grid(True)
   plt.show()

XY8 Sequence for AC Sensing
===========================

The XY8 sequence is sensitive to AC fields at specific frequencies:

.. code-block:: python

   # Sensing frequency f_sense
   # Maximum sensitivity when τ = 1/(2*f_sense)

   f_sense = 1e6  # 1 MHz AC field
   tau = 1 / (2 * f_sense)  # 500 ns

   # XY8 filter function peaks at f_sense and odd harmonics
   freqs = np.linspace(0, 5e6, 1000)
   filter_function = np.sinc(freqs * tau)**2  # Simplified

   plt.figure(figsize=(10, 5))
   plt.plot(freqs / 1e6, filter_function)
   plt.xlabel("Frequency (MHz)")
   plt.ylabel("Sensitivity")
   plt.axvline(f_sense / 1e6, color='r', linestyle='--', label='Target frequency')
   plt.title("XY8 Filter Function")
   plt.legend()
   plt.show()
