================
ODMR Simulation
================

Optically Detected Magnetic Resonance (ODMR) is the standard technique
for reading out NV center spin states. This tutorial shows how to simulate
ODMR spectra.

CW-ODMR
=======

In continuous-wave ODMR, we measure fluorescence while sweeping the
microwave frequency.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from sim import HamiltonianBuilder, LindbladSolver
   from sim.hamiltonian.terms import ZFS, Zeeman, MicrowaveDrive
   from sim.states import ground_state, projector_ms

   # Parameters
   D = 2.87           # GHz
   B = 10             # mT (gives ~280 MHz splitting)
   mw_power = 1       # MHz Rabi frequency

   # Frequency sweep
   freqs = np.linspace(2.6, 3.15, 200)  # GHz
   fluorescence = []

   for f in freqs:
       # Detuning from ms=0 ↔ ms=-1 transition
       # The two transitions are at D ± γB/2π ≈ 2.87 ± 0.28 GHz

       H = HamiltonianBuilder()
       H.add(ZFS(D=D))
       H.add(Zeeman(B=B))
       H.add(MicrowaveDrive(omega=mw_power, detuning=f - D))

       solver = LindbladSolver(H)
       solver.add_t2_dephasing(gamma=5e6)  # T2* = 200 ns

       # Find steady state
       rho_ss = solver.steady_state(t_max=1e-5)

       # Fluorescence ∝ ms=0 population
       # (ms=0 is brighter due to spin-dependent ISC)
       pop_ms0 = np.trace(projector_ms(0) @ rho_ss).real
       fluorescence.append(pop_ms0)

   # Normalize and invert (ODMR shows dips)
   fluorescence = np.array(fluorescence)
   odmr_signal = 1 - 0.3 * (1 - fluorescence)  # 30% contrast

   # Plot
   plt.figure(figsize=(10, 5))
   plt.plot(freqs, odmr_signal)
   plt.xlabel("Microwave Frequency (GHz)")
   plt.ylabel("Fluorescence (a.u.)")
   plt.title("CW-ODMR Spectrum")

   # Mark expected resonances
   gamma_mhz = 28  # MHz/mT
   f_minus = D - gamma_mhz * B / 1000
   f_plus = D + gamma_mhz * B / 1000
   plt.axvline(f_minus, color='r', linestyle='--', label=f'ms=-1: {f_minus:.3f} GHz')
   plt.axvline(f_plus, color='b', linestyle='--', label=f'ms=+1: {f_plus:.3f} GHz')
   plt.legend()
   plt.show()

Hyperfine Structure
===================

Include N14 hyperfine to see the triplet structure:

.. code-block:: python

   from sim.hamiltonian.terms import HyperfineN14

   freqs = np.linspace(2.55, 2.65, 300)  # Zoom on ms=-1 transition
   fluorescence = []

   for f in freqs:
       H = HamiltonianBuilder()
       H.add(ZFS(D=2.87))
       H.add(Zeeman(B=10))
       H.add(HyperfineN14())  # Adds triplet structure
       H.add(MicrowaveDrive(omega=0.5, detuning=f - 2.87))

       solver = LindbladSolver(H)
       solver.add_t2_dephasing(gamma=2e6)  # Narrower linewidth

       rho_ss = solver.steady_state(t_max=2e-5)
       pop_ms0 = np.trace(projector_ms(0) @ rho_ss).real
       fluorescence.append(pop_ms0)

   plt.figure(figsize=(10, 5))
   plt.plot(freqs * 1000, fluorescence)  # Convert to MHz
   plt.xlabel("Microwave Frequency (MHz, relative to D)")
   plt.ylabel("ms=0 Population")
   plt.title("ODMR with N14 Hyperfine Structure")
   plt.show()

You should see three dips separated by ~2.2 MHz (the hyperfine splitting).

Pulsed ODMR
===========

In pulsed ODMR, we apply discrete pulses:

.. code-block:: python

   from sim.states import ground_state

   freqs = np.linspace(2.55, 2.65, 100)
   contrast = []

   for f in freqs:
       # 1. Initialize in ms=0
       rho = ground_state()

       # 2. Apply MW π-pulse
       H = HamiltonianBuilder()
       H.add(ZFS(D=2.87))
       H.add(Zeeman(B=10))
       H.add(MicrowaveDrive(omega=10, detuning=f - 2.87))

       solver = LindbladSolver(H)
       solver.add_t2_dephasing(gamma=1e6)

       # π-pulse duration
       t_pi = 50e-9  # 50 ns for 10 MHz Rabi
       result = solver.evolve(rho, (0, t_pi), n_steps=20)
       rho = result.final_state()

       # 3. Readout
       pop_ms0 = np.trace(projector_ms(0) @ rho).real
       contrast.append(1 - pop_ms0)  # Population transfer

   plt.figure(figsize=(10, 5))
   plt.plot(freqs * 1000, contrast)
   plt.xlabel("Microwave Frequency (MHz)")
   plt.ylabel("Population Transfer")
   plt.title("Pulsed ODMR")
   plt.show()

Magnetic Field Sensing
======================

Use ODMR to determine magnetic field:

.. code-block:: python

   # Unknown field
   B_unknown = 15.3  # mT

   # Simulate ODMR
   freqs = np.linspace(2.4, 3.4, 500)
   signal = []

   for f in freqs:
       H = HamiltonianBuilder()
       H.add(ZFS(D=2.87))
       H.add(Zeeman(B=B_unknown))
       H.add(MicrowaveDrive(omega=1, detuning=f - 2.87))

       solver = LindbladSolver(H)
       solver.add_t2_dephasing(gamma=5e6)
       rho_ss = solver.steady_state()
       signal.append(np.trace(projector_ms(0) @ rho_ss).real)

   signal = np.array(signal)

   # Find dips (minima)
   from scipy.signal import find_peaks
   peaks, _ = find_peaks(-signal, height=-0.9)

   f_dips = freqs[peaks]
   print(f"Detected resonances: {f_dips} GHz")

   # Calculate field
   if len(f_dips) >= 2:
       splitting = abs(f_dips[1] - f_dips[0])  # GHz
       B_measured = splitting / 0.056  # 56 MHz/mT = 0.056 GHz/mT
       print(f"Measured field: {B_measured:.2f} mT")
       print(f"Actual field: {B_unknown:.2f} mT")
