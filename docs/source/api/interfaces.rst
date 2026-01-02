==================
Interfaces Module
==================

The interfaces module provides hardware abstraction layers for
experimental control, with realistic noise models.

AWG Interface
=============

.. automodule:: sim.interfaces.awg
   :members:
   :undoc-members:
   :show-inheritance:

The AWG (Arbitrary Waveform Generator) interface simulates a programmable
pulse generator:

.. code-block:: python

   from sim.interfaces import AWGInterface

   # Create AWG with 1 GS/s sample rate
   awg = AWGInterface(sample_rate=1e9)

   # Add pulses
   awg.add_pulse(
       shape="gaussian",
       amplitude=1.0,          # Normalized amplitude
       duration=50e-9,         # 50 ns
       phase=0
   )
   awg.add_delay(100e-9)       # 100 ns delay
   awg.add_pulse(
       shape="rect",
       amplitude=1.0,
       duration=100e-9,
       phase=np.pi/2
   )

   # Get waveform
   t, waveform = awg.get_waveform()

   # Plot
   import matplotlib.pyplot as plt
   plt.plot(t * 1e9, waveform.real)
   plt.xlabel("Time (ns)")
   plt.ylabel("Amplitude")

Waveform Generation
-------------------

.. code-block:: python

   # Generate IQ waveforms
   I, Q = awg.get_iq_waveform(carrier_freq=2.87e9)

   # Export to file
   awg.save_waveform("sequence.npz")

Laser Interface
===============

.. automodule:: sim.interfaces.laser
   :members:
   :undoc-members:
   :show-inheritance:

The laser interface simulates optical excitation:

.. code-block:: python

   from sim.interfaces import LaserInterface

   # Create laser (532 nm green)
   laser = LaserInterface(wavelength=532e-9)

   # Set power
   laser.set_power(1e-3)  # 1 mW

   # Convert to Rabi frequency
   rabi = laser.power_to_rabi(power=1e-3)

   # Pulse timing
   laser.pulse(duration=1e-6, delay=0)

Photon Counter
==============

.. automodule:: sim.interfaces.photon_counter
   :members:
   :undoc-members:
   :show-inheritance:

The photon counter simulates realistic single-photon detection:

.. code-block:: python

   from sim.interfaces import PhotonCounter

   # Create APD-like detector
   counter = PhotonCounter(
       efficiency=0.1,           # 10% detection efficiency
       dark_count_rate=100,      # 100 Hz dark counts
       dead_time=50e-9,          # 50 ns dead time
       timing_jitter=0.5e-9,     # 500 ps jitter
       afterpulsing_prob=0.01    # 1% afterpulsing
   )

   # Reset counters
   counter.reset()

   # Count photons from density matrix
   for rho in trajectory:
       result = counter.count(
           rho=rho,
           dt=1e-9,              # 1 ns time step
           gamma=8e7             # Emission rate
       )
       print(f"Detected: {result['detected']}")

   # Get total counts
   print(f"Total: {counter.total_counts}")

Detector Types
--------------

**APD (Avalanche Photodiode)**:

.. code-block:: python

   apd = PhotonCounter(
       efficiency=0.1,
       dark_count_rate=100,
       dead_time=50e-9,
       timing_jitter=0.5e-9,
       afterpulsing_prob=0.01
   )

**SNSPD (Superconducting Nanowire)**:

.. code-block:: python

   snspd = PhotonCounter(
       efficiency=0.9,
       dark_count_rate=0.1,
       dead_time=20e-9,
       timing_jitter=50e-12,
       afterpulsing_prob=0.001
   )

Analysis Functions
------------------

.. code-block:: python

   # Count rate
   rate = counter.count_rate(window=1e-6)  # Hz

   # Histogram of detection times
   counts, edges = counter.histogram(
       bin_width=1e-9,
       t_range=(0, 1e-6)
   )

   # g²(τ) autocorrelation
   tau, g2 = counter.calculate_g2(
       tau_max=1e-6,
       n_bins=100
   )

   # Fano factor
   F = counter.fano_factor(window=1e-6)

Complete Measurement
====================

Example: ODMR measurement with all interfaces:

.. code-block:: python

   from sim import HamiltonianBuilder, LindbladSolver
   from sim.hamiltonian.terms import ZFS, Zeeman, MicrowaveDrive
   from sim.states import ground_state, projector_excited
   from sim.interfaces import LaserInterface, PhotonCounter
   import numpy as np

   # Setup
   laser = LaserInterface(wavelength=532e-9)
   counter = PhotonCounter(efficiency=0.1)

   # Sweep MW frequency
   freqs = np.linspace(2.8, 2.94, 100)  # GHz
   counts = []

   for f in freqs:
       # Build Hamiltonian
       H = HamiltonianBuilder()
       H.add(ZFS(D=2.87))
       H.add(Zeeman(B=10))
       H.add(MicrowaveDrive(omega=1, detuning=f-2.87))

       # Simulate
       solver = LindbladSolver(H)
       solver.add_t2_dephasing(gamma=1e6)
       solver.add_optical_decay(gamma=8e7)

       rho0 = ground_state()
       result = solver.evolve(rho0, (0, 1e-6), n_steps=100)

       # Count photons
       counter.reset()
       for rho in result.rho_t:
           counter.count(rho, dt=10e-9, gamma=8e7)

       counts.append(counter.total_counts)

   # Plot ODMR spectrum
   import matplotlib.pyplot as plt
   plt.plot(freqs, counts)
   plt.xlabel("MW Frequency (GHz)")
   plt.ylabel("Photon Counts")
