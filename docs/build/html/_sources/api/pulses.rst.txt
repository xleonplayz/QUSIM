===============
Pulses Module
===============

The pulses module provides tools for creating microwave pulse sequences
for coherent spin control.

Basic Pulses
============

.. automodule:: sim.pulses.basic
   :members:
   :undoc-members:
   :show-inheritance:

Pulse Class
-----------

.. code-block:: python

   from sim.pulses import Pulse

   # Create a pulse
   pulse = Pulse(
       shape="gaussian",      # Envelope shape
       amplitude=10,          # Rabi frequency in MHz
       duration=50e-9,        # Duration in seconds
       phase=0                # Phase in radians
   )

   # Access properties
   print(pulse.area)          # Rotation angle (rad)
   print(pulse.is_pi_pulse)   # True if area ≈ π

Standard Pulses
---------------

.. code-block:: python

   from sim.pulses import pi_pulse, pi_half_pulse

   # π-pulse (180° rotation)
   pi = pi_pulse(rabi_freq_mhz=10, axis='x', shape='rect')

   # π/2-pulse (90° rotation)
   pi2 = pi_half_pulse(rabi_freq_mhz=10, axis='y', shape='gaussian')

Pulse Shapes
============

.. automodule:: sim.pulses.shapes
   :members:
   :undoc-members:
   :show-inheritance:

Available Shapes
----------------

- ``"rect"`` or ``"rectangular"``: Square pulse
- ``"gaussian"``: Gaussian envelope
- ``"drag"``: DRAG pulse (derivative removal by adiabatic gate)
- ``"blackman"``: Blackman window
- ``"flattop"``: Flat top with smooth edges

.. code-block:: python

   from sim.pulses.shapes import get_envelope

   t = np.linspace(0, 100e-9, 1000)

   # Gaussian envelope
   env = get_envelope("gaussian", t, duration=50e-9, sigma=10e-9)

   # DRAG correction
   env_drag = get_envelope("drag", t, duration=50e-9, beta=0.5)

Pulse Sequences
===============

.. automodule:: sim.pulses.sequences
   :members:
   :undoc-members:
   :show-inheritance:

PulseSequence Class
-------------------

.. code-block:: python

   from sim.pulses import PulseSequence, pi_pulse, pi_half_pulse

   seq = PulseSequence()
   seq.add_pulse(pi_half_pulse(10), delay=1e-6)
   seq.add_pulse(pi_pulse(10), delay=1e-6)
   seq.add_pulse(pi_half_pulse(10))

   print(seq.total_duration())

Standard Sequences
------------------

Ramsey
^^^^^^

Free induction decay (T₂* measurement):

.. code-block:: python

   from sim.pulses.sequences import ramsey_sequence

   seq = ramsey_sequence(
       tau=1e-6,              # Free evolution time
       rabi_freq_mhz=10,      # π/2 pulse Rabi frequency
       final_phase=0          # Phase of final π/2
   )

.. math::

   \frac{\pi}{2}_x - \tau - \frac{\pi}{2}_\phi

Spin Echo (Hahn Echo)
^^^^^^^^^^^^^^^^^^^^^

Refocuses static inhomogeneities (T₂ measurement):

.. code-block:: python

   from sim.pulses.sequences import spin_echo_sequence

   seq = spin_echo_sequence(
       tau=100e-6,            # Total evolution time
       rabi_freq_mhz=10
   )

.. math::

   \frac{\pi}{2}_x - \frac{\tau}{2} - \pi_y - \frac{\tau}{2} - \frac{\pi}{2}_x

CPMG
^^^^

Extended coherence with multiple refocusing pulses:

.. code-block:: python

   from sim.pulses.sequences import cpmg_sequence

   seq = cpmg_sequence(
       tau=10e-6,             # Inter-pulse delay
       n_pulses=8,            # Number of π pulses
       rabi_freq_mhz=10
   )

.. math::

   \frac{\pi}{2}_x - \left[\frac{\tau}{2} - \pi_y - \frac{\tau}{2}\right]^N - \frac{\pi}{2}_x

XY4
^^^

Dynamical decoupling robust to pulse errors:

.. code-block:: python

   from sim.pulses.sequences import xy4_sequence

   seq = xy4_sequence(
       tau=1e-6,              # Inter-pulse delay
       n_cycles=4,            # Number of XY4 cycles
       rabi_freq_mhz=10
   )

.. math::

   \frac{\pi}{2} - [\tau - \pi_x - \tau - \pi_y - \tau - \pi_x - \tau - \pi_y]^N - \frac{\pi}{2}

XY8
^^^

More robust variant of XY4:

.. code-block:: python

   from sim.pulses.sequences import xy8_sequence

   seq = xy8_sequence(
       tau=1e-6,
       n_cycles=2,
       rabi_freq_mhz=10
   )

Rabi Oscillation
^^^^^^^^^^^^^^^^

Single pulse of variable duration:

.. code-block:: python

   from sim.pulses.sequences import rabi_sequence

   seq = rabi_sequence(
       duration=100e-9,       # Pulse duration
       rabi_freq_mhz=10
   )

Custom Sequences
================

Build arbitrary sequences:

.. code-block:: python

   from sim.pulses import PulseSequence, Pulse

   def my_sequence(tau, n_pulses, rabi):
       seq = PulseSequence()

       # Initial π/2
       seq.add_pulse(
           Pulse(shape="gaussian", amplitude=rabi,
                 duration=25e-9, phase=0),
           delay=tau
       )

       # Refocusing pulses
       for i in range(n_pulses):
           phase = np.pi/2 if i % 2 == 0 else 0
           seq.add_pulse(
               Pulse(shape="gaussian", amplitude=rabi,
                     duration=50e-9, phase=phase),
               delay=tau
           )

       # Final π/2
       seq.add_pulse(
           Pulse(shape="gaussian", amplitude=rabi,
                 duration=25e-9, phase=0)
       )

       return seq

AWG Integration
===============

Load sequences into an AWG interface:

.. code-block:: python

   from sim.interfaces import AWGInterface

   awg = AWGInterface(sample_rate=1e9)
   seq.to_awg(awg)
   waveform = awg.get_waveform()
