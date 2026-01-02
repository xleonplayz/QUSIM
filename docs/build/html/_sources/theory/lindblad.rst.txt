===========================
Open Quantum System Dynamics
===========================

Real quantum systems interact with their environment, leading to decoherence
and relaxation. The NV center is no exception—spin-lattice relaxation, magnetic
noise, and optical decay all play important roles.

Lindblad Master Equation
========================

The dynamics of an open quantum system are described by the Lindblad master equation:

.. math::

   \frac{d\hat{\rho}}{dt} = -\frac{i}{\hbar}[\hat{H}(t), \hat{\rho}] +
   \sum_k \mathcal{D}[\hat{L}_k](\hat{\rho})

where the **Lindblad dissipator** is:

.. math::

   \mathcal{D}[\hat{L}](\hat{\rho}) = \hat{L}\hat{\rho}\hat{L}^\dagger -
   \frac{1}{2}\left\{\hat{L}^\dagger\hat{L}, \hat{\rho}\right\}

Here:

- :math:`\hat{\rho}`: Density matrix (18×18 for NV center)
- :math:`\hat{H}(t)`: System Hamiltonian (possibly time-dependent)
- :math:`\hat{L}_k`: Lindblad (jump) operators describing dissipation channels
- :math:`\{A, B\} = AB + BA`: Anticommutator

Properties
----------

The Lindblad equation preserves:

1. **Hermiticity**: :math:`\hat{\rho}^\dagger = \hat{\rho}`
2. **Trace**: :math:`\text{Tr}(\hat{\rho}) = 1`
3. **Positivity**: All eigenvalues of :math:`\hat{\rho}` remain non-negative

Vectorization
-------------

For numerical integration, we vectorize the density matrix:

.. math::

   \hat{\rho} \in \mathbb{C}^{18 \times 18} \rightarrow \vec{\rho} \in \mathbb{C}^{324}

The master equation becomes:

.. math::

   \frac{d\vec{\rho}}{dt} = \mathcal{L} \vec{\rho}

where :math:`\mathcal{L}` is the **Liouvillian superoperator** (324×324 matrix).

T₁ Relaxation (Spin-Lattice)
============================

Physical Origin
---------------

Spin-lattice relaxation (T₁) describes energy exchange between the spin system
and the crystal lattice (phonon bath). This drives the spin toward thermal
equilibrium.

For NV centers at room temperature, T₁ is typically 1-10 ms.

Lindblad Operators
------------------

T₁ relaxation is modeled with raising and lowering operators:

.. math::

   \hat{L}_+ &= \sqrt{\gamma_\uparrow} \, \hat{S}_+ \\
   \hat{L}_- &= \sqrt{\gamma_\downarrow} \, \hat{S}_-

where :math:`\gamma_\uparrow` and :math:`\gamma_\downarrow` are transition rates.

At zero temperature (only decay):

.. math::

   \hat{L} = \sqrt{\gamma_{T_1}} \, \hat{S}_-

with :math:`\gamma_{T_1} = 1/T_1`.

Detailed Balance
----------------

At finite temperature :math:`T`:

.. math::

   \frac{\gamma_\uparrow}{\gamma_\downarrow} = e^{-\hbar\omega_0/(k_B T)}

where :math:`\omega_0` is the transition frequency.

For NV at room temperature with :math:`\omega_0 \approx 2\pi \times 2.87\,\text{GHz}`:

.. math::

   \frac{\hbar\omega_0}{k_B T} \approx \frac{(1.05 \times 10^{-34})(2\pi \times 2.87 \times 10^9)}{(1.38 \times 10^{-23})(300)} \approx 0.46

So both absorption and emission are relevant at room temperature.

T₂* Dephasing (Inhomogeneous)
=============================

Physical Origin
---------------

T₂* (or :math:`T_2^*`) describes the decay of spin coherence due to:

1. **Magnetic field fluctuations**: From C13 nuclear bath, paramagnetic impurities
2. **Strain variations**: From lattice defects
3. **Electric field noise**: From surface charges

T₂* is typically 1-10 μs for single NV centers in bulk diamond.

Lindblad Operator
-----------------

Pure dephasing is modeled with:

.. math::

   \hat{L}_{\text{deph}} = \sqrt{\gamma_\phi} \, \hat{S}_z

This causes exponential decay of off-diagonal elements (coherences) while
preserving populations:

.. math::

   \rho_{m_s, m_s'}(t) \propto e^{-\gamma_\phi |m_s - m_s'|^2 t} \rho_{m_s, m_s'}(0)

For :math:`m_s = 0 \leftrightarrow m_s = \pm 1` coherences:

.. math::

   \rho_{0, \pm 1}(t) = e^{-t/T_2^*} \rho_{0, \pm 1}(0)

with :math:`T_2^* = 1/\gamma_\phi`.

Relation to T₁
--------------

The total decoherence rate is:

.. math::

   \frac{1}{T_2^*} = \frac{1}{2T_1} + \frac{1}{T_\phi}

where :math:`T_\phi` is the pure dephasing time. For NV centers, typically
:math:`T_2^* \ll T_1`, so :math:`T_2^* \approx T_\phi`.

T₂ (Homogeneous / Echo)
=======================

Spin Echo
---------

The Hahn echo sequence refocuses static inhomogeneities:

.. math::

   \frac{\pi}{2} - \tau/2 - \pi - \tau/2 - \frac{\pi}{2}

The echo amplitude decays with T₂ > T₂*:

.. math::

   S(\tau) = S_0 e^{-(\tau/T_2)^n}

where :math:`n \approx 1-3` depending on the noise spectrum.

For NV centers: T₂ ~ 100-500 μs (Hahn echo), extendable to >1 ms with dynamical decoupling.

Optical Decay
=============

Physical Origin
---------------

The excited state :math:`{}^3E` decays to the ground state :math:`{}^3A_2` via
spontaneous emission with lifetime :math:`\tau_{\text{rad}} \approx 12\,\text{ns}`.

Lindblad Operators
------------------

For spin-conserving optical decay:

.. math::

   \hat{L}_{m_s, m_I} = \sqrt{\gamma_{\text{opt}}} \, |g, m_s, m_I\rangle \langle e, m_s, m_I|

with :math:`\gamma_{\text{opt}} = 1/\tau_{\text{rad}} \approx 8 \times 10^7\,\text{Hz}`.

There are 9 such operators (one for each :math:`|m_s, m_I\rangle` combination).

Intersystem Crossing (ISC)
==========================

Physical Origin
---------------

Non-radiative decay through singlet states is spin-dependent:

- :math:`m_s = \pm 1`: High ISC rate to singlet
- :math:`m_s = 0`: Low ISC rate

This enables optical spin polarization into :math:`m_s = 0`.

Simplified Model
----------------

We model ISC as effective decay from :math:`|e, m_s = \pm 1\rangle` to :math:`|g, m_s = 0\rangle`:

.. math::

   \hat{L}_{\text{ISC}}^{m_I} = \sqrt{\gamma_{\text{ISC}}} \, |g, 0, m_I\rangle \langle e, \pm 1, m_I|

This captures the essential physics of spin polarization without explicit
modeling of the singlet states.

Nuclear Spin Relaxation
=======================

The N14 nuclear spin has much longer relaxation times than the electron spin:

- :math:`T_{1,n}`: Seconds to minutes
- :math:`T_{2,n}^*`: Milliseconds to seconds

Lindblad operators:

.. math::

   \hat{L}_{I,\pm} = \sqrt{\gamma_n} \, \hat{I}_\pm

where :math:`\gamma_n \ll \gamma_e`.

Numerical Integration
=====================

Solver Methods
--------------

The simulator uses `scipy.integrate.solve_ivp` with adaptive step size control.
Default method: **RK45** (4th-order Runge-Kutta with 5th-order error estimate).

The right-hand side function computes:

.. code-block:: python

   def lindblad_rhs(t, rho_vec):
       rho = rho_vec.reshape((18, 18))

       # Hamiltonian term: -i[H, ρ]
       H = hamiltonian.build(t)
       drho = -1j * (H @ rho - rho @ H)

       # Dissipation terms
       for L in lindblad_operators:
           L_dag = L.conj().T
           L_dag_L = L_dag @ L
           drho += L @ rho @ L_dag - 0.5 * (L_dag_L @ rho + rho @ L_dag_L)

       return drho.flatten()

Tolerance Settings
------------------

Default tolerances for accurate simulation:

- Relative tolerance: :math:`10^{-8}`
- Absolute tolerance: :math:`10^{-10}`

For faster but less accurate simulations, these can be relaxed to :math:`10^{-6}` and :math:`10^{-8}`.

Time Scales
-----------

Typical simulation time scales:

.. list-table:: Characteristic Times
   :widths: 30 30 40
   :header-rows: 1

   * - Process
     - Time Scale
     - Notes
   * - Rabi oscillation
     - 10-100 ns
     - :math:`\pi`-pulse at 10 MHz Rabi
   * - Optical decay
     - ~12 ns
     - Excited state lifetime
   * - T₂* dephasing
     - 1-10 μs
     - Free induction decay
   * - T₂ (echo)
     - 100-500 μs
     - Hahn echo coherence
   * - T₁ relaxation
     - 1-10 ms
     - Spin-lattice

Steady State
============

For time-independent Hamiltonians with dissipation, the system reaches a steady
state :math:`\hat{\rho}_{\text{ss}}` satisfying:

.. math::

   \frac{d\hat{\rho}_{\text{ss}}}{dt} = 0 \Rightarrow \mathcal{L}\hat{\rho}_{\text{ss}} = 0

This can be found by:

1. Long-time evolution from any initial state
2. Direct solution of :math:`\mathcal{L}\hat{\rho}_{\text{ss}} = 0`

Under continuous optical excitation, the steady state shows spin polarization
into :math:`m_s = 0`.
