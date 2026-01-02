======================
Measurement and Readout
======================

Quantum measurements in the NV center system involve projective measurements,
expectation values, and realistic photon detection.

Projective Measurements
=======================

Mathematical Framework
----------------------

A projective measurement is described by a set of projection operators
:math:`\{\hat{P}_k\}` satisfying:

.. math::

   \hat{P}_k^\dagger &= \hat{P}_k \quad \text{(Hermitian)} \\
   \hat{P}_k^2 &= \hat{P}_k \quad \text{(Idempotent)} \\
   \sum_k \hat{P}_k &= \mathbb{1} \quad \text{(Complete)}

The probability of outcome :math:`k` is:

.. math::

   p_k = \text{Tr}(\hat{P}_k \hat{\rho})

Post-measurement state (for outcome :math:`k`):

.. math::

   \hat{\rho}_k = \frac{\hat{P}_k \hat{\rho} \hat{P}_k}{p_k}

Spin State Projectors
---------------------

For electron spin measurements in the ground state manifold:

.. math::

   \hat{P}_{m_s} = |g\rangle\langle g| \otimes |m_s\rangle\langle m_s| \otimes \mathbb{1}_3

Explicitly, the :math:`m_s = 0` projector in the 18-dimensional space:

.. math::

   \hat{P}_{m_s=0}^{(g)} = \sum_{m_I = -1}^{+1} |g, 0, m_I\rangle \langle g, 0, m_I|

This corresponds to a 18×18 diagonal matrix with 1s at indices 3, 4, 5 (ground state :math:`m_s = 0`).

Electronic State Projectors
---------------------------

Ground state projector:

.. math::

   \hat{P}_g = |g\rangle\langle g| \otimes \mathbb{1}_9 = \sum_{n=0}^{8} |n\rangle\langle n|

Excited state projector:

.. math::

   \hat{P}_e = |e\rangle\langle e| \otimes \mathbb{1}_9 = \sum_{n=9}^{17} |n\rangle\langle n|

Nuclear Spin Projectors
-----------------------

For N14 nuclear spin measurements:

.. math::

   \hat{P}_{m_I} = \mathbb{1}_2 \otimes \mathbb{1}_3 \otimes |m_I\rangle\langle m_I|

Expectation Values
==================

For any observable :math:`\hat{O}`, the expectation value is:

.. math::

   \langle \hat{O} \rangle = \text{Tr}(\hat{O} \hat{\rho})

Common Observables
------------------

**Electron spin components:**

.. math::

   \langle \hat{S}_z \rangle &= \text{Tr}(\hat{S}_z \hat{\rho}) \\
   \langle \hat{S}_x \rangle &= \text{Tr}(\hat{S}_x \hat{\rho}) \\
   \langle \hat{S}_y \rangle &= \text{Tr}(\hat{S}_y \hat{\rho})

**Spin populations:**

.. math::

   P_{m_s} = \text{Tr}(\hat{P}_{m_s} \hat{\rho})

**Coherences:**

.. math::

   C_{01} = |\rho_{0,1}| = |\langle m_s=0|\hat{\rho}|m_s=1\rangle|

Bloch Vector
------------

The electron spin state can be visualized on a generalized Bloch sphere:

.. math::

   \vec{S} = (\langle \hat{S}_x \rangle, \langle \hat{S}_y \rangle, \langle \hat{S}_z \rangle)

For a spin-1 system, :math:`|\vec{S}| \leq 1` with equality only for pure states
in the spin subspace.

Purity
------

The purity of a quantum state:

.. math::

   \gamma = \text{Tr}(\hat{\rho}^2)

- :math:`\gamma = 1`: Pure state
- :math:`\gamma = 1/d`: Maximally mixed state (for dimension :math:`d`)
- :math:`\gamma \in [1/18, 1]` for NV center

Optical Readout
===============

Fluorescence-Based Readout
--------------------------

The NV center is read out optically by detecting fluorescence:

1. Illuminate with 532 nm laser
2. Collect red fluorescence (637-800 nm)
3. :math:`m_s = 0` is brighter than :math:`m_s = \pm 1`

The fluorescence rate depends on the spin state:

.. math::

   R_{m_s=0} &= \eta \cdot \gamma_{\text{rad}} \cdot P_e^{(m_s=0)} \\
   R_{m_s=\pm 1} &= \eta \cdot \gamma_{\text{rad}} \cdot P_e^{(m_s=\pm 1)} \cdot (1 - p_{\text{ISC}})

where:

- :math:`\eta`: Collection efficiency
- :math:`\gamma_{\text{rad}}`: Radiative decay rate
- :math:`P_e`: Excited state population
- :math:`p_{\text{ISC}}`: Intersystem crossing probability

Contrast
--------

The readout contrast is defined as:

.. math::

   C = \frac{R_{m_s=0} - R_{m_s=\pm 1}}{R_{m_s=0}}

Typical values: :math:`C \approx 20-30\%`

Signal-to-Noise Ratio
---------------------

For :math:`N` photons detected, the SNR for distinguishing :math:`m_s = 0` from :math:`m_s = \pm 1`:

.. math::

   \text{SNR} = \frac{C \cdot N}{\sqrt{N}} = C \sqrt{N}

Single-shot readout requires :math:`\text{SNR} > 1`, i.e., :math:`N > 1/C^2 \approx 10-25` photons.

Photon Detection
================

Photon Statistics
-----------------

For a coherent optical field, detected photons follow Poisson statistics:

.. math::

   P(n) = \frac{\bar{n}^n e^{-\bar{n}}}{n!}

where :math:`\bar{n}` is the mean photon number.

Detector Models
---------------

The simulator includes realistic detector effects:

**Detection efficiency** :math:`\eta`:

- APD: 5-20%
- SNSPD: 70-95%

**Dark counts**: Background counts even without signal

.. math::

   N_{\text{dark}} = R_{\text{dark}} \cdot T_{\text{int}}

**Dead time** :math:`\tau_d`: Detector is blind after each detection

.. math::

   R_{\text{measured}} = \frac{R_{\text{true}}}{1 + R_{\text{true}} \cdot \tau_d}

**Timing jitter** :math:`\sigma_t`: Uncertainty in photon arrival time

**Afterpulsing**: Spurious counts following a detection

Second-Order Correlation
------------------------

The :math:`g^{(2)}(\tau)` function characterizes photon statistics:

.. math::

   g^{(2)}(\tau) = \frac{\langle I(t) I(t+\tau) \rangle}{\langle I(t) \rangle^2}

For single NV centers:

- :math:`g^{(2)}(0) < 0.5`: Single-photon emission (antibunching)
- :math:`g^{(2)}(\tau \to \infty) = 1`: Uncorrelated at long times

Fano Factor
-----------

The Fano factor characterizes count fluctuations:

.. math::

   F = \frac{\text{Var}(n)}{\langle n \rangle}

- :math:`F = 1`: Poissonian (coherent light)
- :math:`F < 1`: Sub-Poissonian (single photon source)
- :math:`F > 1`: Super-Poissonian (bunched light)

ODMR Spectroscopy
=================

Continuous Wave ODMR
--------------------

In CW-ODMR, the fluorescence is monitored while sweeping the microwave frequency:

.. math::

   S(f_{\text{MW}}) = S_0 \left(1 - C \cdot L(f_{\text{MW}} - f_0)\right)

where :math:`L(f)` is a Lorentzian lineshape:

.. math::

   L(f) = \frac{\Gamma^2/4}{f^2 + \Gamma^2/4}

with linewidth :math:`\Gamma \approx 1/(\pi T_2^*)`.

The resonance frequencies are:

.. math::

   f_\pm = D \pm \gamma_e B_z / (2\pi)

Pulsed ODMR
-----------

In pulsed ODMR:

1. Initialize with laser pulse → :math:`m_s = 0`
2. Apply MW :math:`\pi`-pulse
3. Readout with laser pulse

The signal shows the population transfer efficiency as a function of MW frequency.

Ramsey Interferometry
---------------------

The Ramsey sequence measures free precession:

.. math::

   P_{m_s=0}(\tau) = \frac{1}{2}\left(1 + e^{-\tau/T_2^*} \cos(\Delta \omega \cdot \tau)\right)

where :math:`\Delta\omega = \omega_{\text{MW}} - \omega_0` is the detuning.

From the oscillation frequency, the local field can be determined with sensitivity:

.. math::

   \delta B \approx \frac{1}{\gamma_e \sqrt{T_2^* \cdot T_{\text{meas}}}}
