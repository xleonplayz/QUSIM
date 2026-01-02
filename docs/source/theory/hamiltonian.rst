====================
Hamiltonian Terms
====================

The total NV center Hamiltonian is composed of several terms:

.. math::

   \hat{H} = \hat{H}_{\text{ZFS}} + \hat{H}_{\text{Zeeman}} + \hat{H}_{\text{HF}} +
             \hat{H}_{\text{MW}} + \hat{H}_{\text{optical}} + \hat{H}_{\text{Stark}} +
             \hat{H}_{\text{strain}} + \hat{H}_{\text{JT}}

Each term is implemented as an 18×18 matrix in the full Hilbert space.

Zero-Field Splitting (ZFS)
==========================

Physical Origin
---------------

The zero-field splitting arises from spin-spin dipolar interaction between the
two unpaired electrons in the NV center. This creates an energy splitting even
in the absence of external magnetic fields.

Mathematical Form
-----------------

The ZFS Hamiltonian is:

.. math::

   \hat{H}_{\text{ZFS}} = D \hat{S}_z^2 + E(\hat{S}_x^2 - \hat{S}_y^2)

where:

- :math:`D`: Axial ZFS parameter (splitting between :math:`m_s=0` and :math:`m_s=\pm 1`)
- :math:`E`: Transverse ZFS parameter (strain-induced splitting of :math:`m_s=\pm 1`)

**Typical values:**

- :math:`D_{\text{gs}} = 2.87\,\text{GHz}` (ground state at 300 K)
- :math:`D_{\text{es}} = 1.42\,\text{GHz}` (excited state)
- :math:`E = 0` to :math:`\sim 10\,\text{MHz}` (depends on local strain)

Energy Levels
-------------

For :math:`E = 0`:

.. math::

   E(m_s = 0) &= 0 \\
   E(m_s = \pm 1) &= D

For :math:`E \neq 0`:

.. math::

   E(m_s = 0) &= 0 \\
   E(m_s = +1) &= D + E \\
   E(m_s = -1) &= D - E

The :math:`E` term lifts the degeneracy between :math:`m_s = +1` and :math:`m_s = -1`.

Matrix Representation
---------------------

In the :math:`\{|+1\rangle, |0\rangle, |-1\rangle\}` basis:

.. math::

   \hat{S}_z^2 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}

.. math::

   \hat{S}_x^2 - \hat{S}_y^2 = \begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ 1 & 0 & 0 \end{pmatrix}

Therefore:

.. math::

   \hat{H}_{\text{ZFS}} = \begin{pmatrix} D & 0 & E \\ 0 & 0 & 0 \\ E & 0 & D \end{pmatrix}

Temperature Dependence
----------------------

The D parameter has a temperature dependence:

.. math::

   D(T) \approx D_0 + \alpha (T - 300\,\text{K})

with :math:`\alpha \approx -74\,\text{kHz/K}`. This enables NV-based thermometry.

Zeeman Effect
=============

Physical Origin
---------------

The Zeeman effect describes the interaction between the electron spin magnetic
moment and an external magnetic field.

Mathematical Form
-----------------

.. math::

   \hat{H}_{\text{Zeeman}} = \gamma_e \vec{B} \cdot \hat{\vec{S}} =
   \gamma_e (B_x \hat{S}_x + B_y \hat{S}_y + B_z \hat{S}_z)

where :math:`\gamma_e = g_e \mu_B / \hbar \approx 1.76 \times 10^{11}\,\text{rad/(s·T)}`.

Energy Shifts
-------------

For a field :math:`\vec{B} = B_z \hat{z}` along the NV axis:

.. math::

   E(m_s) = \gamma_e B_z m_s

The splitting between adjacent :math:`m_s` levels:

.. math::

   \Delta E = \gamma_e B_z \approx 28\,\text{MHz/mT} \cdot B[\text{mT}]

The splitting between :math:`m_s = +1` and :math:`m_s = -1`:

.. math::

   \Delta E_{\pm 1} = 2\gamma_e B_z \approx 56\,\text{MHz/mT} \cdot B[\text{mT}]

Combined with ZFS
-----------------

The full energy levels with ZFS and Zeeman:

.. math::

   E(m_s = 0) &= 0 \\
   E(m_s = +1) &= D + \gamma_e B_z \\
   E(m_s = -1) &= D - \gamma_e B_z

This creates two distinct ODMR frequencies:

.. math::

   f_+ &= D + \gamma_e B_z / (2\pi) \\
   f_- &= D - \gamma_e B_z / (2\pi)

Transverse Fields
-----------------

For non-axial fields, :math:`B_x, B_y \neq 0`, there is state mixing. The
off-diagonal elements couple :math:`m_s = 0` to :math:`m_s = \pm 1`:

.. math::

   \hat{H}_{\text{Zeeman}}^{\perp} = \frac{\gamma_e}{\sqrt{2}} \begin{pmatrix}
   0 & B_x - iB_y & 0 \\
   B_x + iB_y & 0 & B_x - iB_y \\
   0 & B_x + iB_y & 0
   \end{pmatrix}

N14 Hyperfine Coupling
======================

Physical Origin
---------------

The N14 nucleus (:math:`I = 1`) at the vacancy site couples to the electron
spin through:

1. **Fermi contact interaction**: Isotropic coupling from electron density at nucleus
2. **Dipolar interaction**: Anisotropic coupling from magnetic dipole-dipole interaction

Additionally, N14 has a nuclear electric quadrupole moment that interacts with
the local electric field gradient.

Mathematical Form
-----------------

.. math::

   \hat{H}_{\text{HF}} = A_\parallel \hat{S}_z \hat{I}_z +
                          A_\perp (\hat{S}_x \hat{I}_x + \hat{S}_y \hat{I}_y) +
                          P \left(\hat{I}_z^2 - \frac{2}{3}\right)

where:

- :math:`A_\parallel = -2.14\,\text{MHz}`: Parallel hyperfine coupling
- :math:`A_\perp = -2.70\,\text{MHz}`: Perpendicular hyperfine coupling
- :math:`P = -5.0\,\text{MHz}`: Nuclear quadrupole parameter

Hyperfine Tensor
----------------

The hyperfine interaction can be written as:

.. math::

   \hat{H}_{\text{HF}} = \hat{\vec{S}} \cdot \mathbf{A} \cdot \hat{\vec{I}}

where the hyperfine tensor (in the NV frame) is:

.. math::

   \mathbf{A} = \begin{pmatrix}
   A_\perp & 0 & 0 \\
   0 & A_\perp & 0 \\
   0 & 0 & A_\parallel
   \end{pmatrix}

Energy Shifts
-------------

For :math:`m_s = 0` (only quadrupole contributes):

.. math::

   E(m_s=0, m_I) = P \left(m_I^2 - \frac{2}{3}\right)

For :math:`m_s = \pm 1` (hyperfine + quadrupole):

.. math::

   E(m_s, m_I) = A_\parallel m_s m_I + P \left(m_I^2 - \frac{2}{3}\right)

This creates a characteristic triplet structure in ODMR spectra.

9×9 Matrix
----------

In the :math:`|m_s\rangle \otimes |m_I\rangle` basis (9-dimensional):

.. math::

   \hat{H}_{\text{HF}} = A_\parallel (\hat{S}_z \otimes \hat{I}_z) +
                          A_\perp (\hat{S}_x \otimes \hat{I}_x + \hat{S}_y \otimes \hat{I}_y) +
                          P (\mathbb{1}_3 \otimes \hat{I}_z^2) - \frac{2P}{3}\mathbb{1}_9

C13 Hyperfine Coupling
======================

Physical Origin
---------------

Carbon-13 nuclei (:math:`I = 1/2`, 1.1% natural abundance) in the diamond lattice
couple to the NV electron spin through dipolar interaction. The coupling strength
depends on the distance and orientation to the NV center.

Mathematical Form
-----------------

For a C13 nucleus at position :math:`\vec{r}` from the NV:

.. math::

   \hat{H}_{\text{C13}} = \hat{\vec{S}} \cdot \mathbf{A}_{\text{dip}} \cdot \hat{\vec{I}}

The dipolar tensor:

.. math::

   A_{\text{dip}}^{ij} = \frac{\mu_0}{4\pi} \frac{\gamma_e \gamma_{\text{C13}} \hbar}{r^3}
   \left(\delta_{ij} - 3\frac{r_i r_j}{r^2}\right)

Secular Approximation
---------------------

In the high-field limit (:math:`B \gg A/\gamma_e`), only the secular terms survive:

.. math::

   \hat{H}_{\text{C13}}^{\text{sec}} = A_{zz} \hat{S}_z \hat{I}_z

where:

.. math::

   A_{zz} = \frac{\mu_0}{4\pi} \frac{\gamma_e \gamma_{\text{C13}} \hbar}{r^3}
   (1 - 3\cos^2\theta)

with :math:`\theta` the angle between :math:`\vec{r}` and the NV axis.

Typical Coupling Strengths
--------------------------

.. list-table:: C13 Hyperfine Couplings
   :widths: 30 30 40
   :header-rows: 1

   * - Position
     - Distance
     - Coupling
   * - Nearest neighbor
     - 1.54 Å
     - ~130 MHz
   * - 2nd shell
     - 2.5 Å
     - ~15 MHz
   * - 3rd shell
     - 3.5 Å
     - ~3 MHz

Microwave Drive
===============

Physical Origin
---------------

Microwave radiation at frequencies near :math:`D \pm \gamma_e B` drives transitions
between :math:`m_s = 0` and :math:`m_s = \pm 1`.

Mathematical Form (Rotating Frame)
----------------------------------

In the rotating wave approximation:

.. math::

   \hat{H}_{\text{MW}} = \frac{\Omega}{2} \left(\hat{S}_+ e^{-i\phi} + \hat{S}_- e^{i\phi}\right) + \Delta \hat{S}_z

where:

- :math:`\Omega`: Rabi frequency (proportional to MW amplitude)
- :math:`\phi`: MW phase
- :math:`\Delta = \omega_{\text{MW}} - \omega_0`: Detuning from resonance

On Resonance (:math:`\Delta = 0`)
---------------------------------

.. math::

   \hat{H}_{\text{MW}} = \frac{\Omega}{2} \left(\hat{S}_+ e^{-i\phi} + \hat{S}_- e^{i\phi}\right)

For :math:`\phi = 0` (X rotation):

.. math::

   \hat{H}_{\text{MW}} = \Omega \hat{S}_x

For :math:`\phi = \pi/2` (Y rotation):

.. math::

   \hat{H}_{\text{MW}} = \Omega \hat{S}_y

Rabi Oscillations
-----------------

Starting from :math:`|m_s = 0\rangle`, the population in :math:`|m_s = +1\rangle`:

.. math::

   P_{+1}(t) = \sin^2\left(\frac{\Omega t}{2}\right)

For a :math:`\pi`-pulse (:math:`\Omega t = \pi`), full population transfer occurs.

Pulse Durations
---------------

For a given Rabi frequency :math:`\Omega`:

.. math::

   t_{\pi/2} &= \frac{\pi}{2\Omega} \\
   t_{\pi} &= \frac{\pi}{\Omega}

**Example**: For :math:`\Omega = 2\pi \times 10\,\text{MHz}`:

- :math:`t_{\pi/2} = 25\,\text{ns}`
- :math:`t_{\pi} = 50\,\text{ns}`

Optical Coupling
================

Mathematical Form
-----------------

The optical drive couples ground and excited states:

.. math::

   \hat{H}_{\text{opt}} = \frac{\Omega_{\text{opt}}}{2} \sum_{m_s, m_I}
   \left(|e, m_s, m_I\rangle \langle g, m_s, m_I| + \text{h.c.}\right)

For spin-conserving transitions. Spin-selective transitions can be achieved with
polarization control.

Stark Effect
============

DC Stark Effect
---------------

Electric fields shift the ZFS through:

.. math::

   \hat{H}_{\text{Stark}}^{\text{DC}} = d_\perp E_\perp (\hat{S}_x^2 - \hat{S}_y^2) + d_\parallel E_z \hat{S}_z^2

where:

- :math:`d_\perp \approx 17\,\text{Hz/(V/m)}`
- :math:`d_\parallel \approx 3.5\,\text{Hz/(V/m)}`

AC Stark Effect
---------------

Off-resonant optical fields create effective spin Hamiltonians through the AC Stark shift.

Strain
======

Physical Origin
---------------

Lattice strain from defects or external stress perturbs the NV electronic structure.

Mathematical Form
-----------------

.. math::

   \hat{H}_{\text{strain}} = \sum_{i,j} \varepsilon_{ij} \frac{\{\hat{S}_i, \hat{S}_j\}}{2}

where :math:`\varepsilon_{ij}` is the strain tensor and :math:`\{A, B\} = AB + BA`.

The primary effect is through transverse strain :math:`E_1, E_2`:

.. math::

   \hat{H}_{\text{strain}} \approx E_1 (\hat{S}_x^2 - \hat{S}_y^2) + E_2 (\hat{S}_x \hat{S}_y + \hat{S}_y \hat{S}_x)

Jahn-Teller Effect
==================

In the excited state, the orbital degeneracy leads to vibronic coupling:

.. math::

   \hat{H}_{\text{JT}} = \lambda_{\text{JT}} \hat{Q} \cdot \hat{\sigma}

where :math:`\hat{Q}` are phonon coordinates and :math:`\hat{\sigma}` are pseudo-spin
operators in the orbital space.

This leads to dynamic averaging of the excited state ZFS at room temperature.
