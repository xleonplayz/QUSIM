===============================
The Nitrogen-Vacancy Center
===============================

Physical Structure
==================

The nitrogen-vacancy (NV) center is a point defect in diamond consisting of a
substitutional nitrogen atom adjacent to a lattice vacancy. The NV center
possesses :math:`C_{3v}` symmetry with the symmetry axis along the
:math:`\langle 111 \rangle` crystallographic direction.

.. figure:: /_static/nv_structure.png
   :align: center
   :width: 400px
   :alt: NV center structure

   Schematic of the NV center in the diamond lattice. The nitrogen atom (blue)
   is adjacent to a vacancy (white), surrounded by three carbon atoms.

Electronic Structure
====================

The negatively charged NV⁻ center has six electrons: three from dangling bonds
of carbon atoms, two from the nitrogen, and one captured from the environment.
These electrons form a spin-1 system.

Energy Levels
-------------

The electronic structure consists of:

**Ground State** :math:`{}^3A_2`
   A spin triplet with :math:`S=1`. The orbital singlet has :math:`A_2` symmetry.
   Zero-field splitting :math:`D_{\text{gs}} = 2.87\,\text{GHz}` at room temperature.

**Excited State** :math:`{}^3E`
   A spin triplet with orbital doublet symmetry. Zero-field splitting
   :math:`D_{\text{es}} = 1.42\,\text{GHz}`.

**Singlet States** :math:`{}^1A_1` and :math:`{}^1E`
   Intermediate states enabling intersystem crossing (ISC) and spin polarization.

The optical transition between :math:`{}^3A_2` and :math:`{}^3E` occurs at:

.. math::

   \lambda_{\text{ZPL}} = 637\,\text{nm} \quad (1.945\,\text{eV})

Hilbert Space
=============

The full Hilbert space of the NV center in this simulator is:

.. math::

   \mathcal{H} = \mathcal{H}_{\text{electronic}} \otimes \mathcal{H}_{\text{spin}} \otimes \mathcal{H}_{\text{nuclear}}

with dimensions:

- :math:`\mathcal{H}_{\text{electronic}}`: 2-dimensional (ground :math:`|g\rangle`, excited :math:`|e\rangle`)
- :math:`\mathcal{H}_{\text{spin}}`: 3-dimensional (:math:`|m_s = +1\rangle, |0\rangle, |-1\rangle`)
- :math:`\mathcal{H}_{\text{nuclear}}`: 3-dimensional (:math:`|m_I = +1\rangle, |0\rangle, |-1\rangle` for N14)

**Total dimension**: :math:`2 \times 3 \times 3 = 18`

Basis States
------------

The computational basis is ordered as:

.. math::

   |n\rangle = |g/e\rangle \otimes |m_s\rangle \otimes |m_I\rangle

with the index mapping:

.. math::

   n = 9 \cdot \delta_{e} + 3 \cdot (1 - m_s) + (1 - m_I)

where :math:`\delta_e = 1` for excited state, :math:`\delta_e = 0` for ground state.

Explicitly:

.. list-table:: Basis state ordering
   :widths: 10 30 30
   :header-rows: 1

   * - Index
     - State
     - Description
   * - 0
     - :math:`|g, +1, +1\rangle`
     - Ground, ms=+1, mI=+1
   * - 1
     - :math:`|g, +1, 0\rangle`
     - Ground, ms=+1, mI=0
   * - 2
     - :math:`|g, +1, -1\rangle`
     - Ground, ms=+1, mI=-1
   * - 3
     - :math:`|g, 0, +1\rangle`
     - Ground, ms=0, mI=+1
   * - 4
     - :math:`|g, 0, 0\rangle`
     - Ground, ms=0, mI=0
   * - 5
     - :math:`|g, 0, -1\rangle`
     - Ground, ms=0, mI=-1
   * - 6
     - :math:`|g, -1, +1\rangle`
     - Ground, ms=-1, mI=+1
   * - 7
     - :math:`|g, -1, 0\rangle`
     - Ground, ms=-1, mI=0
   * - 8
     - :math:`|g, -1, -1\rangle`
     - Ground, ms=-1, mI=-1
   * - 9-17
     - :math:`|e, m_s, m_I\rangle`
     - Excited state analogues

Spin-1 Operators
================

The electron spin operators for :math:`S=1` in the :math:`\{|+1\rangle, |0\rangle, |-1\rangle\}` basis:

.. math::

   \hat{S}_x = \frac{1}{\sqrt{2}} \begin{pmatrix}
   0 & 1 & 0 \\
   1 & 0 & 1 \\
   0 & 1 & 0
   \end{pmatrix}

.. math::

   \hat{S}_y = \frac{1}{\sqrt{2}} \begin{pmatrix}
   0 & -i & 0 \\
   i & 0 & -i \\
   0 & i & 0
   \end{pmatrix}

.. math::

   \hat{S}_z = \begin{pmatrix}
   1 & 0 & 0 \\
   0 & 0 & 0 \\
   0 & 0 & -1
   \end{pmatrix}

These satisfy the angular momentum commutation relations:

.. math::

   [\hat{S}_i, \hat{S}_j] = i\epsilon_{ijk}\hat{S}_k

and the Casimir relation:

.. math::

   \hat{S}^2 = \hat{S}_x^2 + \hat{S}_y^2 + \hat{S}_z^2 = S(S+1)\mathbb{1} = 2\mathbb{1}

Raising and lowering operators:

.. math::

   \hat{S}_+ = \hat{S}_x + i\hat{S}_y, \quad \hat{S}_- = \hat{S}_x - i\hat{S}_y

with:

.. math::

   \hat{S}_\pm |m_s\rangle = \sqrt{S(S+1) - m_s(m_s \pm 1)} |m_s \pm 1\rangle

Physical Constants
==================

.. list-table:: NV Center Physical Constants
   :widths: 40 30 30
   :header-rows: 1

   * - Parameter
     - Value
     - Unit
   * - Ground state ZFS :math:`D_{\text{gs}}`
     - 2.87
     - GHz
   * - Excited state ZFS :math:`D_{\text{es}}`
     - 1.42
     - GHz
   * - Electron g-factor :math:`g_e`
     - 2.0028
     - —
   * - Gyromagnetic ratio :math:`\gamma_e`
     - :math:`1.76 \times 10^{11}`
     - rad/(s·T)
   * - N14 parallel hyperfine :math:`A_\parallel`
     - -2.14
     - MHz
   * - N14 perpendicular hyperfine :math:`A_\perp`
     - -2.70
     - MHz
   * - N14 quadrupole :math:`P`
     - -5.0
     - MHz
   * - Optical lifetime :math:`\tau_{\text{rad}}`
     - ~12
     - ns
   * - ZPL wavelength
     - 637
     - nm
   * - Temperature coefficient :math:`dD/dT`
     - -74
     - kHz/K

Optical Transitions
===================

Spin-Conserving Transitions
---------------------------

The primary optical transitions are spin-conserving:

.. math::

   |g, m_s\rangle \xrightarrow{637\,\text{nm}} |e, m_s\rangle

These transitions are used for:

- **Initialization**: Optical pumping into :math:`|g, m_s=0\rangle`
- **Readout**: Spin-dependent fluorescence

Intersystem Crossing (ISC)
--------------------------

Non-radiative decay through singlet states enables spin polarization:

.. math::

   |e, m_s = \pm 1\rangle \xrightarrow{\text{ISC}} |{}^1A_1\rangle \rightarrow |{}^1E\rangle \xrightarrow{\text{ISC}} |g, m_s = 0\rangle

The ISC rate is spin-dependent:

- :math:`m_s = \pm 1`: High ISC probability (:math:`\sim 30-50\%`)
- :math:`m_s = 0`: Low ISC probability (:math:`\sim 0\%`)

This spin-dependent shelving enables:

1. **Optical spin polarization**: After ~1 μs of illumination, >90% population in :math:`|m_s=0\rangle`
2. **Spin readout**: :math:`m_s=0` is brighter than :math:`m_s=\pm 1`

Contrast
--------

The fluorescence contrast between spin states is:

.. math::

   C = \frac{I_{m_s=0} - I_{m_s=\pm 1}}{I_{m_s=0}} \approx 20-30\%

This contrast is the basis for optically detected magnetic resonance (ODMR).
