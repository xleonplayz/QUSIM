.. QUSIM documentation master file

======================================
QUSIM - NV Center Quantum Simulator
======================================

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://opensource.org/licenses/MIT

**QUSIM** is a hyperrealistic quantum simulator for nitrogen-vacancy (NV) centers
in diamond. It provides a complete framework for simulating NV center physics,
including spin dynamics, optical transitions, and realistic hardware interfaces.

.. note::

   This documentation includes the complete mathematical framework used in the
   simulator, with all equations derived from first principles.

Key Features
------------

- **18-dimensional Hilbert space**: Full :math:`|g/e\rangle \otimes |m_s\rangle \otimes |m_I\rangle` basis
- **Realistic Hamiltonians**: ZFS, Zeeman, Hyperfine (N14/C13), MW drive, Optical, Stark, Strain
- **Open quantum systems**: Lindblad master equation with T₁, T₂*, optical decay
- **Pulse sequences**: Ramsey, Spin Echo, CPMG, XY4, XY8
- **Hardware interfaces**: AWG, Laser, Photon Counter with realistic noise models

Quick Start
-----------

.. code-block:: python

   from sim import HamiltonianBuilder, LindbladSolver
   from sim.hamiltonian.terms import ZFS, Zeeman, HyperfineN14
   from sim.states import ground_state, projector_ms

   # Build NV Hamiltonian
   H = HamiltonianBuilder()
   H.add(ZFS(D=2.87))           # 2.87 GHz zero-field splitting
   H.add(Zeeman(B=10))          # 10 mT magnetic field
   H.add(HyperfineN14())        # N14 hyperfine structure

   # Setup dissipation
   solver = LindbladSolver(H)
   solver.add_t1_relaxation(gamma=1e3)   # T1 = 1 ms
   solver.add_t2_dephasing(gamma=1e6)    # T2* = 1 μs

   # Evolve from ground state
   rho0 = ground_state()
   result = solver.evolve(rho0, t_span=(0, 1e-6), n_steps=100)

   # Measure ms=0 population
   pop = result.population(projector_ms(0))

Physical Background
-------------------

The NV center consists of a substitutional nitrogen atom adjacent to a vacancy
in the diamond lattice. The negatively charged NV⁻ center has a spin-1 ground
state with remarkable properties:

.. math::

   \hat{H}_{\text{NV}} = \hat{H}_{\text{ZFS}} + \hat{H}_{\text{Zeeman}} +
                          \hat{H}_{\text{HF}} + \hat{H}_{\text{MW}} + \cdots

The electronic structure features:

- **Ground state** :math:`{}^3A_2`: Spin triplet with :math:`S=1`
- **Excited state** :math:`{}^3E`: Optically accessible at 637 nm (ZPL)
- **Singlet states** :math:`{}^1A_1, {}^1E`: Enable spin polarization

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Theory & Physics

   theory/nv_center
   theory/hamiltonian
   theory/lindblad
   theory/measurement

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   tutorials/installation
   tutorials/quickstart
   tutorials/odmr
   tutorials/pulsed

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/hamiltonian
   api/dynamics
   api/states
   api/pulses
   api/interfaces

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
--------

If you use QUSIM in your research, please cite:

.. code-block:: bibtex

   @software{qusim2024,
     author = {Kaiser, Leon},
     title = {QUSIM: Hyperrealistic NV Center Quantum Simulator},
     year = {2024},
     institution = {MSQC, Goethe University Frankfurt},
     url = {https://github.com/xleonplayz/QUSIM}
   }

References
----------

.. [Doherty2013] Doherty, M. W. et al. "The nitrogen-vacancy colour centre in diamond."
   *Physics Reports* **528**, 1-45 (2013).

.. [Maze2011] Maze, J. R. et al. "Properties of nitrogen-vacancy centers in diamond:
   the group theoretic approach." *New J. Phys.* **13**, 025025 (2011).

.. [Childress2013] Childress, L. & Hanson, R. "Diamond NV centers for quantum computing
   and quantum networks." *MRS Bulletin* **38**, 134 (2013).
