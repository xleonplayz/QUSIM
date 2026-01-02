============
Installation
============

Requirements
============

QUSIM requires Python 3.9 or later and the following dependencies:

- NumPy ≥ 1.24
- SciPy ≥ 1.10
- Matplotlib ≥ 3.7 (optional, for plotting)

Installation
============

From Source
-----------

Clone the repository and install:

.. code-block:: bash

   git clone https://github.com/xleonplayz/QUSIM.git
   cd QUSIM
   pip install -e .

Or install dependencies manually:

.. code-block:: bash

   pip install numpy scipy matplotlib

Verification
============

Test the installation:

.. code-block:: python

   from sim import HamiltonianBuilder
   from sim.hamiltonian.terms import ZFS

   H = HamiltonianBuilder()
   H.add(ZFS(D=2.87))
   print(H.build().shape)  # Should print (18, 18)

Optional Dependencies
=====================

For the mock server integration:

.. code-block:: bash

   pip install fastapi uvicorn jax jaxlib

For documentation building:

.. code-block:: bash

   pip install sphinx furo myst-parser
