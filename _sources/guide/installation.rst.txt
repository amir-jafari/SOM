Installation
============

Requirements
------------

* Python 3.8 or later
* NumPy
* SciPy
* Matplotlib
* NetworkX

Installing from PyPI
--------------------

.. code-block:: bash

   pip install NNSOM

This installs the ``NNSOM`` package and all runtime dependencies.

Installing from source
----------------------

.. code-block:: bash

   git clone https://github.com/amir-jafari/SOM.git
   cd SOM
   pip install -e .

The ``-e`` flag installs in *editable* mode so any changes to the source are
immediately reflected without reinstalling.

Optional dependencies
---------------------

For GPU acceleration (NVIDIA CUDA required):

.. code-block:: bash

   pip install cupy

For running the example Jupyter notebooks:

.. code-block:: bash

   pip install NNSOM notebook

For building the documentation locally:

.. code-block:: bash

   pip install NNSOM[docs]
   cd src/docs
   make html

For running the test suite:

.. code-block:: bash

   pip install NNSOM[dev]
   pytest

Verifying the installation
--------------------------

.. code-block:: python

   import NNSOM
   from NNSOM.plots import SOMPlots
   print("NNSOM imported successfully")

Getting help
------------

If you encounter a bug or have a question, please open an issue on the
`GitHub issue tracker <https://github.com/amir-jafari/SOM/issues>`_.
Useful bug reports include:

* A minimal reproducible code example
* The error message or unexpected output
* Your NNSOM version (``pip show NNSOM``)
* Your Python and NumPy versions