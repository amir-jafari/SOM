SOM — CPU Implementation
========================

The :class:`~NNSOM.som.SOM` class is the core NumPy-based Self-Organizing Map.
All public symbols are importable directly from ``NNSOM``:

.. code-block:: python

   from NNSOM.som import SOM

   som = SOM((8, 8))
   som.init_w(data)
   som.train(data, init_neighborhood=3, epochs=200, steps=100)

.. currentmodule:: NNSOM.som

.. autoclass:: SOM
   :no-members:

Initialization and training
----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SOM.init_w
   SOM.train
   SOM.sim_som

Clustering
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SOM.cluster_data

Quality metrics
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SOM.quantization_error
   SOM.topological_error
   SOM.distortion_error

Persistence
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SOM.save_pickle
   SOM.load_pickle