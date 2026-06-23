SOMGpu — GPU Implementation
============================

:class:`~NNSOM.som_gpu.SOMGpu` mirrors the :class:`~NNSOM.som.SOM` API but
offloads computation to the GPU via `CuPy <https://cupy.dev>`_.  It is
selected automatically by :class:`~NNSOM.plots.SOMPlots` when CuPy is
available.

.. code-block:: python

   from NNSOM.som_gpu import SOMGpu   # requires CuPy

   som = SOMGpu((8, 8))
   som.init_w(data)
   som.train(data, init_neighborhood=3, epochs=200, steps=100)

.. note::

   CuPy requires an NVIDIA CUDA-capable GPU.  If CuPy is not installed,
   use :class:`~NNSOM.som.SOM` or :class:`~NNSOM.plots.SOMPlots` instead —
   they fall back to NumPy automatically.

.. currentmodule:: NNSOM.som_gpu

.. autoclass:: SOMGpu
   :no-members:

Initialization and training
----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SOMGpu.init_w
   SOMGpu.train
   SOMGpu.sim_som

Clustering
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SOMGpu.cluster_data

Quality metrics
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SOMGpu.quantization_error
   SOMGpu.topological_error
   SOMGpu.distortion_error

Persistence
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SOMGpu.save_pickle
   SOMGpu.load_pickle
