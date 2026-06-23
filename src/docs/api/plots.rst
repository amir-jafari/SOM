SOMPlots — Visualization
========================

:class:`~NNSOM.plots.SOMPlots` extends either :class:`~NNSOM.som.SOM` or
:class:`~NNSOM.som_gpu.SOMGpu` (whichever is available) with a full suite
of visualization methods.  It is the recommended entry point for most users.

.. code-block:: python

   from NNSOM.plots import SOMPlots

   som = SOMPlots((8, 8))
   som.init_w(data)
   som.train(data, init_neighborhood=3, epochs=200, steps=100)

   fig, ax, patches = som.plot('neuron_dist')

.. currentmodule:: NNSOM.plots

.. autoclass:: SOMPlots
   :no-members:

Constructor
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SOMPlots.__init__

Topology plots
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SOMPlots.plt_top
   SOMPlots.plt_top_num

Hit and distance maps
---------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SOMPlots.hit_hist
   SOMPlots.neuron_dist_plot
   SOMPlots.gray_hist

Cluster and class plots
-----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SOMPlots.cmplx_hit_hist
   SOMPlots.custom_cmplx_hit_hist
   SOMPlots.simple_grid
   SOMPlots.plt_nc

Color-coded and component maps
-------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SOMPlots.color_hist
   SOMPlots.plt_stem
   SOMPlots.plt_pie
   SOMPlots.plt_wgts
   SOMPlots.plt_histogram
   SOMPlots.plt_boxplot
   SOMPlots.plt_violin_plot
   SOMPlots.plt_scatter
   SOMPlots.component_positions
   SOMPlots.component_planes
   SOMPlots.weight_as_image

Interactive plot
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SOMPlots.plot