Quick Start
===========

This page walks through the **five-step SOM workflow** — from raw data to a
trained, saved, and visualized Self-Organizing Map.

Step 1 — Prepare your data
---------------------------

NNSOM expects a 2-D NumPy array with shape ``(features, samples)``.

.. code-block:: python

   import numpy as np
   from sklearn.preprocessing import MinMaxScaler

   # Example: 3000 samples, 10 features
   data = np.random.rand(3000, 10)

   # Normalize to [-1, 1] (recommended)
   scaler = MinMaxScaler(feature_range=(-1, 1))
   data_norm = scaler.fit_transform(data.T).T   # fit_transform works on (samples, features)

Alternatively, use the built-in :func:`~NNSOM.utils.preminmax` utility:

.. code-block:: python

   from NNSOM.utils import preminmax

   data_norm, data_min, data_max = preminmax(data)

Step 2 — Choose the grid size
------------------------------

The SOM grid is defined by ``(rows, cols)``.  A common rule of thumb is to
set the total number of neurons to around 5 × √N, where N is the number of
samples.

.. code-block:: python

   from NNSOM.plots import SOMPlots

   dimensions = (8, 8)   # 64 neurons for 3000 samples
   som = SOMPlots(dimensions)

Step 3 — Initialize and train
------------------------------

Weight initialization uses PCA to align the initial map with the principal
directions of the data, which speeds up convergence.

.. code-block:: python

   som.init_w(data_norm)

   som.train(
       data_norm,
       init_neighborhood=3,   # starting neighbourhood radius
       epochs=200,            # training iterations
       steps=100,             # neighbourhood decay steps
   )

Step 4 — Cluster the data
--------------------------

After training, assign each data point to its best-matching neuron:

.. code-block:: python

   clust, clust_dist, max_clust_dist = som.cluster_data(data_norm)

``clust[i]`` contains the indices of all data points mapped to neuron ``i``,
sorted by their distance to the neuron centre.

Step 5 — Visualize the map
---------------------------

:class:`~NNSOM.plots.SOMPlots` provides a unified ``plot`` method that
dispatches to the right visualization based on the plot type string.

.. code-block:: python

   import matplotlib.pyplot as plt

   # Neuron distance map (U-matrix)
   fig, ax, patches = som.plot('neuron_dist')
   plt.show()

   # Hit histogram — how many samples map to each neuron
   fig, ax, patches = som.plot('hit_hist', data_norm)
   plt.show()

   # Topology with numbered neurons
   fig, ax, patches = som.plt_top_num()
   plt.show()

Save and reload the trained model
----------------------------------

.. code-block:: python

   som.save_pickle("my_som.pkl", "/path/to/models/")

   som2 = SOMPlots(dimensions)
   som2 = som2.load_pickle("my_som.pkl", "/path/to/models/")

Quality metrics
---------------

Three standard metrics are available to evaluate map quality after training:

.. code-block:: python

   outputs = som.sim_som(data_norm)
   dist    = np.min(
       np.linalg.norm(data_norm[:, :, None] - som.w.T[None, :, :], axis=1),
       axis=1
   )

   qe = som.quantization_error(dist)
   te = som.topological_error(data_norm)
   de = som.distortion_error(data_norm)

   print(f"Quantization error : {qe:.4f}")
   print(f"Topological error  : {te:.4f}")
   print(f"Distortion error   : {de:.4f}")