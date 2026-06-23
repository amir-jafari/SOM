Utility Functions
=================

All utility functions are importable from ``NNSOM.utils``:

.. code-block:: python

   from NNSOM.utils import preminmax, get_cluster_avg, get_perc_cluster

They are also re-exported at the top level of ``NNSOM``.

.. currentmodule:: NNSOM.utils

Data normalization
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   preminmax

Topology geometry
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   calculate_positions
   distances
   normalize_position
   spread_positions
   get_hexagon_shape
   get_edge_shape

Coordinate transforms
---------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   cart2pol
   pol2cart
   rotate_xy

Cluster data extraction
-----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_cluster_data
   get_cluster_array
   get_cluster_avg
   count_classes_in_cluster
   cal_class_cluster_intersect

Class and label analysis
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   closest_class_cluster
   majority_class_cluster
   get_perc_cluster
   get_color_labels
   get_edge_widths

Error and misclassification analysis
-------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_ind_misclassified
   get_perc_misclassified
   get_conf_indices
   get_dominant_class_error_types

Miscellaneous
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   flatten
   get_global_min_max
   create_buttons
   calculate_button_positions