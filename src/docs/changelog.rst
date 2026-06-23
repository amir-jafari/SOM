Changelog
=========

1.8.3 (2025)
------------

* Declared missing runtime dependencies (``numpy``, ``scipy``, ``matplotlib``,
  ``networkx``) in ``pyproject.toml`` so ``pip install NNSOM`` installs them
  automatically.
* Overhauled Sphinx documentation: RTD theme, full API reference generated
  from docstrings, Quick Start and Installation guides, Changelog page.
* Added NumPy-style docstrings to all previously undocumented utility
  functions in ``utils.py``.

1.8.2 (2024)
------------

* Removed author email from package metadata.
* Minor documentation and packaging improvements.

1.8.1 (2024)
------------

* Patch release: corrected wheel and source distribution builds.
* Added Forest example dataset under ``examples/Tabular/Forest/``.

1.8.0 (2024)
------------

* Restructured package layout to ``src/NNSOM`` standard.
* Added :class:`~NNSOM.plots.SOMPlots` visualization class with interactive
  plot support via :meth:`~NNSOM.plots.SOMPlots.plot`.
* Added GPU-accelerated :class:`~NNSOM.som_gpu.SOMGpu` with automatic CuPy
  detection; :class:`~NNSOM.plots.SOMPlots` falls back to NumPy when CuPy is
  unavailable.
* New quality metrics: :meth:`~NNSOM.som.SOM.quantization_error`,
  :meth:`~NNSOM.som.SOM.topological_error`,
  :meth:`~NNSOM.som.SOM.distortion_error`.
* Utility functions for cluster statistics:
  :func:`~NNSOM.utils.get_cluster_avg`,
  :func:`~NNSOM.utils.majority_class_cluster`,
  :func:`~NNSOM.utils.get_perc_cluster`, and more.
* Sub-clustering support via ``SOM.sub_som``.
* Pickle-based model persistence:
  :meth:`~NNSOM.som.SOM.save_pickle` /
  :meth:`~NNSOM.som.SOM.load_pickle`.
* Iris dataset example notebooks included.