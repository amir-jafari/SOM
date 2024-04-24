CuPy Functionality
===========================

NNSOM leverages CuPy, a library for GPU-accelerated computing with a NumPy-compatible interface, to enhance the performance of SOM computations. By utilizing the parallel processing power of GPUs, NNSOM can process large datasets significantly faster than using CPU-based implementations. This document outlines how to use NNSOM with CuPy and the benefits it offers.

Requirements
------------

Before using CuPy with NNSOM, ensure you have the following:

- A compatible GPU (NVIDIA CUDA-capable GPU)
- CUDA Toolkit (check compatibility with your CuPy version)
- CuPy library installed in your Python environment

To install CuPy, you can use pip:

.. code-block:: bash

   pip install cupy

Usage
-----

To utilize CuPy with NNSOM, import the CuPy-enabled version of the SOM:

.. code-block:: python

   from NNSOM.som_gpu import SOMGpu

Here is an example of initializing and training a SOM on GPU:

.. code-block:: python

   dimensions = (10, 10)  # Dimensions of the SOM grid
   som = SOMGpu(dimensions)  # Initialize SOM
   data = np.random.rand(1000, 10)  # Generate some random data
   som.init_w(data)  # Initialize weights
   som.train(data)  # Train SOM

Benefits
--------

Using CuPy with NNSOM provides the following benefits:

- **Faster computation**: Leveraging GPU for the training and simulation processes of SOMs speeds up computations, especially beneficial for large datasets.
- **Efficient memory usage**: CuPy efficiently manages memory on GPUs, which can be more restrictive than CPU memory.
- **Seamless integration**: Works with existing NumPy arrays and functions, minimizing the need to modify your existing codebase.

Limitations
-----------

While CuPy accelerates computations, there are a few limitations to consider:

- **Hardware dependency**: Requires an NVIDIA GPU; not compatible with other GPU brands.
- **Software compatibility**: Ensure compatibility with CUDA versions and the operating system.
- **Debugging and profiling**: GPU debugging can be more complex compared to CPU.

Automatic CuPy Detection
------------------------

The `SOMPlots` class in NNSOM is designed to automatically detect if CuPy is available in the user's environment. If CuPy is detected, `SOMPlots` will utilize GPU acceleration for all applicable operations, optimizing performance without requiring manual configuration from the user.

.. code-block:: python

   from NNSOM.som_plots import SOMPlots

   # SOMPlots instantiation automatically checks for CuPy
   som_plots = SOMPlots(dimensions=(10, 10))
   data = np.random.rand(1000, 10)  # Example data
   som_plots.init_w(data)  # Initialize weights with potential GPU acceleration
   som_plots.train(data)  # Train SOM with potential GPU acceleration

If CuPy is not available, `SOMPlots` will automatically fall back to a CPU-based implementation, ensuring that SOM computations can still be performed, albeit at a slower speed compared to GPU acceleration. This feature provides a seamless user experience by adapting to the available system resources.


Conclusion
----------

Integrating CuPy with NNSOM allows users to exploit the computational power of GPUs, making it a suitable choice for large-scale and real-time data processing scenarios in machine learning and

