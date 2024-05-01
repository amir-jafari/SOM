Quick Start Guide for NNSOM
===========================

This guide will help you get started with the NNSOM library, a tool for training Self-Organizing Maps (SOMs).

Basic Usage
-----------

Here's a quick overview of how to set up and train an SOM model using NNSOM.

1. Import the necessary libraries and prepare your data as a NumPy matrix.

.. code-block:: python

    import numpy as np
    np.random.seed(42)
    data = np.random.rand(3000, 10)  # Random data for demonstration

2. Define the normalizing function for the data.

.. code-block:: python

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    norm_func = scaler.fit_transform()

3. Set the SOM grid and training parameters.

.. code-block:: python

    SOM_Row_Num = 4
    SOM_Col_Num = 4
    Dimensions = (SOM_Row_Num, SOM_Col_Num)
    Epochs = 200
    Steps = 100
    Init_neighborhood = 3

4. Initialize and train the SOM.

.. code-block:: python

    from NNSOM.plots import SOMPlots
    som = SOMPlots(Dimensions)
    som.init_w(data, norm_func=norm_func)   # data is normalized based on user specific normalized function
    som.train(data, Init_neighborhood, Epochs, Steps)

5. Save and load the trained model.

.. code-block:: python

    file_name = "my_som_model.pkl"
    model_path = "/path/to/models/"
    som.save_pickle(file_name, model_path)
    som = som.load_pickle(file_name, model_path)

Visualizing Results
-------------------

After training, you can visualize the SOM using the built-in plotting functions.

.. code-block:: python

    import matplotlib.pyplot as plt
    fig, ax, patches = som.plot('neuron_dist')
    plt.show()

For more details on the NNSOM package and its functionalities, refer to the official documentation or the GitHub repository.

