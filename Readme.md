# NNSOM

## Self-Organizing Maps

NNSOM is a Python library that provides an implementation of Self-Organizing Maps (SOM) using NumPy and CuPy.
SOM is a type of Artificial Neural Network that can transform complex, nonlinear statistical relationships between high-dimensional data into simple topological relationships on a low-dimensional display (typically 2-dimensional).

The library is designed with two main goals in mind:

- Extensibility: NNSOM aims to provide a solid foundation for researchers to build upon and extend its functionality according to their specific requirements.
- Educational Value: The implementation is structured in a way that allows students to quickly understand the inner workings of SOM, fostering a better grasp of the algorithm's details.

With NNSOM, researchers and students alike can leverage the power of SOM for various applications, such as data visualization, clustering, and dimensionality reduction, while benefiting from the flexibility and educational value offered by this library.

## Installation

You can install the NNSOM by just using pip:

```angular2html
pip install NNSOM
```

## How to use it

You can see the example file with Iris dataset on Jupyter Notebook [here](https://github.com/amir-jafari/SOM/blob/main/examples/Tabular/Iris/notebook/Iris_training.ipynb).

### Data Preparation
To use the NNSOM library effectively, format your data as a NumPy matrix where each row is an observation. 
```bash
import numpy as np
np.random.seed(42)
data = np.random.rand(3000, 10)
```

Alternatively, you can provide the data as a list of lists, following this structure:
```bash
data = [
  [value1, value2, value3, ..., valueN], # Observation 1
  [value1, value2, value3, ..., valueN], # Observation 2
  ...,
  [value1, value2, value3, ..., valueN], # Observation M
]
```

### Customize Your Normalization
Depending on your data's specific characteristics, you may opt to define a custom normalization function. 
Here's how to normalize your data using sklearn's MinMaxScaler:
```bash 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
norm_func = scaler.fit_transform
```

### Configurate the SOM Grid Parameters
Then, you can configurate the SOM Grid Parameters as follows:
```bash
SOM_Row_Num = 4  # The number of rows in the SOM grid
SOM_Col_Num = 4  # The number of columns in the SOM grid
Dimensions = (SOM_Row_Num, SOM_Col_Num) # The two-dimensional layout of the SOM grid 
```

### Configurate the Training Parameters 
Next, you can configurate the Training Parameters as follows:
```bash
Epochs = 200  # The total number of training epochs 
Steps = 100  #  The granularity of the weight update process within each epoch.
Init_neighborhood = 3 # Initial size of the neighborhood radius   
```

### Train the SOM
Then, you can train NNSOM just as follows:
```bash
from NNSOM.plots import SOMPlots
som = SOMPlots(Dimensions)  # Initialization of 4x4 SOM
som.init_w(data, norm_func=norm_func) # Initialize the weight
som.train(data, Init_neighborhood, Epochs, Steps)
```

### Export a SOM and load it again
A model can be saved using pickle as follows:
```bash
file_name = "..."
model_path = ".../"

som.save_pickle(file_name, model_path)
```
and can be loaded as follows:
```bash
from NNSOM.plots import SOMPlots
som = SOMPlots(Dimensions)  # Use the same dimension with the stored model.
som = som.load_pickle(file_name, model_path)
```

### Post-Training Data Clustering with NNSOM
After training SOM with NNSOM, you can leverage the trained model to cluster new or existing data. 
```bash
clust, dist, mdist, clusterSizes = som.cluster_data(data)
```
- clust: This is a list where each sublist contains the indices of data points that are assigned to the same cluster.
- dist: This list mirrors the structure of the "clust" list, with each sublist containing the distances of the corresponding data points. in "clust" from their Best Matching Unit.
- mdist: An array where each element represents the maximum distance between the SOM neuron.
- clusterSizes: An array listing the number of data points in each cluster.

### Error Analysis
NNSOM offers comprehensive tools to assess the quality and reliability of the trained SOM through various error metrics. 
Understanding these errors can help refine the SOM's configuration and interpret its performance effectively. 
Below are the three types of error measures provided by NNSOM:

#### 1. Quantization Error
Quantization error measures the average distance between each data point and its Best Matching Unit (BMU). This error provides insight into the SOM's ability to accurately represent the data space. A lower quantization error generally indicates a better representation.

Examples:
```bash
# Find quantization error
clust, dist, mdist, clusterSizes = som.cluster_data(data)
quant_err = som.quantization_error(dist)
print('Quantization error: ' + str(quant_err))
```

#### 2. Topological Error
Topological error evaluates the SOM's preservation of the data's topological structure. It is calculated by checking if adjacent data points in the input space are mapped to adjacent neurons in the SOM. This metric is split into two:

- Topological Error (1st neighbor): Measures the proportion of data points whose first nearest neighbor in the input space is not their neighbor on the map.
- Topological Error (1st and 2nd neighbor): Extends this to the first and second nearest neighbors.

Examples:
```bash
# Find topological error
top_error_1, top_error_1_2 =  som.topological_error(data)
print('Topological Error (1st neighbor) = ' + str(top_error_1) + '%')
print('Topological Error (1st and 2nd neighbor) = ' + str(top_error_1_2) + '%')
```

#### 3. Distortion Error
Distortion error calculates the total distance between each data point and its corresponding BMU, scaled by the data density around each BMU. This error helps to understand how well the SOM covers the distribution of the dataset and identifies areas where the map might be over or under-fitting.

Examples:
```bash
# Find Distortion Error
som.distortion_error(data)
```

### Visualize the SOM

To effectively understand and interpret the results of your SOM training, visualizing the SOM grid is crucial.
The NNSOM library offers a variety of plotting functions that allow you to visualize different aspects of the SOM and the training process.

#### The Generic Plot Function [[source]](https://github.com/amir-jafari/SOM/blob/main/src/NNSOM/plots.py#L1391)
This generic plot function can be used to generate multiple types of visualizations depending on the specified plot type.

Usage of the Plot Function:
```bash
som.plot('plot_type', data_dict=None, ind=None, target_class=None, use_add_array=False)
```

Parameters:
- plot_type: A string indicating the type of plot to generate. Options include 'top', 'neuron_dist', 'hit_hist', etc.
- data_dict: Optional dictionary containing data needed for specific plots.
- ind: Optional index for targeted plotting.
- target_class: Optional parameter to specify a target class for the plot.
- use_add_array: Boolean flag to indicate whether additional arrays in data_dict should be used.

Structure of data_dict:

The data_dict parameter should be structured as follows to provide necessary data for the plots:
```bash
data_dict = {
  "data": data,          # Main dataset used in SOM training or the new inputs data
  "target": y,           # Target variable, if applicable
  "clust": clust,        # Clustering results from SOM
  "add_1d_array": [],    # Additional 1D arrays for enhanced plotting
  "add_2d_array": [],    # Additional 2D arrays for enhanced plotting
}
```

The source code for the plot function can be found [here](https://github.com/amir-jafari/SOM/blob/main/src/NNSOM/plots.py#L1391).

#### Examples of Common Visualizations

1. Topological Grid
    
    Visualize the topological grid of the SOM to understand the layout and structure of the neurons.
```bash
import matplotlib.pyplot as plt
fig, ax, patches = som.plot('top')
plt.show()
```

2. Neuron Distance Map (U-Map)
    
    Display a distance map (U-Map) to see the distances between neighboring neurons, highlighting potential clusters.
```bash
fig, ax, pathces = som.plot('neuron_dist')
plt.show()
```

3. Hit Histogram

    Generate a hit histogram to visualize the frequency of each neuron being the best matching unit.
```bash
fig, ax, patches, text = som.plot('hit_hist', data_dict)
plt.show()
```
