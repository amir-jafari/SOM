import matplotlib.pyplot as plt

from NNSOM.plots import SOMPlots
from NNSOM.utils import *

from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

"""## Set Up the parameters used for NNSOM"""

# SOM Parameters
SOM_Row_Num = 4  # The number of row used for the SOM grid.
Dimensions = (SOM_Row_Num, SOM_Row_Num) # The dimensions of the SOM grid.

# Training Parameters
Epochs = 500
Steps = 100
Init_neighborhood = 3

# Random State
from numpy.random import default_rng
SEED = 1234567
rng = default_rng(SEED)


iris = load_iris()
X = iris.data
y = iris.target

# Preprocessing data
X = X[rng.permutation(len(X))]
y = y[rng.permutation(len(X))]

scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)
X_scaled = np.transpose(X_scaled)

# Training SOM

# Determine model dir and file name
model_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..", "..", "Model"))
Trained_SOM_File = "SOM_Model_iris_Epoch_" + str(Epochs) + '_Seed_'  + str(SEED) + '_Size_' + str(SOM_Row_Num) + '.pkl'

# Load som instance
som = SOMPlots(Dimensions)
som = som.load_pickle(Trained_SOM_File, model_dir + os.sep)

"""## Extract SOM Cluster Details

After training the SOM, information on which clusters the training data were classified into can be obtained. This can be used to visualize various additional variables on the topology of the SOM.

**clust**: sequence of vectors with indices of input data that are in each cluster sorted by distance from cluster center.

dist: sequence of vectors with distance of input data that are in each cluster sorted by distance fro cluster center.

mdist: 1d array with maximum distance in each cluster

clustSize: 1d array with number of items in each cluster
"""

clust, dist, mdist, clustSize = extract_cluster_details(som, X_scaled)

"""# Visualize the SOM with the generic plot function

Generic Plot Function:

- Parameters:
  - plot_type (str): the type of plot to be generated.
  - data_dict (dict): a dictionary containing the data to be plotted.

    The key of the dictionary is predefined:
      - "original_data": the raw data trained
      - "input_data": the original data scaled from -1 to 1
      - "target": the target class of orinal data
      - "clust": the sequence of vector with indices
      - "cat_1darray": the additional 1d-array with categorical feature
      - "cat_2darray": the additional 2d-array with categorical feature
      - "num_1darray": the additional 1d-array with numerical feature

  - ind (int, tuple, or list): the index (indices) to be plotted on original data
  - target_class (int): the target class to be plotted on target or cat_1darray
  - use_add_1darray (bool): if true, the additional 1-d array to be used
  - use_add_2darray (bool): if true, the additional 2-d array to be used
  - **kwargs (dict): Additional argument to be passed to the interactive plot function

"""

# 1. Plots that do not require an Argument

#Topology

fig, ax, pathces = som.plot("top")
plt.show()

fig, ax, patches, text = som.plot('top_num')
plt.show()

fig, ax, patches = som.plot('neuron_connection')
plt.show()

fig, ax, patches = som.plot('neuron_dist')
plt.show()

fig, ax, h_axes = som.plot('wgts')
plt.show()

# 2. Plots requiring Additional Variable

# Data Preparation

data_dict ={
    "original_data": X, # iris.data shuffled with the specific random state
    'input_data': X_scaled, # iris.data scaled from -1 to 1
    "target": y,  # iris.target shuffled with the specific random state
    "clust": clust, # sequence of vectors with indices. This clust is corresponding to original data
    "cat_1darray": None,
    "cat_2darray": None,
    "num_1darray": None
}

# Hit histogram
fig, ax, patches, text = som.plot('hit_hist', data_dict)
plt.show()

# Gray Hist without using additional categorical 1-d array.
fig, ax, patches, text = som.plot('gray_hist', data_dict, target_class=0)
plt.suptitle("Gray Histogram without an additional variable - Sentosa", fontsize=16)
plt.show()

fig, ax, patches, text = som.plot('gray_hist', data_dict, target_class=1)
plt.suptitle("Gray Histogram without an additional variable - Verginica", fontsize=16)
plt.show()

# If the len(cat_1darray) == len(original_data), the class percentage is calculated in the function.
# In this case, the user requied to provide target_class where the user want to check.
data_dict['cat_1darray'] = y
fig, ax, patches, text = som.plot('gray_hist', data_dict, target_class=0, use_add_1darray=True)
plt.suptitle("Gray Histogram with non-pre processed additional variable - Sentosa", fontsize=16)
plt.show()

# If the len(cat_1darray) == numNeurons,
# the 1-d array is passes the plot directory without specifying target_class
sent_perc = get_perc_cluster(y, 0, clust)
data_dict['cat_1darray'] = sent_perc
fig, ax, patches, text = som.plot('gray_hist', data_dict, use_add_1darray=True)
plt.suptitle("Gray Histogram with the pre-processed additional array - Sentosa", fontsize=16)
plt.show()

fig, ax, patches, text, cbar = som.plot('color_hist', data_dict, ind=0)
plt.suptitle("Color Histogram without an additional variable - Sentosa", fontsize=16)
plt.show()

fig, ax, patches, text, cbar = som.plot('color_hist', data_dict, ind=2)
plt.suptitle("Color Histogram without an additional variable - Virginica", fontsize=16)
plt.show()


data_dict['num_1darray'] = X[:, 0]
fig, ax, patches, text, cbar = som.plot('color_hist', data_dict, use_add_1darray=True)
plt.suptitle("Color Histogram with non-preprocessed additional variable - Sentosa", fontsize=16)
plt.show()

data_dict['num_1darray'] = get_cluster_avg(X[:, 0], clust)
fig, ax, patches, text, cbar = som.plot('color_hist', data_dict, use_add_1darray=True)
plt.suptitle("Color Histogram with a preprocessed additional variable - Sentosa", fontsize=16)
plt.show()

fig, ax, patches, text = som.plot("complex_hist", data_dict, target_class=0)
plt.suptitle("Complex Hit Histogram without additional variables", fontsize=16)
plt.show()

face_labels = majority_class_cluster(y, clust)
edge_labels = closest_class_cluster(y, clust)
edge_width = get_edge_widths(np.where(y == 0)[0], clust)
data_dict['cat_2darray'] = np.transpose(np.array([face_labels, edge_labels, edge_width]))

fig, ax, patches, text = som.plot("complex_hist", data_dict, use_add_2darray=True)
plt.suptitle("Complex Hit Histogram with additional variables", fontsize=16)
plt.show()

fig, ax, patches, cbar = som.plot('simple_grid', data_dict, ind=0, target_class=0)
plt.suptitle("Simple Grid without Additional Variable")
plt.show()

# num_feature: len(num_1darray) == numNeurons
data_dict['cat_1darray'] = sent_perc
# cut_feature: len(cat_1darray) == numNeurons
data_dict['num_1darray'] = get_cluster_avg(X[:, 0], clust)

fig, ax, patches, cbar = som.plot('simple_grid', data_dict, ind=0, target_class=0, use_add_1darray=True)
plt.suptitle("Simple Grid with Additional Variable - numNeuron")
plt.show()

# num_feature: len(num_1darray) == len(original_data)
data_dict['num_1darray'] = X[:, 0]
# cut_feature: len(cat_1darray) == len(original_data)
data_dict['cat_1darray'] = y
fig, ax, patches, cbar = som.plot('simple_grid', data_dict,target_class=0, use_add_1darray=True)
plt.suptitle("Simple Grid with Additional Variable - len(original_data)")
plt.show()

# Pie Chart
fig, ax, h_axes = som.plot('pie', data_dict)
plt.suptitle("Pie Chart with target distribution")
plt.show()

data_dict['cat_2darray'] = count_classes_in_cluster(y, clust)
fig, ax, h_axes = som.plot('pie', data_dict, use_add_2darray=True)
plt.suptitle("Pie Chart with additional variable distribution")
plt.show()

fig, ax, h_axes = som.plot('stem', data_dict)
plt.suptitle("Stem Plot with target distribution")
plt.show()

fig, ax, h_axes = som.plot('stem', data_dict, use_add_2darray=True)
plt.suptitle("Stem Plot with additional variable distribution")
plt.show()

fig, ax, h_axes = som.plot('hist', data_dict, ind=0)
plt.suptitle("Histogram - Sepal Length")
plt.show()

fig, ax, h_axes = som.plot('hist', data_dict, ind=1)
plt.suptitle("Histogram - Sepal Width")
plt.show()

fig, ax, h_axes = som.plot('box', data_dict, ind=0)
plt.suptitle("Box Plot - Sepal Length")
plt.show()

fig, ax, h_axes = som.plot('box', data_dict, ind= [0, 1])
plt.suptitle("Box Plot without additional variable - Sepal Length and Sepal Width")
plt.show()

fig, ax, h_axes = som.plot('box', data_dict)
plt.suptitle("Box Plot without additional variable - Iris Features")
plt.show()

fig, ax, h_axes = som.plot('violin', data_dict, ind=0)
plt.suptitle("Violin Plot - Sepal Length")
plt.show()

fig, ax, h_axes = som.plot('violin', data_dict, ind= [1, 2])
plt.suptitle("Violin Plot - Sepal Length and Sepal Width")
plt.show()

fig, ax, h_axes = som.plot('violin', data_dict)
plt.suptitle("Violin Plot - Iris Features")
plt.show()

fig, ax, h_axes = som.plot('scatter', data_dict, ind=[0, 1])
plt.suptitle("Scatter Plot - Sepal Length vs Sepal Width")
plt.show()

fig, ax, h_axes = som.plot('scatter', data_dict, ind=[2, 3])
plt.suptitle("Scatter Plot - Petal Length vs. Petal Width")
plt.show()

som.plot('component_planes', data_dict)

som.plot('component_positions', data_dict)