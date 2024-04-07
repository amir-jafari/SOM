from NNSOM.plots import SOMPlots
from NNSOM.utils import *

from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression

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

"""Training SOM"""

# Determine model dir and file name
model_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..", "..", "Model"))
Trained_SOM_File = "SOM_Model_iris_Epoch_" + str(Epochs) + '_Seed_'  + str(SEED) + '_Size_' + str(SOM_Row_Num) + '.pkl'

# Load som instance
som = SOMPlots(Dimensions)
som = som.load_pickle(Trained_SOM_File, model_dir + os.sep)

"""Extract SOM Cluster Details"""

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

# 1. Plots that do not require an Argument (Information about SOM itself)

#Topology
fig, ax, pathces = som.plot("top")
plt.show()

# Toplology with neuron numbers
fig, ax, patches, text = som.plot('top_num')
plt.show()

# neuron connection
fig, ax, patches = som.plot('neuron_connection')
plt.show()

# neuron dist
fig, ax, patches = som.plot('neuron_dist')
plt.show()

# weight as line
fig, ax, h_axes = som.plot('wgts')
plt.show()

# 2. Plots that requires input data

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

# Gray Hist
fig, ax, patches, text = som.plot('gray_hist', data_dict, target_class=0)
plt.suptitle("Gray Histogram - The percentage of Sentosa", fontsize=16)
plt.show()

fig, ax, patches, text = som.plot('gray_hist', data_dict, target_class=1)
plt.suptitle("Gray Histogram - The percentage of Versicolor", fontsize=16)
plt.show()

# Color Histogram
fig, ax, patches, text, cbar = som.plot('color_hist', data_dict, ind=0)
plt.suptitle("Color Histogram  - Sepal Length", fontsize=16)
plt.show()

fig, ax, patches, text, cbar = som.plot('color_hist', data_dict, ind=1)
plt.suptitle("Color Histogram - Sepal Width", fontsize=16)
plt.show()

# Complex Hit Histogram
# sentosa: blue, vesicolor: blue, virginica: red
fig, ax, patches, text = som.plot("complex_hist", data_dict, target_class=0)
plt.suptitle("Complex Hit Histogram", fontsize=16)
plt.show()

# Simple Grid
fig, ax, patches, cbar = som.plot('simple_grid', data_dict, ind=0, target_class=0)
plt.suptitle("Simple Grid - color shade: sepal length, size: perc sentosa")
plt.show()

# Basic Plot Family
# Pie Chart
fig, ax, h_axes = som.plot('pie', data_dict)
plt.suptitle("Pie Chart with target distribution")
plt.show()

# Stem
fig, ax, h_axes = som.plot('stem', data_dict)
plt.suptitle("Stem Plot with target distribution")
plt.show()

# Histogram
fig, ax, h_axes = som.plot('hist', data_dict, ind=0)
plt.suptitle("Histogram - Sepal Length")
plt.show()

fig, ax, h_axes = som.plot('hist', data_dict, ind=1)
plt.suptitle("Histogram - Sepal Width")
plt.show()

# Box Plot
fig, ax, h_axes = som.plot('box', data_dict, ind=0)
plt.suptitle("Box Plot - Sepal Length")
plt.show()

fig, ax, h_axes = som.plot('box', data_dict, ind= [0, 1])
plt.suptitle("Box Plot without additional variable - Sepal Length and Sepal Width")
plt.show()

fig, ax, h_axes = som.plot('box', data_dict)
plt.suptitle("Box Plot without additional variable - Iris Features")
plt.show()

# Violin Plot
fig, ax, h_axes = som.plot('violin', data_dict, ind=0)
plt.suptitle("Violin Plot - Sepal Length")
plt.show()

fig, ax, h_axes = som.plot('violin', data_dict, ind= [1, 2])
plt.suptitle("Violin Plot - Sepal Length and Sepal Width")
plt.show()

fig, ax, h_axes = som.plot('violin', data_dict)
plt.suptitle("Violin Plot - Iris Features")
plt.show()

# Scatter Plot
fig, ax, h_axes = som.plot('scatter', data_dict, ind=[0, 1])
plt.suptitle("Scatter Plot - Sepal Length vs Sepal Width")
plt.show()

fig, ax, h_axes = som.plot('scatter', data_dict, ind=[2, 3])
plt.suptitle("Scatter Plot - Petal Length vs. Petal Width")
plt.show()

# Component Plane
som.plot('component_planes', data_dict)

# Component Positioins
som.plot('component_positions', data_dict)

# 3. Plot that required Additional Variable (Post Training Analysis)
# Training Logistic Regression
logit = LogisticRegression(solver='lbfgs', multi_class='multinomial')
logit.fit(np.transpose(X_scaled), y)
results = logit.predict(np.transpose(X_scaled))

# Data Preparation
perc_misclassified = get_perc_misclassified(y, results, clust)
sent_tp, sent_tn, sent_fp, sent_fn = get_conf_indices(y, results, 0)
sentosa_conf = cal_class_cluster_intersect(clust, sent_tp, sent_tn, sent_fp, sent_fn)

data_dict = {
    "original_data": X,
    "input_data": X_scaled,
    "target": y,
    "clust": clust,
    'cat_1darray': perc_misclassified,
    # percentage of the misclassified items in each clust, len(cat_1d_array) == numNeuron, gray_hist , simple grid (perc)
    'cat_2darray': sentosa_conf,
    # the number of each conf metrix of sentosa in each cluster, len(cat_2d_array) == numNeuron, pie, stem, complex hit hist
    'num_1darray': perc_misclassified,
    # percentage of the specifi categorical variable, len(num_1d_array) == numNeuron, color_hist, simple grid (avg)
}

# Gray Hist
fig, ax, patches, text = som.plot('gray_hist', data_dict, use_add_1darray=True)
plt.suptitle("Gray Hist - Percentage of misclassified")
plt.show()

# Color Hist
fig, ax, patches, text, cbar = som.plot('color_hist', data_dict, use_add_1darray=True)
plt.suptitle("Gray Hist - Percentage of misclassified")
plt.show()

# Pie Chart
fig, ax, h_axes = som.plot('pie', data_dict, use_add_2darray=True)  # Additional 2d array has numbers of each conf matrix value.
plt.suptitle("Pie Chart - the number of tp, tn, fp, fn in sentosa")
plt.show()

# Stem Plot
fig, ax, h_axes = som.plot('stem', data_dict, use_add_2darray=True)  # Additional 2d array has numbers of each conf matrix value.
plt.suptitle("Stem Plot - the number of tp, tn, fp, fn in sentosa")
plt.show()

# For complex hit histogram, it requires sort of preprocesssing.
ind_misclassified = get_ind_misclassified(y, results) # find the index misclassified
face_color = get_color_labels(clust, ind_misclassified) # based on the misclassified index, create list with 0 or 1. If misclassified index is majority 1, else 0.
edge_color = get_color_labels(clust, sent_tn, sent_fn)  # based on the index of tn and fn of sentosa, if tn is majority assign 1, else 0
edge_width = get_edge_widths(ind_misclassified, clust)  # based on the misclassified index, define the edge widths.
cat_2darray = np.transpose(np.array([face_color, edge_color, edge_width]))
data_dict['cat_2darray'] = cat_2darray

# Complex Hit Histogram
fig, ax, patche, text = som.plot('complex_hist', data_dict, use_add_2darray=True)
plt.suptitle('Complex Hit Hist for Error Analysis')
plt.show()
