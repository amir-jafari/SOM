from NNSOM.plots import SOMPlots
from NNSOM.utils import extract_cluster_details, get_perc_cluster, closest_class_cluster, \
    majority_class_cluster, get_cluster_array, count_classes_in_cluster, get_cluster_data

import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# SOM Parameters
SOM_Row_Num = 4  # The number of row used for the SOM grid.
Dimensions = (SOM_Row_Num, SOM_Row_Num) # The dimensions of the SOM grid.

# Random State
SEED = 1234567
rng = default_rng(SEED)

# Data Preprocessing
iris = load_iris()
X = iris.data
y = iris.target
X = X[rng.permutation(len(X))]
y = y[rng.permutation(len(X))]
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)
X = np.transpose(X)

# Define the directory path for saving the model outside the repository
model_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..", "..", "Model"))
trained_file_name = "SOM_Model_iris_Epoch_500_Seed_1234567_Size_4.pkl"

som = SOMPlots(Dimensions)
som = som.load_pickle(trained_file_name, model_dir + os.sep)

# Data post processing
clust, dist, mdist, clustSize = extract_cluster_details(som, X)

# Extract information from input to pass plot functions
perc_sentosa = get_perc_cluster(y, 0, clust)
closest_class_array = closest_class_cluster(y, clust)
majority_class_array = majority_class_cluster(y, clust)
target_in_cluster = get_cluster_array(y, clust)
iris_class_counts_cluster_array = count_classes_in_cluster(y, clust)
iris_class_align = np.tile([0, 1, 2], (len(clust), 1))
sepal_length_in_cluster = get_cluster_array(np.transpose(X)[:, 0], clust)
sepal_width_in_cluster = get_cluster_array(np.transpose(X)[:, 1], clust)
iris_cluster = get_cluster_data(iris.data, clust)

clust, dist, mdist, clustSize = extract_cluster_details(som, X)

# Data Preparation to pass additional variables
# Extract Information from input to pass plot functions
perc_sentosa = get_perc_cluster(y, 0, clust)
closest_class_array = closest_class_cluster(y, clust)
majority_class_array = majority_class_cluster(y, clust)
target_in_cluster = get_cluster_array(y, clust)
iris_class_counts_cluster_array = count_classes_in_cluster(y, clust)
iris_class_align = np.tile([0, 1, 2], (len(clust), 1))
sepal_length_in_cluster = get_cluster_array(np.transpose(X)[:, 0], clust)
sepal_width_in_cluster = get_cluster_array(np.transpose(X)[:, 1], clust)
iris_cluster = get_cluster_data(iris.data, clust)

# Visualization
# Grey Hist
fig, ax, pathces, text = som.gray_hist(X, perc_sentosa)
plt.show()

# Color Hist
fig2, ax2, pathces2, text2 = som.color_hist(X, perc_sentosa)
plt.show()


# Pie Chart
fig3, ax3, h_axes3 = som.multiplot('pie', "Class Distribution", perc_sentosa, iris_class_counts_cluster_array, False)
plt.show()

# Stem Plot
fig4, ax4, h_axes4 = som.multiplot('stem', iris_class_align, iris_class_counts_cluster_array)
plt.show()

# Histogram
fig, ax, h_axes = som.multiplot('hist', sepal_length_in_cluster)
plt.show()

# Boxplot
fig, ax, h_axes = som.multiplot('boxplot', sepal_length_in_cluster)
plt.show()

# Violin plot
fig, ax, h_axes = som.multiplot('violin', iris_cluster)
plt.show()

# Scatter Plot
fig, axes, h_axes = som.plt_scatter(sepal_length_in_cluster, sepal_width_in_cluster)
plt.show()

# Components Plane
som.component_planes(X)
