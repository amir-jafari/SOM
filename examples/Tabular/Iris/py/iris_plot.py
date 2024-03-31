from NNSOM.plots import SOMPlots
from NNSOM.utils import *
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Random State
SEED = 1234567
rng = default_rng(SEED)

iris = load_iris()
X = iris.data
y = iris.target

# Preprocessing data
X = X[rng.permutation(len(X))]
y = y[rng.permutation(len(X))]

scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)
X = np.transpose(X)

# Define the directory path for saving the model outside the repository
model_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..", "..", "Model/"))
trained_file_name = "SOM_Model_iris_Epoch_500_Seed_1234567_Size_4.pkl"

# Define the path for loading the model
model_path = os.path.join(model_dir, trained_file_name)

# SOM Parameters
SOM_Row_Num = 4  # The number of row used for the SOM grid.
Dimensions = (SOM_Row_Num, SOM_Row_Num) # The dimensions of the SOM grid.

som = SOMPlots(Dimensions)
som = som.load_pickle(trained_file_name, model_path)

clust, dist, mdist, clustSize = extract_cluster_details(som, X)

# Visualization
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


# Grey Hist
fig, ax, pathces, text = som.gray_hist(X, perc_sentosa)
plt.show()

# Color Hist
fig, ax, pathces, text = som.color_hist(X, perc_sentosa)
plt.show()


# Multi Plot - Pie Chart
# The distribution of three classes in each cluster
fig, ax, h_axes = som.multiplot('pie', "Class Distribution", perc_sentosa, iris_class_counts_cluster_array, False)
plt.show()

# Multi Plot - Stem Plot
# Distribution of Categories
# x: Categories (0: sentosa, 1: versicolor, 2: virginica)
# y: count for each class
fig, ax, h_axes = som.multiplot('stem', iris_class_align, iris_class_counts_cluster_array)
plt.show()

# Multiplot - Histogram

fig, ax, h_axes = som.multiplot('hist', sepal_length_in_cluster, clust)
plt.show()

# Multiplot - Boxplot

fig, ax, h_axes = som.multiplot('boxplot', sepal_length_in_cluster, clust)
plt.show()

# Multiplot - Violin Plot
fig, ax, h_axes = som.multiplot('violin', iris_cluster, clust)
plt.show()

# Scatter Plot
fig, axes, h_axes = som.plt_scatter(sepal_length_in_cluster, sepal_width_in_cluster)
plt.show()

# Component planes
som.component_planes(X)


