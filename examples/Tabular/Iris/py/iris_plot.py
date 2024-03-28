from NNSOM.plots import SOMPlots

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

# Visualization
# Data Preparation
# persentage of sentosa
proportion_sentosa = []
for i in range(som.numNeurons):  # S is the total number of neurons
    cluster_indices = som.clust[i]
    sentosa_in_cluster = 0

    # Count how many indices in this cluster are in sentosa
    for index in cluster_indices:
        if y[index] == 0:
            sentosa_in_cluster += 1

    # Calculate the proportion of sentosa in this cluster
    if len(cluster_indices) > 0:
        proportion = sentosa_in_cluster / len(cluster_indices)
    else:
        proportion = 0  # Avoid division by zero if the cluster is empty

    # Add the calculated proportion to the list
    proportion_sentosa.append(proportion)

# Get percentatges for each neuron
perc_sentosa = np.array(proportion_sentosa) * 100

# Closest Class for each cluster
closest_class = []
for i in range(som.numNeurons):
    cluster_indices = som.clust[i]
    if len(cluster_indices) > 0:
        closest_class.append(y[cluster_indices[0]])
    else:
        closest_class.append(None)

# Target value in each cluster
target_in_cluster = []
for i in range(som.numNeurons):
    cluster_indices = som.clust[i]
    if len(cluster_indices) > 0:
        target_in_cluster.append(y[cluster_indices])
    else:
        target_in_cluster.append(None)

# Grey Hist
fig, ax, pathces, text = som.gray_hist(X, perc_sentosa)
plt.show()

# Color Hist
fig, ax, pathces, text = som.color_hist(X, perc_sentosa)
plt.show()

# Simple Grid
# Plot color code for sentosa in each cluster
fig, ax, patches, cbar = som.simple_grid(perc_sentosa, proportion_sentosa)
plt.title('Simple grid for sentosa', fontsize=16)
plt.show()

# Multi Plot - Pie Chart
# Data Preprocessing for additional Variable
# Using the target in cluster, create the size list for pie chart for each cluster
pie_chart_sizes = []
for cluster in target_in_cluster:
    if cluster is not None:
        # Count the occurrences of each target value
        size = [list(cluster).count(i) for i in range(3)]  # Assuming 3 unique values: 0, 1, 2
        pie_chart_sizes.append(size)
    else:
        pie_chart_sizes.append([0, 0, 0])  # Represent an empty cluster

fig, ax, h_axes = som.multiplot('pie', "Class Distribution", perc_sentosa, pie_chart_sizes)

# Multi Plot - Stem Plot
dist_1 = []
dist_2 = []
for cluster in target_in_cluster:
    if cluster is not None:
        # Use numpy to count occurrences of each class
        counts = [np.sum(cluster == 0), np.sum(cluster == 1), np.sum(cluster == 2)]
        dist_1.append([0, 1, 2])
        dist_2.append(counts)
    else:
        dist_1.append([0, 1, 2])
        dist_2.append([0, 0, 0])

fig, ax, h_axes = som.multiplot('stem', dist_1, dist_2)
plt.show()

# Multiplot - Histogram
sepal_length_in_cluster = []
for i in range(som.numNeurons):
    cluster_indices = som.clust[i]
    if len(cluster_indices) > 0:
        sepal_length_in_cluster.append(iris.data[cluster_indices, 0])
    else:
        sepal_length_in_cluster.append([0])

fig, ax, h_axes = som.multiplot('hist', sepal_length_in_cluster)
plt.show()

# Multiplot - Boxplot
fig, ax, h_axes = som.multiplot('boxplot', sepal_length_in_cluster)
plt.show()

# Multiplot - Violin Plot
# Input Data Preprocessing
# Create the matrix for each cluster where the items in the hit cluster
iris_cluster = []
for i in range(som.numNeurons):
    cluster_indices = som.clust[i]
    if len(cluster_indices) > 0:
        # Make sure cluster_indices are integers and within the range of iris.data
        cluster_indices = np.array(cluster_indices, dtype=int)
        # Index iris.data using cluster_indices
        cluster_data = iris.data[cluster_indices]
        iris_cluster.append(cluster_data)
    else:
        iris_cluster.append(np.array([]))  # Use an empty array for empty clusters

fig, ax, h_axes = som.multiplot('violin', iris_cluster)
plt.show()

# Scatter Plot
fig, axes, h_axes = som.plt_scatter(X, (0, 1))
plt.show()


