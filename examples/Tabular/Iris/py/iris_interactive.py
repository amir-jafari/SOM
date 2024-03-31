from NNSOM.plots import SOMPlots
from NNSOM.utils import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from numpy.random import default_rng
import os

# SOM Parameters
SOM_Row_Num = 4  # The number of row used for the SOM grid.
Dimensions = (SOM_Row_Num, SOM_Row_Num)  # The dimensions of the SOM grid.

# Training Parameters
Epochs = 500
Steps = 100
Init_neighborhood = 3

# Random State
SEED = 1234567
rng = default_rng(SEED)

# Data Preparation
iris = load_iris()
X = iris.data
y = iris.target

# Preprocessing data
X = X[rng.permutation(len(X))]
y = y[rng.permutation(len(X))]
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)
X = np.transpose(X)

# Determine model dir and file name
# model_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..", "..", "Model"))
# Trained_SOM_File = "SOM_Model_iris_Epoch_" + str(Epochs) + '_Seed_'  + str(SEED) + '_Size_' + str(SOM_Row_Num) + '.pkl'

# Load som instance
som = SOMPlots(Dimensions)
# Train som instance
som.init_w(X)
som.train(X, Init_neighborhood, Epochs, Steps)
# som = som.load_pickle(Trained_SOM_File, model_dir + os.sep)

# Data Processing
clust, dist, mdist, clustSize = extract_cluster_details(som, X)
num1 = get_cluster_array(X[0], clust)
num2 = get_cluster_array(X[1], clust)
cat = count_classes_in_cluster(y, clust)
height = count_classes_in_cluster(y, clust)  # height for stem
align = get_align_cluster(y, clust)  # align for stem

kwargs = {
    'data': X,
    "labels": "",  # For labels on the plot
    'clust': clust,
    'target': y,
    'num1': num1,  # For hist, box, violin and scatter
    'num2': num2,  # For box, violin and scatter
    'cat': cat,  # For pie chart (sizes)
    'align': align,  # For stem plot
    'height': height,  # For stem plot
    'topn': 5,
}

# Interactive hit hist
fig, ax, patches, text = som.hit_hist(X, textFlag=True, mouse_click=True, **kwarg)
plt.show()

# Interactive neuron dist
fig, ax, patches, text = som.neuron_dist_plot(mouse_click=True, **kwarg)
plt.show()

# Interactive pie plot
# Data Prep for pie
perc_sentosa = get_perc_cluster(y, 0, clust)
iris_class_counts_cluster_array = count_classes_in_cluster(y, clust)

fig, ax, h_axes = som.plt_pie('Pie Chart', perc_sentosa, iris_class_counts_cluster_array, mouse_click=True, **kwargs)
plt.show()

