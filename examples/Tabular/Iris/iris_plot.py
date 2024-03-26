# Importing Library
from NNSOM.plots import SOMPlots
from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from NNSOM.plots import SOMPlots
from NNSOM.utils import *
import matplotlib.pyplot as plt

# SOM Parameters
SOM_Row_Num = 4  # The number of row used for the SOM grid.
Dimensions = (SOM_Row_Num, SOM_Row_Num) # The dimensions of the SOM grid.

# Random State
from numpy.random import default_rng
SEED = 1234567
rng = default_rng(SEED)

from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()
X = iris.data
y = iris.target

# Preprocessing data
X = X[rng.permutation(len(X))]
y = y[rng.permutation(len(X))]

scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)
X = np.transpose(X)

"""Loading Pre-trained SOM"""

import os

model_path = "/Users/sravya/Desktop/Capstone/SOM/examples/Tabular/Iris/"
trianed_file_name = "SOM_Model_iris_Epoch_500_Seed_1234567_Size_4.pkl"

# SOM Parameters
SOM_Row_Num = 4  # The number of row used for the SOM grid.
Dimensions = (SOM_Row_Num, SOM_Row_Num) # The dimensions of the SOM grid.

som = SOMPlots(Dimensions)
som = som.load_pickle(trianed_file_name, model_path)

# Find quantization error
quant_err = som.quantization_error()
print('Quantization error: ' + str(quant_err))

# Find topological error
top_error_1, top_error_1_2 =  som.topological_error(X)
print('Topological Error (1st neighbor) = ' + str(top_error_1) + '%')
print('Topological Error (1st and 2nd neighbor) = ' + str(top_error_1_2) + '%')

# Find Distortion Error
som.distortion_error(X)

"""Visualization

Data Preparation to pass additional variables
"""

# persentage of sentosa
numNeurons = som.numNeurons

perc_sentosa = []

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

"""Hit Histogram"""

# Hit Histogram
fig3, ax3, patches3, text3 = som.hit_hist(X, True)
plt.show()

"""Gray Histogram

Darker: Less sentosa
"""

# Visualization
fig, ax, patches, text = som.gray_hist(X, perc_sentosa)
plt.show()

"""Color Hist"""

fig1, ax1, patches1, text1 = som.color_hist(X, perc_sentosa)
plt.show()

"""Complex Hist"""

fig2, ax2, patches2, text2 = som.cmplx_hit_hist(X, perc_sentosa, y[0])
plt.show()

"""Simple Grid"""

# Plot color code for sentosa in each cluster
fig51, ax51, patches51, cbar51 = som.simple_grid(perc_sentosa, proportion_sentosa)
plt.title('Simple grid for sentosa', fontsize=16)
plt.show()


"""Multiplot - pie"""

for i in range(len(proportion_sentosa)):
    Title = f'Pie Plot {i}'  # Title for the current plot
    fig66, ax66, handles66 = som.multiplot('pie', Title, proportion_sentosa[i])
    plt.show()

"""Multiplot - dist"""

fig9, ax9, h_axes9 = som.multiplot('dist',proportion_sentosa)
plt.show()

"""Multiplot - hist"""

fig10, ax10, h_axes10 = som.multiplot('hist',proportion_sentosa)
plt.show()

"""Multiplot - boxplot"""

fig11, ax11, h_axes11 = som.multiplot('boxplot',proportion_sentosa)
plt.show()

"""Multiplot - fanchart"""



"""Multiplot - violin"""

