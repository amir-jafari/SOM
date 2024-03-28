# Importing Library
import os

from NNSOM.plots import SOMPlots
from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Set the parameters
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

# Data Preparation
iris = load_iris()
X = iris.data
y = iris.target

# Preprocessing data
X = X[rng.permutation(len(X))]
y = y[rng.permutation(len(X))]

scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)
X = np.transpose(X_scaled)

# Training
som = SOMPlots(Dimensions)
som.init_w(X)
som.train(X, Init_neighborhood, Epochs, Steps)


import os

# Define the directory path for saving the model outside the repository
model_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..", "..", "Model/"))
# Create the directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
Trained_SOM_File = "SOM_Model_iris_Epoch_" + str(Epochs) + '_Seed_'  + str(SEED) + '_Size_' + str(SOM_Row_Num) + '.pkl'


# Define the path for saving the model
model_path = os.path.join(model_dir, Trained_SOM_File)

# Save the model
som.save_pickle(Trained_SOM_File, model_path)


# Visualization
# SOM Topology
fig1, ax1, patches1 = som.plt_top()
plt.show()

# SOM Topology with neruon numbers
fig2, ax2, pathces2, text2 = som.plt_top_num()
plt.show()

# Hit Histogram
fig3, ax3, patches3, text3 = som.hit_hist(X, True)
plt.show()

# Neighborhood Connection Map
fig4, ax4, patches4 = som.plt_nc()
plt.show()

# Distance Map
fig5, ax5, patches5 = som.neuron_dist_plot()
plt.show()

# Weight as Line
fig6, ax6, h_axes = som.multiplot('wgts')
plt.show()

# Weight positions
som.component_positions(X_scaled)

# Weight Planes
som.component_planes(X)