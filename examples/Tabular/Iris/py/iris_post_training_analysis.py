import matplotlib
from NNSOM.plots import SOMPlots
from NNSOM.utils import *
import os
from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

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

# SOM Parameters
SOM_Row_Num = 4  # The number of row used for the SOM grid.
Dimensions = (SOM_Row_Num, SOM_Row_Num)     # The dimensions of the SOM grid.

# Define the directory path for saving the model outside the repository
model_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "..", "..", "Model/"))
trianed_file_name = "SOM_Model_iris_Epoch_500_Seed_1234567_Size_4.pkl"

# Define the path for loading the model
model_path = os.path.join(model_dir, trianed_file_name)

som = SOMPlots(Dimensions)
som = som.load_pickle(trianed_file_name, model_path)

# Extract SOM cluster details
clust, dist, mdist, clustSizes = extract_cluster_details(som, X)

# Train the classifier with Iris dataset
# Train Logistic Regression on Iris
logit = LogisticRegression(random_state=SEED)
logit.fit(np.transpose(X), y)
results = logit.predict(np.transpose(X))

# Visualization
import matplotlib.pyplot as plt

# Data Preparation for plots
ind_misclassified = get_ind_misclassified(y, results)
perc_misclassified = get_perc_misclassified(y, results, clust)
num_classes = count_classes_in_cluster(y, clust)
num_sentosa = num_classes[:, 0]
sen_tp, sen_tn, sen_fp, sen_fn = get_conf_indices(y, results, 0)
sentosa_conf = cal_class_cluster_intersect(clust, sen_tp, sen_tn, sen_fp, sen_fn)
sentosa_conf_align = np.tile([0, 1, 2, 3], (len(clust), 1))



# Grey Hist

# In this case, the color shade indicateds the propotion of Misclassified Neuron.
fig1, ax1, patches1, text1 = som.gray_hist(X, perc_misclassified)
plt.show()

# Color hist

# The color colose to red indicates more likely to be misclassified
fig2, ax2, patches2, text2 = som.color_hist(X, perc_misclassified)
plt.show()

# Complex hit hist
# Inner Color: Whether there are majority correctly or incorrectly classified classes. (Green misclassified dominant, blue correctly classified dominant)
# Edge Color: Majority Error Type for Sentosa (Type 1 Error (red) or Type 2 Error (pink))
# Edge Thickness: Number of Error
fig, ax, patches, text = som.cmplx_hit_hist(X, clust, perc_misclassified, ind_misclassified, sen_fp, sen_fn)
plt.show()

# Simple grid
# color: misclassified percentages
# size: the number of sentosa
# Input Data Preparation

fig, ax, patches, cbar = som.simple_grid(perc_misclassified, num_sentosa)
plt.show()

# Pie Chart
#
# Scale: Larger pie chart indicates more misclassified data.
#
# Distribution: the size of tp, tn, fp, and fn

fig, ax, h_axes = som.multiplot("pie", 'Pie Chart - tp, tn, fp, fn', perc_misclassified, sentosa_conf)
plt.show()

# Stem Plot
#
# Align: tp, tn, fp, fn
#
# Height: the number of tp, tn, fp, and fn

fig, ax, h_axes = som.multiplot("stem", sentosa_conf_align, sentosa_conf)
plt.show()

