from NNSOM.plots import SOMPlots
from NNSOM.utils import extract_cluster_details, get_ind_misclassified, \
    get_perc_misclassified, get_conf_indices, count_classes_in_cluster, cal_class_cluster_intersect

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
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
clust, dist, mdist, clustSizes = extract_cluster_details(som, X)

# Train Logistic Regression on Iris
logit = LogisticRegression(random_state=SEED)
logit.fit(np.transpose(X), y)
results = logit.predict(np.transpose(X))

# Date Preparation for plots
ind_misclassified = get_ind_misclassified(y, results)
perc_misclassified = get_perc_misclassified(y, results, clust)
num_classes = count_classes_in_cluster(y, clust)
num_sentosa = num_classes[:, 0]
sen_tp, sen_tn, sen_fp, sen_fn = get_conf_indices(y, results, 0)
sentosa_conf = cal_class_cluster_intersect(clust, sen_tp, sen_tn, sen_fp, sen_fn)
sentosa_conf_align = np.tile([0, 1, 2, 3], (len(clust), 1))

# Visualization
# Gray Hist
fig1, ax1, patches1, text1 = som.gray_hist(X, perc_misclassified)
plt.show()

# Color Hist
fig2, ax2, patches2, text2 = som.color_hist(X, perc_misclassified)
plt.show()

# Complex Hit hist
fig, ax, patches, text = som.cmplx_hit_hist(X, clust, perc_misclassified, ind_misclassified, sen_fp, sen_fn)
plt.show()

# Simple Grid
fig, ax, patches, cbar = som.simple_grid(perc_misclassified, num_sentosa)
plt.show()

# Pie chart
fig, ax, h_axes = som.multiplot("pie", 'Pie Chart - tp, tn, fp, fn', perc_misclassified, sentosa_conf)
plt.show()

# Step Plot
fig, ax, h_axes = som.multiplot("stem", sentosa_conf_align, sentosa_conf)
plt.show()


