from NNSOM import SOM
import numpy as np
import pickle
from datetime import datetime
from scipy.spatial.distance import cdist
import pandas as pd
now = datetime.now()
from numpy.random import default_rng
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
import os

# Flag to initialize the som (True), or load previously initialized (False)
Init_Flag = False

# Flag to save initialized model
Init_Save_Flag = False

# Flag to train the som, or load previously trained
Train_Flag = False

# Flag to save trained model
Save_SOM_Flag = False

# Set parameters
SOM_Row_Num = 4
data_path = os.getcwd() + os.sep + 'data' + os.sep
model_path = os.getcwd() + os.sep + 'models' + os.sep
output_path = os.getcwd() + os.sep + 'output' + os.sep

Dimensions = (SOM_Row_Num, SOM_Row_Num)
Epochs = 200
Steps = 100
Init_neighborhood = 3
SEED = 1234567
rng = default_rng(SEED)

Init_Som_File = model_path + "SOM_init_f4_ep_50_Seed_" + str(SEED) + '_Size_' + str(SOM_Row_Num) + ".pkl"
Trained_SOM_File = model_path + "SOM_Model_f4_ep_50_Epoch_" + str(Epochs) + '_Seed_' + str(SEED) + '_Size_' + str(
    SOM_Row_Num) + ".pkl"

# Load data
# (needed to be updated the method to make user get the data)
# You can download the data from the following link:
# (https://drive.google.com/file/d/1Jc5DFLza0W05gfXd56hj9hSyVAqa3aG8)
input_file = data_path + 'cv_electra_f4_ep_50_Features.npy'
X = np.load(input_file)

tot_num = len(X)

# Randomize to get different results
X = X[rng.permutation(tot_num)]

# Initializing can take a long time for larege data sets
# Reduce size here. X1 is used for initialization, X is used for training.
X1 = X[:int(tot_num/8)]
X1 = np.transpose(X1)

X = np.transpose(X)


# Train SOM, or load pretrained SOM
if Train_Flag:
    if Init_Flag:
        # Initialize weights of SOM
        som_net = SOM(Dimensions)
        som_net.init_w(X1)
        if Init_Save_Flag:
            with open(Init_Som_File, 'wb') as filehandler:
                pickle.dump(som_net, filehandler)
    else:
        # Read in initialized SOM
        with open(Init_Som_File, 'rb') as filehandler:
            som_net = pickle.load(filehandler)
    # Train network
    som_net.train(X, Init_neighborhood, Epochs, Steps)
else:
    # Read in trained network
    with open(Trained_SOM_File, 'rb') as filehandler:
        som_net = pickle.load(filehandler)

if Save_SOM_Flag:
    with open(Trained_SOM_File, 'wb') as filehandler:
        pickle.dump(som_net, filehandler, protocol=pickle.HIGHEST_PROTOCOL)

# Compute statistics
# Distance between each input and each weight
x_w_dist = cdist(som_net.w, np.transpose(X), 'euclidean')

# Find the index of the weight closest to the input
ind1 = np.argmin(x_w_dist,axis=0)

shapw = som_net.w.shape
S = shapw[0]
shapx = X.shape
Q = shapx[1]
net_ones = np.ones(S)
same_size = 100*np.ones(S)

Clust = []
dist = []
mdist = np.zeros(S)
clustSize = []

for i in range(S):
    # Find which inputs are closest to the current weight (in cluster i)
    tempclust = np.where(ind1==i)[0]

    # Save distance of each input in the cluster to cluster center (weight)
    tempdist = x_w_dist[i, tempclust]
    indsort = np.argsort(tempdist)
    tempclust = tempclust[indsort]
    tempdist = tempdist[indsort]

    # Add to distance array sorted distances
    dist.append(tempdist)

    # Add to Cluster array sorted indices
    Clust.append(tempclust)

    # Cluster size
    num = len(tempclust)
    clustSize.append(num)

    # Save the maximum distance to any input in the cluster from cluster center
    if num>0:
        mdist[i] = tempdist[-1]


# Find quantization error
quant_err = np.array([ 0 if len(item)==0 else np.mean(item) for item in dist]).mean()
print('Quantization error = ' + str(quant_err))

# Topological Error - Percent inputs where closest center and next closest center
# are not neighbors
ndist = som_net.neuron_dist
sort_dist = np.argsort(x_w_dist,axis=0)
top_dist = [ndist[sort_dist[0,ii],sort_dist[1,ii]] for ii  in range(sort_dist.shape[1])]
neighbors = np.where(np.array(top_dist)>1.1)
top_error = 100*len(neighbors[0])/x_w_dist.shape[1]
print('Topological Error (1st neighbor) = ' + str(top_error) + '%')
neighbors = np.where(np.array(top_dist)>2.1)
top_error = 100*len(neighbors[0])/x_w_dist.shape[1]
print('Topological Error (1st and 2nd neighbor) = ' + str(top_error) + '%')


# Distortion
dd = [1, 2, 3] # neighborhood distances
ww = som_net.w
wwdist = cdist(ww, ww, 'euclidean')
sst  = ndist[:, ind1]
for d in dd:
    factor1 = 2*d*d
    factor2 = Q*d*np.sqrt(2*np.pi)
    temp = np.exp(-np.multiply(sst,sst)/factor1)
    distortion = np.sum(np.multiply(temp,x_w_dist))/factor2
    print('Distortion (d='+str(d)+') = ' + str(distortion))

