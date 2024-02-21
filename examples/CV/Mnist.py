# Importing the required libraries
from NNSOM.som import SOM
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from keras.datasets import mnist
from numpy.random import default_rng

# Setting Parameters
SOM_Row_Num = 8
Dimensions = (SOM_Row_Num, SOM_Row_Num)
Epochs = 1000
Steps = 100
Init_neighborhood = 3
SEED = 1234567
rng = default_rng(SEED)

# Loading the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data Preprocessing
plt.gray()
plt.figure(figsize=(4, 4))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[i])
    plt.axis('off')

# Normalizing the forest
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshaping the forest
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Input forest
tot_num = len(x_train)
# Randomize to get different results
X = x_train[rng.permutation(tot_num)]

# Initializing can take a long time for larege forest sets
# Reduce size here. X1 is used for initialization, X is used for training.
X1 = X[:int(tot_num/8)]
X1 = np.transpose(X1)

X = np.transpose(X)

# Training the SOM
som_net = SOM(Dimensions)
som_net.init_w(X1)
som_net.train(X, Init_neighborhood, Epochs, Steps)

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
sst = ndist[:, ind1]
for d in dd:
    factor1 = 2*d*d
    factor2 = Q*d*np.sqrt(2*np.pi)
    temp = np.exp(-np.multiply(sst,sst)/factor1)
    distortion = np.sum(np.multiply(temp,x_w_dist))/factor2
    print('Distortion (d='+str(d)+') = ' + str(distortion))

# Plotting the SOM
# Plot the topology
fig, ax, pathces = som_net.plt_top()
plt.title("SOM Topology", fontsize=16)
plt.show()

# SOM Topology with cluster numbers
fig, ax, pathces, text = som_net.plt_top_num()
plt.title("SOM Topology", fontsize=16)
plt.show()

# Plot the hit histogram
fig, ax, pathces, text = som_net.hit_hist(X, True)
plt.title("Hit Histogram", fontsize=16)
plt.show()

# Plot distance between cluster
fig, ax, pathces = som_net.neuron_dist_plot()
plt.title("SOM Neighbor Weight Distances", fontsize=16)
plt.show()

# Simple Grid
fig, ax, patches, cbar = som_net.simple_grid(mdist, net_ones)
plt.title("Maximum radius for each cluster", fontsize=16)
plt.show()

# Plot weights on SOM
fig, ax, h_axes = som_net.plt_wgts()
plt.title("SOM Weights", fontsize=16)
plt.show()

# Plot topology with images
ww = som_net.w
print(ww.shape)

data = ww

# Assuming `forest` is your numpy array with shape (64, 784)
# Reshape forest to (64, 28, 28) to make it easier to handle
images = data.reshape(-1, 28, 28)

# Set up the figure size and grid
fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(10, 10))

# Adjust layout
plt.subplots_adjust(hspace=0.5, wspace=0.5)

for i, ax in enumerate(axes.flat):
    # Display image
    ax.imshow(images[i], cmap='gray')

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

# Show the plot
plt.show()