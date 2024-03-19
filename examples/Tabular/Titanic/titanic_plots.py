import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy.spatial.distance import cdist
import os
# Set parameters
SOM_Row_Num = 4
file_path = os.getcwd() + os.sep
os.chdir('../../../')
abs_path = os.getcwd() + os.sep
data_path = abs_path + 'Data' + os.sep + 'Tabular' + os.sep + 'Titanic' + os.sep
model_path = file_path + os.sep + 'Models' + os.sep
output_path = file_path + os.sep + 'Output' + os.sep
os.chdir(file_path)

input_file = data_path + 'titanic.csv'
titanic_df = pd.read_csv(input_file)

titanic_df.dropna(inplace=True)
titanic_df = titanic_df[['Pclass', 'Age', 'Fare', 'Survived']]

# Randomize to get different results
titanic_df = titanic_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Convert the dataset to a NumPy array
X = titanic_df.values

# Initializing can take a long time for large datasets
# Reduce size here. X1 is used for initialization, X is used for training.
X1 = X[:int(len(X) / 8)]  # Adjust the fraction as needed
X1 = np.transpose(X1)

X = np.transpose(X)

Trained_SOM_File = model_path + "SOM_Model_titanic.pkl"

# Read in trained network
file_to_read = open(Trained_SOM_File, "rb")
som_net = pickle.load(file_to_read)

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

Category = ['0_Yes', '1_No']

fig0, ax0, patch0 = som_net.plt_wgts()
plt.show()

# Plot distribution of categories
fig0, ax0, patch0 = som_net.plt_dist(cats_n.transpose())
plt.show()

# Plot the topology
fig0, ax0, patch0, text0 = som_net.plt_top_num()
plt.title('SOM Topology')
plt.show()

# Plot the hit histogram
fig4, ax4, patch4, text4 = som_net.hit_hist(X, True)
plt.show()

# Plot distances between clusters
fig55, ax55, patch55 = som_net.neuron_dist_plot()
plt.title('SOM Neighbor Weight Distances', fontsize=16)
plt.show()

# Plot color code for maximum distance of each cluster
fig51, ax51, patches51, cbar51 = som_net.simple_grid(mdist, net_ones)
plt.title('Maximum radius for each cluster', fontsize=16)
plt.show()

for i in range(8):
    # Plot the pie plots showing tp, fn, tn, fp for each cluster, with same size for each hexagon
    Title  = ''
    # Title = 'TP(g), FN(y), TN(b), FP(r) for '  +  Category[i]
    fig2, ax2, handles2 = som_net.plt_pie(Title, same_size, tp_n[i], fn_n[i], tn_n[i], fp_n[i])
    plt.show()

for i in range(8):
    # Plot color code for false positives
    fig2, ax2, handles2, cbar2 = som_net.simple_grid(fp_n[i], net_ones)
    plt.show()

for i in range(8):
    # Plot color code for percent positive for each category in  each hexagon
    fig5, ax5, patches5, cbar5 = som_net.simple_grid(Perc_pos[i], net_ones)
    plt.title('Percent Positive, Category ' + Category[i], fontsize=16)
    plt.show()






