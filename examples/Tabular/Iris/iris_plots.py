import numpy as np
import matplotlib.pyplot as plt
import pickle
from NNSOM import utils
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os
# Set parameters
SOM_Row_Num = 4
file_path = os.getcwd() + os.sep
os.chdir('../../../')
abs_path = os.getcwd() + os.sep
model_path = file_path + os.sep + 'Models' + os.sep
output_path = file_path + os.sep + 'Output' + os.sep
os.chdir(file_path)


iris = load_iris()

# Create DataFrame
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns=iris['feature_names'] + ['target'])

# Map target values to label names
label_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
data['target'] = data['target'].map(label_names)

# Separate features and target
X = data[data.columns[:-1]]
y = data['target']

# Normalize the data using preminmax
normalized_data, min_values, max_values = utils.preminmax(X)
df = pd.DataFrame(normalized_data, columns=iris.feature_names)

X = np.transpose(df)


# Missclassification Indices
Ind_missClasses = []    # indices of misclassified inputs

for i in range(len(y)):
    if y[i] != result[i]:
        Ind_missClasses.append(i)

cm = confusion_matrix(y, result)
# Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
cm_df = pd.DataFrame(cm,
                     index = ['SETOSA','VERSICOLR','VIRGINICA'],
                     columns = ['SETOSA','VERSICOLR','VIRGINICA'])

type1_error_index = []  # False Positives for 'SETOSA'
type2_error_index = []  # False Negatives for 'SETOSA'

for i in range(len(y)):
    if y[i] == 0 and result[i] != 0:
        # This is a Type 2 error for 'SETOSA' (False Negative)
        type2_error_index.append(i)
    elif y[i] != 0 and result[i] == 0:
        # This is a Type 1 error for 'SETOSA' (False Positive)
        type1_error_index.append(i)

X = np.transpose(X)

Trained_SOM_File = model_path + "SOM_Trained_Iris.pkl"

# Read in trained network
file_to_read = open(Trained_SOM_File, "rb")
som_net = pickle.load(file_to_read)

ww = som_net.w
# Compute statistics
# Distance between each input and each weight
x_w_dist = cdist(ww, np.transpose(X), 'euclidean')

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
tp = [[] for _ in range(8)]
fp = [[] for _ in range(8)]
fn = [[] for _ in range(8)]
tn = [[] for _ in range(8)]

Perc_pos = np.zeros((8,S))
Perc_neg = np.zeros((8,S))
num_pos = np.zeros((8,S))
tp_n = np.zeros((8,S))
fp_n = np.zeros((8,S))
fn_n = np.zeros((8,S))
tn_n = np.zeros((8,S))
cats_n = np.zeros((8,S))

mdist = np.zeros(S)

clustSize = []

for i in range(S):
    # print('i = ' + str(i))
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

    # Find FP, FN, TP, TN for each category
    for cati in range(8):
        # False positives
        fpos = np.intersect1d(tempclust, Ind21[cati])
        fp[cati].append(fpos)
        fp_n[cati][i] = len(fpos)

        # False negatives
        fneg = np.intersect1d(tempclust, Ind12[cati])
        fn[cati].append(fneg)
        fn_n[cati][i] = len(fneg)
        # True positives
        temp = np.union1d(fp[cati][i], fn[cati][i])
        temp = np.setdiff1d(tempclust, temp)
        tpos = np.intersect1d(temp, index_c[cati])
        tp[cati].append(tpos)
        tp_n[cati][i] = len(tpos)
        # True negatives
        tneg = np.intersect1d(temp, index_nc[cati])
        tn[cati].append(tneg)
        tn_n[cati][i] = len(tneg)
        # Postives
        cats_n[cati][i] = len(np.intersect1d(tempclust, index_c[cati]))

    if num!=0:
        num = float(num)
        # Percentage of good binders in each cluster
        for cati in range(8):
            num_pos[cati][i] = len(np.intersect1d(tempclust, index_c[cati]))
            Perc_pos[cati][i] = 100 * len(np.intersect1d(tempclust, index_c[cati])) / num
            Perc_neg[cati][i] = 100 * len(np.intersect1d(tempclust, index_nc[cati])) / num

    else:
        print('Num = 0, i = ' + str(i))


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
wwdist = cdist(ww, ww, 'euclidean')
sst  = ndist[:, ind1]
for d in dd:
    factor1 = 2*d*d
    factor2 = Q*d*np.sqrt(2*np.pi)
    temp = np.exp(-np.multiply(sst,sst)/factor1)
    distortion = np.sum(np.multiply(temp,x_w_dist))/factor2
    print('Distortion (d='+str(d)+') = ' + str(distortion))

Category = ['0_Setosa', '1_Versicoclor', '2_Virginicia']

input_data_points_in_clusters = []
for i in range(len(Clust)):
    cluster_indices = Clust[0]  # Indices of input data points in cluster i
    input_data_points = X[:, cluster_indices]  # Input data points in cluster i
    input_data_points_in_clusters.append(input_data_points)

# Now, input_data_points_in_clusters contains input data points for each cluster

fig0, ax0, patch0 = som_net.plt_hist(input_data_points_in_clusters)
plt.show()

# Plot distribution of categories
fig1, ax1, patch1 = som_net.plt_dist(cats_n.transpose())
plt.show()

# Plot the topology
fig2, ax2, patch2, text2 = som_net.plt_top_num()
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

for i in range(3):
    # Plot the pie plots showing tp, fn, tn, fp for each cluster, with same size for each hexagon
    Title  = ''
    # Title = 'TP(g), FN(y), TN(b), FP(r) for '  +  Category[i]
    fig2, ax2, handles2 = som_net.plt_pie(Title, same_size, tp_n[i], fn_n[i], tn_n[i], fp_n[i])
    plt.show()

for i in range(3):
    # Plot color code for false positives
    fig2, ax2, handles2, cbar2 = som_net.simple_grid(fp_n[i], net_ones)
    plt.show()

for i in range(3):
    # Plot color code for percent positive for each category in  each hexagon
    fig5, ax5, patches5, cbar5 = som_net.simple_grid(Perc_pos[i], net_ones)
    plt.title('Percent Positive, Category ' + Category[i], fontsize=16)
    plt.show()



