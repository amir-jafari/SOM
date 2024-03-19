# B. For cases where there is one additional discrete variable (finite number of types, e.g. label #) associated with each item:

# This dataset contains information about passengers aboard the Titanic, including discrete variables like passenger class, sex, and whether they survived or not.

from NNSOM import SOM
import numpy as np
import pickle
from datetime import datetime
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
now = datetime.now()
from numpy.random import default_rng
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

# Path Settings
file_path = os.getcwd() + os.sep
os.chdir('../../../')
abs_path = os.getcwd() + os.sep
data_path = abs_path + 'Data' + os.sep + 'Tabular' + os.sep + 'Titanic' + os.sep
model_path = file_path + os.sep + 'Models' + os.sep
output_path = file_path + os.sep + 'Output' + os.sep
os.chdir(file_path)

# Load Titanic dataset
titanic_df = pd.read_csv(data_path + 'titanic.csv')

# Flag to initialize the som (True), or load previously initialized (False)
Init_Flag = True
# Flag to save initialized model
Init_Save_Flag = False
# Flag to train the som, or load previously trained
Train_Flag = True
# Flag to save trained model
Save_SOM_Flag = False

# Set parameters
SOM_Row_Num = 4
SOM_Col_Num = 4
N = 3

Dimensions = (SOM_Row_Num, SOM_Col_Num, N+1)
Epochs = 1000
Steps = 100
Init_neighborhood = 3
SEED = 1234567
rng = default_rng(SEED)


# Preprocess the data
titanic_df.dropna(inplace=True)  # Remove rows with missing values for simplicity
titanic_df = titanic_df[['Pclass', 'Age', 'Fare', 'Survived']]  # Select relevant features
print(titanic_df.head())
scaler = StandardScaler()
X = scaler.fit_transform(titanic_df.values)
X = pd.DataFrame(X, columns=['Pclass', 'Age', 'Fare', 'Survived' ])
tot_num = len(X)

# Randomize to get different results
X = X[rng.permutation(tot_num)]

# Initializing can take a long time for larege data sets
# Reduce size here. X1 is used for initialization, X is used for training.
X1 = X[:int(tot_num/8)]
X1 = np.transpose(X1)

X = np.transpose(X)


Init_Som_File = model_path + "SOM_init_titanic.pkl"
Trained_SOM_File = model_path + "SOM_Model_titanic.pkl"

# Train SOM or load pretrained SOM
if Train_Flag:
    if Init_Flag:
        # Initialize SOM
        som_net = SOM(Dimensions)
        som_net.init_w(X)
    else:
        # Load initialized SOM
        with open(Init_Som_File, "rb") as file_to_read:
            som_net = pickle.load(file_to_read)
    # Train the SOM
    som_net.train(X, Init_neighborhood, Epochs, Steps)

    # Check if the directory exists, if not, create it
    model_dir = os.path.dirname(Trained_SOM_File)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save trained SOM
    with open(Trained_SOM_File, 'wb') as filehandler:
        pickle.dump(som_net, filehandler, pickle.HIGHEST_PROTOCOL)

print(X)
print(som_net.w)
# Compute statistics
# Distance between each input and each weight
x_w_dist = cdist(som_net.w, X, 'euclidean')

# Find the index of the weight closest to the input
ind1 = np.argmin(x_w_dist, axis=0)

# Compute quantization error
quant_err = np.mean(np.min(x_w_dist, axis=0))
print('Quantization error = ' + str(quant_err))

# Compute topological error
ndist = som_net.neuron_dist
sort_dist = np.argsort(x_w_dist, axis=0)
top_dist = [ndist[sort_dist[0, ii], sort_dist[1, ii]] for ii in range(sort_dist.shape[1])]
top_error_1st_neighbor = np.mean(np.array(top_dist) > 1.1) * 100
top_error_2nd_neighbor = np.mean(np.array(top_dist) > 2.1) * 100
print('Topological Error (1st neighbor) = ' + str(top_error_1st_neighbor) + '%')
print('Topological Error (1st and 2nd neighbor) = ' + str(top_error_2nd_neighbor) + '%')

# Compute distortion
dd = [1, 2, 3]  # neighborhood distances
ww = som_net.w
wwdist = cdist(ww, ww, 'euclidean')
sst = ndist[:, ind1]
Q = X.shape[1]
for d in dd:
    factor1 = 2 * d * d
    factor2 = Q * d * np.sqrt(2 * np.pi)
    temp = np.exp(-np.multiply(sst, sst) / factor1)
    distortion = np.sum(np.multiply(temp, x_w_dist)) / factor2
    print('Distortion (d=' + str(d) + ') = ' + str(distortion))
