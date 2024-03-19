# B. For cases where there is one additional discrete variable (finite number of types, e.g. label #) associated with each item:

# This dataset contains information about passengers aboard the Titanic, including discrete variables like passenger class, sex, and whether they survived or not.

from NNSOM import SOM
import numpy as np
import pickle
from datetime import datetime
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

# Load data
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

X = np.transpose(X)  # Transpose if necessary for your model


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
else:
    # Read in trained network
    file_to_read = open(Trained_SOM_File, "rb")
    som_net = pickle.load(file_to_read)


if Save_SOM_Flag:
    filehandler = open(Trained_SOM_File, 'wb')
    pickle.dump(som_net, filehandler, pickle.HIGHEST_PROTOCOL)

# Compute statistics
# Distance between each input and each weight
x_w_dist = cdist(som_net.w, np.transpose(X), 'euclidean')

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
