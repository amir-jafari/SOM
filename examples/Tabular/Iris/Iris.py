

import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.datasets import load_iris
from som import SOM
from utils import preminmax
import matplotlib.pyplot as plt

# Load the Iris dataset
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
normalized_data, min_values, max_values = preminmax(X)
df = pd.DataFrame(normalized_data, columns=iris.feature_names)

df = np.transpose(df)

# Initialize variables
SOM_Row_Num = 4
Dimensions = (SOM_Row_Num, SOM_Row_Num)
Epochs = 200
Steps = 100
Init_neighborhood = 3
SEED = 1234567

# Initialize random number generator
rng = default_rng(SEED)

# Train SOM
som = SOM(Dimensions)
som.init_w(df)
som.train(df, Init_neighborhood, Epochs, Steps)




