import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from NNSOM import SOM
from sklearn.datasets import load_iris

# Assuming you have your data stored in a variable named 'data'
# Load the Iris dataset
iris = load_iris()

# Create DataFrame
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# Map target values to label names
label_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
data['target'] = data['target'].map(label_names)

# Separate features and target
X = data[data.columns[:-1]]
y = data['target']


data = np.transpose(X)

# Train your SOM
som = SOM(dimensions=(10, 10))  # Adjust dimensions as needed
som.init_w(data)
som.train(data)

# Get cluster assignments
cluster_assignments = np.argmax(som.outputs, axis=0)

# Group data by cluster
clustered_data = {}
for i, cluster_id in enumerate(cluster_assignments):
    if cluster_id not in clustered_data:
        clustered_data[cluster_id] = []
    clustered_data[cluster_id].append(data[i])

# Variables to plot (adjust as needed)
variables_to_plot = [0, 1, 2]  # Assuming you want to plot the first 3 variables

# Create boxplots for each variable within each cluster
for cluster_id, cluster_data in clustered_data.items():
    plt.figure(figsize=(10, 6))
    plt.title(f'Cluster {cluster_id} Boxplots')
    plt.boxplot(np.array(cluster_data)[:, variables_to_plot], labels=[f'Variable {i+1}' for i in variables_to_plot])
    plt.xlabel('Variable')
    plt.ylabel('Value')
    plt.show()
