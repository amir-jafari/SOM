# Roadmap
Goal: To create a sophisticated SOM package. 
The main methods would be simulation, training, saving and loading a model, calculating distances, handling neighborhoods, and creating plots.
Regarding the Plot, we need the functionality not only for displaying the result of the cluster of the dataset but also for post-training analysis of other trained models.

## Time Schedule
- week 1
- week 2
- week 3
- week 4
- week 5: Now here
- week 6: 
- week 7: 
- week 8: 
- week 9: 
- week 10: 
- week 11: 
- week 12: 
- week 13:

## 1. SOM Packages Functionality

### 1.1 Basic Plots
- [ ] Color Higtogram: Single number (different color maps as arg)
- [ ] hit histogram & complex hit hist (combine them in one plot function): Size of the histogram and interior color of histogram and edge color.
- [ ] U map (neuron dist plot)
- [ ] Multiple plot (anuy plot in each hex) pie, bar, stem, etc.

### 1.2 Distance Functions
- [ ] Euclidean
- [ ] Manhattan
- [ ] Cosine
- [ ] Chebyshev

### 1.3 Save and Load
- [X] Save the trained SOM as pickle
- [X] Load the trained SOM as pickle

### 1.4 Interactive Plots
- [ ] Implement interactive plots with Plotly
- [ ] Right click on the cluster to see the cluster information
    - [ ] Plot the cluster center (line or image)
    - [ ] Plot / list the 5 closet inputs to the cluster center

### 1.5 CuPy (GPU usability)
- [ ] Implement CuPy for faster computation

### 1.6 Documentation
- [ ] Add documentation to the functions

### 1.7 Subclustering
- [ ] Train a subcluster -> Save som and json mapping


## 2. SOM Packages Testing with example data sets

### 2.1 Test the SOM packages as clustering method with some example datasets (text, CV, tabular, etc)
- [ ] Test the SOM packages with some example datasets with labels
- [ ] Get the Plot with them.

### 2.2 Test the SOM as post training tools for a trained model
- [ ] Train a neural network model with some example datasets and make a prediction.
- [ ] Get the confusion matrix of the prediction for the rained model
- [ ] Plot them with SOM cluster and compare them to get insight
