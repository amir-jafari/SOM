# Roadmap
Goal: To create a sophisticated SOM package. 
The main methods would be simulation, training, saving and loading a model, calculating distances, and creating plots. 
Regarding the plot methods, we need the functionality for displaying trained SOM information and for post-training analysis of other trained models.

## Time Schedule
- week 1: Topic Proposal
- week 2: 
- week 3:
- week 4:
- week 5:
- week 6 (~ Mar 12): Implementing Plot Functions in SOM class and  testing with Iris data
- week 7 (~ Mar 19): Implementing Plot Functions in SOM class and testing with Iris data / Midterm presentation
- week 8 (~ Mar 26):
- week 9 (~ Apr 2): 
- week 10 (~ Apr 9): 
- week 11 (~ Apr 16): 
- week 12 (~ Apr 23): Final Presentation, Final Journal Submission

## 1. SOM Packages Functionality

### 1.1 Plot functions
- [ ] A. For cases in which there is one additional continuous variable associated with each item: (Lakshmi by March 19)
    - [ ] A.1 Shade the hexagon with a greyscale or colorcode using the average of the variable accross the items iin each cluster or with the standard deviation of the variable across the items in the cluster.
    - [ ] A.2 Make a histogram of the variable in each cluster
    - [ ] A.3 Make a boxplot of the variable in ach cluster
    - [ ] A.4 Make a dispersin fan diagram of the variable in each cluster.
    - [ ] A.5 Make a violin plot of the variable in each cluster.
          
- [ ] B. For cases in which there are one additional discrete variable (finite number of types, e.g. label # ) assosciate with each item: (Lakshmim by March 19)
    - [ ] B.1 Shade the hexagon with a greyscale or colorcode using the average of the variable accross the items iin each cluster or with the standard deviation of the variable across the items in the cluster.
    - [ ] B.2 Make a bar chart or stem plot of the numbers of each variable type in each cluster.
    - [ ] B.3 Make a pie chart of the percentage of each varriable type in each cluster.
    - [ ] B.4 Make a pie chart of the percentage of each variable type in each cluster.

- [X] C. For cases in which there are two continuous variables associated with each item: (Ei by March 19)
    - [X] C.1 make a scatter plot of the two variables in each cluster.
    - [X] C.2 Perform a regression between the two variables and plot the regression line in each cluster.
    - [ ] I want to discuss how should the user pass the input data and variables they want to see. (Ei) 
    
- [ ] D. For cases with two discrete variables in each cluster (e.g., predicted and true class) (Ei by March 19)
    - [ ] D.1 Confusino matrix type grid.
    - [ ] D.2 Grid heatmap.
          
- [X] E. Hit Histogram (the hit histogram clud bee made with the original training data, or with a new set of data.

- [X] F. Neighbor distance plot (U-matrix)

- [ ] G. Weight planes (Lakshmi by March 19)

- [ ] H. Weight positions (Lakshmi by March 19)

- [X] I. SOM Topology

- [ ] J. Neighbor connections. (Ei by March 19)

- [X] K. Complex Hit histogram: For hit histogram can ake interior of hexagon color coded to some additional variable and edges of hecagons couded to some other variable. Witdth of items in the cluster.
      
- [ ] L. For any of plots, could have the size of the plots related to the number of items in the clusters. (Sometimes make the radius of the pie plots proportioanl to the number of items in the cluster.)

- [X] N. Plot a grey scale or color code with the hexagon related to the radius of the cluster (e.g., maximum distance of an item from the cluster center) 

- [ ] O. Things to do by right cliking on a cluster and selecting from a pop up menu. (Ei by March 19)
    - [ ] O.1 Plot the 5 (or any number) closest items to the cluster center.
    - [ ] O.2 Compute a suc-SOM for the items in the selected cluster
    - [ ] O.3 Save the indices of the items in the cluster. 

### 1.2 Save and Load
- [X] Save the trained SOM as pickle
- [X] Load the trained SOM as pickle

### 1.3 CuPy (GPU usability)
- [ ] Implement CuPy for faster computation

### 1.4 Documentation
- [ ] Add documentation to the functions

### 1.5 Subclustering
- [ ] Train a subcluster -> Save som and json mapping

## 2. SOM Packages Testing with example data sets

### 2.1 Test the SOM packages as clustering method with some example datasets (text, CV, tabular, etc)
- [ ] Test the SOM packages with some example datasets with labels
- [ ] Get the Plot with them.

### 2.2 Test the SOM as post training tools for a trained model
- [ ] Train a neural network model with some example datasets and make a prediction.
- [ ] Get the confusion matrix of the prediction for the rained model
- [ ] Plot them with SOM cluster and compare them to get insight
