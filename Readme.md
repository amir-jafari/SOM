# NNSOM

## Self-Organizing Maps

NNSOM is a Python library that provides an implementation of Self-Organizing Maps (SOM) using NumPy and CuPy.
SOM is a type of Artificial Neural Network that can transform complex, nonlinear statistical relationships between high-dimensional data into simple topological relationships on a low-dimensional display (typically 2-dimensional).

The library is designed with two main goals in mind:

- Extensibility: NNSOM aims to provide a solid foundation for researchers to build upon and extend its functionality according to their specific requirements.
- Educational Value: The implementation is structured in a way that allows students to quickly understand the inner workings of SOM, fostering a better grasp of the algorithm's details.

With NNSOM, researchers and students alike can leverage the power of SOM for various applications, such as data visualization, clustering, and dimensionality reduction, while benefiting from the flexibility and educational value offered by this library.

## Installation

You can install the NNSOM by just using pip:

```angular2html
pip install NNSOM
```

## How to use it

### Data Preparation
To utilize the NNSOM library, your data needs to be structured in a specific format. It should be prepared as a NumPy matrix, where each row represents an individual observation. 
```bash
import numpy as np
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
data = np.random.rand(3000, 10)

# Define the normalize function
scaler = MinMaxScaler(feature_range(-1, 1))
norm_func = scaler.fit_transform
```

Alternatively, you can provide the data as a list of lists, following this structure:
```bash
data = [
  [value1, value2, value3, ..., valueN], # Observation 1
  [value1, value2, value3, ..., valueN], # Observation 2
  ...,
  [value1, value2, value3, ..., valueN], # Observation M
]
```

### Configurate the SOM Grid Parameters
Then, you can configurate the SOM Grid Parameters as follows:
- SOM_Row_Num: Specifies the number of rows in the SOM grid.
- SOM_Col_Num: Specifies the number of columns in the SOM grid.
- Dimensions: Defines the two-dimensional layout of the SOM grid.
```bash
SOM_Row_Num = 4  
SOM_Col_Num = 4  
Dimensions = (SOM_Row_Num, SOM_Col_Num)  
```

### Configurate the Training Parameters 
Next, you can configurate the Training Parameters as follows:
- Epochs: The total number of training epochs.
- Steps: This parameter controls the granularity of the weight update process within each epoch.
- Init_neighborhood: Initial size of the neighborhood radius. This radius affects the update extent around the winning neuron in the grid and typically decreases over time during training.
```bash
Epochs = 200  
Steps = 100  
Init_neighborhood = 3  
```

### Train the SOM
Then, you can train NNSOM just as follows:
```bash
from NNSOM.plots import SOMPlots
som = SOMPlots(Dimensions)  # Initialization of 4x4 SOM
som.init_w(data, norm_func=norm_func) # Initialize the weight
som.train(data, Init_neighborhood, Epochs, Steps)
```

### Export a SOM and load it again
A model can be saved using pickle as follows:
```bash
file_name = "..."
model_path = ".../"

som.save_pickle(file_name, model_path)
```
and can be loaded as follows:
```bash
from NNSOM.plots import SOMPlots
som = SOMPlots(Dimensions)
som = som.load_pickle(file_name, model_path)
```