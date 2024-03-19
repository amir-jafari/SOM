# A.For cases in which there is one additional continuous variable associated with each item
# We can use the load_diabetes dataset,
# which contains ten baseline variables (age, sex, body mass index, average blood pressure, and six blood serum measurements)
# and a quantitative measure of disease progression one year after baseline.
from datetime import datetime
from som import SOM
from plots import Plots

from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np

now = datetime.now()
from numpy.random import default_rng
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
import os


from sklearn.datasets import load_diabetes
import pandas as pd

# Load the diabetes dataset
diabetes = load_diabetes()

# Create a pandas DataFrame
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

# Add the target variable (disease progression)
df['disease_progression'] = diabetes.target

# Display the first few rows of the DataFrame
print(df.head())

# Define independent variable (e.g., body mass index)
X = df[['bmi']]  # Selecting only the 'bmi' column as the independent variable

# Define dependent variable
y = df['disease_progression']

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

# Plot the hit histogram
plots = Plots()
fig0, ax0, patch0, text0 = plots.hit_hist( som, df, True)
plt.show()
