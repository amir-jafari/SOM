import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def forest_eda(data):
    """
    Perform exploratory data analysis on a subset of data to visualize the distributions
    and relationships between three selected columns.

    This function creates a grid of plots with histograms on the diagonal to show the
    distribution of each variable and scatter plots in the upper triangle to show the
    relationships between pairs of variables. The lower triangle is left blank.

    Parameters:
    -----------
    data : numpy.ndarray
        A 2D NumPy array where each row represents an observation and each column
        represents a variable. The function specifically uses columns 7, 8, and 9 from this array.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The Figure object containing the grid of plots.

    axes : numpy.ndarray of matplotlib.axes._subplots.AxesSubplot
        An array of Axes objects representing the subplots in the grid.

    Notes:
    ------
    - The function is hardcoded to use columns 7, 8, and 9 from the input data. Ensure
      that the input array has at least 9 columns.
    - The histograms use 10 bins and are colored 'skyblue' with 'black' edgecolor.
    - Scatter plots use a 'darkblue' color for points with an alpha value of 0.5 for
      transparency.
    - The layout of plots is automatically adjusted to fit into the figure space neatly.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)
    df = pd.DataFrame(data[:, 6:9], columns=['7', '8', '9'])
    n = len(df.columns)
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(8, 8))
    # Loop through rows
    for i in range(n):
        # Loop through columns
        for j in range(n):
            # Hide lower triangle and diagonal
            if i > j:
                axes[i, j].axis('off')
            # Diagonal: plot histograms
            if i == j:
                axes[i, j].hist(df[df.columns[i]], bins=10, color='skyblue', edgecolor='black')
                axes[i, j].set_ylabel('Frequency')
            # Upper triangle: plot scatter
            if i < j:
                axes[i, j].scatter(df[df.columns[i]], df[df.columns[j]], alpha=0.5, color='darkblue')
                axes[j, i].set_xlabel(df.columns[i])  # Set x-axis label for corresponding lower triangle plot
                axes[i, j].set_ylabel(df.columns[j])  # Set y-axis label

    plt.tight_layout()
    plt.show()

    return fig, axes

