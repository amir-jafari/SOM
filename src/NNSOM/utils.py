from scipy.spatial.distance import cdist
from scipy import sparse
import numpy as np
import networkx as nx
from matplotlib.widgets import Button


def preminmax(p):
    # Normalize the inputs to be in the range [-1, 1]
    minp = np.amin(p, 1)
    maxp = np.amax(p, 1)

    equal = np.equal(minp, maxp)
    nequal = np.logical_not(equal)

    if sum(equal) != 0:
        print('Some maximums and minimums are equal. Those inputs won''t be transformed.')
        minp0 = minp*nequal - 1*equal
        maxp0 = maxp*nequal + 1*equal
    else:
        minp0 = minp
        maxp0 = maxp

    minp0 = np.expand_dims(minp0, axis=1)
    maxp0 = np.expand_dims(maxp0, axis=1)
    pn = 2*(p-minp0)/(maxp0-minp0) - 1
    return pn, minp, maxp


def calculate_positions(dim):
    # Calculate the positions of the neurons in the SOM
    dims = len(dim)
    position = np.zeros((dims, np.prod(dim)))
    len1 = 1

    center = 0

    for i in range(2):
        dimi = dim[i]
        newlen = len1 * dimi
        offset = np.sqrt(1 - center*center)

        if i == 1:
            for j in range(1, dimi):
                ishift = [center * (j % 2)]
                doshift = np.array(ishift*len1)
                position[0, np.arange(len1)+len1*j] = position[0, np.arange(len1)] + doshift

        posi = np.array(range(dimi)) * offset
        ind2 = np.floor(np.arange(newlen) / len1)
        ind2 = ind2.astype(int)
        position[i, np.arange(newlen)] = posi[ind2]

        len1 = newlen
        center = 0.5

    return position


def cart2pol(x, y):
    # Convert cartesian coordinates to polar coordinates
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    # Convert polar coordinates to cartesian coordinates
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_xy(x1, y1, angle):
    # Rotate the coordinates x1, y1 by angle
    [a, r] = cart2pol(x1, y1)
    a = a + angle
    x2, y2 = pol2cart(a, r)
    return x2, y2


def normalize_position(position):
    # Normalize the positions of the neurons to be in the range [-1, 1]
    shap = position.shape
    numPos = shap[1]
    minPos = np.ndarray.min(position,axis=1)
    maxPos = np.ndarray.max(position,axis=1)
    difPos = maxPos - minPos
    equal = np.equal(minPos, maxPos)
    difPos[equal] = 1
    minPos = np.expand_dims(minPos, axis=1)
    minPos = np.repeat(minPos, numPos, axis=1)
    difPos = np.expand_dims(difPos, axis=1)
    difPos = np.repeat(difPos, numPos, axis=1)
    posit = 2 * ((position - minPos)/difPos) - 1
    return posit


def spread_positions(position, positionMean, positionBasis):
    # Spread the positions of the neurons
    shappos = position.shape
    numPos = shappos[1]
    position1 = np.repeat(positionMean, numPos, axis=1) + np.matmul(positionBasis, position)
    return position1


def distances(pos):
    # Compute the distances between the neurons in the SOM topology
    posT = np.transpose(pos)
    dist = cdist(posT, posT, 'euclidean')

    link = dist <= 1.00001
    link = sparse.csr_matrix(1.0*link)

    g = nx.DiGraph(link)
    dist = nx.floyd_warshall_numpy(g)

    return dist


def get_hexagon_shape():
    # Determine the shape of the hexagon to represent each cluster
    shapex = np.array([-1, 0, 1, 1, 0, -1]) * 0.5
    shapey = np.array([1, 2, 1, -1, -2, -1]) * np.sqrt(0.75) / 3
    return shapex, shapey


def get_edge_shape():
    # Determine the shape of the elongated hexagon to represent edge between each cluster
    edgex = np.array([-1, 0, 1, 0]) * 0.5
    edgey = np.array([0, 1, 0, - 1]) * np.sqrt(0.75) / 3

    return edgex, edgey


# Helper function to extract cluster details obtained from the SOM and input data
def extract_cluster_details(som, data):
    """
    Returns Cluster Index Array, Cluster Distance Array,
    Maximum Distance Array, and Cluster Size Array.

    Parameters
    ----------
    som : SOM object
        A SOM object trained on the input data.
    data : numpy array
        Input data to be clustered.

    Returns
    -------
    clust : list
        A cluster array of indices sorted by distances.
    dist : list
        A cluster array of distances sorted by distances.
    mdist : numpy array
        A list of maimum distance to any input in the cluster from cluster center.
    clustSize : list
        Cluster array sizes.
    """

    # Assertions (check if the input data and som weights have the same number of features)
    if data.shape[0] != som.w.shape[1]:
        raise ValueError('The number of features in the input data and the SOM weights do not match.')

    w = som.w
    shapw = w.shape
    S = shapw[0]

    x_w_dist = cdist(som.w, np.transpose(data), 'euclidean')
    ind1 = np.argmin(x_w_dist, axis=0)

    clust = []  # a cluster array of indices sorted by distances
    dist = []  # a cluster array of distances sorted by distances
    mdist = np.zeros(S)  # a list of maimum distance to any input in the cluster from cluster center
    clustSize = []  # cluster array sizes

    for i in range(S):
        # Find which inputs are closest to each weight (in cluster i)
        tempclust = np.where(ind1 == i)[0]

        # Save distance of each input in the cluster to cluster center (weight)
        tempdist = x_w_dist[i, tempclust]
        indsort = np.argsort(tempdist)
        tempclust = tempclust[indsort]  # Sort indices
        tempdist = tempdist[indsort]

        # Add to distance array sorted distances
        dist.append(tempdist)

        # Add to Cluster array sorted indices
        clust.append(tempclust)

        # Cluster size
        num = len(tempclust)
        clustSize.append(num)

        # Save the maximum distance to any input in the cluster from cluster center
        if num > 0:
            mdist[i] = tempdist[-1]

    return clust, dist, mdist, clustSize


# Helper Functions to extract information from the input data for passing to the plot
def get_cluster_data(data, clust):
    """
    For each cluster, extract the corresponding data points and return them in a list.

    Parameters
    ----------
    data : numpy array
        The dataset from which to extract the clusters' data.
    clust : list of arrays
        A list where each element is an array of indices for data points in the corresponding cluster.

    Returns
    -------
    cluster_data_list : list of numpy arrays
        A list where each element is a numpy array containing the data points of a cluster.
    """
    cluster_data_list = []
    for cluster_indices in clust:
        if len(cluster_indices) > 0:
            # Ensure cluster_indices are integers and within the range of data
            cluster_indices = np.array(cluster_indices, dtype=int)
            cluster_data = data[cluster_indices]
            cluster_data_list.append(cluster_data)
        else:
            cluster_data_list.append(np.array([]))  # Use an empty array for empty clusters

    return cluster_data_list


def get_cluster_array(feature, clust):
    """
    Returns a NumPy array of objects, each containing the feature values for each cluster.

    Parameters
    ----------
    feature : array-like
        Feature array.
    clust : list
        A list of cluster arrays, each containing indices sorted by distances.

    Returns
    -------
    cluster_array : numpy.ndarray
        A NumPy array where each element is an array of feature values for that cluster.
    """
    cluster_array = np.empty(len(clust), dtype=object)

    for i, cluster_indices in enumerate(clust):
        if len(cluster_indices) > 0:
            cluster_array[i] = feature[cluster_indices]
        else:
            cluster_array[i] = np.array([])  # Store an empty array if the cluster is empty

    return cluster_array


def closest_class_cluster(cat_feature, clust):
    """
    Returns the cluster array with the closest class for each cluster.

    Paramters
    ----------
    cat_feature : array-like
        Categorical feature array.
    clust : list
        A cluster array of indices sorted by distances.

    Returns
    -------
    closest_class : numpy array
        A cluster array with the closest class for each cluster.
    """
    closest_class = np.zeros(len(clust))
    for i in range(len(clust)):
        cluster_indices = clust[i]
        if len(cluster_indices) > 0:
            closest_class[i] = cat_feature[cluster_indices[0]]
        else:
            closest_class[i] = None  # Avoid division by zero if the cluster is empty

    return closest_class


def majority_class_cluster(cat_feature, clust):
    """
    Returns the cluster array with the majority class for each cluster.

    Paramters
    ----------
    cat_feature : array-like
        Categorical feature array.
    clust : list
        A cluster array of indices sorted by distances.

    Returns
    -------
    majority_class : numpy array
        A cluster array with the majority class
    """
    majority_class = np.zeros(len(clust))
    for i in range(len(clust)):
        cluster_indices = clust[i]
        if len(cluster_indices) > 0:
            majority_class[i] = np.argmax(np.bincount(cat_feature[cluster_indices]))
        else:
            majority_class[i] = None  # Avoid division by zero if the cluster is empty

    return majority_class


def get_perc_cluster(cat_feature, target, clust):
    """
    Return cluster array with the percentage of a specific target class in each cluster.

    Parameters
    ----------
    cat_feature : array-like
        Categorical feature array.
    target : int or str
        Target class to calculate the percentage.
    clust : list
        A cluster array of indices sorted by distances.

    Returns
    -------
    cluster_array : numpy array
        A cluster array with the percentage of target class.
    """
    # Create Cluster Array with the percentage of target class
    cluster_array = np.zeros(len(clust))
    for i in range(len(clust)):
        cluster_indices = clust[i]
        if len(cluster_indices) > 0:
            cluster_array[i] = np.sum(cat_feature[cluster_indices] == target) / len(cluster_indices)
        else:
            cluster_array[i] = 0  # Avoid division by zero if the cluster is empty

    return cluster_array * 100


def count_classes_in_cluster(cat_feature, clust):
    """
    Count the occurrences of each class in each cluster using vectorized operations
    for efficiency.

    Parameters
    ----------
    cat_feature : array-like
        Categorical feature array.
    clust : list
        A list of arrays, each containing the indices of elements in a cluster.

    Returns
    -------
    cluster_counts : numpy array
        A 2D array with counts of each class in each cluster.
    """
    unique_classes, _ = np.unique(cat_feature, return_counts=True)
    num_classes = len(unique_classes)

    # Initialize the array to hold class counts for each cluster
    cluster_counts = np.zeros((len(clust), num_classes), dtype=int)

    # Loop over clusters to count class occurrences
    for i, indices in enumerate(clust):
        if len(indices) > 0:
            # Count occurrences of each class in the cluster
            cluster_counts[i] = [np.sum(cat_feature[indices] == cls) for cls in unique_classes]
        else:
            cluster_counts[i] = np.zeros(num_classes, dtype=int)

    return cluster_counts


def get_align_cluster(cat_feature, clust):
    """
    Create an array of alignments for each cluster based on class indices.

    Parameters
    ----------
    cat_feature : array-like
        Categorical feature array.
    clust : list
        A list of arrays, each containing the indices of elements in a cluster.

    Returns
    -------
    alignment_cluster : numpy array
        A 2D array with alignments for each cluster.
    """
    unique_classes = np.unique(cat_feature)
    num_classes = len(unique_classes)

    # Initialize the array to hold alignments for each cluster
    alignment_cluster = np.zeros((len(clust), num_classes), dtype=int)

    # Populate the alignment array
    # Since the alignment seems to be the index of the class, we can directly use the unique_classes for alignment
    for i in range(len(clust)):
        alignment_cluster[i] = unique_classes

    return alignment_cluster


def cal_class_cluster_intersect(clust, *args):
    """
    Calculate the intersection sizes of each class with each neuron cluster.

    This function computes the size of the intersection between each given class
    (represented by arrays of indices) and each neuron cluster (represented by
    a list of lists of indices). The result is a 2D array where each row corresponds
    to a neuron cluster, and each column corresponds to one of the classes.

    Parameters
    ----------
    clust : list of lists
        A collection of neuron clusters, where each neuron cluster is a list of indices.
    *args : sequence of array-like
        A variable number of arrays, each representing a class with indices.

    Returns
    -------
    numpy.ndarray
        A 2D array where the entry at position (i, j) represents the number of indices
        in the j-th class that are also in the i-th neuron cluster.

    Examples
    --------
    >>> clust = [[4, 5, 9], [1, 7], [2, 10, 11], [3, 6, 8]]
    >>> ind1 = np.array([1, 2, 3])
    >>> ind2 = np.array([4, 5, 6])
    >>> ind3 = np.array([7, 8, 9])
    >>> ind4 = np.array([10, 11, 12])
    >>> get_sizes_clust(clust, ind1, ind2, ind3, ind4)
    array([[0, 2, 1, 0],
           [1, 0, 1, 0],
           [1, 0, 0, 2],
           [1, 1, 1, 0]])
    """
    numst = list(args)

    cluster_sizes_matrix = np.zeros((len(numst), len(clust)))

    for i, ind in enumerate(numst):
        for j, cluster in enumerate(clust):
            cluster_sizes_matrix[i][j] = np.sum(np.isin(cluster, ind))

    return cluster_sizes_matrix.T


# Helper function  to extract information from the post-training model for passing to the plot
def get_ind_misclassified(target, prediction):
    """
    Get the indices of misclassified items.

    Parameters
    ----------
    target : array-like
        The true target values.
    prediction : array-like
        The predicted values.

    Returns
    -------
    misclassified_indices : list
        List of indices of misclassified items.
    """
    misclassified_indices = np.where(target != prediction)[0]

    return misclassified_indices


def get_perc_misclassified(target, prediction, clust):
    """
    Calculate the percentage of misclassified items in each cluster and return as a numpy array.

    Parameters
    ----------
    target : array-like
        The true target values.
    prediction : array-like
        The predicted values.
    clust : array-like
        List of arrays, each containing the indices of elements in a cluster.

    Returns
    -------
    proportion_misclassified : numpy array
        Percentage of misclassified items in each cluster.
    """
    # Get the indices of misclassified items.
    misclassified_indices = get_ind_misclassified(target, prediction)

    # Initialize array to store proportion of misclassified items
    proportion_misclassified = np.zeros(len(clust))

    for i, cluster_indices in enumerate(clust):
        if len(cluster_indices) > 0:
            # Compute intersection of cluster indices and misclassified indices
            misclassified_count = np.intersect1d(cluster_indices, misclassified_indices).size
            proportion_misclassified[i] = (misclassified_count / len(cluster_indices)) * 100

    return proportion_misclassified


def get_conf_indices(target, results, target_class):
    """
    Get the indices of True Positive, True Negative, False Positive, and False Negative for a specific target class.

    Parameters
    ----------
    target : array-like
        The true target values.
    results : array-like
        The predicted values.
    target_class : int
        The target class for which to get the confusion indices.

    Returns
    -------
    tp_index : numpy array
        Indices of True Positives.
    tn_index : numpy array
        Indices of True Negatives.
    fp_index : numpy array
        Indices of False Positives.
    fn_index : numpy array
        Indices of False Negatives.
    """
    tp_index = np.where((target == target_class) & (results == target_class))[0]
    tn_index = np.where((target != target_class) & (results != target_class))[0]
    fp_index = np.where((target != target_class) & (results == target_class))[0]
    fn_index = np.where((target == target_class) & (results != target_class))[0]

    return tp_index, tn_index, fp_index, fn_index

def flatten(data):
    """
    Recursively flattens a nested list structure of numbers into a single list.

    Args:
        data: A number (int or float) or a nested list of numbers. The data to be flattened.

    Returns:
        A list of numbers, where all nested structures in the input have been
        flattened into a single list.
    """
    if isinstance(data, (int, float, np.float64)):  # base case for numbers
        return [data]
    else:
        flat_list = []
        for item in data:
            flat_list.extend(flatten(item))  # recursive call to flatten
        return flat_list


def get_global_min_max(data):
    """
    Finds the global minimum and maximum values in a nested list structure.

    This function flattens the input data into a single list and then
    determines the minimum and maximum values.

    Args:
        data: A nested list of integers. The structure can be of any depth.

    Returns:
        A tuple (min_value, max_value) where min_value is the minimum value
        in the data, and max_value is the maximum value.
    """
    flat_list = flatten(data)
    return min(flat_list), max(flat_list)


def get_edge_width(indices, clust):
    """ Calculate edge width for each cluster based on the number of indices in the cluster.

    Args:
        indices: 1-d array
            Array of indices for the specific class.
        clust: sequence of vectors
            A sequence of vectors, each containing the indices of elements in a cluster.

    Returns:
        lwidth: 1-d array
            Array of edge widths for each cluster.
    """

    lwidth = np.zeros(len(clust))

    for i in range(len(clust)):
        if len(clust[i]) != 0:
            if len(np.intersect1d(clust[i], indices)) > 0:
                lwidth[i] = 20. * len(np.intersect1d(clust[i], indices)) / len(clust[i])
            else:
                lwidth[i] = None
        else:
            lwidth[i] = None

    return lwidth


def get_edge_color(clust, *args):
    """ Calculate edge color for each cluster based on the number of indices in the cluster.

    Args:
        clust: sequence of vectors
            A sequence of vectors, each containing the indices of elements in a cluster.

        *args: 1-d array
            A list of indices where the specific class is present.
    """

    # unpack the args
    numst = list(args)

    # Initialize the edge color array
    edge_color = np.zeros(len(clust))

    # Detect the intersection of the cluster and each list of indices (class),
    # and get the majority class in the cluster.
    # Append the majority class to the edge color array.
    for i in range(len(clust)):
        if len(clust[i]) != 0:
            intersection = [len(np.intersect1d(clust[i], numst[j])) for j in range(len(numst))]
            edge_color[i] = np.argmax(intersection)
        else:
            edge_color[i] = None

    return edge_color


# Helper functions to create button objects in the interactive plot
def create_buttons(fig, button_types):
    sidebar_width = 0.2
    button_config = calculate_button_positions(len(button_types), sidebar_width)

    buttons = {}
    for i, button_type in enumerate(button_types):
        button_ax = fig.add_axes(button_config[i])
        buttons[button_type] = Button(button_ax, button_type.capitalize(), hovercolor='0.975')

    return buttons


def calculate_button_positions(num_buttons, sidebar_width):
    # Calculate button positions and sizes
    button_ratio = 16 / 9
    single_button_width = sidebar_width * 0.8
    single_button_height = single_button_width / button_ratio
    margin = 0.05
    total_buttons_height = num_buttons * single_button_height + (num_buttons - 1) * margin

    button_config = []
    for i in range(num_buttons):
        y_pos = (1 - total_buttons_height) / 2 + (num_buttons - 1 - i) * (single_button_height + margin)
        x_centered = 0.8 + (0.2 - single_button_width) / 2
        button_config.append([x_centered, y_pos, single_button_width, single_button_height])

    return button_config


