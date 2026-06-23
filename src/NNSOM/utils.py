from scipy.spatial.distance import cdist
from scipy import sparse
import numpy as np
import networkx as nx
from matplotlib.widgets import Button


def preminmax(p):
    """Normalize input data row-wise to the range [-1, 1].

    If a row's minimum equals its maximum, that row is mapped to the
    range ``[-1, 1]`` by treating the single value as the midpoint
    (min shifted by -1, max shifted by +1).

    Parameters
    ----------
    p : np.ndarray, shape (R, Q)
        Input matrix with R features and Q samples.

    Returns
    -------
    pn : np.ndarray, shape (R, Q)
        Normalized input matrix with values in [-1, 1].
    minp : np.ndarray, shape (R,)
        Per-row minimum values of the original data.
    maxp : np.ndarray, shape (R,)
        Per-row maximum values of the original data.
    """
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
    """Compute the 2-D hexagonal grid positions of SOM neurons.

    Lays out neurons in a hexagonal topology where odd columns are
    offset by half a unit in the x-direction.

    Parameters
    ----------
    dim : array-like of int, length 2
        Grid dimensions ``(rows, cols)`` of the SOM.

    Returns
    -------
    position : np.ndarray, shape (2, rows*cols)
        X/Y coordinates of each neuron.  Row 0 is the x-axis,
        row 1 is the y-axis.
    """
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
    """Convert Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x : float or np.ndarray
        X-coordinate(s).
    y : float or np.ndarray
        Y-coordinate(s).

    Returns
    -------
    theta : float or np.ndarray
        Angle in radians, measured counter-clockwise from the positive
        x-axis (output of ``np.arctan2``).
    rho : float or np.ndarray
        Radial distance(s) from the origin.
    """
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    """Convert polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    theta : float or np.ndarray
        Angle(s) in radians.
    rho : float or np.ndarray
        Radial distance(s) from the origin.

    Returns
    -------
    x : float or np.ndarray
        X-coordinate(s).
    y : float or np.ndarray
        Y-coordinate(s).
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_xy(x1, y1, angle):
    """Rotate 2-D coordinates by a given angle.

    Parameters
    ----------
    x1 : float or np.ndarray
        Original X-coordinate(s).
    y1 : float or np.ndarray
        Original Y-coordinate(s).
    angle : float
        Rotation angle in radians (counter-clockwise).

    Returns
    -------
    x2 : float or np.ndarray
        Rotated X-coordinate(s).
    y2 : float or np.ndarray
        Rotated Y-coordinate(s).
    """
    [a, r] = cart2pol(x1, y1)
    a = a + angle
    x2, y2 = pol2cart(a, r)
    return x2, y2


def normalize_position(position):
    """Normalize neuron positions to the range [-1, 1] along each axis.

    If the minimum and maximum positions along an axis are equal, that
    axis is left unchanged (division by 1 instead of 0).

    Parameters
    ----------
    position : np.ndarray, shape (2, N)
        Raw neuron positions; row 0 is x, row 1 is y.

    Returns
    -------
    posit : np.ndarray, shape (2, N)
        Normalized positions with values in [-1, 1].
    """
    shap = position.shape
    numPos = shap[1]
    minPos = np.ndarray.min(position, axis=1)
    maxPos = np.ndarray.max(position, axis=1)
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
    """Map normalized neuron positions into data space using a basis.

    Applies the affine transform ``positionMean + positionBasis @ position``
    to project unit-square positions into the principal-component subspace
    of the training data.

    Parameters
    ----------
    position : np.ndarray, shape (2, N)
        Normalized neuron positions.
    positionMean : np.ndarray, shape (R, 1)
        Mean of the training data (column vector), used as the translation.
    positionBasis : np.ndarray, shape (R, 2)
        Basis matrix (two principal components) used for the linear map.

    Returns
    -------
    position1 : np.ndarray, shape (R, N)
        Neuron positions expressed in data space.
    """
    shappos = position.shape
    numPos = shappos[1]
    position1 = np.repeat(positionMean, numPos, axis=1) + np.matmul(positionBasis, position)
    return position1


def distances(pos):
    """Compute shortest-path distances between all pairs of SOM neurons.

    Builds an adjacency graph where two neurons are connected if their
    Euclidean distance is at most 1 (i.e., immediate hex neighbours),
    then uses Floyd-Warshall to compute all-pairs shortest paths.

    Parameters
    ----------
    pos : np.ndarray, shape (2, N)
        Neuron positions as returned by :func:`calculate_positions`.

    Returns
    -------
    dist : np.ndarray, shape (N, N)
        Matrix of topological (graph-hop) distances between neurons.
    """
    posT = np.transpose(pos)
    dist = cdist(posT, posT, 'euclidean')

    link = dist <= 1.00001
    link = sparse.csr_matrix(1.0*link)

    g = nx.DiGraph(link)
    dist = nx.floyd_warshall_numpy(g)

    return dist


def get_hexagon_shape():
    """Return the vertex offsets for drawing a unit hexagon.

    The hexagon is centred at the origin. Add the neuron's ``(x, y)``
    position to these offsets to draw its hexagonal patch.

    Returns
    -------
    shapex : np.ndarray, shape (6,)
        X-offsets of the six hexagon vertices.
    shapey : np.ndarray, shape (6,)
        Y-offsets of the six hexagon vertices.
    """
    shapex = np.array([-1, 0, 1, 1, 0, -1]) * 0.5
    shapey = np.array([1, 2, 1, -1, -2, -1]) * np.sqrt(0.75) / 3
    return shapex, shapey


def get_edge_shape():
    """Return the vertex offsets for drawing the elongated diamond between two neurons.

    Used to render the edge between adjacent clusters in topology plots.

    Returns
    -------
    edgex : np.ndarray, shape (4,)
        X-offsets of the four diamond vertices.
    edgey : np.ndarray, shape (4,)
        Y-offsets of the four diamond vertices.
    """
    edgex = np.array([-1, 0, 1, 0]) * 0.5
    edgey = np.array([0, 1, 0, -1]) * np.sqrt(0.75) / 3

    return edgex, edgey


# Helper Functions to extract information from the input data for passing to the plot
def get_cluster_data(data, clust):
    """For each cluster, extract the corresponding data points.

    Parameters
    ----------
    data : np.ndarray
        The dataset from which to extract cluster data.
    clust : list of array-like
        A list where each element holds the indices of data points
        belonging to that cluster.

    Returns
    -------
    cluster_data_list : list of np.ndarray
        A list where each element is an array of data points for one
        cluster.  Empty clusters produce an empty array.
    """
    cluster_data_list = []
    for cluster_indices in clust:
        if len(cluster_indices) > 0:
            cluster_indices = np.array(cluster_indices, dtype=int)
            cluster_data = data[cluster_indices]
            cluster_data_list.append(cluster_data)
        else:
            cluster_data_list.append(np.array([]))

    return cluster_data_list


def get_cluster_array(feature, clust):
    """Return an object array whose elements are per-cluster feature values.

    Parameters
    ----------
    feature : array-like, shape (N,)
        Feature values for all data points.
    clust : list of array-like
        A list of cluster index arrays sorted by distance to the winning
        neuron.

    Returns
    -------
    cluster_array : np.ndarray of object, shape (numNeurons,)
        Each element is a 1-D array of feature values for that cluster.
        Empty clusters store an empty array.
    """
    cluster_array = np.empty(len(clust), dtype=object)

    for i, cluster_indices in enumerate(clust):
        if len(cluster_indices) > 0:
            cluster_array[i] = feature[cluster_indices]
        else:
            cluster_array[i] = np.array([])

    return cluster_array


def get_cluster_avg(feature, clust):
    """Compute the mean feature value for each cluster.

    Parameters
    ----------
    feature : array-like, shape (N,)
        Feature values for all data points.
    clust : list of array-like
        A list of cluster index arrays sorted by distance to the winning
        neuron.

    Returns
    -------
    cluster_avg : np.ndarray, shape (numNeurons,)
        Mean feature value per cluster.  Zero for empty clusters.
    """
    cluster_array = get_cluster_array(feature, clust)
    cluster_avg = np.zeros(len(cluster_array))
    for i in range(len(cluster_array)):
        if len(cluster_array[i]) > 0:
            cluster_avg[i] = np.mean(cluster_array[i])

    return cluster_avg


def closest_class_cluster(cat_feature, clust):
    """Return the class of the closest data point in each cluster.

    Parameters
    ----------
    cat_feature : array-like, shape (N,)
        Categorical feature values for all data points.
    clust : list of array-like
        A list of cluster index arrays sorted by distance to the winning
        neuron (closest first).

    Returns
    -------
    closest_class : np.ndarray, shape (numNeurons,)
        Class label of the nearest data point in each cluster.
        ``None`` for empty clusters.
    """
    closest_class = np.zeros(len(clust))
    for i in range(len(clust)):
        cluster_indices = clust[i]
        if len(cluster_indices) > 0:
            closest_class[i] = cat_feature[cluster_indices[0]]
        else:
            closest_class[i] = None

    return closest_class


def majority_class_cluster(cat_feature, clust):
    """Return the majority class label for each cluster.

    Parameters
    ----------
    cat_feature : array-like of int, shape (N,)
        Integer-encoded categorical feature for all data points.
    clust : list of array-like
        A list of cluster index arrays sorted by distance to the winning
        neuron.

    Returns
    -------
    majority_class : np.ndarray, shape (numNeurons,)
        Most frequent class label in each cluster.
        ``None`` for empty clusters.
    """
    majority_class = np.zeros(len(clust))
    for i in range(len(clust)):
        cluster_indices = clust[i]
        if len(cluster_indices) > 0:
            majority_class[i] = np.argmax(np.bincount(cat_feature[cluster_indices]))
        else:
            majority_class[i] = None

    return majority_class


def get_perc_cluster(cat_feature, target, clust):
    """Return the percentage of a target class in each cluster.

    Parameters
    ----------
    cat_feature : array-like, shape (N,)
        Categorical feature values for all data points.
    target : int or str
        The class label whose proportion is computed.
    clust : list of array-like
        A list of cluster index arrays sorted by distance to the winning
        neuron.

    Returns
    -------
    cluster_array : np.ndarray, shape (numNeurons,)
        Percentage (0–100) of ``target`` class in each cluster.
        Zero for empty clusters.
    """
    cluster_array = np.zeros(len(clust))
    for i in range(len(clust)):
        cluster_indices = clust[i]
        if len(cluster_indices) > 0:
            cluster_array[i] = np.sum(cat_feature[cluster_indices] == target) / len(cluster_indices)
        else:
            cluster_array[i] = 0

    return cluster_array * 100


def count_classes_in_cluster(cat_feature, clust):
    """Count occurrences of each class in every cluster.

    Parameters
    ----------
    cat_feature : array-like, shape (N,)
        Categorical feature values for all data points.
    clust : list of array-like
        A list of cluster index arrays.

    Returns
    -------
    cluster_counts : np.ndarray of int, shape (numNeurons, numClasses)
        Entry ``[i, j]`` is the count of class ``j`` in cluster ``i``.
    """
    unique_classes, _ = np.unique(cat_feature, return_counts=True)
    num_classes = len(unique_classes)

    cluster_counts = np.zeros((len(clust), num_classes), dtype=int)

    for i, indices in enumerate(clust):
        if len(indices) > 0:
            cluster_counts[i] = [np.sum(cat_feature[indices] == cls) for cls in unique_classes]
        else:
            cluster_counts[i] = np.zeros(num_classes, dtype=int)

    return cluster_counts


def cal_class_cluster_intersect(clust, *args):
    """Calculate the intersection sizes of each class with each neuron cluster.

    Parameters
    ----------
    clust : list of list of int
        A collection of neuron clusters; each element is a list of
        data-point indices assigned to that neuron.
    *args : array-like of int
        One array per class, each holding the data-point indices that
        belong to that class.

    Returns
    -------
    cluster_sizes_matrix : np.ndarray, shape (numNeurons, numClasses)
        Entry ``[i, j]`` is the number of data points in class ``j``
        that are also in neuron cluster ``i``.

    Examples
    --------
    >>> clust = [[4, 5, 9], [1, 7], [2, 10, 11], [3, 6, 8]]
    >>> ind1 = np.array([1, 2, 3])
    >>> ind2 = np.array([4, 5, 6])
    >>> ind3 = np.array([7, 8, 9])
    >>> ind4 = np.array([10, 11, 12])
    >>> cal_class_cluster_intersect(clust, ind1, ind2, ind3, ind4)
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


# Helper function to extract information from the post-training model for passing to the plot
def get_ind_misclassified(target, prediction):
    """Return indices of misclassified samples.

    Parameters
    ----------
    target : array-like, shape (N,)
        True class labels.
    prediction : array-like, shape (N,)
        Predicted class labels.

    Returns
    -------
    misclassified_indices : np.ndarray of int
        Indices where ``target != prediction``.
    """
    misclassified_indices = np.where(target != prediction)[0]

    return misclassified_indices


def get_perc_misclassified(target, prediction, clust):
    """Compute the percentage of misclassified samples in each cluster.

    Parameters
    ----------
    target : array-like, shape (N,)
        True class labels.
    prediction : array-like, shape (N,)
        Predicted class labels.
    clust : list of array-like
        A list of cluster index arrays.

    Returns
    -------
    proportion_misclassified : np.ndarray, shape (numNeurons,)
        Percentage (0–100) of misclassified samples per cluster.
    """
    misclassified_indices = get_ind_misclassified(target, prediction)

    proportion_misclassified = np.zeros(len(clust))

    for i, cluster_indices in enumerate(clust):
        if len(cluster_indices) > 0:
            misclassified_count = np.intersect1d(cluster_indices, misclassified_indices).size
            proportion_misclassified[i] = (misclassified_count / len(cluster_indices)) * 100

    return proportion_misclassified


def get_conf_indices(target, results, target_class):
    """Return TP, TN, FP, FN indices for a given class in a binary sense.

    Parameters
    ----------
    target : array-like, shape (N,)
        True class labels.
    results : array-like, shape (N,)
        Predicted class labels.
    target_class : int
        The class label treated as the positive class.

    Returns
    -------
    tp_index : np.ndarray of int
        Indices of True Positives.
    tn_index : np.ndarray of int
        Indices of True Negatives.
    fp_index : np.ndarray of int
        Indices of False Positives.
    fn_index : np.ndarray of int
        Indices of False Negatives.
    """
    tp_index = np.where((target == target_class) & (results == target_class))[0]
    tn_index = np.where((target != target_class) & (results != target_class))[0]
    fp_index = np.where((target != target_class) & (results == target_class))[0]
    fn_index = np.where((target == target_class) & (results != target_class))[0]

    return tp_index, tn_index, fp_index, fn_index


def get_dominant_class_error_types(dominant_classes, error_types):
    """Map each cluster's dominant class to its majority error type.

    Parameters
    ----------
    dominant_classes : array-like, shape (numNeurons,)
        Dominant class label for each cluster.  May contain ``NaN``
        for empty clusters.
    error_types : list of array-like, each shape (numNeurons,)
        One array per class; ``error_types[k][i]`` is the majority
        error type in cluster ``i`` for class ``k``.

    Returns
    -------
    output_error_types : np.ndarray, shape (numNeurons,)
        Majority error type per cluster, selected according to the
        dominant class.  ``NaN`` where the dominant class is ``NaN``.

    Raises
    ------
    ValueError
        If fewer ``error_types`` arrays are provided than required by
        the maximum dominant class index, or if a dominant class index
        is out of bounds.
    """
    if len(error_types) < np.max([dc for dc in dominant_classes if not np.isnan(dc)]) + 1:
        raise ValueError("Not enough error type arrays provided for all classes.")

    output_error_types = np.full(len(dominant_classes), np.nan)

    for idx, dominant_class in enumerate(dominant_classes):
        if not np.isnan(dominant_class):
            if dominant_class < len(error_types):
                output_error_types[idx] = error_types[int(dominant_class)][idx]
            else:
                raise ValueError(f"Class {dominant_class} is out of the provided error type array bounds.")
        else:
            output_error_types[idx] = np.nan

    return output_error_types


def flatten(data):
    """Recursively flatten a nested list of numbers into a single list.

    Parameters
    ----------
    data : int, float, or nested list
        A number or arbitrarily nested list of numbers.

    Returns
    -------
    flat_list : list of float
        All numbers from ``data`` in depth-first order.
    """
    if isinstance(data, (int, float, np.float64)):
        return [data]
    else:
        flat_list = []
        for item in data:
            flat_list.extend(flatten(item))
        return flat_list


def get_global_min_max(data):
    """Find the global minimum and maximum in a nested list.

    Parameters
    ----------
    data : nested list of numbers
        Input data of arbitrary nesting depth.

    Returns
    -------
    min_value : float
        Minimum value across all elements.
    max_value : float
        Maximum value across all elements.
    """
    flat_list = flatten(data)
    return min(flat_list), max(flat_list)


def get_edge_widths(indices, clust):
    """Calculate edge line widths per cluster based on class membership fraction.

    Width is proportional to the fraction of cluster members that belong
    to the given class, scaled to a maximum of 20.

    Parameters
    ----------
    indices : array-like of int
        Indices of data points belonging to the target class.
    clust : list of array-like
        A list of cluster index arrays.

    Returns
    -------
    lwidth : np.ndarray, shape (numNeurons,)
        Line width for each cluster edge.  ``None`` for empty clusters
        or clusters with no class members.
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


def get_color_labels(clust, *listOfIndices):
    """Generate a colour label per cluster based on majority class membership.

    When a single class index array is provided, clusters where that
    class is the majority are labelled ``1``; others are labelled ``0``.
    When multiple arrays are provided, each cluster is labelled with the
    index of the class that has the most members in it.

    Parameters
    ----------
    clust : list of array-like
        A list of cluster index arrays.
    *listOfIndices : array-like of int
        One array per class holding data-point indices for that class.
        At least one array is required.

    Returns
    -------
    color_labels : np.ndarray, shape (numNeurons,)
        Integer colour label for each cluster.  ``None`` for empty
        clusters.

    Raises
    ------
    ValueError
        If no class index arrays are provided, or if any argument is
        not a list or ``np.ndarray``.
    """
    if len(listOfIndices) == 0:
        raise ValueError('At least one class indices must be provided.')

    numst = []
    for arg in listOfIndices:
        if not isinstance(arg, (list, np.ndarray)):
            raise ValueError('The arguments must be a list or numpy array.')
        else:
            numst.append(arg)

    color_labels = np.zeros(len(clust))

    if len(numst) == 1:
        indices = numst[0]
        for i in range(len(clust)):
            if len(clust[i]) != 0:
                num_class = len(np.intersect1d(clust[i], indices))
                if num_class > len(clust[i]) / 2:
                    color_labels[i] = 1
                else:
                    color_labels[i] = 0
            else:
                color_labels[i] = None

    else:
        for i in range(len(clust)):
            if len(clust[i]) != 0:
                intersection = [len(np.intersect1d(clust[i], numst[j])) for j in range(len(numst))]
                color_labels[i] = np.argmax(intersection)
            else:
                color_labels[i] = None

    return color_labels


# Helper functions to create button objects in the interactive plot
def create_buttons(fig, button_types):
    """Create labelled matplotlib Button widgets in a sidebar.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to which buttons are added.
    button_types : list of str
        Labels for the buttons to create.

    Returns
    -------
    buttons : dict of {str: matplotlib.widgets.Button}
        Mapping from button label to its ``Button`` widget.
    """
    sidebar_width = 0.2
    button_config = calculate_button_positions(len(button_types), sidebar_width)

    buttons = {}
    for i, button_type in enumerate(button_types):
        button_ax = fig.add_axes(button_config[i])
        buttons[button_type] = Button(button_ax, button_type.capitalize(), hovercolor='0.975')

    return buttons


def calculate_button_positions(num_buttons, sidebar_width):
    """Calculate evenly-spaced button positions within a figure sidebar.

    Parameters
    ----------
    num_buttons : int
        Number of buttons to lay out.
    sidebar_width : float
        Fractional width of the sidebar relative to the figure width
        (e.g. ``0.2`` for 20 %).

    Returns
    -------
    button_config : list of [float, float, float, float]
        List of ``[left, bottom, width, height]`` axes rectangles in
        figure coordinates, one per button.
    """
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
