try:
    import cupy as cp
    print("You are using GPU acceleration with Cupy")
except ImportError:
    print("CuPy is not available. For CPU-based operations, you can use the NumPy version of this SOM.")
    print("Please consider installing the 'NNSOM' package, and use 'from NNSOM.som import SOM' for a NumPy-based SOM implementation.")
    raise SystemExit

from .utils import calculate_positions, distances

import numpy as np
from datetime import datetime
import pickle
import warnings


class SOMGpu:
    """
    Represents a Self-Organizing Map (SOM) using GPU acceleration with CuPy.

    A Self-Organizing Map (SOM) is an artificial neural network used for unsupervised learning,
    which projects high-dimensional data into a lower-dimensional (typically two-dimensional) space.
    It is trained using a competitive learning approach to produce a discretized representation
    of the input space of training samples.

    Attributes:
        dimensions (tuple, list, np.ndarray): Dimensions of the SOM grid, defining the layout and number of neurons.
        numNeurons (int): Total number of neurons, computed as the product of the grid dimensions.
        pos (np.ndarray): Positions of neurons within the grid.
        neuron_dist (np.ndarray): Precomputed Euclidean distances between neurons in the grid.
        w (np.ndarray): Weight matrix representing the feature vectors of the neurons.
        sim_flag (bool): Indicates if the SOM has been simulated/trained.
        output (np.ndarray): Output from the latest simulation.
        norm_func (callable): Function used to normalize input data.
        sub_som (dict): Optional sub-clustering using additional SOMs at neuron positions.

    Methods:
        __init__(self, dimensions): Initializes the SOM with the specified dimensions.
        init_w(self, x, norm_func=None): Initializes the weights using PCA on input data `x`.
        sim_som(self, x): Simulates SOM processing for input `x`, identifying activated neurons.
        train(self, x, init_neighborhood=3, epochs=200, steps=100, norm_func=None):
            Trains the SOM using batch SOM algorithm on input data `x`.
        quantization_error(self, dist): Calculates the quantization error of the model.
        topological_error(self, data): Calculates the topological error of the model.
        distortion_error(self, data): Calculates the distortion error of the model.
        save_pickle(self, filename, path, data_format='pkl'): Saves the SOM object to a file in pickle format.
        load_pickle(self, filename, path, data_format='pkl'): Loads the SOM object from a file in pickle format.
        _normalize_position(self, position): Helper method to normalize neuron positions.
        _spread_positions(self, position, positionMean, positionBasis): Helper method to adjust neuron positions.
        _euclidean_distance(self, XA, XB): Computes Euclidean distances between two sets of vectors.
        _to_categorical(self, x, num_classes=None): Converts class vector to binary class matrix.

    Raises:
        ImportError: If CuPy is not available, suggests using the NNSOM package for a NumPy-based implementation.

    Example:
        >>> dimensions = (10, 10)
        >>> som = SOMGpu(dimensions)
        >>> data = np.random.rand(100, 10)
        >>> som.init_w(data, norm_func=None)
        >>> som.train(data, norm_func=None)
        >>> output = som.sim_som(data)
    """
    def __init__(self, dimensions):
        """
        Initializes the SOM with the specified dimensions and calculates the positions and distances between neurons in the SOM grid.

        Parameters
        ----------
        dimensions : tuple, list, or np.ndarray
                    The dimensions (shape) of the SOM grid.
        """

        self.dimensions = dimensions  # Processing as numpy
        self.numNeurons = np.prod(dimensions) # Processing as numpy
        # Calculate positions of neurons
        self.pos = calculate_positions(dimensions)
        # Calculate distances between neurons
        self.neuron_dist = distances(self.pos)
        # Initialize the weight matrix with empty list
        self.w = []
        # Set simulation flag to True,  needs to do simulation
        self.sim_flag = True
        # Initialize the output of simulation
        self.output = None
        # Initialize a normalize() function
        self.norm_func = None

        # Initialize the dictionary of sub-cluster. {neuron_number(int): sub-clustering SOM obj}
        self.sub_som = {}

    def init_w(self, x, norm_func=None):
        """
        Initializes the weights of the SOM using principal components analysis (PCA) on the input data x.

        Parameters
        ----------
        x : np.ndarray
            The input data used for weight initialization.
        """
        # Initialize SOM weights using principal components

        # Print Beginning time for initialization
        print('Beginning Initialization')
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)

        # Normalize the input data
        x = self.normalize(x, norm_func)
        x = cp.asarray(x)

        sz = x.shape

        posMean = cp.mean(x, axis=1)
        posMean = cp.expand_dims(posMean, axis=1)

        xc = x - posMean

        components, gains, encodedInputsT = cp.linalg.svd(xc)

        encodedInputsT = cp.transpose(encodedInputsT)

        basis = components * gains
        stdev = cp.std(encodedInputsT, axis=0)
        stdev = stdev[:len(basis)]
        posBasis = 2.5 * basis * stdev

        numNeurons = self.numNeurons
        numDimensions = len(self.dimensions)
        dimensions = self.dimensions
        sampleSize = sz[1]
        inputSize = sz[0]
        dimOrder = cp.argsort(cp.asarray(dimensions))

        restoreOrder = cp.concatenate((cp.sort(dimOrder), cp.arange(numDimensions, cp.minimum(inputSize, sampleSize))))

        if numDimensions > inputSize:
            posBasis = cp.concatenate((posBasis, cp.random.rand(inputSize, inputSize) * 0.001))

        posBasis = posBasis[restoreOrder]

        pos1 = self.pos
        pos1 = cp.asarray(pos1)

        if sampleSize < inputSize:
            posBasis = cp.concatenate((posBasis, cp.zeros((inputSize, inputSize - sampleSize))))

        if inputSize > numDimensions:
            pos1 = cp.concatenate((pos1, cp.zeros((inputSize - numDimensions, numNeurons))), axis=0)

        pos2 = self._normalize_position(pos1)

        pos3 = self._spread_positions(pos2, posMean, posBasis)

        self.w = cp.asnumpy(cp.transpose(pos3))

        # Print Ending time for initialization
        print('Ending Initialization')
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)

    def sim_som(self, x):
        """
        Simulates the SOM with x as the input, determining which neurons are activated by the input vectors.

        Parameters
        ----------
        x : np.ndarray
            The input data to simulate the SOM with.

        Returns
        -------
        np.ndarray
            The simulated output of the SOM.
        """
        # Simulate the SOM, with x as the input
        # Transform np.ndarray into cp.ndarray
        w = cp.asarray(self.w)
        x = cp.asarray(x)

        # Compute the negative distance from the inputs to each center
        n = -self._euclidean_distance(w, cp.transpose(x))

        # Find out which center was closest to the input
        maxRows = cp.argmax(n, axis=0)
        a = self._to_categorical(maxRows, num_classes=n.shape[0]) #made correction-added number of class
        a = cp.asnumpy(a)

        return np.transpose(a)

    def train(self, x, init_neighborhood=3, epochs=200, steps=100, norm_func=None):
        """
        Trains the SOM using the batch SOM algorithm on the input data x.

        Parameters
        ----------
        x : np.ndarray
            The input data to train the SOM with.
        init_neighborhood : int, optional
            The initial neighborhood size.
        epochs : int, optional
            The number of epochs to train for.
        steps : int, optional
            The number of steps for training.

        Returns
        -------
        None
        """
        # Normalize the input data
        x = self.normalize(x, norm_func)
        x = cp.asarray(x)

        # Train the SOM using the batch SOM algorithm

        w = self.w
        w = cp.asarray(w)
        shapw = w.shape
        S = shapw[0]
        shapx = x.shape
        Q = shapx[1]

        step = 0

        print('Beginning Training')
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        # Train the network
        for i in range(epochs):

            # network output
            a = self.sim_som(x)
            a = cp.asarray(a)

            # neighborhood distance
            nd = 1 + (init_neighborhood-1) * (1 - step/steps)
            neighborhood = self.neuron_dist <= nd
            neighborhood = cp.asarray(neighborhood)
            #print(nd)
            # remove some outputs at random
            a = a * (cp.random.rand(S, Q) < 0.90)
            a2 = cp.matmul(neighborhood, a) + a

            # find how many times each neuron won
            # (The winning neuron is the one that exhibits the smallest distance or similarity to the input data)
            suma2 = cp.sum(a2, axis=1)
            loserIndex = cp.squeeze(cp.asarray(suma2 == 0))
            suma2[loserIndex] = 1
            suma2 = cp.expand_dims(suma2,axis = 1)
            a3 = a2 / cp.repeat(suma2, Q, axis=1)

            neww = cp.matmul(a3, cp.transpose(x))

            dw = neww - w

            dw[loserIndex] = 0

            w = w + cp.array(dw)

            step = step + 1

            if step % 50 == 0:
                print(step)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("Current Time =", current_time)

        self.w = cp.asnumpy(w)
        self.outputs = self.sim_som(x)
        self.sim_flag = False

        print('Ending Training')
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)

    def cluster_data(self, x):
        """
        Cluster the input data based on the trained SOM reference vectors.

        Parameters
        ----------
        x : ndarray (normalized)
            The input data to be clustered.

        Returns
        -------
        clusters : list of lists
            A list containing sub-lists, where each sublist represents a cluster.
            The indices of the input data points belonging to the same cluster
            are stored in the corresponding sublist, sorted by their proximity
            to the cluster center.

        cluster_distances : list of lists
            A list containing sub-lists, where each sublist represents the distances
            of the input data points to the corresponding cluster center, sorted in
            the same order as the indices in the `clusters` list.

        max_cluster_distances : ndarray
            A list containing the maximum distance between each cluster center
            and the data points belonging to that cluster.

        cluster_sizes : ndarray
            A list containing the number of data points in each cluster.

        Raises
        -------
        ValueError
            If the SOM has not been trained.

        ValueError
            If the number of features in the input data and the SOM weights do not match.
        """

        if self.sim_flag:
            raise ValueError("SOM has not been trained.")

        if x.shape[1] != self.w.shape[1]:
            raise ValueError('The number of features in the input data and the SOM weights do not match.')

        # Normalize the input data
        x = self.normalize(x, self.norm_func)
        x = cp.asarray(x)

        w = self.w
        w = cp.asarray(w)
        shapw = w.shape
        S = shapw[0]

        x_w_dist = self._euclidean_distance(w, cp.transpose(x))
        ind1 = cp.argmin(x_w_dist, axis=0)

        clusters = []  # a cluster array of indices sorted by distances
        cluster_distances = []  # a cluster array of distances sorted by distances
        max_cluster_distances = cp.zeros(S)  # a list of maimum distance to any input in the cluster from cluster center
        cluster_sizes = []  # cluster array sizes

        for i in range(S):
            # Find which inputs are closest to each weight (in cluster i)
            tempclust = cp.where(ind1 == i)[0]

            # Save distance of each input in the cluster to cluster center (weight)
            tempdist = x_w_dist[i, tempclust]
            indsort = cp.argsort(tempdist)
            tempclust = tempclust[indsort]  # Sort indices
            tempdist = tempdist[indsort]

            # Add to distance array sorted distances
            cluster_distances.append(tempdist)

            # Add to Cluster array sorted indices
            clusters.append(tempclust)

            # Cluster size
            num = len(tempclust)
            cluster_sizes.append(num)

            # Save the maximum distance to any input in the cluster from cluster center
            if num > 0:
                max_cluster_distances[i] = tempdist[-1]

        # Convert clusters to a list of lists
        clusters = [clust.get().tolist() for clust in clusters]
        # Convert cluster_distances to a list of lists
        cluster_distances = [dist.get().tolist() for dist in cluster_distances]
        # Convert max_cluster_distances to a NumPy array
        max_cluster_distances = max_cluster_distances.get()
        # Convert cluster_sizes to a NumPy array
        cluster_sizes = np.array(cluster_sizes)

        return clusters, cluster_distances, max_cluster_distances, cluster_sizes

    def normalize(self, x, norm_func=None):
        """
        Normalize the input data using a custom function.

        Parameters
        ----------
        x: array-like
            The input data to be normalized.
        norm_func: callable, optional
            A custom normalization or standardization function to be applied to the input data.
            If provided, it should take the input data as its argument and return the preprocessed data.
            Default is None, in which case the input data is returned as-is.

        Returns
        -------
        x_preprocessed: array-like
            The preprocessed input data.

        Raises
        ------
        Warning
            If `norm_func` is None, a warning is raised to indicate the potential inefficiency in SOM training.

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.feature_extraction.text import TfidfVectorizer
        >>> from sklearn.preprocessing import StandardScaler

        >>> # Case 1: Tabular data (without normalization)
        >>> iris = load_iris()
        >>> X = iris.data
        >>> som = SOM(dimensions=(5, 5))
        >>> X_norm = som.normalize(X)
        >>> print(np.allclose(np.transpose(X_norm), X))
        True

        >>> # Case 2: Image data (using custom normalization)
        >>> image_data = np.random.randint(0, 256, size=(28, 28))
        >>> som = SOM(dimensions=(10, 10))
        >>> custom_norm_func = lambda x: x / 255  # Custom normalization function
        >>> image_data_norm = som.normalize(image_data, norm_func=custom_norm_func)
        >>> print(image_data_norm.min(), image_data_norm.max())
        0.0 1.0

        >>> # Case 3: Text data (without normalization)
        >>> text_data = ["This is a sample text.", "Another example sentence."]
        >>> vectorizer = TfidfVectorizer()
        >>> tfidf_matrix = vectorizer.fit_transform(text_data)
        >>> som = SOM(dimensions=(8, 8))
        >>> text_data_norm = som.normalize(tfidf_matrix.toarray())
        >>> print(np.allclose(np.transpose(text_data_norm), tfidf_matrix.toarray()))
        True
        """

        if norm_func is not None:
            x_norm = norm_func(x)  # Use the provided custom normalization function
            self.norm_func = norm_func
        else:
            warnings.warn(
                "Without normalization function: SOM training may be inefficient if you are not normalized.",
                UserWarning, stacklevel=2)
            x_norm = x  # Return the input data as-is

        return np.transpose(x_norm)

    def quantization_error(self, dist):
        """
        Calculate quantization error
        """
        quant_err = np.array([0 if len(item) == 0 else np.mean(item) for item in dist]).mean()

        return quant_err

    def topological_error(self, x):
        """
        Calculate topological error
        """
        w = self.w
        ndist = self.neuron_dist

        # Normalize Input
        x = self.normalize(x, self.norm_func)

        # Calculate the distance between item vs. cluster center
        x_w_dist = self._euclidean_distance(w, np.transpose(x))

        sort_dist = np.argsort(x_w_dist, axis=0)
        top_dist = [ndist[sort_dist[0, ii], sort_dist[1, ii]] for ii in range(sort_dist.shape[1])]
        neighbors = np.where(np.array(top_dist) > 1.1)
        top_error_1st = 100 * len(neighbors[0]) / x_w_dist.shape[1]
        neighbors = np.where(np.array(top_dist) > 2.1)
        top_error_1st_and_2nd = 100 * len(neighbors[0]) / x_w_dist.shape[1]

        return top_error_1st, top_error_1st_and_2nd

    def distortion_error(self, x):
        """
        Calculate distortion
        """
        # Normalize input data
        x = self.normalize(x, self.norm_func)

        shapx = x.shape
        Q = shapx[1]  # Number of samples

        ww = self.w
        ndist = self.neuron_dist
        x_w_dist = self._euclidean_distance(ww, np.transpose(x))
        ind1 = np.argmin(x_w_dist, axis=0)

        dd = [1, 2, 3]  # neighborhood distances
        wwdist = self._euclidean_distance(ww, ww)
        sst = ndist[:, ind1]

        for d in dd:
            factor1 = 2 * d * d
            factor2 = Q * d * np.sqrt(2 * np.pi)
            temp = np.exp(-np.multiply(sst, sst) / factor1)
            distortion = np.sum(np.multiply(temp, x_w_dist)) / factor2
            print('Distortion (d=' + str(d) + ') = ' + str(distortion))

    def save_pickle(self, filename, path, data_format='pkl'):
        """ Save the SOM object to a file using pickle.

        Parameters
        ----------
        filename : str
            The name of the file to save the SOM object to.

        path : str
            The path to the file to save the SOM object to.

        data_format : str
            The format to save the SOM object in. Must be one of: pkl

        Returns
        -------
        None
        """
        if data_format not in ['pkl']:
            raise ValueError('data_format must be one of: pkl')

        if data_format == 'pkl':
            with open(path + filename, 'wb') as f:
                pickle.dump(self, f)

    def load_pickle(self, filename, path, data_format='pkl'):
        """ Load the SOM object from a file using pickle.

        Parameters
        ----------
        filename : str
            The name of the file to load the SOM object from.

        path : str
            The path to the file to load the SOM object from.

        data_format : str
            The format to load the SOM object from. Must be one of: pkl

        Returns
        -------
        None
        """
        if data_format not in ['pkl']:
            raise ValueError('data_format must be one of: pkl')

        if data_format == 'pkl':
            with open(path + filename, 'rb') as f:
                som = pickle.load(f)

        return som

    def _normalize_position(self, position):  # Implement from utils
        # Normalize the positions of the neurons to be in the range [-1, 1]
        shap = position.shape
        numPos = shap[1]
        minPos = cp.ndarray.min(position,axis=1)
        maxPos = cp.ndarray.max(position,axis=1)
        difPos = maxPos - minPos
        equal = cp.equal(minPos, maxPos)
        difPos[equal] = 1
        minPos = cp.expand_dims(minPos, axis=1)
        minPos = cp.repeat(minPos, numPos, axis=1)
        difPos = cp.expand_dims(difPos, axis=1)
        difPos = cp.repeat(difPos, numPos, axis=1)
        posit = 2 * ((position - minPos)/difPos) - 1
        return posit

    def _spread_positions(self, position, positionMean, positionBasis): # Implement from utils
        # Spread the positions of the neurons
        shappos = position.shape
        numPos = shappos[1]
        position1 = cp.repeat(positionMean, numPos, axis=1) + cp.matmul(positionBasis, position)
        return position1

    def _euclidean_distance(self, XA, XB):
        """ Compute distance between each pair of the two collections of inputs.

        Parameters
        ----------
        XA : array_like
            An :math:`m_A` by :math:`n` array of :math:`m_A`
            original observations in an :math:`n`-dimensional space.
            Inputs are converted to float type.
        XB : array_like
            An :math:`m_B` by :math:`n` array of :math:`m_B`
            original observations in an :math:`n`-dimensional space.
            Inputs are converted to float type.

        Returns
        -------
        Y : ndarray
            A :math:`m_A` by :math:`m_B` distance matrix is returned.
            For each :math:`i` and :math:`j`, the metric
            ``dist(u=XA[i], v=XB[j])`` is computed and stored in the
            :math:`ij` th entry.

        Raises
        ------
        ValueError
            An exception is thrown if `XA` and `XB` do not have
            the same number of columns.


        """
        XA = cp.asarray(XA)
        XB = cp.asarray(XB)

        s = XA.shape
        sB = XB.shape

        if len(s) != 2:
            raise ValueError('XA must be a 2-dimensional array.')
        if len(sB) != 2:
            raise ValueError('XB must be a 2-dimensional array.')
        if s[1] != sB[1]:
            raise ValueError('XA and XB must have the same number of columns '
                            '(i.e. feature dimension.)')

        XA2 = cp.sum(XA**2, axis=1).reshape(-1, 1)  # Squares of norms of x, reshaped to column vector
        XB2 = cp.sum(XB**2, axis=1).reshape(1, -1)  # Squares of norms of y, reshaped to row vector
        xy = cp.dot(XA, XB.T)  # Matrix product of x and transpose of y
        distances = cp.sqrt(cp.maximum(0, XA2 + XB2 - 2 * xy))  # Ensure non-negative for sqrt

        return distances

    def _to_categorical(self, x, num_classes=None):
        """ Converts a class vector (integers) to binary class matrix.

        Args:
        x: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
            as `max(x) + 1`. Defaults to `None`.

        Returns:
        A binary matrix representation of the input as a NumPy or Cupy array. The class
        axis is placed last.

        Examples:

        >>> a = self._to_categorical([0, 1, 2, 3], num_classes=4)
        >>> a
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])
        """
        x = cp.array(x, dtype="int64")
        input_shape = x.shape

        # Shrink the last dimension if the shape is (..., 1).
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])

        x = x.reshape(-1)
        if not num_classes:
            num_classes = cp.max(x) + 1
        batch_size = x.shape[0]
        categorical = cp.zeros((batch_size, num_classes))
        categorical[cp.arange(batch_size), x] = 1
        output_shape = input_shape + (num_classes,)
        categorical = cp.reshape(categorical, output_shape)

        return categorical
