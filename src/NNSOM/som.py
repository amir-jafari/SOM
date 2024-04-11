from .utils import *
import numpy as np
from tensorflow.keras.utils import to_categorical
from datetime import datetime
from scipy.spatial.distance import cdist
import pickle
import warnings

class SOM():
    """
    A class to represent a Self-Organizing Map (SOM), a type of artificial neural network
    trained using unsupervised learning to produce a two-dimensional, discretized representation
    of the input space of the training samples.

    Attributes
    ----------
    dimensions : tuple, list, or array-like
        The dimensions of the SOM grid. Determines the layout and number of neurons in the map.
    numNeurons : int
        The total number of neurons in the SOM, calculated as the product of the dimensions.
    pos : array-like
        The positions of the neurons in the SOM grid.
    neuron_dist : array-like
        The distances between neurons in the SOM.
    w : array-like
        The weight matrix of the SOM, representing the feature vectors of the neurons.
    sim_flag : bool
        A flag indicating whether the SOM has been simulated or not.

    Methods
    -------
    __init__(self, dimensions):
        Initializes the SOM with the specified dimensions.

    init_w(self, x):
        Initializes the weights of the SOM using principal components analysis on the input data x.

    sim_som(self, x):
        Simulates the SOM with x as the input, determining which neurons are activated by the input vectors.

    train(self, x, init_neighborhood=3, epochs=200, steps=100):
        Trains the SOM using the batch SOM algorithm on the input data x.

    quantization_error(self, dist)
        Calculate quantization error

    topological_error(self, data)
        Calculate 1st and 1st-2nd toplogical error

    distortion_error(self, data)
        Calculate distortion error

    save_pickle(self, filename, path, data_format='pkl'):
        Saves the SOM object to a file using the pickle format.

    load_pickle(self, filename, path, data_format='pkl'):
        Loads a SOM object from a file using the pickle format.
    """
    def __init__(self, dimensions):
        """
        Initializes the SOM with the specified dimensions and calculates the positions and distances between neurons in the SOM grid.

        Parameters
        ----------
        dimensions : tuple, list, or np.ndarray
                    The dimensions (shape) of the SOM grid.
        """

        self.dimensions = dimensions
        self.numNeurons = np.prod(dimensions)
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

        sz = x.shape

        posMean = np.mean(x, axis=1)
        posMean = np.expand_dims(posMean, axis=1)

        xc = x - posMean

        components, gains, encodedInputsT = np.linalg.svd(xc)

        encodedInputsT = np.transpose(encodedInputsT)

        basis = components * gains
        stdev = np.std(encodedInputsT, axis=0)
        stdev = stdev[:len(basis)]
        posBasis = 2.5 * basis * stdev

        numNeurons = self.numNeurons
        numDimensions = len(self.dimensions)
        sampleSize = sz[1]
        inputSize = sz[0]
        dimOrder = np.argsort(self.dimensions)

        restoreOrder = np.concatenate((np.sort(dimOrder), np.arange(numDimensions, np.minimum(inputSize, sampleSize))))

        if numDimensions > inputSize:
            posBasis = np.concatenate((posBasis, np.random.rand(inputSize, inputSize) * 0.001))

        posBasis = posBasis[restoreOrder]

        pos1 = self.pos

        if sampleSize < inputSize:
            posBasis = np.concatenate((posBasis, np.zeros((inputSize, inputSize - sampleSize))))

        if inputSize > numDimensions:
            pos1 = np.concatenate((pos1, np.zeros((inputSize - numDimensions, numNeurons))), axis=0)

        pos2 = normalize_position(pos1)

        pos3 = spread_positions(pos2, posMean, posBasis)

        self.w = np.transpose(pos3)

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
        shapx = x.shape   # shapes of the input x
        shapw = self.w.shape # weights of the SOM

        # Compute the negative distance from the inputs to each center
        n = -cdist(self.w, np.transpose(x), 'euclidean')
        # n = np.empty((shapw[0], shapx[1]))
        # for jj in range(shapw[0]):
        #     wj = self.w[jj]
        #     wj = np.expand_dims(wj, axis=1)
        #     n[jj] = np.sum((x - wj)**2, axis=0)

        # n = -np.sqrt(n)
        #print(n)

        # Find out which center was closest to the input
        maxRows = np.argmax(n, axis=0)
        a = to_categorical(maxRows, num_classes=n.shape[0]) #made correction-added number of class
        # a = tf.constant(a, shape=[np.transpose(n).shape[0],np.transpose(n).shape[1]])  # made change
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

        # Train the SOM using the batch SOM algorithm

        w = self.w
        shapw = w.shape
        S = shapw[0]
        shapx = x.shape
        Q = shapx[1]

        step = 0

        now = datetime.now()

        print('Beginning Training')
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        # Train the network
        for i in range(epochs):

            # network output
            a = self.sim_som(x)

            # neighborhood distance
            nd = 1 + (init_neighborhood-1) * (1 - step/steps)
            neighborhood = self.neuron_dist <= nd
            #print(nd)
            # remove some outputs at random
            a = a * (np.random.rand(S, Q) < 0.90)
            a2 = np.matmul(neighborhood,a) + a

            # find how many times each neuron won
            # (The winning neuron is the one that exhibits the smallest distance or similarity to the input data)
            suma2 = np.sum(a2, axis=1)
            loserIndex = np.squeeze(np.asarray(suma2 == 0))
            suma2[loserIndex] = 1
            suma2 = np.expand_dims(suma2,axis = 1)
            a3 = a2 / np.repeat(suma2, Q, axis=1)

            neww = np.matmul(a3, np.transpose(x))

            dw = neww - w

            dw[loserIndex] = 0

            w = w + np.array(dw)

            step = step + 1

            if step%50==0:
                print(step)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("Current Time =", current_time)

        self.w = w
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

        w = self.w
        shapw = w.shape
        S = shapw[0]

        x_w_dist = cdist(w, np.transpose(x), 'euclidean')
        ind1 = np.argmin(x_w_dist, axis=0)

        clusters = []  # a cluster array of indices sorted by distances
        cluster_distances = []  # a cluster array of distances sorted by distances
        max_cluster_distances = np.zeros(S)  # a list of maimum distance to any input in the cluster from cluster center
        cluster_sizes = []  # cluster array sizes

        for i in range(S):
            # Find which inputs are closest to each weight (in cluster i)
            tempclust = np.where(ind1 == i)[0]

            # Save distance of each input in the cluster to cluster center (weight)
            tempdist = x_w_dist[i, tempclust]
            indsort = np.argsort(tempdist)
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
        x_w_dist = cdist(w, np.transpose(x), 'euclidean')

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
        x_w_dist = cdist(ww, np.transpose(x), 'euclidean')
        ind1 = np.argmin(x_w_dist, axis=0)

        dd = [1, 2, 3]  # neighborhood distances
        wwdist = cdist(ww, ww, 'euclidean')
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
