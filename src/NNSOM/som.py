from .utils import *
import numpy as np
from tensorflow.keras.utils import to_categorical
from datetime import datetime
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

    save_pickle(self, filename, path, data_format='pkl'):
        Saves the SOM object to a file using the pickle format.

    load_pickle(self, filename, path, data_format='pkl'):
        Loads a SOM object from a file using the pickle format.

    Visualization methods (hit_hist, neuron_dist_plot, cmplx_hit_hist, gray_hist, color_hist,
    plt_top, plt_top_num, plt_pie, plt_wgts, simple_grid, plt_dist):
        Generate various plots to visualize aspects of the SOM, such as the distribution of input
        data across neurons, the distances between neurons, and the weights of the neurons.
    """
    def __init__(self, dimensions):

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

    def init_w(self, x):
        # Initialize SOM weights using principal components

        # Print Beginning time for initialization
        print('Beginning Initialization')
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)

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
        a = to_categorical(maxRows,num_classes=n.shape[0])#made correction-added number of class
        # a = tf.constant(a, shape=[np.transpose(n).shape[0],np.transpose(n).shape[1]])  # made change
        return np.transpose(a)


    def train(self, x, init_neighborhood=3, epochs=200, steps=100):
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

    def hit_hist(self, x, textFlag):
        # Basic hit histogram
        # x contains the input data
        # If textFlag is true, the number of members of each cluster
        # is printed on the cluster.
        w = self.w
        pos = self.pos
        numNeurons = self.numNeurons

        # Determine the shape of the hexagon to represent each cluster
        z = np.sqrt(0.75)
        shapex = np.array([-1, 0, 1, 1, 0, -1]) * 0.5
        shapey = np.array([1, 2, 1, -1, -2, -1]) * (z / 3)

        # Get the figure, remove the frame, and find the limits
        # of the axis that will fit all of the hexagons
        #       # fig, ax = plt.subplots(frameon=False, figsize=(8, 8))
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.axis('equal')
        xmin = np.min(pos[0]) + np.min(shapex)
        xmax = np.max(pos[0]) + np.max(shapex)
        ymin = np.min(pos[1]) + np.min(shapey)
        ymax = np.max(pos[1]) + np.max(shapey)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        # Plot the outer hexgons
        for i in range(numNeurons):
            plt.fill(pos[0, i] + shapex, pos[1, i] + shapey, facecolor=(1, 1, 1), edgecolor=(0.8, 0.8, 0.8))

        # Plot the inner hexagon
        patches = []
        for i in range(numNeurons):
            temp = plt.fill(pos[0, i] + shapex, pos[1, i] + shapey, edgecolor=None)
            patches.append(temp)

        # Add the size of the cluster, if needed
        text = []
        if textFlag:
            for i in range(numNeurons):
                temp = plt.text(pos[0, i], pos[1, i], '9', horizontalalignment='center', verticalalignment='center', color='w')
                temp._fontproperties._weight = 'bold'
                temp._fontproperties._size = 12.0
                text.append(temp)

        # Compute the SOM outputs for the data set
        if self.sim_flag:
            outputs = self.sim_som(x)
            self.outputs = outputs
            self.sim_flag = False
        else:
            outputs = self.outputs

        # Find out how many inputs fall into each cluster
        hits = np.sum(outputs, axis=1)
        norm_hits = np.sqrt(hits/np.amax(hits))

        shapex1 = np.append(shapex, shapex[0])
        shapey1 = np.append(shapey, shapey[0])

        # Make the size of the inner hexagon proportional to the cluster size
        for i in range(numNeurons):
            patches[i][0]._facecolor = (0.4, 0.4, 0.6, 1.0)
            patches[i][0]._edgecolor = (0.2, 0.2, 0.3, 1.0)
            patches[i][0]._path._vertices[:, 0] = pos[0, i] + shapex1 * norm_hits[i]
            patches[i][0]._path._vertices[:, 1] = pos[1, i] + shapey1 * norm_hits[i]
            if textFlag:
                text[i]._text = str(int(hits[i]))

        # Get rid of extra white space on sides
        plt.axis('off')
        fig.tight_layout()

        return fig, ax, patches, text

    def plt_nc(self):
        # Neighborhood Connection Map. The gray hexagons represent cluster centers.
        pos = self.pos
        numNeurons = self.numNeurons

        # Determine the hexagon shape
        shapex, shapey = get_hexagon_shape()
        shapex, shapey = shapex * 0.3, shapey * 0.3

        # Determine the elongated hexagon shape
        edgex, edgey = get_edge_shape()

        # Set up edges
        neighbors = np.zeros((numNeurons, numNeurons))
        neighbors[self.neuron_dist <= 1.001] = 1.0
        neighbors = np.tril(neighbors - np.identity(numNeurons))

        # Get the figure and axes
        fig, ax = plt.subplots(figsize=(8,8), frameon=False)
        ax.axis('equal')
        ax.axis('off')
        xmin = np.min(pos[0]) + np.min(shapex)
        xmax = np.max(pos[0]) + np.max(shapex)
        ymin = np.min(pos[1]) + np.min(shapey)
        ymax = np.max(pos[1]) + np.max(shapey)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        # Draw elongated hexagons between neurons
        patches = []
        for i in range(numNeurons):
            for j in np.where(neighbors[:, i] == 1.0)[0]:
                pdiff = pos[:, j] - pos[:, i]
                angle = np.arctan2(pdiff[1], pdiff[0])
                ex, ey = rotate_xy(edgex, edgey, angle)
                edgePos = (pos[:, i] + pos[:, j]) * 0.5
                p1 = (2 * pos[:, i] + 1 * pos[:, j]) / 3
                p2 = (1 * pos[:, i] + 2 * pos[:, j]) / 3
                temp = ax.fill(edgePos[0] + ex, edgePos[1] + ey, facecolor='none', edgecolor=(0.8, 0.8, 0.8))
                patches.append(temp)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=[1, 0, 0])

        # Setup neurons. Place gray hexagon at neuron locations.
        for i in range(numNeurons):
            ax.fill(pos[0, i] + shapex, pos[1, i] + shapey, facecolor=(0.4, 0.4, 0.6), edgecolor=(0.8, 0.8, 0.8))

        return fig, ax, patches

    def neuron_dist_plot(self):
        # Distance map. The gray hexagons represent cluster centers.
        # The colors of the elongated hexagons between the cluster
        # centers represent the distance between the centers. The
        # darker the color the larger the distance.

        pos = self.pos
        numNeurons = self.numNeurons

        # Determine the shape of the hexagon to represent each cluster
        shapex, shapey = get_hexagon_shape()
        shapex = shapex * 0.3
        shapey = shapey * 0.3

        # Determine the shape of the elongated hexagon
        edgex, edgey = get_edge_shape()

        # Set up edges
        neighbors = np.zeros((numNeurons,numNeurons))
        neighbors[self.neuron_dist<=1.001] = 1.0
        neighbors = np.tril(neighbors - np.identity(numNeurons))

        # Get the figure, remove the frame, and find the limits
        # of the axis that will fit all of the hexagons
        #       # fig, ax = plt.subplots(frameon=False, figsize=(8, 8))
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.axis('equal')
        plt.axis('off')
        xmin = np.min(pos[0]) + np.min(shapex)
        xmax = np.max(pos[0]) + np.max(shapex)
        ymin = np.min(pos[1]) + np.min(shapey)
        ymax = np.max(pos[1]) + np.max(shapey)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        # Draw elongated hexagons between neurons
        numEdges = int(np.sum(neighbors))
        patches = []
        for i in range(numNeurons):
            for j in np.where(neighbors[:,i]==1.0)[0]:
                pdiff = pos[:, j]-pos[:, i]
                angle = np.arctan2(pdiff[1], pdiff[0])
                ex, ey = rotate_xy(edgex, edgey, angle)
                edgePos = (pos[:, i] + pos[:, j])*0.5
                p1 = (2 * pos[:, i] + 1 * pos[:, j])/ 3
                p2 = (1 * pos[:, i] + 2 * pos[:, j])/ 3
                temp  = plt.fill(edgePos[0]+ex, edgePos[1]+ey, facecolor=np.random.rand(1,3), edgecolor='none')
                patches.append(temp)
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=[1, 0, 0])
        # Setup neurons. Place gray hexagon at neuron locations.
        for i in range(numNeurons):
            plt.fill(pos[0, i] + shapex, pos[1, i] + shapey, facecolor=(0.4, 0.4, 0.6), edgecolor=(0.8, 0.8, 0.8))

        # Find the distance between neighboring weights.
        weights = self.w
        levels = np.zeros(numEdges)
        k = 0
        for i in range(numNeurons):
            for j in np.where(neighbors[:,i]==1.0)[0]:
                levels[k] = np.sqrt(np.sum((weights[i,:] - weights[j,:]) ** 2))
                k = k + 1
        mn = np.amin(levels)
        mx = np.amax(levels)
        if mx==mn:
            levels = np.zeros(1,numEdges) + 0.5
        else:
            levels  = (levels - mn)/(mx - mn)

        # Make the face color black for the maximum distance and
        # yellow for the minimum distance. The middle distance  will
        # be red.
        k = 0
        for i in range(numNeurons):
            for j in np.where(neighbors[:,i]==1.0)[0]:
                level =  1 - levels[k]
                red = np.amin([level * 2, 1])
                green = np.amax([level * 2 - 1, 0])
                c = (red, green, 0, 1.0)
                patches[k][0]._facecolor =  c
                k = k + 1

        # Get rid of extra white space on sides
        fig.tight_layout()

        return fig, ax, patches

    def cmplx_hit_hist(self, x, perc_gb, clust, ind_missClass, ind21, ind12):
        # This is a modified hit histogram, indicating if a cluster contains a
        # majority of good binders, and indicating how many/type errors occur in
        # each cluster
        #
        # Inputs are
        #  x - data set
        #  perc_gb - percent of good binders in each cluster
        #  clust - list of indices of inputs that belong in each cluster
        #  ind_missClass - indices of consistently misclassified inputs
        #  ind21 - indices of false positive cases
        #  ind12 - indices of false negative cases
        # Make hit histogram
        fig, ax, patches, text = self.hit_hist(x, True)

        numNeurons = self.numNeurons

        for neuron in range(numNeurons):

            # Make face color green if majority of cluster are good binders
            if (perc_gb[neuron] >= 50):
                patches[neuron][0]._facecolor = (0.0, 1.0, 0.0, 1.0)

            if len(np.intersect1d(clust[neuron], ind_missClass)) != 0:
                # If there are errors in the cluster, change width of
                # hexagon edge in proportion to number of errors
                lwidth = 20. * len(np.intersect1d(clust[neuron], ind_missClass)) / len(clust[neuron])

                if len(np.intersect1d(clust[neuron], ind12)) > len(np.intersect1d(clust[neuron], ind21)):
                    # Make edge color red if most errors are false positive
                    color = (1.0, 0.0, 0.0, 1.0)
                else:
                    # Make edge color purple if most errors are false negative
                    color = (1.0, 0.0, 1.0, 1.0)
            else:
                lwidth = 0
                color = (1.0, 1.0, 1.0, 0.0)

            patches[neuron][0]._linewidth = lwidth
            patches[neuron][0]._edgecolor = color

        # Get rid of extra white space on sides
        fig.tight_layout()

        return fig, ax, patches, text

    def gray_hist(self, x, perc):
        # Make another hit histogram figure, and change the colors of the hexagons
        # to indicate the perc of pdb (or gb) ligands in each cluster. Lighter color
        # means more PDB ligands, darker color means more well-docked bad binders.

        numNeurons = self.numNeurons

        fig, ax, patches, text = self.hit_hist(x, False)

        # Scale the gray scale to the perc value
        for neuron in range(numNeurons):
            scale = perc[neuron] / 100.0
            color = [scale for i in range(3)]
            color.append(1.0)
            color = tuple(color)
            patches[neuron][0]._facecolor = color

        # Get rid of extra white space on sides
        fig.tight_layout()

        return fig, ax, patches, text

    def color_hist(self, x, avg):
        # Plot an SOM figure where the size of the hexagons is related to
        # the number of elements in the clusters, and the color of the
        # inner hexagon is coded to the variable avg, which could be the
        # average number of a certain type of bond in the cluster

        # Find the maximum value of avg across all clusters
        dmax = np.amax(np.abs(avg))
        numNeurons = self.numNeurons

        fig, ax, patches, text = self.hit_hist(x, False)

        # Use the jet color map
        cmap = plt.get_cmap('jet')
        xx = np.zeros(numNeurons)

        # Adjust the color of the hexagon according to the avg value
        for neuron in range(numNeurons):
            xx[neuron] = avg[neuron] / dmax
            color = cmap(xx[neuron])
            patches[neuron][0]._facecolor = color

        fig.tight_layout()

        # # Add a color bar the the figure to indicate levels
        # # create an axes on the right side of ax. The width of cax will be 5%
        # # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        #
        # cbar = plt.colorbar(ax, cax=cax, cmap=cmap)

        cax = cm.ScalarMappable(cmap=cmap)
        cax.set_array(xx)
        cbar = fig.colorbar(cax)

        # Adjust the tick labels to the correct scale
        ticklab = cbar.ax.get_yticks()
        numticks = len(ticklab)
        ticktext = []
        for i in range(numticks):
            ticktext.append('%.2f' % (dmax * ticklab[i]))

        cbar.ax.set_yticklabels(ticktext)

        # Get rid of extra white space on sides
        fig.tight_layout()

        return fig, patches, text, cbar

    def plt_top(self):
        # Plot the topology of the SOM
        w = self.w
        pos = self.pos
        numNeurons = self.numNeurons
        z = np.sqrt(0.75)
        shapex = np.array([-1, 0, 1, 1, 0, -1]) * 0.5
        shapey = np.array([1, 2, 1, -1, -2, -1]) * (z / 3)

        fig, ax = plt.subplots(frameon=False)
        plt.axis('equal')
        xmin = np.min(pos[0]) + np.min(shapex)
        xmax = np.max(pos[0]) + np.max(shapex)
        ymin = np.min(pos[1]) + np.min(shapey)
        ymax = np.max(pos[1]) + np.max(shapey)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        patches = []
        for i in range(numNeurons):
            temp = plt.fill(pos[0, i] + shapex, pos[1, i] + shapey, facecolor=(1, 1, 1), edgecolor=(0.8, 0.8, 0.8))
            patches.append(temp)

        # Get rid of extra white space on sides
        fig.tight_layout()

        return fig, ax, patches

    def plt_top_num(self):
        # Plot the topology of the SOM with numbers for neurons
        w = self.w
        pos = self.pos
        numNeurons = self.numNeurons
        z = np.sqrt(0.75)
        shapex = np.array([-1, 0, 1, 1, 0, -1]) * 0.5
        shapey = np.array([1, 2, 1, -1, -2, -1]) * (z / 3)

        fig, ax = plt.subplots(frameon=False)
        ax.axis('off')
        plt.axis('equal')
        xmin = np.min(pos[0]) + np.min(shapex)
        xmax = np.max(pos[0]) + np.max(shapex)
        ymin = np.min(pos[1]) + np.min(shapey)
        ymax = np.max(pos[1]) + np.max(shapey)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        patches = []
        for i in range(numNeurons):
            temp = plt.fill(pos[0, i] + shapex, pos[1, i] + shapey, facecolor=(1, 1, 1), edgecolor=(0.8, 0.8, 0.8))
            patches.append(temp)

        text = []
        for i in range(numNeurons):
            temp = plt.text(pos[0, i], pos[1, i], str(i), horizontalalignment='center', verticalalignment='center', color='b')
            temp._fontproperties._weight = 'bold'
            temp._fontproperties._size = 12.0
            text.append(temp)

        # Get rid of extra white space on sides
        fig.tight_layout()

        return fig, ax, patches, text

    def plt_pie(self, title, perc, *argv):
        # Plots pie charts on SOM cluster locations. The size of the chart
        # is related to the percentage of PDB or WD in the cluster.
        # The arguments are tp, fn, tn and fp.
        # perc is the percentage of PDB (or WD) in each cluster

        pos = self.pos

        # Pull out the statistics (tp, fn, tn, fp) from the arguments
        numst = []
        for arg in argv:
            numst.append(arg)

        # If there are 4 arguments, it is for the PDB case
        pdb = False
        if len(numst)==4:
            pdb = True

        # Find the locations and size for each neuron in the SOM
        w = self.w
        numNeurons = self.numNeurons
        z = np.sqrt(0.75)
        shapex = np.array([-1, 0, 1, 1, 0, -1]) * 0.5
        shminx = np.min(shapex)
        shmaxx = np.max(shapex)
        shapey = np.array([1, 2, 1, -1, -2, -1]) * (z / 3)
        shminy = np.min(shapey)
        shmaxy = np.max(shapey)

        # Create the figure and get the transformations from data
        # to pixel and from pixel to axes.
        fig, ax = plt.subplots(frameon=False, figsize=(8,8))
        ax.axis('off')
        plt.axis('equal')
        transDat = ax.transData
        transAxi = ax.transAxes.inverted()

        # Find how big the axes needs to be to fit the hexagons
        xmin = np.min(pos[0]) + np.min(shapex)
        xmax = np.max(pos[0]) + np.max(shapex)
        ymin = np.min(pos[1]) + np.min(shapey)
        ymax = np.max(pos[1]) + np.max(shapey)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax+0.5])

        h_axes = [0] * numNeurons

        # Assign the colors for the pie chart (tp, fn, tn, fp)
        if pdb:
            clrs = ['lawngreen', 'yellow', 'blue', 'red']
        else:
            # Only two colors for well docked bad binders (tn, fp)
            clrs = ['blue', 'red']

        for neuron in range(numNeurons):

            # Scale the size of the pie chart according to the percent of PDB
            # data (or WD data) in that cell
            if pdb:
                scale = np.sqrt(perc[neuron]/100)
            else:
                scale = np.sqrt((100-perc[neuron]) / 100)

            if scale==0:
                scale = 0.01

            # Find the size of the cell in data units
            minx = pos[0, neuron] + shminx
            maxx = pos[0, neuron] + shmaxx
            miny = pos[1, neuron] + shminy
            maxy = pos[1, neuron] + shmaxy

            # Convert the size of the cell to axes units
            minxyDis = transDat.transform([minx, miny])
            maxxyDis = transDat.transform([maxx, maxy])
            minxyAx = transAxi.transform(minxyDis)
            maxxyAx = transAxi.transform(maxxyDis)

            # Find the width and height of the cell
            width = maxxyAx[0] - minxyAx[0]
            height = maxxyAx[1] - minxyAx[1]

            # Find the center point of the cell
            xavg = np.average([minxyAx[0], maxxyAx[0]])
            yavg = np.average([minxyAx[1], maxxyAx[1]])

            # Scale the width and height according to percent PDB or WD
            width = scale*width
            height = scale*height

            # Locate the beginning point of the cell
            x1 = xavg - (width/2)
            y1 = yavg - (height/2)

            # Place the axis in the cell for the pie chart
            place = [x1, y1, width, height]
            h_axes[neuron] = plt.axes(place)

            # Set numbers (tp, fn, tn, fp) for pie chart
            if pdb:
                nums = [numst[0][neuron], numst[1][neuron], numst[2][neuron], numst[3][neuron]]
            else:
                nums = [numst[0][neuron], numst[1][neuron]]

            # Make pie chart
            if np.sum(nums)==0:
                nums = [0.0, 1.0, 0.0, 0.0]
            h_axes[neuron].pie(nums, colors=clrs)

            # Leave some buffer space between cells
            h_axes[neuron].margins(0.05)

            # Make scales on both axes equal
            plt.axis('equal')

        # Get rid of extra white space on sides
        # fig.tight_layout()
        plt.suptitle(title, fontsize=16)

        # Return handles to figure, axes and pie charts
        return fig, ax, h_axes

    def plt_wgts(self):
        # Plots weights on SOM cluster locations.
        #

        pos = self.pos

        # Find the locations and size for each neuron in the SOM
        w = self.w
        numNeurons = self.numNeurons
        z = np.sqrt(0.75)
        shapex = np.array([-1, 0, 1, 1, 0, -1]) * 0.5
        shminx = np.min(shapex)
        shmaxx = np.max(shapex)
        shapey = np.array([1, 2, 1, -1, -2, -1]) * (z / 3)
        shminy = np.min(shapey)
        shmaxy = np.max(shapey)

        # Create the figure and get the transformations from data
        # to pixel and from pixel to axes.
        fig, ax = plt.subplots(frameon=False, figsize=(8,8))
        ax.axis('off')
        plt.axis('equal')
        transDat = ax.transData
        transAxi = ax.transAxes.inverted()

        # Find how big the axes needs to be to fit the hexagons
        xmin = np.min(pos[0]) + np.min(shapex)
        xmax = np.max(pos[0]) + np.max(shapex)
        ymin = np.min(pos[1]) + np.min(shapey)
        ymax = np.max(pos[1]) + np.max(shapey)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax+0.5])

        h_axes = [0] * numNeurons

        for neuron in range(numNeurons):

            # Find the size of the cell in data units
            minx = pos[0, neuron] + shminx
            maxx = pos[0, neuron] + shmaxx
            miny = pos[1, neuron] + shminy
            maxy = pos[1, neuron] + shmaxy

            # Convert the size of the cell to axes units
            minxyDis = transDat.transform([minx, miny])
            maxxyDis = transDat.transform([maxx, maxy])
            minxyAx = transAxi.transform(minxyDis)
            maxxyAx = transAxi.transform(maxxyDis)

            # Find the width and height of the cell
            width = maxxyAx[0] - minxyAx[0]
            height = maxxyAx[1] - minxyAx[1]

            # Find the center point of the cell
            xavg = np.average([minxyAx[0], maxxyAx[0]])
            yavg = np.average([minxyAx[1], maxxyAx[1]])

            # Scale the width and height
            scale = 0.75
            width = scale*width
            height = scale*height


            # Locate the beginning point of the cell
            x1 = xavg - (width/2)
            y1 = yavg - (height/2)

            # Place the axis in the cell for the pie chart
            place = [x1, y1, width, height]
            h_axes[neuron] = plt.axes(place)

            # Make graph
            h_axes[neuron].plot(w[neuron])
            title = 'Cluster Centers as Lines'
            plt.axis('off')


            # Leave some buffer space between cells
            h_axes[neuron].margins(0.05)

            # # Make scales on both axes equal
            # plt.axis('equal')

        # Get rid of extra white space on sides
        fig.tight_layout()
        plt.suptitle(title, fontsize=16)

        # Return handles to figure, axes and pie charts
        return fig, ax, h_axes

    def simple_grid(self, avg, sizes):
        # Basic hexagon grid plot
        # Colors are selected from avg array.
        # Sizes of inner hexagons are selected from sizes array.
        w = self.w
        pos = self.pos
        numNeurons = self.numNeurons

        # Determine the shape of the hexagon to represent each cluster
        z = np.sqrt(0.75)
        shapex = np.array([-1, 0, 1, 1, 0, -1]) * 0.5
        shapey = np.array([1, 2, 1, -1, -2, -1]) * (z / 3)

        # Get the figure, remove the frame, and find the limits
        # of the axis that will fit all of the hexagons
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.axis('equal')
        plt.axis('off')
        xmin = np.min(pos[0]) + np.min(shapex)
        xmax = np.max(pos[0]) + np.max(shapex)
        ymin = np.min(pos[1]) + np.min(shapey)
        ymax = np.max(pos[1]) + np.max(shapey)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        # Plot the outer hexgons
        for i in range(numNeurons):
            plt.fill(pos[0, i] + shapex, pos[1, i] + shapey, facecolor=(1, 1, 1), edgecolor=(0.8, 0.8, 0.8))

        # Plot the inner hexagon
        patches = []
        for i in range(numNeurons):
            temp = plt.fill(pos[0, i] + shapex, pos[1, i] + shapey, edgecolor=None)
            patches.append(temp)

        shapex1 = np.append(shapex, shapex[0])
        shapey1 = np.append(shapey, shapey[0])

        # Make the size of the inner hexagon proportional to the desired size
        sizes = np.sqrt(sizes / np.amax(sizes))
        for i in range(numNeurons):
            patches[i][0]._facecolor = (0.4, 0.4, 0.6, 1.0)
            patches[i][0]._edgecolor = (0.2, 0.2, 0.3, 1.0)
            patches[i][0]._path._vertices[:, 0] = pos[0, i] + shapex1 * sizes[i]
            patches[i][0]._path._vertices[:, 1] = pos[1, i] + shapey1 * sizes[i]

        # Get rid of extra white space on sides
        fig.tight_layout()

        # Find the maximum value of avg across all clusters
        # dmax = np.amax(np.abs(avg))
        dmax = np.amax(avg)
        dmin = np.amin(avg)
        drange = dmax - dmin


        # Use the jet color map
        cmap = plt.get_cmap('jet')
        xx = np.zeros(numNeurons)

        # Adjust the color of the hexagon according to the avg value
        for neuron in range(numNeurons):
            #        xx[neuron] = avg[neuron] / dmax
            xx[neuron] = (avg[neuron]-dmin) / drange
            color = cmap(xx[neuron])
            patches[neuron][0]._facecolor = color

        fig.tight_layout()

        # # Add a color bar the the figure to indicate levels
        # # create an axes on the right side of ax. The width of cax will be 5%
        # # of ax and the padding between cax and ax will be fixed at 0.05 inch.

        cax = cm.ScalarMappable(cmap=cmap)
        cax.set_array(xx)
        #cbar = fig.colorbar(cax)
        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        # plt.colorbar(im, fraction=0.046, pad=0.04)
        #
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        #
        # plt.colorbar(im, cax=cax)

        # Adjust the tick labels to the correct scale
        ticklab = cbar.ax.get_yticks()
        numticks = len(ticklab)
        ticktext = []
        for i in range(numticks):
            ticktext.append('%.2f' % (drange * ticklab[i] + dmin))

        cbar.ax.set_yticklabels(ticktext)

        # Get rid of extra white space on sides
        fig.tight_layout()

        return fig, ax, patches, cbar

    def plt_dist(self, dist):
        # Plots distributions of categories on SOM cluster locations.
        #

        pos = self.pos

        # Find the locations and size for each neuron in the SOM
        w = self.w
        numNeurons = self.numNeurons
        z = np.sqrt(0.75)
        shapex = np.array([-1, 0, 1, 1, 0, -1]) * 0.5
        shminx = np.min(shapex)
        shmaxx = np.max(shapex)
        shapey = np.array([1, 2, 1, -1, -2, -1]) * (z / 3)
        shminy = np.min(shapey)
        shmaxy = np.max(shapey)

        # Create the figure and get the transformations from data
        # to pixel and from pixel to axes.
        fig, ax = plt.subplots(frameon=False, figsize=(8,8))
        ax.axis('off')
        plt.axis('equal')
        transDat = ax.transData
        transAxi = ax.transAxes.inverted()

        # Find how big the axes needs to be to fit the hexagons
        xmin = np.min(pos[0]) + np.min(shapex)
        xmax = np.max(pos[0]) + np.max(shapex)
        ymin = np.min(pos[1]) + np.min(shapey)
        ymax = np.max(pos[1]) + np.max(shapey)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax+0.5])

        h_axes = [0] * numNeurons

        for neuron in range(numNeurons):

            # Find the size of the cell in data units
            minx = pos[0, neuron] + shminx
            maxx = pos[0, neuron] + shmaxx
            miny = pos[1, neuron] + shminy
            maxy = pos[1, neuron] + shmaxy

            # Convert the size of the cell to axes units
            minxyDis = transDat.transform([minx, miny])
            maxxyDis = transDat.transform([maxx, maxy])
            minxyAx = transAxi.transform(minxyDis)
            maxxyAx = transAxi.transform(maxxyDis)

            # Find the width and height of the cell
            width = maxxyAx[0] - minxyAx[0]
            height = maxxyAx[1] - minxyAx[1]

            # Find the center point of the cell
            xavg = np.average([minxyAx[0], maxxyAx[0]])
            yavg = np.average([minxyAx[1], maxxyAx[1]])

            # Scale the width and height
            # scale = 0.75
            scale = 0.65
            width = scale*width
            height = scale*height

            # Locate the beginning point of the cell
            x1 = xavg - (width/2)
            y1 = yavg - (height/2)

            # Place the axis in the cell for the pie chart
            place = [x1, y1, width, height]
            h_axes[neuron] = plt.axes(place)

            # Make graph
            h_axes[neuron].stem(dist[neuron])
            title = ''
            plt.axis('off')

            # Leave some buffer space between cells
            h_axes[neuron].margins(0.05)

            # # Make scales on both axes equal
            # plt.axis('equal')

        # Get rid of extra white space on sides
        #fig.tight_layout()
        plt.suptitle(title, fontsize=16)

        # Return handles to figure, axes and pie charts
        return fig, ax, h_axes

    def plt_scatter(self, x, indices, clust, reg_line=True):
        """ Generate Scatter Plot for Each Neuron.

        Args:
            x: input data
            indices: array-like indices e.g. (0, 1) or [0, 1]
            Clust: list of indices of input data for each cluster
            reg_line: Flag

        Returns:
            fig:
            ax:
            h_axes:
        """
        pos = self.pos
        numNeurons = self.numNeurons

        # Data preprocessing
        # This should be updated!!!!
        x1 = x[indices[0], :]
        x2 = x[indices[1], :]

        # Determine the shape
        shapex, shapey = get_hexagon_shape()

        # Create the figure and set axis properties
        fig, main_ax = plt.subplots(figsize=(8, 8),
                                    frameon=False,
                                    layout='constrained')
        xmin = np.min(pos[0, :]) + np.min(shapex)
        xmax = np.max(pos[0, :]) + np.max(shapex)
        ymin = np.min(pos[1, :]) + np.min(shapey)
        ymax = np.max(pos[1, :]) + np.max(shapey)
        main_ax.set_xlim([xmin, xmax])
        main_ax.set_ylim([ymin, ymax])
        main_ax.set_aspect(1)
        main_ax.axis('off')

        # Loop over each neuron to draw hexagons
        for neuron in range(numNeurons):
            main_ax.fill(pos[0, neuron] + shapex,
                         pos[1, neuron] + shapey,
                         facecolor=(1, 1, 1),
                         edgecolor=(0.8, 0.8, 0.8))

        # Loop over each neuron for scatter plots
        h_axes = [0] * numNeurons   # create a container for sub_axes

        for neuron in range(numNeurons):
            # Find the size of the cell in data
            minx = pos[0, neuron] + np.min(shapex)
            maxx = pos[0, neuron] + np.max(shapex)
            miny = pos[1, neuron] + np.min(shapey)
            maxy = pos[1, neuron] + np.max(shapey)

            # Convert the size of the cell to axes units
            minxyDis = main_ax.transData.transform((minx, miny))
            maxxyDis = main_ax.transData.transform((maxx, maxy))
            minxyAx = main_ax.transAxes.inverted().transform(minxyDis)
            maxxyAx = main_ax.transAxes.inverted().transform(maxxyDis)

            # Find the width and height of the cell
            width = maxxyAx[0] - minxyAx[0]
            height = maxxyAx[1] - minxyAx[1]

            # Find the center point of the cell
            xavg = np.average([minxyAx[0], maxxyAx[0]])
            yavg = np.average([minxyAx[1], maxxyAx[1]])

            # Scale the width and height
            scale = np.sqrt(0.75) / 3 * 2  # Just fit-in the hexagon
            width = width * scale
            height = height * scale

            # Locate the beginning point of the cell
            x0 = xavg - (width / 2)
            y0 = yavg - (height / 2)

            if len(clust[neuron]) > 0:
                h_axes[neuron] = inset_axes(main_ax, width='100%', height='100%', loc=3,
                                            bbox_to_anchor=(x0, y0, width, height),
                                            bbox_transform=main_ax.transAxes, borderpad=0)
                h_axes[neuron].set(xticks=[], yticks=[])
                h_axes[neuron].set_frame_on(False)
            else:
                h_axes[neuron] = None

            # Make Scatter Plot for each neuron
            if len(clust[neuron]) > 0:
                # Pick specific rows based on the Clust
                x1_temp = x1[clust[neuron]]
                x2_temp = x2[clust[neuron]]
                h_axes[neuron].scatter(x1_temp, x2_temp, s=1, c='k')

                if reg_line:
                    m, p = np.polyfit(x1_temp, x2_temp, 1)
                    h_axes[neuron].plot(x1_temp, m * x1_temp + p, c='r', linewidth=1)
                    title = "Scatter Plot for each neuron with regression lines"
                else:
                    title = "Scatter Plot for each neuron without regression lines"
            else:
                h_axes[neuron] = None

        plt.suptitle(title, fontsize=16)

        return fig, main_ax, h_axes



