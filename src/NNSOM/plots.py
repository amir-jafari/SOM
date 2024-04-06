from .som import SOM
from .utils import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
class SOMPlots(SOM):
    """
    SOMPlots extends the SOM class by adding visualization capabilities to
    the Self-Organizing Map (SOM). It allows for the graphical representation
    of the SOM's structure, the distribution of input data across its neurons,
    and various other analytical visualizations that aid in the interpretation
    of the SOM's behavior and characteristics.

    Attributes:
        dimensions (tuple): The dimensions of the SOM grid.

    Methods:
        plt_top():
            Plots the topology of the SOM using hexagonal units.
        plt_top_num():
            Plots the topology of the SOM with numbered neurons.
        hit_hist(x, textFlag):
            Plots a hit histogram showing how many data points are mapped to each neuron.
        gray_hist(x, perc):
            Plots a histogram with neurons colored in shades of gray based on a given percentage value.
        color_hist(x, avg):
            Plots a color-coded histogram based on the average values provided for each neuron.
        cmplx_hit_hist(x, perc_gb, clust, ind_missClass, ind21, ind12):
            Plots a complex hit histogram showing the distribution of data and misclassifications.
        plt_nc():
            Plots the neighborhood connections between the SOM neurons.
        neuron_dist_plot():
            Plots the distances between neurons to visualize the SOM's topology.
        simple_grid(avg, sizes):
            Plots a simple hexagonal grid with varying colors and sizes based on provided data.
        setup_axes():
            Sets up the axes for plotting individual neuron statistics.
        plt_dist(dist):
            Plots distributions of values across the SOM neurons.
        plt_wgts():
            Plots the weights of the SOM neurons as line graphs.
        plt_pie(title, perc, *argv):
            Plots pie charts for each neuron to show data distribution in categories.
        plt_histogram(som, data):
            Plots histograms for each neuron to show the distribution of data.
        plt_boxplot(data):
            Plots boxplots for each neuron to show the distribution of data.
        plt_dispersion_fan_plot(data):
            Plots dispersion or fan plots for each neuron.
        plt_violin_plot(som, data):
            Plots violin plots for each neuron to show the distribution of data.
        plt_scatter(x, indices, clust, reg_line=True):
            Plots scatter graphs for each neuron to show the distribution of two variables.
        multiplot(plot_type, *args):
            Facilitates plotting of multiple types of graphs based on the plot_type argument.
    """
    def __init__(self, dimensions):
        super().__init__(dimensions)

    def plt_top(self, mouse_click=False, connect_pick_event=True, **kwargs):
        """ Plots the topology of the SOM using hexagonal units.

        Args:
            mouse_click: bool
                If true, the interactive plot and sub-clustering functionalities to be activated
            connect_pick_event: bool
                If true, the pick event is connected to the plot
            kwarg: dict
                Additional arguments to be passed to the onpick function
                Possible keys include:
                    'data', 'clust', 'target', 'num1', 'num2', 'cat', 'align', 'height' and 'topn'

        Returns:
            fig, ax, pathces
        """
        w = self.w
        pos = self.pos
        numNeurons = self.numNeurons

        shapex, shapey = get_hexagon_shape()

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
            temp, = ax.fill(pos[0, i] + shapex, pos[1, i] + shapey, facecolor=(1, 1, 1), edgecolor=(0.8, 0.8, 0.8),
                            picker=True)
            patches.append(temp)

        # Assign the cluster number for each hexagon
        hexagon_to_neuron = {hex: neuron for neuron, hex in enumerate(patches)}

        # Mouse Click Functionality
        if mouse_click and connect_pick_event:
            fig.canvas.mpl_connect(
                'pick_event', lambda event: self.onpick(event, patches, hexagon_to_neuron, **kwargs)
            )

        # Get rid of extra white space on sides
        plt.tight_layout()

        return fig, ax, patches

    def plt_top_num(self, mouse_click=False, connect_pick_event=True, **kwargs):
        """ Plots the topology of the SOM with numbered neurons.

        Args:
            mouse_click: bool
                If true, the interactive plot and sub-clustering functionalities to be activated
            connect_pick_event: bool
                If true, the pick event is connected to the plot
            kwarg: dict
                Additional arguments to be passed to the onpick function
                Possible keys include:
                    'data', 'clust', 'target', 'num1', 'num2', 'cat', 'align', 'height' and 'topn'

        Returns:
            fig, ax, pathces, text
        """
        w = self.w
        pos = self.pos
        numNeurons = self.numNeurons

        shapex, shapey = get_hexagon_shape()

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
            temp, = ax.fill(pos[0, i] + shapex, pos[1, i] + shapey, facecolor=(1, 1, 1), edgecolor=(0.8, 0.8, 0.8),
                            picker=True)
            patches.append(temp)

        # Assign the cluster number for each hexagon
        hexagon_to_neuron = {hex: neuron for neuron, hex in enumerate(patches)}

        text = []
        for i in range(numNeurons):
            temp = plt.text(pos[0, i], pos[1, i], str(i), horizontalalignment='center', verticalalignment='center',
                            color='b')
            temp._fontproperties._weight = 'bold'
            temp._fontproperties._size = 12.0
            text.append(temp)

        # Mouse Click Functionality
        if mouse_click and connect_pick_event:
            fig.canvas.mpl_connect(
                'pick_event', lambda event: self.onpick(event, patches, hexagon_to_neuron, **kwargs)
            )

        # Get rid of extra white space on sides
        plt.tight_layout()

        return fig, ax, patches, text

    def hit_hist(self, x, textFlag=True, mouse_click=False, connect_pick_event=True, **kwargs):
        """ Generate Hit Histogram

        Parameters
        ----------
        x: array-like
            The input data to be clustered
        textFlag: bool
            If true, the number of members of each cluster is printed on the cluster.
        mouse_click: bool
            If true, the interactive plot and sub-clustering functionalities to be activated
        connect_pick_event
            If true, the pick event is connected to the plot
        kwargs: dict
            Additional arguments to be passed to the on_pick function
            Possible keys includes:
            'data', 'labels', 'clust', 'target', 'num1', 'num2',
            'cat', 'align', 'height' and 'topn'

        Returns
        -------
            fig, ax, patches, text
        """

        pos = self.pos
        numNeurons = self.numNeurons

        # Determine the shape of the hexagon to represent each cluster
        shapex, shapey = get_hexagon_shape()

        # Create the main figure and axes
        # Set the main axes properties
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.set_axis_off()
        xmin = np.min(pos[0]) + np.min(shapex)
        xmax = np.max(pos[0]) + np.max(shapex)
        ymin = np.min(pos[1]) + np.min(shapey)
        ymax = np.max(pos[1]) + np.max(shapey)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        # Create the cluster hexagons
        hexagons = []
        for i in range(numNeurons):
            hex, = ax.fill(pos[0, i] + shapex, pos[1, i] + shapey,
                             facecolor=(1, 1, 1), edgecolor=(0.8, 0.8, 0.8),
                             picker=True)
            hexagons.append(hex)

        # Assign cluster number for each hexagon
        hexagon_to_neuron = {hex: neuron for neuron, hex in enumerate(hexagons)}

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
            outputs = outputs
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
        # plt.axis('off')
        #fig.tight_layout()

        # Mouse Click Functionality
        if mouse_click and connect_pick_event:
            fig.canvas.mpl_connect(
                'pick_event', lambda event: self.onpick(event, hexagons, hexagon_to_neuron, **kwargs)
            )

        return fig, ax, patches, text

    def gray_hist(self, x, perc, mouse_click=False, **kwargs):
        # Make another hit histogram figure, and change the colors of the hexagons
        # to indicate the perc of pdb (or gb) ligands in each cluster. Lighter color
        # means more PDB ligands, darker color means more well-docked bad binders.

        numNeurons = self.numNeurons

        fig, ax, patches, text = self.hit_hist(x, False, mouse_click, **kwargs)

        # Scale the gray scale to the perc value
        for neuron in range(numNeurons):
            scale = perc[neuron] / 100.0
            color = [scale for i in range(3)]
            color.append(1.0)
            color = tuple(color)
            patches[neuron][0]._facecolor = color

        # Get rid of extra white space on sides
        plt.tight_layout()

        return fig, ax, patches, text

    def color_hist(self, x, avg, mouse_click=False, **kwargs):
        # Plot an SOM figure where the size of the hexagons is related to
        # the number of elements in the clusters, and the color of the
        # inner hexagon is coded to the variable avg, which could be the
        # average number of a certain type of bond in the cluster

        # Find the maximum value of avg across all clusters
        dmax = np.amax(np.abs(avg))
        numNeurons = self.numNeurons

        fig, ax, patches, text = self.hit_hist(x, False, mouse_click, **kwargs)

        # Use the jet color map
        cmap = plt.get_cmap('jet')
        xx = np.zeros(numNeurons)

        # Adjust the color of the hexagon according to the avg value
        for neuron in range(numNeurons):
            xx[neuron] = avg[neuron] / dmax
            color = cmap(xx[neuron])
            patches[neuron][0]._facecolor = color

        plt.tight_layout()

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

        return fig, ax, patches, text, cbar

    def cmplx_hit_hist(self, x, clust, perc, ind_missClass, ind21, ind12, mouse_click=False, **kwargs):
        """ Generates a complex hit histogram.
        It indicates what the majority class in each cluster is, and how many specific class occur in each cluster.

        Args:
            x: array-like
                The input data to be clustered
            clust: list
                List of indices of inputs that belong in each cluster
            perc: array-like
                Percent of the specific class in each cluster
            ind_missClass: array-like
                Indices of consistently misclassified inputs
            ind21: array-like
                Indices of false positive cases
            ind12: array-like
                Indices of false negative cases
        """

        numNeurons = self.numNeurons

        if mouse_click:
            kwargs['clust'] = clust

        # Make hit histogram
        fig, ax, patches, text = self.hit_hist(x, True, mouse_click, **kwargs)

        for neuron in range(numNeurons):

            # Make face color green if majority of class in cluster are good binders
            if (perc[neuron] >= 50):
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
        plt.tight_layout()

        return fig, ax, patches, text

    def custom_cmplx_hit_hist(self, x, face_labels, edge_labels, edge_width, mouse_click=False, **kwargs):
        """ Generate cmplex hit hist
        Users can specify the face color, edge width and edge color for each neuron.

        x: array-like or sequence of vectors
            The input data to be clustered
        face_labels: array-like
            class labels to determine the face color of the hexagons
        edge_labels: array-like
            class labels to determine the edge color of the hexagons
        edge_width: array-like
            A list of edge_width standerdised between (1 - 20).
            You can call get_edge_width to get the standardised edge width in the utils function.
            len(edge_width) must be equal to the number of neurons
        mouse_click: bool
            If true, the interactive plot and sub-clustering functionalities to be activated
        kwargs: dict
            Additional arguments to be passed to the onpick function
            Possible keys include:
            'data', 'clust', 'target', 'num1', 'num2',
            'cat', 'align', 'height' and 'topn'

        Returns:
            fig, ax, patches, text
        """
        numNeurons = self.numNeurons

        x = np.asarray(x, np.float32)
        face_labels = np.asarray(face_labels, np.float32)
        edge_labels = np.asarray(edge_labels, np.float32)
        edge_width = np.asarray(edge_width, np.float32)

        # Check if the input data is a sequence of vectors
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")

        # Check if the input data can be cdist with self.w
        # the input data must be transposed
        if x.shape[0] != self.w.shape[1]:
            raise ValueError("The input data must have the same number of features as the SOM")

        # Check if the face color, line width and edge color are 1D arrays
        if face_labels.ndim != 1 or edge_width.ndim != 1 or edge_labels.ndim != 1:
            raise ValueError("fcolor, lwidth and ecolor must be 1D arrays")

        # Check if the length of fcolor, lwidth and ecolor are equal to the number of neurons
        if len(face_labels) != numNeurons or len(edge_width) != numNeurons or len(edge_labels) != numNeurons:
            raise ValueError("The length of x, fcolor, lwidth and ecolor must be equal to the number of neurons")

        # Make hit histogram
        fig, ax, patches, text = self.hit_hist(x, True, mouse_click, **kwargs)

        # Exclude nan values for the unique color count
        unique_fcolor = np.unique(face_labels[~np.isnan(face_labels)])
        unique_ecolor = np.unique(edge_labels[~np.isnan(edge_labels)])

        # Create the colormaps
        cmap1 = plt.get_cmap('jet', len(unique_fcolor))
        cmap2 = plt.get_cmap('cool', len(unique_ecolor))

        for neuron in range(numNeurons):
            if not np.isnan(face_labels[neuron]):
                # Normalize the class label to the colormap index
                color1_idx = np.argwhere(unique_fcolor == face_labels[neuron])[0][0] / (len(unique_fcolor) - 1)
                color2_idx = np.argwhere(unique_ecolor == edge_labels[neuron])[0][0] / (len(unique_ecolor) - 1)

                # Get the corresponding color from the colormap
                patches[neuron][0]._facecolor = cmap1(color1_idx)
                patches[neuron][0]._linewidth = edge_width[neuron]
                patches[neuron][0]._edgecolor = cmap2(color2_idx)

        # Get rid of extra white space on sides
        plt.tight_layout()

        return fig, ax, patches, text

    def plt_nc(self, mouse_click=False, connect_pick_event=True, **kwargs):
        """ Generates neighborhood connection map.
        The gray hexagons represent cluster centers.

        Args:
            mouse_click: bool
                If true, the interactive plot and sub-clustering functionalities to be activated
            connect_pick_event: bool
                If true, the pick event is connected to the plot
            kwarg: dict
                Additional arguments to be passed to the onpick function
                Possible keys include:
                    'data', 'clust', 'target', 'num1', 'num2', 'cat', 'align', 'height' and 'topn'

        Returns:
            fig, ax, pathces
        """
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
        fig, ax = plt.subplots(figsize=(8, 8), frameon=False)
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
        hexagons = []
        for i in range(numNeurons):
            hex, = ax.fill(pos[0, i] + shapex, pos[1, i] + shapey, facecolor=(0.4, 0.4, 0.6), edgecolor=(0.8, 0.8, 0.8),
                           picker=True)
            hexagons.append(hex)

        # Assign the cluster number for each hexagon
        hexagon_to_neuron = {hex: neuron for neuron, hex in enumerate(hexagons)}

        if mouse_click and connect_pick_event:
            fig.canvas.mpl_connect(
                'pick_event', lambda event: self.onpick(event, hexagons, hexagon_to_neuron, **kwargs)
            )

        return fig, ax, patches

    def neuron_dist_plot(self, mouse_click=False, connect_pick_event=True, **kwargs):
        """ Generates distance map.
        The gray hexagons represent cluster centers.
        The colors of the elongated hexagons between the cluster centers represent the distance between the centers.
        The darker the color the larget the distance.

        Args:
            mouse_click: bool
                If true, the interactive plot and sub-clustering functionalities to be activated
            connect_pick_event: bool
                If true, the pick event is connected to the plot
            kwarg: dict
                Additional arguments to be passed to the onpick function
                Possible keys include:
                    'data', 'clust', 'target', 'num1', 'num2', 'cat', 'align', 'height' and 'topn'
        Returns:
            fig, ax, pathces, text
        """

        pos = self.pos
        numNeurons = self.numNeurons

        # Determine the shape of the hexagon to represent each cluster
        symmetry = 6
        z = np.sqrt(0.75)/3
        shapex = np.array([-1, 0, 1, 1, 0, -1]) * 0.5
        shapey = np.array([1, 2, 1, -1, -2, -1]) * z
        edgex = np.array([-1, 0, 1, 0]) * 0.5
        edgey = np.array([0, 1, 0, - 1]) * z
        shapex = shapex * 0.3
        shapey = shapey * 0.3

        # Set up edges
        neighbors = np.zeros((numNeurons, numNeurons))
        neighbors[self.neuron_dist <= 1.001] = 1.0
        neighbors = np.tril(neighbors - np.identity(numNeurons))

        # Get the figure, remove the frame, and find the limits
        # of the axis that will fit hexagons
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.set_axis_off()
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
            for j in np.where(neighbors[:, i] == 1.0)[0]:
                pdiff = pos[:, j]-pos[:, i]
                angle = np.arctan2(pdiff[1], pdiff[0])
                ex, ey = rotate_xy(edgex, edgey, angle)
                edgePos = (pos[:, i] + pos[:, j]) * 0.5
                p1 = (2 * pos[:, i] + 1 * pos[:, j]) / 3
                p2 = (1 * pos[:, i] + 2 * pos[:, j]) / 3
                temp = plt.fill(edgePos[0]+ex, edgePos[1]+ey, facecolor=np.random.rand(1,3), edgecolor='none')
                patches.append(temp)
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=[1, 0, 0])

        # Setup neurons. Place gray hexagon at neuron locations.
        hexagons = []
        for i in range(numNeurons):
            hex, = ax.fill(pos[0, i] + shapex, pos[1, i] + shapey, facecolor=(0.4, 0.4, 0.6),
                           edgecolor=(0.8, 0.8, 0.8), picker=True)
            hexagons.append(hex)

        # Assign the cluster number for each hexagon
        hexagon_to_neuron = {hex: neuron for neuron, hex in enumerate(hexagons)}

        # Find the distance between neighboring weights.
        weights = self.w
        levels = np.zeros(numEdges)
        k = 0
        for i in range(numNeurons):
            for j in np.where(neighbors[:, i] == 1.0)[0]:
                levels[k] = np.sqrt(np.sum((weights[i, :] - weights[j, :]) ** 2))
                k = k + 1
        mn = np.amin(levels)
        mx = np.amax(levels)
        if mx == mn:
            levels = np.zeros(1, numEdges) + 0.5
        else:
            levels = (levels - mn)/(mx - mn)

        # Make the face color black for the maximum distance and
        # yellow for the minimum distance. The middle distance  will
        # be red.
        k = 0
        for i in range(numNeurons):
            for j in np.where(neighbors[:, i] == 1.0)[0]:
                level = 1 - levels[k]
                red = np.amin([level * 2, 1])
                green = np.amax([level * 2 - 1, 0])
                c = (red, green, 0, 1.0)
                patches[k][0]._facecolor = c
                k = k + 1

        # Mouse Click Functionality
        if mouse_click and connect_pick_event:
            fig.canvas.mpl_connect(
                'pick_event', lambda event: self.onpick(event, hexagons, hexagon_to_neuron, **kwargs)
            )

        return fig, ax, patches

    def simple_grid(self, avg, sizes, mouse_click=False, connect_pick_event=True, **kwargs):
        """ Basic hexagon grid plot
        Colors are selected from avg array.
        Sizes of inner hexagons are selected rom sizes array.

        Args:
            avg: array-like
                Average values for each neuron
            sizes: array-like
                Sizes of inner hexagons
            mouse_click: bool
                If true, the interactive plot and sub-clustering functionalities to be activated
            connect_pick_event: bool
                If true, the pick event is connected to the plot
            kwarg: dict
                Additional arguments to be passed to the onpick function
                Possible keys include:
                    'data', 'clust', 'target', 'num1', 'num2', 'cat', 'align', 'height' and 'topn'

        Returns:
            fig, ax, pathces, cbar
        """

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

        # Draw the outer hexagons
        hexagons = []
        for i in range(numNeurons):
            hex, = ax.fill(pos[0, i] + shapex, pos[1, i] + shapey, facecolor=(0.4, 0.4, 0.6), edgecolor=(0.8, 0.8, 0.8),
                           picker=True)
            hexagons.append(hex)

        # Assign the cluster number for each hexagon
        hexagon_to_neuron = {hex: neuron for neuron, hex in enumerate(hexagons)}

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
            xx[neuron] = (avg[neuron] - dmin) / drange
            color = cmap(xx[neuron])
            patches[neuron][0]._facecolor = color

        plt.tight_layout()

        # # Add a color bar the the figure to indicate levels
        # # create an axes on the right side of ax. The width of cax will be 5%
        # # of ax and the padding between cax and ax will be fixed at 0.05 inch.

        cax = cm.ScalarMappable(cmap=cmap)
        cax.set_array(xx)
        # cbar = fig.colorbar(cax)
        cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)

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

        if mouse_click and connect_pick_event:
            fig.canvas.mpl_connect(
                'pick_event', lambda event: self.onpick(event, hexagons, hexagon_to_neuron, **kwargs)
            )

        plt.tight_layout()

        return fig, ax, patches, cbar

    def setup_axes(self):
        # Setup figure, axes and sub-axes for plots
        pos = self.pos
        numNeurons = self.numNeurons

        # Determine the hexagon shape
        shapex, shapey = get_hexagon_shape()
        shminx = np.min(shapex)
        shmaxx = np.max(shapex)
        shminy = np.min(shapey)
        shmaxy = np.max(shapey)

        # Create the figure and get the transformations from data
        # to pixel and from pixel to axes.
        fig, ax = plt.subplots(frameon=False,
                               figsize=(8, 8),
                               layout='constrained')

        # Set the main axes properties
        xmin = np.min(pos[0]) + np.min(shapex)
        xmax = np.max(pos[0]) + np.max(shapex)
        ymin = np.min(pos[1]) + np.min(shapey)
        ymax = np.max(pos[1]) + np.max(shapey)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax + 0.5])
        ax.axis('off')
        ax.set_aspect('equal')

        # Draw hexagon
        hexagons = []
        for neuron in range(numNeurons):
            hex, = ax.fill(pos[0, neuron] + shapex, pos[1, neuron] + shapey,
                           facecolor=(1, 1, 1), edgecolor=(0.8, 0.8, 0.8), picker=True)
            hexagons.append(hex)

        # Assign the cluster number for each hexagon
        hexagon_to_neuron = {hex: neuron for neuron, hex in enumerate(hexagons)}

        # Loop over to create sub-axe in each cluster
        h_axes = [0] * numNeurons   # A container for sub-axes

        for neuron in range(numNeurons):
            # Find the size of the cell in data units
            minx = pos[0, neuron] + shminx
            maxx = pos[0, neuron] + shmaxx
            miny = pos[1, neuron] + shminy
            maxy = pos[1, neuron] + shmaxy

            # Convert the size of the cell to axes units
            minxyDis = ax.transData.transform([minx, miny])
            maxxyDis = ax.transData.transform([maxx, maxy])
            minxyAx = ax.transAxes.inverted().transform(minxyDis)
            maxxyAx = ax.transAxes.inverted().transform(maxxyDis)

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

            # Create sub-axes
            h_axes[neuron] = inset_axes(ax, width='100%', height='100%', loc=3,
                                        bbox_to_anchor=(x0, y0, width, height),
                                        bbox_transform=ax.transAxes, borderpad=0)
            h_axes[neuron].set(xticks=[], yticks=[])
            h_axes[neuron].set_frame_on(False)

        return fig, ax, h_axes, hexagons, hexagon_to_neuron

    def plt_stem(self, x, y, mouse_click=False, connect_pick_event=True, **kwargs):
        """ Generate stem plot for each neuron.

        Args:
            x: array-like
                The x-axis values (align)
            y: array-like or sequence of vectors
                The y-axis values (height)
            mouse_click: bool
                If true, the interactive plot and sub-clustering functionalities to be activated
            connect_pick_event: bool
                If true, the pick event is connected to the plot
            kwarg: dict
                Additional arguments to be passed to the onpick function
                Possible keys include:
                    'data', 'clust', 'target', 'num1', 'num2', 'cat', 'align', 'height' and 'topn'

        Returns:
            fig, ax, h_axes
        """

        numNeurons = self.numNeurons

        # Setup figure, axes, and sub-axes
        fig, ax, h_axes, hexagons, hexagon_to_neuron = self.setup_axes()

        # Draw stem plot
        for neuron in range(numNeurons):
            # Make graph
            h_axes[neuron].stem(x, y[neuron])

        if mouse_click and connect_pick_event:
            kwargs['align'] = x
            kwargs['height'] = y
            fig.canvas.mpl_connect(
                'pick_event', lambda event: self.onpick(event, hexagons, hexagon_to_neuron, **kwargs)
            )

        return fig, ax, h_axes

    def plt_wgts(self, mouse_click=False, connect_pick_event=True, **kwargs):
        """ Generate line plot for each neuron.

        Args:
            mouse_click: bool
                If true, the interactive plot and sub-clustering functionalities to be activated
            connect_pick_event: bool
                If true, the pick event is connected to the plot
            kwarg: dict
                Additional arguments to be passed to the onpick function
                Possible keys include:
                    'data', 'clust', 'target', 'num1', 'num2', 'cat', 'align', 'height' and 'topn'

        Returns:
            fig, ax, h_axes
        """

        numNeurons = self.numNeurons
        w = self.w

        # Setup figure, main axes, and sub-axes
        fig, ax, h_axes, hexagons, hexagon_to_neuron = self.setup_axes()

        # Draw line plots
        for neuron in range(numNeurons):
            # Make graph
            h_axes[neuron].plot(w[neuron])

        if mouse_click and connect_pick_event:
            fig.canvas.mpl_connect(
                'pick_event', lambda event: self.onpick(event, hexagons, hexagon_to_neuron, **kwargs)
            )

        return fig, ax, h_axes

    def plt_pie(self, x, s=None, mouse_click=False, connect_pick_event=True, **kwargs):
        """ Generate pie plot for each neuron.

        Args:
            x: Array or sequence of vectors.
                The wedge size
            s: 1-D array-like, optional
                Scale the size of the pie chart according to the percent of the specific class in that cell. (0-100)
                (default: None)
            mouse_click: bool
                If true, the interactive plot and sub-clustering functionalities to be activated
            connect_pick_event: bool
                If true, the pick event is connected to the plot
            kwarg: dict
                Additional arguments to be passed to the onpick function
                Possible keys include:
                    'data', 'clust', 'target', 'num1', 'num2', 'cat', 'align', 'height' and 'topn'

        Returns:
            fig, ax, h_axes
        """
        # Validate the length of x (array or sequence of vectors)
        if len(x) != self.numNeurons:
            raise ValueError("The length of x must be equal to the number of neurons.")

        # Validate perc values
        if s is not None:
            s = np.array(s)
            if np.any(s < 0) or np.any(s > 100):
                raise ValueError("Percentage values must be between 0 and 100.")

        # Validate the length of perc
        if s is not None and len(s) != self.numNeurons:
            raise ValueError("The length of s must be equal to the number of neurons.")

        numNeurons = self.numNeurons

        # Determine the number of colors needed
        shapclust = x.shape
        num_colors = shapclust[1]

        # Generate a color list using a colormap
        cmap = cm.get_cmap('plasma', num_colors)  # Use any suitable
        clrs = [cmap(i) for i in range(num_colors)]

        # Setup figure, main axes, and sub-axes
        fig, ax, h_axes, hexagons, hexagon_to_neuron = self.setup_axes()

        # Draw pie plot in each neuron
        for neuron in range(numNeurons):
            # Determine the scale of the pie chart
            if s is None:
                scale = 1
            else:
                scale = np.sqrt(s[neuron] / 100)
                scale = max(scale, 0.01)  # Ensure minimum scale

            # Make pie chart
            if np.sum(x[neuron]) != 0:
                h_axes[neuron].pie(x[neuron], colors=clrs, radius=scale)
            else:
                h_axes[neuron] = None

        if mouse_click and connect_pick_event:
            kwargs['cat'] = x
            fig.canvas.mpl_connect(
                'pick_event', lambda event: self.onpick(event, hexagons, hexagon_to_neuron, **kwargs)
            )

        return fig, ax, h_axes

    def plt_histogram(self, x, mouse_click=False, connect_pick_event=True, **kwargs):
        """ Generate histogram for each neuron.

        Args:
            x: array-like
                The input data to be plotted in histogram
            mouse_click: bool
                If true, the interactive plot and sub-clustering functionalities to be activated
            connect_pick_event: bool
                If true, the pick event is connected to the plot
            kwarg: dict
                Additional arguments to be passed to the onpick function
                Possible keys include:
                    'data', 'clust', 'target', 'num1', 'num2', 'cat', 'align', 'height' and 'topn'

        Returns:
            fig, ax, h_axes
        """

        numNeurons = self.numNeurons

        # Setup figure, main axes, and sub-axes
        fig, ax, h_axes, hexagons, hexagon_to_neuron = self.setup_axes()

        # Draw histogram
        for neuron in range(numNeurons):
            if len(x[neuron]) > 0:
                # Make graph
                h_axes[neuron].hist(x[neuron])

                # Enable the axes for this histogram
                h_axes[neuron].set_frame_on(True)
                h_axes[neuron].tick_params(axis='both', which='both', length=5)  # Show tick marks
                h_axes[neuron].set(xticks=[], yticks=[])

                # Show only the left and bottom spines
                h_axes[neuron].spines['top'].set_visible(False)
                h_axes[neuron].spines['right'].set_visible(False)
                h_axes[neuron].spines['left'].set_visible(True)
                h_axes[neuron].spines['bottom'].set_visible(True)
            else:
                h_axes[neuron] = None

        if mouse_click and connect_pick_event:
            kwargs['num1'] = x
            fig.canvas.mpl_connect(
                'pick_event', lambda event: self.onpick(event, hexagons, hexagon_to_neuron, **kwargs)
            )

        return fig, ax, h_axes

    def plt_boxplot(self, x, mouse_click=False, connect_pick_event=True, **kwargs):
        """ Generate box plot for each neuron.

        Args:
            x: array-like
                The input data to be plotted in box plot
            mouse_click: bool
                If true, the interactive plot and sub-clustering functionalities to be activated
            connect_pick_event: bool
                If true, the pick event is connected to the plot
            kwarg: dict
                Additional arguments to be passed to the onpick function
                Possible keys include:
                    'data', 'clust', 'target', 'num1', 'num2', 'cat', 'align', 'height' and 'topn'

        Returns:
            fig, ax, h_axes
        """

        numNeurons = self.numNeurons

        # Setup figure, main axes, and sub-axes
        fig, ax, h_axes, hexagons, hexagon_to_neuron = self.setup_axes()

        # Find global min and max across all neuron's data
        global_min, global_max = get_global_min_max(x)

        for neuron in range(numNeurons):
            if len(x[neuron]) > 0:
                # Make graph
                h_axes[neuron].boxplot(x[neuron])

                # Set the same y axis limits for all subplots
                h_axes[neuron].set_ylim(global_min, global_max)
                # h_axes[neuron].set_yticks(np.linspace(global_min, global_max, 5))
            else:
                h_axes[neuron] = None

        if mouse_click and connect_pick_event:
            kwargs['num1'] = x
            fig.canvas.mpl_connect(
                'pick_event', lambda event: self.onpick(event, hexagons, hexagon_to_neuron, **kwargs)
            )

        return fig, ax, h_axes

    def plt_violin_plot(self, x, mouse_click=False, connect_pick_event=True, **kwargs):
        """ Generate violin plot for each neuron.

        Args:
            x: array-like
                The input data to be plotted in violin plot
            mouse_click: bool
                If true, the interactive plot and sub-clustering functionalities to be activated
            connect_pick_event: bool
                If true, the pick event is connected to the plot
            kwarg: dict
                Additional arguments to be passed to the onpick function
                Possible keys include:
                    'data', 'clust', 'target', 'num1', 'num2', 'cat', 'align', 'height' and 'topn'

        Returns:
            fig, ax, h_axes
        """

        numNeurons = self.numNeurons

        # Setup figure, main axes, and sub-axes
        fig, ax, h_axes, hexagons, hexagon_to_neuron = self.setup_axes()

        # Find global min and max across all neuron's data
        global_min, global_max = get_global_min_max(x)

        for neuron in range(numNeurons):
            if len(x[neuron]) > 0:
                # Make graph on the appropriate sub-axes
                h_axes[neuron].violinplot(x[neuron])

                # Set the same y axis limits for all subplots
                h_axes[neuron].set_ylim(global_min, global_max)
                # h_axes[neuron].set_yticks(np.linspace(global_min, global_max, 5))
            else:
                h_axes[neuron] = None

        if mouse_click and connect_pick_event:
            kwargs['num1'] = x
            fig.canvas.mpl_connect(
                'pick_event', lambda event: self.onpick(event, hexagons, hexagon_to_neuron, **kwargs)
            )

        return fig, ax, h_axes

    def multiplot(self, plot_type, *args):
        # Dictionary mapping plot types to corresponding plotting methods
        plot_functions = {
            'wgts' : self.plt_wgts,
            'pie': self.plt_pie,
            'stem': self.plt_stem,
            'hist': self.plt_histogram,
            'box': self.plt_boxplot,
            'violin': self.plt_violin_plot,
            'scatter': self.plt_scatter
        }

        selected_plot = plot_functions.get(plot_type)
        if selected_plot:
            return selected_plot(*args)
        else:
            raise ValueError("Invalid function type")

    def plt_scatter(self, x, y, reg_line=True, mouse_click=False, connect_pick_event=True, **kwargs):
        """ Generate Scatter Plot for Each Neuron.

        Args:
            x: input data
            indices: array-like indices e.g. (0, 1) or [0, 1]
            clust: list of indices of input data for each cluster
            reg_line: Flag

        Returns:
            fig, ax, h_axes
        """
        pos = self.pos
        numNeurons = self.numNeurons

        # Setup figure, main axes, and sub-axes
        fig, ax, h_axes, hexagons, hexagon_to_neuron = self.setup_axes()

        # Loop over each neuron for hexagons and scatter plots
        for neuron in range(numNeurons):
            # Make Scatter Plot for each neuron
            if len(x[neuron]) > 0 and len(y[neuron]) > 0:
                h_axes[neuron].scatter(x[neuron], y[neuron], s=1, c='k')

                if reg_line:
                    m, p = np.polyfit(x[neuron], y[neuron], 1)
                    h_axes[neuron].plot(x[neuron], m * x[neuron] + p, c='r', linewidth=1)

            else:
                h_axes[neuron] = None

        if mouse_click and connect_pick_event:
            kwargs['num1'] = x
            kwargs['num2'] = y
            fig.canvas.mpl_connect(
                'pick_event', lambda event: self.onpick(event, hexagons, hexagon_to_neuron, **kwargs)
            )

        return fig, ax, h_axes

    def component_positions(self, X_scaled):
        """
        Plots the SOM weight vectors, the Iris dataset input vectors, and connects neighboring neurons.

        Parameters:
        - som: A trained SOM instance with attributes 'w' for weight vectors and 'dimensions' for grid dimensions.
        - X_scaled: The normalized Iris dataset input vectors.
        """
        # Extract the trained weight vectors and the SOM grid dimensions
        weight_vectors = self.w
        grid_x, grid_y = self.dimensions

        # Plot the SOM weight vectors as gray dots
        plt.scatter(weight_vectors[:, 0], weight_vectors[:, 1], color='gray', s=50, label='Weight Vectors')
        # for i, vec in enumerate(weight_vectors):
        #     plt.annotate(str(i), (vec[0], vec[1]), textcoords="offset points", xytext=(0,5), ha='center')

        # Plot the Iris data points as blue dots
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], color='green', s=20, label='Input Vectors', alpha=0.5)

        # Draw red lines to connect neighboring neurons
        for i in range(grid_x):
            for j in range(grid_y):
                index = i * grid_y + j  # Calculate the linear index of the neuron in the SOM
                neuron = weight_vectors[index]

                # Connect to the right neighbor if it exists
                if j < grid_y - 1:
                    right_index = i * grid_y + (j + 1)
                    right_neighbor = weight_vectors[right_index]
                    plt.plot([neuron[0], right_neighbor[0]], [neuron[1], right_neighbor[1]], color='red')

                # Connect to the bottom neighbor if it exists
                if i < grid_x - 1:
                    bottom_index = (i + 1) * grid_y + j
                    bottom_neighbor = weight_vectors[bottom_index]
                    plt.plot([neuron[0], bottom_neighbor[0]], [neuron[1], bottom_neighbor[1]], color='red')

        # Set labels and legend
        plt.xlabel('Weight 1')
        plt.ylabel('Weight 2')
        plt.title('SOM Weight Positions')
        plt.legend()
        plt.grid(False)
        plt.show()

    def component_planes(self, X):
        w = self.w
        pos = self.pos
        numNeurons = self.numNeurons
        z = np.sqrt(0.75)
        shapex = np.array([-1, 0, 1, 1, 0, -1]) * 0.5
        shapey = np.array([1, 2, 1, -1, -2, -1]) * (z / 3)

        num_features = X.shape[0]
        grid_size = int(np.ceil(np.sqrt(num_features)))  # Calculate grid size

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

        for i, ax in enumerate(axes.flatten()):
            if i < num_features:
                ax.axis('equal')
                xmin = np.min(pos[0]) + np.min(shapex)
                xmax = np.max(pos[0]) + np.max(shapex)
                ymin = np.min(pos[1]) + np.min(shapey)
                ymax = np.max(pos[1]) + np.max(shapey)
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])

                # Get the weights for the current feature
                feature_weights = w[:, i]

                # Normalize the weights to range between 0 and 1
                norm = mcolors.Normalize(vmin=np.min(feature_weights), vmax=np.max(feature_weights))

                for j in range(numNeurons):
                    color = plt.cm.viridis(norm(feature_weights[j]))  # Choose colormap as viridis
                    inverted_color = tuple(
                        1 - np.array(color[:3]))  # Invert the color to make darker colors represent larger weights
                    ax.fill(pos[0, j] + shapex, pos[1, j] + shapey, facecolor=inverted_color, edgecolor=(0.8, 0.8, 0.8))

        plt.show()

    # Generic Plot Function
    def plot(self, plot_type, data_dict=None, ind=None, target_class=None, use_add_1darray=False,
             use_add_2darray=False, **kwargs):
        """ Generic Plot Function.
        It generates a plot based on the plot type and data provides.

        Parameters
        ----------
        plot_type : str
            The type of plot to be generated:
            ["top", "top_num", "hit_hist", "gray_hist",
            "color_hist", "complex_hist", "nc", "neuron_dist",
            "simple_grid", "stem", "pie", "wgts", "pie", "hist",
            "box", "violin", "scatter", "component_positions",
            "component_planes"]

        data_dict: dict (optional)
            A dictionary containing the data to be plotted.
            The key is prefixed with the data type and the value is the data itself.
            {"original_data", "input_data", "target", "clust","cat_1darray", "cat_2darray",
            "num_1darray"}

        ind : int, str or array-like (optional)
            The indices of the data to be plotted.

        target_class: int (optional)
            The target class to be plotted.

        use_add_1darray: bool (optional)
            If true, the additional array to be used.

        use_add_2darray: bool (optional)
            If true, the additional 2D array to be used.

        **kwargs : dict
            Additional arguments to be passed to the interactive plot function.
        """
        # Plot Types allowed
        plot_types = ["top", "top_num",
                      "hit_hist", "gray_hist", "color_hist", "complex_hist",
                      "neuron_connection", "neuron_dist",
                      "simple_grid",
                      "stem", "pie", "wgts", "hist", "box", "violin", "scatter",
                      "component_positions", "component_planes"]

        # Validate the plot type
        if plot_type not in plot_types:
            raise ValueError(f"Invalid plot type: {plot_type}")

        # Validate the data_dict
        if data_dict is None and plot_type not in ["top", "top_num",
                                                   "neuron_connection",
                                                   "neuron_dist", "wgts"]:
            raise ValueError("data_dict is required for this plot type.")

        # Validate the plot function
        plot_functions = {
            "top": self.plt_top,
            "top_num": self.plt_top_num,
            "hit_hist": self.hit_hist,
            "gray_hist": self.gray_hist,
            "color_hist": self.color_hist,
            "complex_hist": self.custom_cmplx_hit_hist,
            "neuron_connection": self.plt_nc,
            "neuron_dist": self.neuron_dist_plot,
            "simple_grid": self.simple_grid,
            "stem": self.plt_stem,
            "pie": self.plt_pie,
            "wgts": self.plt_wgts,
            "hist": self.plt_histogram,
            "box": self.plt_boxplot,
            "violin": self.plt_violin_plot,
            "scatter": self.plt_scatter,
            "component_positions": self.component_positions,
            "component_planes": self.component_planes
        }

        # Assign the plot function
        selected_plot = plot_functions.get(plot_type)

        # Error Handling if the plot function recieve the appropriate arguments
        def validate_data_dict(keys):
            for key in keys:
                if key not in data_dict:
                    raise ValueError(f"{key} is required for this plot type.")

        # ======== Topology, Neuron Connection, Neuron Distance, and Weight Plot ==========
        if plot_type in ["top", "top_num", "neuron_connection", "neuron_dist", "wgts"]:
            # Call the plot function
            return selected_plot(**kwargs)

        # ======== Components Plane Family ==========
        # Need to be modified
        elif plot_type in ['component_positions', 'component_planes']:
            # Error Handling if the data_dict have the scaled input data X
            validate_data_dict(["input_data"])
            # Data Preparation
            x_scaled = data_dict['input_data']
            # Invoke Function
            return selected_plot(x_scaled)

        # =====================  Hit Histogram Family =====================
        elif plot_type in ['hit_hist', 'gray_hist', 'color_hist', 'complex_hist']:
            # Validate the data_dict have the scaled input data
            validate_data_dict(["input_data"])

            # Extract input data
            x = data_dict['input_data']

            if plot_type in ['hit_hist']:
                # Invoke the function
                return selected_plot(x, True, **kwargs)

            elif plot_type in ['gray_hist']:
                if use_add_1darray:  # Check if user want to use additional 1D array

                    # Validate the data_dict have the additional 1D array and original data
                    validate_data_dict(["cat_1darray", "original_data"])

                    # Validate the length of additional 1D array is equal to number of neuron or length of original data.
                    if len(data_dict['cat_1darray']) != self.numNeurons and len(data_dict['cat_1darray']) != len(
                            data_dict['original_data']):
                        raise ValueError(
                            "The additional 1D array must have the same length as the clust data or original data.")

                    # If the length of additional 1D array is equal to the length of clust data just extract data
                    elif len(data_dict['cat_1darray']) == self.numNeurons:
                        perc = data_dict['cat_1darray']

                    # If the length of additional 1D array is equal to the length of original data then preprocess the data
                    elif len(data_dict['cat_1darray']) == len(data_dict['original_data']):

                        # Validate the data_dict have the clust data
                        validate_data_dict(["clust"])

                        if target_class is None:
                            raise ValueError("The target class is required. Eg. target_class = 1")

                        # Preprocess and extract the data
                        perc = get_perc_cluster(data_dict['cat_1darray'], target_class, data_dict['clust'])

                else:
                    # Error Handling if the target data not provided
                    validate_data_dict(["target", "clust"])

                    # Validate the data_dict have the clust data
                    if target_class is None:
                        raise ValueError("The target class is required. Eg. target_class = 1")

                    perc = get_perc_cluster(data_dict['target'], target_class, data_dict['clust'])

                # Invoke the gray hist function
                return selected_plot(x, perc, **kwargs)

            elif plot_type in ['color_hist']:
                # Check if user want to use additional 1D array
                if use_add_1darray:
                    # Check if the additional 1D array and original data are provided
                    validate_data_dict(["num_1darray", "original_data"])

                    num_feature = data_dict['num_1darray']
                    original_data_size = len(data_dict['original_data'])

                    if len(num_feature) != original_data_size and len(num_feature) != self.numNeurons:
                        raise ValueError(
                            "The additional 1D array must have the same length as the clust data or original data.")

                    elif len(num_feature) == original_data_size:
                        # Check if the clust data is provided
                        validate_data_dict(["clust"])

                        clust = data_dict['clust']
                        avg = get_cluster_avg(num_feature, clust)

                    elif len(num_feature) == self.numNeurons:

                        avg = num_feature

                else:
                    # Error Handling if the original data and clust are provided
                    validate_data_dict(["original_data", "clust"])

                    if ind is None:
                        raise ValueError(
                            "The indices is required for this plot type. Which numerical feature in the original data you want to plot? Eg. ind = 0")

                    feature = data_dict['original_data'][:, ind]

                    avg = get_cluster_avg(feature, data_dict['clust'])

                return selected_plot(x, avg, **kwargs)

            elif plot_type in ['complex_hist']:
                if use_add_2darray:
                    # Validate the data_dict have the additional 2D array
                    validate_data_dict(["cat_2darray"])

                    cat_2darray = data_dict['cat_2darray']

                    if cat_2darray.shape[0] != self.numNeurons:
                        raise ValueError(
                            "The additional 2D array must have the same length as the number of neurons.")

                    # Validate the additional 2D array have 3 features
                    if cat_2darray.shape[1] != 3:
                        raise ValueError(
                            "The additional 2D array must have 3 features. E.g. [numNeurons, [face_labels, edge_labels, edge_widths]]")

                    # Extract Data
                    face_labels = cat_2darray[:, 0]
                    edge_labels = cat_2darray[:, 1]
                    edge_widths = cat_2darray[:, 2]

                else:
                    # Error Handling if the target and clust data not provided
                    validate_data_dict(["target", "clust"])
                    # Error Handling if the target class not provided
                    if target_class is None:
                        raise ValueError("The target class is required")

                    # Generate Data
                    target = data_dict['target']
                    clust = data_dict['clust']

                    face_labels = majority_class_cluster(target, clust)
                    edge_labels = closest_class_cluster(target, clust)
                    target_ind = np.where(target == target_class)[0]
                    edge_widths = get_edge_widths(target_ind, clust)

                return selected_plot(x, face_labels, edge_labels, edge_widths, **kwargs)

        # ===================== Simple Grid =====================
        elif plot_type in ['simple_grid']:
            # Validate the data_dict have the original data
            validate_data_dict(["original_data"])

            original_data = data_dict['original_data']

            # Check if there're additional variables
            if use_add_1darray:
                # Error Handling if the additional 1D arrays and clust not provided
                validate_data_dict(["cat_1darray", "num_1darray", "clust"])

                cat_feature = data_dict['cat_1darray']
                num_feature = data_dict['num_1darray']
                clust = data_dict['clust']

                # ================================
                # Extract avg from num_1darray
                # potential input of num_1darray
                # 1. the specific column of the original data
                # 2. the pre-processes 1-d array with average value for each cluster
                # ================================
                # Validate the length of additional 1D array is equal to the number of neuron or length of original data.
                if len(num_feature) != self.numNeurons and len(num_feature) != len(original_data):
                    raise ValueError(
                        "The additional 1D array must have the same length as the clust data or original data.")
                elif len(num_feature) == self.numNeurons:
                    avg = num_feature
                elif len(num_feature) == len(original_data):
                    avg = get_cluster_avg(num_feature, clust)

                # ===============================
                # Extract sizes from cat_1darray
                # potential input of cat_1darray
                # 1. The specific feature of the original data (e.g. the specific column) <- it'requres target class
                # 2. The specific feature of the pre-processes 1-d array with magnitude for each cluster (e.g. percentage)
                # ===============================
                if len(cat_feature) != self.numNeurons and len(cat_feature) != len(original_data):
                    raise ValueError(
                        "The additional 1D array must have the same length as the clust data or original data.")

                elif len(cat_feature) == self.numNeurons:
                    sizes = cat_feature

                elif len(cat_feature) == len(original_data):
                    if target_class is None:
                        raise ValueError("The target class is required")
                    sizes = get_perc_cluster(cat_feature, target_class, clust)

            else:
                # Error Handling if the target and clust data not provided
                validate_data_dict(["clust", "target"])

                if ind is None:
                    raise ValueError("The indices is required for this plot type.")

                if target_class is None:
                    raise ValueError("The target class is required")

                clust = data_dict['clust']

                # Extract avg from the original data
                num_feature = original_data[:, ind]
                avg = get_cluster_avg(num_feature, clust)

                # Extract size from the target
                target = data_dict['target']
                sizes = get_perc_cluster(target, target_class, clust)

            return selected_plot(avg, sizes, **kwargs)

        # ===================== Basic Plot Family =====================
        elif plot_type in ['stem', 'pie']:
            # Error Handling if the clust data not provided
            validate_data_dict(["target", "clust"])

            # Extract Information
            clust = data_dict['clust']

            # If the user wanna plot input data
            if use_add_2darray:
                # Error Handling if the additional 2D array not provided
                validate_data_dict(["cat_2darray"])

                # Error Handling if additional vategorical variable has correct length
                if len(data_dict["cat_2darray"]) != len(clust):
                    raise ValueError("The additional categorical data must have the same length as the clust data.")

                #  Get Additional Data
                sizes = data_dict['cat_2darray']

            else:
                target = data_dict['target']
                sizes = count_classes_in_cluster(target, clust)

            if plot_type == 'pie':

                # =============================================
                # It needs to handle scale (need to implement)
                # =============================================

                # Call the pie plot
                return selected_plot(sizes, **kwargs)

            elif plot_type == 'stem':
                # Extract Align
                if use_add_1darray:
                    align = [i for i in range(sizes.shape[1])]
                else:
                    align = [i for i in range(len(np.unique(target)))]
                # Call the stem plot
                return selected_plot(align, sizes, **kwargs)

        elif plot_type in ['hist']:
            # Error Handling if the index not provided
            if ind is None:
                raise ValueError("The indices is required for this plot type.")

            # Error Handling if the original data not provided
            validate_data_dict(["original_data", "clust"])

            # Extract the feature from the original data
            clust = data_dict['clust']
            feature = data_dict['original_data'][:, ind]

            x = get_cluster_array(feature, clust)

            return selected_plot(x, **kwargs)

        elif plot_type in ['scatter']:
            # Error Handling if the index not provided
            if ind is None:
                raise ValueError("The indices is required for this plot type.")

            if len(ind) != 2:
                raise ValueError("The indices must contain exactly two elements. Eg. [0, 1]")

            # Error Handling if the original data and clust not provided
            validate_data_dict(["original_data", "clust"])

            # Extract the feature from the original data
            x = data_dict['original_data'][:, ind[0]]
            y = data_dict['original_data'][:, ind[1]]
            clust = data_dict['clust']
            x = get_cluster_array(x, clust)
            y = get_cluster_array(y, clust)

            # Call the scatter plot function
            return selected_plot(x, y, **kwargs)

        elif plot_type in ['box', 'violin']:
            # Error Handling if the original data not provided
            validate_data_dict(["original_data", "clust"])

            # Extract the feature from the original data
            clust = data_dict['clust']

            if ind is None:  # Extract All data
                data = data_dict['original_data']
                x = get_cluster_data(data, clust)
            elif isinstance(ind, int):  # index just have 1 index
                data = data_dict['original_data'][:, ind]
                x = get_cluster_array(data, clust)
            elif isinstance(ind, (list, np.ndarray)):  # index have multiple indices
                data = data_dict['original_data'][:, ind]
                x = get_cluster_data(data, clust)

            # Call the box plot and violin function
            return selected_plot(x, **kwargs)

    # Interactive Functionality
    def onpick(self, event, hexagons, hexagon_to_neuron, **kwargs):
        """
        Interactive Plot Function
        Args:
            event: event
                a mouse click event
            hexagons: list
                a list of hexagons
            hexagon_to_neuron: dict
                a dictionary mapping hexagons to neurons
            **kwargs:

        Returns:
            None
        """
        if event.artist not in hexagons:
            return

        # Detect the clicked hexagon
        thishex = event.artist
        neuron_ind = hexagon_to_neuron[thishex]

        if len(kwargs["clust"][neuron_ind]) <= 0:
            print('No data in this cluster')
            return

        # Create a new window
        fig, ax1 = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(right=0.8)

        # Button Configuration
        button_types = self.determine_button_types(**kwargs)
        buttons = create_buttons(fig, button_types)

        # Set up button click events
        for button_type, button in buttons.items():
            button.on_clicked(lambda event, b=button_type:
                              self.button_click_event(b, ax1, neuron_ind, **kwargs))

        plt.show()

    def button_click_event(self, button_type, ax, neuron_ind, **kwargs):

        # Handle button click event by calling the appropriate plot function
        if button_type == 'pie':
            # Pre-process categorical variables
            sizes = kwargs['cat']
            sizes = sizes[neuron_ind][:kwargs['topn']]
            self.plot_pie(ax, sizes, neuron_ind)
        elif button_type == 'stem':
            self.plot_stem(ax, kwargs['align'], kwargs['height'], neuron_ind)

        elif button_type == 'hist':
            num1 = kwargs['num1'][neuron_ind][:kwargs['topn']]
            self.plot_hist(ax, num1, neuron_ind)

        elif button_type == 'box':
            # Pre-process continuous variables
            nums = []
            for key in kwargs:
                if key.startswith('num'):
                    nums.append(kwargs[key][neuron_ind][:kwargs['topn']])
            self.plot_box(ax, nums, neuron_ind)

        elif button_type == 'violin':
            # Pre-process continuous variables
            nums = []
            for key in kwargs:
                if key.startswith('num'):
                    nums.append(kwargs[key][neuron_ind][:kwargs['topn']])
            self.plot_violin(ax, nums, neuron_ind)

        elif button_type == 'scatter':
            # Pre-process continuous variables
            nums = []
            for key in kwargs:
                if key.startswith('num'):
                    nums.append(kwargs[key][neuron_ind][:kwargs['topn']])
            self.plot_scatter(ax, nums[0], nums[1], neuron_ind)

        elif button_type == 'sub_cluster':
            cluster_data = get_cluster_data(np.transpose(kwargs['data']), kwargs['clust'])
            sub_clust_data = cluster_data[neuron_ind]  # Get the data for the
            self.sub_clustering(sub_clust_data, neuron_ind)

        else:
            print(f"Unknown button type: {button_type}")

    def determine_button_types(self, **kwargs):
        button_types = []

        # Check for categorical data for pie charts
        if 'cat' in kwargs and kwargs['cat'] is not None:
            button_types.append('pie')

        # Check for alignment and height data for stem plots
        if 'align' in kwargs and 'height' in kwargs:
            button_types.append('stem')

        # Check for numerical data and decide which buttons to add
        num_keys = [key for key in kwargs if
                    key.startswith('num') and isinstance(kwargs[key], (list, np.ndarray)) and len(kwargs[key]) > 0]

        if num_keys:
            button_types.extend(['hist', 'box', 'violin'])

            # Add 'scatter' button only if there are at least two numerical columns
            if len(num_keys) >= 2:
                button_types.append('scatter')

        # Assuming sub-clustering is always an option
        button_types.append('sub_cluster')

        return button_types

    # Helper function to create charts
    def plot_pie(self, ax, data, neuronNum):
        """
        Helper function to plot pie chart in the interactive plots
        Args:
            ax:
            data:
            neuronNum:

        Returns:

        """
        # Clear the axes
        ax.clear()
        # Pie chart plot logic here
        # Determine the number of colors needed
        num_colors = len(data)
        cmap = cm.get_cmap('plasma', num_colors)
        clrs = [cmap(i) for i in range(num_colors)]
        ax.pie(data, colors=clrs, autopct='%1.1f%%')
        ax.set_title('Pie Chart inside the Cluster ' + str(neuronNum))
        # Redraw the figure
        ax.figure.canvas.draw_idle()

    def plot_stem(self, ax, align, height, neuronNum):
        # Clear the axes
        ax.clear()
        # Stem plot
        ax.stem(align[neuronNum], height[neuronNum])  # x: cat, y: data
        ax.set_title('Stem Plot inside the Clluster ' + str(neuronNum))
        # Redraw the figure
        ax.figure.canvas.draw_idle()

    def plot_hist(self, ax, data, neuronNum):
        """
        Helper function to plot histogram in the interactive plots
        Args:
            ax:
            data:
            neuronNum:

        Returns:

        """
        # Clear the axes
        ax.clear()
        # Histogram plot logic here
        ax.hist(data)
        ax.set_title('Histogram inside the Cluster ' + str(neuronNum))
        # Redraw the figure
        ax.figure.canvas.draw_idle()

    def plot_box(self, ax, data, neuronNum):
        # Clear the axes
        ax.clear()
        # Box plot logic here
        ax.boxplot(data)
        ax.set_title("Box plot in the Cluster " + str(neuronNum))
        # Redraw the figure
        ax.figure.canvas.draw_idle()

    def plot_violin(self, ax, data, neuronNum):
        # Clear the axes
        ax.clear()
        # Violin plot logic here
        ax.violinplot(data)
        ax.set_title('Violin Plot inside the Cluster ' + str(neuronNum))
        # Redraw the figure
        ax.figure.canvas.draw_idle()

    def plot_scatter(self, ax, num1, num2, neuronNum):
        """
        Helper function to display scatter plot in the interactive plots
        Args:
            ax:
            data:
            num1:
            num2:
            neuronNum:

        Returns:

        """
        # Clear the axes
        ax.clear()
        # Scatter plot logic here
        ax.scatter(num1, num2)
        ax.set_title('Scatter Plot inside the Cluster ' + str(neuronNum))
        # Redraw the figure
        ax.figure.canvas.draw_idle()

    def sub_clustering(self, data, neuron_ind):
        """
        Helper function for interactive function which create the sub-cluster
        Args:
            data:
            neuron_ind:

        Returns:

        """
        if len(data) <= 1:
            print("There is no enough data to create sub-cluster")
            return

        # Data Prep
        sub_x = np.transpose(data)

        if neuron_ind in self.sub_som:
            print('Sub clustering already done')
            sub_clust = self.sub_som[neuron_ind]
        else:
            # Training Sub Cluster
            sub_clust = SOMPlots((2, 2))
            sub_clust.init_w(sub_x)
            sub_clust.train(sub_x, 3, 500, 100)

            self.sub_som[neuron_ind] = sub_clust

        # Plot the sub cluster <- Can we h
        fig, ax, patches, text = sub_clust.hit_hist(sub_x, True, connect_pick_event=False)

        plt.show()
