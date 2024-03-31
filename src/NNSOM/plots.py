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

    def hit_hist(self, x, textFlag, mouse_click=False, connect_pick_event=True, **kwargs):
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

    def gray_hist(self, x, perc, mouse_click=False, connect_pick_event=True, **kwargs):
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

    def color_hist(self, x, avg, mouse_click=False, connect_pick_event=True, **kwargs):
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

        return fig, patches, text, cbar

    # Need to be generalized
    def cmplx_hit_hist(self, x, clust, perc, ind_missClass, ind21, ind12, mouse_click=False, connect_pick_event=True,
                       **kwargs):
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
                The x-axis values
            y: array-like
                The y-axis values
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
            h_axes[neuron].stem(x[neuron], y[neuron])

        title = 'dist plot'
        plt.suptitle(title, fontsize=16)

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

        title = 'Cluster Centers as Lines'
        plt.suptitle(title, fontsize=16)

        if mouse_click and connect_pick_event:
            fig.canvas.mpl_connect(
                'pick_event', lambda event: self.onpick(event, hexagons, hexagon_to_neuron, **kwargs)
            )

        return fig, ax, h_axes

    def plt_pie(self, title, perc, sizes_cluster, scaleFlag=False, mouse_click=False, connect_pick_event=True, **kwargs):
        """ Generate pie plot for each neuron.

        Args:
            title: str
                The title of the plot
            perc: array-like
                The percentage of a specific class in each cluster. It handle the scale of the pie plot.
            sizes_cluster: 2D array-like
                The size of classes in each cluster.
                It should be a 2D array with the shape of (numNeurons, numClasses)
            scaleFlag: bool
                If true, the size of the pie plot is scaled based on the perc value.
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

        pos = self.pos
        numNeurons = self.numNeurons

        # Pull out the statistics (tp, fn, tn, fp) from the arguments
        # numst = []
        # for arg in argv:
        #     numst.append(arg)

        # If there are 4 arguments, it is for the PDB case
        # pdb = False
        # if len(numst) == 4:
        #     pdb = True

        # Assign the colors for the pie chart (tp, fn, tn, fp)
        # if pdb:
        #     clrs = ['lawngreen', 'yellow', 'blue', 'red']
        # else:
        #     # Only two colors for well docked bad binders (tn, fp)
        #     clrs = ['blue', 'red']

        # Set default scale
        scale = 1

        # Determine the number of colors needed
        shapclust = sizes_cluster.shape
        num_colors = shapclust[1]

        # Generate a color list using a colormap
        cmap = cm.get_cmap('plasma', num_colors)  # Use any suitable
        clrs = [cmap(i) for i in range(num_colors)]

        # Setup figure, main axes, and sub-axes
        fig, ax, h_axes, hexagons, hexagon_to_neuron = self.setup_axes()

        # Draw pie plot in each neuron
        for neuron in range(numNeurons):
            # Scale the size of the pie chart according to the percent of PDB
            # data (or WD data) in that cell
            # # if pdb:
            #     scale = np.sqrt(perc[neuron] / 100)
            # else:
            #     scale = np.sqrt((100 - perc[neuron]) / 100)
            if scaleFlag:
                scale = np.sqrt(perc[neuron] / 100)
                scale = max(scale, 0.01)  # Ensure minimum scale

            # if scale == 0:
            #     scale = 0.01
                # Set numbers (tp, fn, tn, fp) for pie chart
                # if pdb:
                #     nums = [numst[0][neuron], numst[1][neuron], numst[2][neuron], numst[3][neuron]]
                # else:
                #     nums = [numst[0][neuron], numst[1][neuron]]

            # Make pie chart
            if np.sum(sizes_cluster[neuron]) != 0:
                h_axes[neuron].pie(sizes_cluster[neuron], colors=clrs, radius=scale)
            else:
                h_axes[neuron] = None

        if mouse_click and connect_pick_event:
            kwargs['cat'] = sizes_cluster
            fig.canvas.mpl_connect(
                'pick_event', lambda event: self.onpick(event, hexagons, hexagon_to_neuron, **kwargs)
            )

        plt.suptitle(title, fontsize=16)

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
            else:
                h_axes[neuron] = None

        title = 'Cluster Centers as Histogram'
        plt.suptitle(title, fontsize=16)

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

        for neuron in range(numNeurons):
            if len(x[neuron]) > 0:
                # Make graph
                h_axes[neuron].boxplot(x[neuron])
            else:
                h_axes[neuron] = None

        title = 'Cluster Centers as BoxPlot'
        plt.suptitle(title, fontsize=16)

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

        for neuron in range(numNeurons):
            if len(x[neuron]) > 0:
                # Make graph on the appropriate sub-axes
                h_axes[neuron].violinplot(x[neuron])
            else:
                h_axes[neuron] = None

        title = 'Violin plot'
        plt.suptitle(title, fontsize=16)

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
            'boxplot': self.plt_boxplot,
            'violin': self.plt_violin_plot
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
                    title = "Scatter Plot for each neuron with regression lines"
                else:
                    title = "Scatter Plot for each neuron without regression lines"
            else:
                h_axes[neuron] = None

        plt.suptitle(title, fontsize=16)

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
