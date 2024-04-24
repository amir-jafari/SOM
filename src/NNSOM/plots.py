# Dynamically choose a parent class for your subclass based on whether CuPy is available
try:
    import cupy as cp
    from .som_gpu import SOMGpu
    base_class = SOMGpu
except ImportError:
    from .som import SOM
    base_class = SOM

from .utils import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors


class SOMPlots(base_class):
    """
    A subclass of either SOM or SOMGpu (based on the availability of CuPy), designed to provide visualization
    and interactive plotting capabilities for Self-Organizing Maps (SOMs). This class is intended to enrich the analysis
    of SOMs by offering a variety of advanced visualization techniques to explore the trained SOM topology,
    distribution of data points, and various statistics derived from the SOM's learning process.

    Attributes:
        dimensions: A Gird of the SOM topology.

    Methods:
        The class includes methods for plotting the topology of the SOM, generating hit histograms,
        displaying cluster information, and more. These methods support interactive features through
        mouse clicks, allowing users to engage with the visualizations dynamically. Each method
        can also handle additional parameters for customization and handles various plotting styles
        like hexagonal units, numbered neurons, color gradients, and complex cluster histograms.
        The class also provides a generic plot method to handle different types of SOM visualizations
        and an event handling method to manage user interactions during the plotting sessions.

    This class supports interactivity and offers multiple visualization methods to deeply understand
    and analyze the behaviors and results of SOMs. It's especially useful for gaining insights into
    the topology, data distribution, and classification performance of SOMs.
    """

    def __init__(self, dimensions):
        """
        Initializes the SOMPlots class with specified dimensions for the SOM grid. This constructor
        sets up the underlying SOM or SOMGpu infrastructure, depending on the availability of CuPy.

        Parameters
        ----------
        dimensions : array-like, tuple, or int
            A tuple specifying the dimensions (rows, columns) of the SOM grid.
        """
        super().__init__(dimensions)

    def plt_top(self, mouse_click=False, connect_pick_event=True, **kwargs):
        """
        Plots the topology of the SOM using hexagonal units. This method visualizes the position and the
        boundaries of each neuron within the grid, allowing for interaction if enabled.

        Parameters
        ----------
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing interactive functionality such as
            querying or modifying neuron data, by default False.
        connect_pick_event : bool, optional
            If True, connects a pick event that triggers when a neuron (hexagon) is clicked, by default True.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler `onpick` when an interactive
            element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        patches : list
            A list of matplotlib.patches.Patch objects representing the hexagonal units of the SOM.
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
        """
        Plots the topology of the SOM with each neuron numbered. This method visualizes each neuron as a hexagon with a
        number indicating its index, which is useful for identifying and referencing specific neurons during analysis.

        Parameters
        ----------
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing for interaction such as detailed queries or
            data manipulation associated with specific neurons, by default False.
        connect_pick_event : bool, optional
            If True, connects a pick event that triggers when a neuron is clicked, by default True.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler `onpick` when an interactive
            element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        patches : list
            A list of matplotlib.patches.Patch objects representing the hexagonal units of the SOM.
        text : list
            A list of matplotlib.text.Text objects displaying the neuron indices.
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
        """
        Generates a hit histogram for the SOM, which displays the frequency of data points assigned to each neuron.
        Each neuron is represented as a hexagon, and the size of each hexagon is proportional to the number of hits.
        Optionally, the actual number of hits can be displayed within each hexagon.

        Parameters
        ----------
        x : array-like
            The input data to be visualized in the histogram.
        textFlag : bool, optional
            If True, displays the count of hits within each hexagon, by default True.
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing for interactive functionality such as
            querying or modifying neuron data, by default False.
        connect_pick_event : bool, optional
            If True, connects a pick event that triggers when a neuron is clicked, by default True.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler `onpick` when an interactive
            element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        patches : list
            A list of matplotlib.patches.Patch objects representing the inner hexagons colored based on hit counts.
        text : list, optional
            A list of matplotlib.text.Text objects displaying the hit counts, included if textFlag is True.
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
            x = self.normalize(x, self.norm_func)
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
        """
        Generates a grayscale histogram for the SOM, where the shade of each hexagon represents the corresponding value from the provided percentage array.

        Parameters
        ----------
        x : array-like
            The input data to be visualized in the histogram.
        perc : array-like
            An array containing the percentage values to be represented by the grayscale shades, where higher values correspond to darker shades.
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing for interactive functionality such as querying or modifying neuron data, by default False.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler onpick when an interactive element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        patches : list
            A list of matplotlib.patches.Patch objects representing the inner hexagons colored based on the grayscale values.
        text : list, optional
            A list of matplotlib.text.Text objects displaying the hit counts, included if textFlag is True in the underlying `hit_hist` method.
        """
        numNeurons = self.numNeurons
        dmax = np.amax(np.abs(perc))    # Find the maximum value of perc across all clusters

        fig, ax, patches, text = self.hit_hist(x, False, mouse_click, **kwargs)

        # Scale the gray scale to the perc value
        for neuron in range(numNeurons):
            scale = perc[neuron] / dmax
            color = [scale for i in range(3)]  # Create a gray color based on the scaled value
            color.append(1.0)  # Add alpha value
            patches[neuron][0]._facecolor = tuple(color)  # Apply the color to the patch

        # Get rid of extra white space on sides
        plt.tight_layout()

        return fig, ax, patches, text

    def color_hist(self, x, avg, mouse_click=False, **kwargs):
        """
        Generates a colored histogram for the SOM, where the color of each hexagon represents the corresponding value from the provided average array.

        Parameters
        ----------
        x : array-like
            The input data to be visualized in the histogram.
        avg : array-like
            An array containing the average values to be represented by the color map, where higher values correspond to warmer colors (e.g., red) and lower values correspond to cooler colors (e.g., blue).
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing for interactive functionality such as querying or modifying neuron data, by default False.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler onpick when an interactive element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        patches : list
            A list of matplotlib.patches.Patch objects representing the inner hexagons colored based on the average values.
        text : list, optional
            A list of matplotlib.text.Text objects displaying the hit counts, included if textFlag is True in the underlying `hit_hist` method.
        cbar : matplotlib.colorbar.Colorbar
            The Colorbar object attached to the plot, representing the color mapping.
        """

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
        cbar = fig.colorbar(cax, ax=ax)

        # Adjust the tick labels to the correct scale
        ticklab = cbar.ax.get_yticks()
        numticks = len(ticklab)
        ticktext = []
        for i in range(numticks):
            ticktext.append('%.2f' % (dmax * ticklab[i]))

        # Set the ticks first
        cbar.ax.set_yticks(ticklab)

        cbar.ax.set_yticklabels(ticktext)

        # Get rid of extra white space on sides
        fig.tight_layout()

        return fig, ax, patches, text, cbar

    def cmplx_hit_hist(self, x, clust, perc, ind_missClass, ind21, ind12, mouse_click=False, **kwargs):
        """
        Generates a complex hit histogram for the SOM, incorporating information about cluster quality, misclassifications, and false positives/negatives.

        Parameters
        ----------
        x : array-like
            The input data to be visualized in the histogram.
        clust : list or array-like
            A list or array containing the cluster assignments for each data point.
        perc : array-like
            An array containing the percentage values for each cluster, representing the proportion of good binders.
        ind_missClass : array-like
            An array containing the indices of misclassified data points.
        ind21 : array-like
            An array containing the indices of false positive data points (classified as good binders but are actually bad binders).
        ind12 : array-like
            An array containing the indices of false negative data points (classified as bad binders but are actually good binders).
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing for interactive functionality such as querying or modifying neuron data, by default False.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler onpick when an interactive element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        patches : list
            A list of matplotlib.patches.Patch objects representing the inner hexagons colored and styled based on cluster quality and misclassifications.
        text : list
            A list of matplotlib.text.Text objects displaying the hit counts within each hexagon.
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
        """
        Generates a custom complex hit histogram for the SOM, allowing for flexible customization of the hexagon face colors, edge colors, and edge widths.

        Parameters
        ----------
        x : array-like
            The input data to be visualized in the histogram.
        face_labels : array-like
            An array containing the labels or values to be represented by the face colors of the hexagons.
        edge_labels : array-like
            An array containing the labels or values to be represented by the edge colors of the hexagons.
        edge_width : array-like
            An array containing the values for the edge widths of the hexagons.
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing for interactive functionality such as querying or modifying neuron data, by default False.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler onpick when an interactive element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        patches : list
            A list of matplotlib.patches.Patch objects representing the inner hexagons colored and styled based on the provided face labels, edge labels, and edge widths.
        text : list
            A list of matplotlib.text.Text objects displaying the hit counts within each hexagon.

        Raises
        ------
        ValueError
            If the input data or label arrays have incorrect dimensions or lengths.
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
        if x.shape[1] != self.w.shape[1]:
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
        """
        Generates a Neighborhood Connection Map for the SOM, displaying the connections between neighboring neurons.

        Parameters
        ----------
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing for interactive functionality such as querying or modifying neuron data, by default False.
        connect_pick_event : bool, optional
            If True, connects a pick event that triggers when a neuron is clicked, by default True.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler onpick when an interactive element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        patches : list
            A list of matplotlib.patches.Patch objects representing the edges between connected neurons.
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
        """
        Generates a neuron distance plot that visualizes the distances between neighboring neurons in the SOM grid.

        Parameters
        ----------
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing for interactive functionality such as querying or modifying neuron data, by default False.
        connect_pick_event : bool, optional
            If True, connects a pick event that triggers when a neuron is clicked, by default True.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler onpick when an interactive element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        patches : list
            A list of matplotlib.patches.Patch objects representing the edges between connected neurons, colored based on the distance between their respective neuron weights.
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
        """
        Generates a simple grid plot that visualizes the SOM neurons as hexagons with varying sizes and colors.

        Parameters
        ----------
        avg : array-like
            An array containing the average values to be represented by the color map, where higher values correspond to warmer colors (e.g., red) and lower values correspond to cooler colors (e.g., blue).
        sizes : array-like
            An array containing the sizes to be used for the inner hexagons within each neuron, where larger values result in larger hexagons.
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing for interactive functionality such as querying or modifying neuron data, by default False.
        connect_pick_event : bool, optional
            If True, connects a pick event that triggers when a neuron is clicked, by default True.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler onpick when an interactive element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        patches : list
            A list of matplotlib.patches.Patch objects representing the inner hexagons colored based on the average values and sized based on the provided sizes.
        cbar : matplotlib.colorbar.Colorbar
            The Colorbar object attached to the plot, representing the color mapping.
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
        cbar = fig.colorbar(cax, ax = ax, fraction=0.046, pad=0.04)

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

        cbar.ax.set_yticks(ticklab)

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
        """
        Generates a stem plot visualization for the SOM, displaying the input data and neuron responses.

        Parameters
        ----------
        x : array-like
            The input data or independent variable for the stem plot.
        y : array-like
            The neuron responses or dependent variable for the stem plot, where each row corresponds to a neuron.
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing for interactive functionality such as querying or modifying neuron data, by default False.
        connect_pick_event : bool, optional
            If True, connects a pick event that triggers when a neuron is clicked, by default True.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler onpick when an interactive element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        h_axes : list
            A list of matplotlib.axes.Axes objects, each containing a stem plot for a single neuron.
        """

        numNeurons = self.numNeurons

        # Setup figure, axes, and sub-axes
        fig, ax, h_axes, hexagons, hexagon_to_neuron = self.setup_axes()

        # Draw stem plot
        for neuron in range(numNeurons):
            # Make graph
            h_axes[neuron].stem(x, y[neuron])

        if mouse_click and connect_pick_event:
            kwargs['cat'] = y
            fig.canvas.mpl_connect(
                'pick_event', lambda event: self.onpick(event, hexagons, hexagon_to_neuron, **kwargs)
            )

        return fig, ax, h_axes

    def plt_wgts(self, mouse_click=False, connect_pick_event=True, **kwargs):
        """
        Generates a line plot visualization for the SOM weights, displaying the weight vectors for each neuron.

        Parameters
        ----------
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing for interactive functionality such as querying or modifying neuron data, by default False.
        connect_pick_event : bool, optional
            If True, connects a pick event that triggers when a neuron is clicked, by default True.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler onpick when an interactive element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        h_axes : list
            A list of matplotlib.axes.Axes objects, each containing a line plot for a single neuron's weight vector.
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
        """
        Generates a pie chart visualization for the SOM, displaying the composition of each neuron's data or cluster.

        Parameters
        ----------
        x : array-like
            A 2D array or sequence of vectors, where each row represents the composition or category values for a single neuron.
        s : array-like, optional
            An array containing the percentage values to be used for scaling the pie chart sizes, by default None.
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing for interactive functionality such as querying or modifying neuron data, by default False.
        connect_pick_event : bool, optional
            If True, connects a pick event that triggers when a neuron is clicked, by default True.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler onpick when an interactive element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        h_axes : list
            A list of matplotlib.axes.Axes objects, each containing a pie chart for a single neuron's composition.

        Raises
        ------
        ValueError
            If the length of `x` or `s` (if provided) does not match the number of neurons, or if the percentage values in `s` are not between 0 and 100.
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
        """
        Generates a histogram visualization for the SOM, displaying the data distribution within each neuron's cluster.

        Parameters
        ----------
        x : array-like
            A 2D array or sequence of vectors, where each row represents the data points assigned to a single neuron.
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing for interactive functionality such as querying or modifying neuron data, by default False.
        connect_pick_event : bool, optional
            If True, connects a pick event that triggers when a neuron is clicked, by default True.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler onpick when an interactive element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        h_axes : list
            A list of matplotlib.axes.Axes objects, each containing a histogram for a single neuron's data distribution.
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
        """
        Generates a boxplot visualization for the SOM, displaying the statistical summary of the data distribution within each neuron's cluster.

        Parameters
        ----------
        x : array-like
            A 2D array or sequence of vectors, where each row represents the data points assigned to a single neuron.
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing for interactive functionality such as querying or modifying neuron data, by default False.
        connect_pick_event : bool, optional
            If True, connects a pick event that triggers when a neuron is clicked, by default True.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler onpick when an interactive element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        h_axes : list
            A list of matplotlib.axes.Axes objects, each containing a boxplot for a single neuron's data distribution.
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
        """
        Generates a violin plot visualization for the SOM, displaying the distribution of data within each neuron's cluster.

        Parameters
        ----------
        x : array-like
            A 2D array or sequence of vectors, where each row represents the data points assigned to a single neuron.
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing for interactive functionality such as querying or modifying neuron data, by default False.
        connect_pick_event : bool, optional
            If True, connects a pick event that triggers when a neuron is clicked, by default True.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler onpick when an interactive element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        h_axes : list
            A list of matplotlib.axes.Axes objects, each containing a violin plot for a single neuron's data distribution.
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

    def plt_scatter(self, x, y, reg_line=True, mouse_click=False, connect_pick_event=True, **kwargs):
        """
        Generates a scatter plot visualization for the SOM, displaying the data points assigned to each neuron and an optional regression line.

        Parameters
        ----------
        x : array-like
            A 2D array or sequence of vectors, where each row represents the x-coordinate data points assigned to a single neuron.
        y : array-like
            A 2D array or sequence of vectors, where each row represents the y-coordinate data points assigned to a single neuron.
        reg_line : bool, optional
            If True, a regression line is plotted for each neuron's data, by default True.
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing for interactive functionality such as querying or modifying neuron data, by default False.
        connect_pick_event : bool, optional
            If True, connects a pick event that triggers when a neuron is clicked, by default True.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler onpick when an interactive element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        h_axes : list
            A list of matplotlib.axes.Axes objects, each containing a scatter plot for a single neuron's data.
        """

        pos = self.pos
        numNeurons = self.numNeurons

        # Setup figure, main axes, and sub-axes
        fig, ax, h_axes, hexagons, hexagon_to_neuron = self.setup_axes()

        # Determine the global minimum and maximum of x and y for the axes limits
        x_min, x_max = get_global_min_max(x)
        y_min, y_max = get_global_min_max(y)

        # Loop over each neuron for hexagons and scatter plots
        for neuron in range(numNeurons):
            # Make Scatter Plot for each neuron
            if len(x[neuron]) > 0 and len(y[neuron]) > 0:
                h_axes[neuron].scatter(x[neuron], y[neuron], s=1, c='k')

                if reg_line:
                    m, p = np.polyfit(x[neuron], y[neuron], 1)
                    h_axes[neuron].plot(x[neuron], m * x[neuron] + p, c='r', linewidth=1)

                # Set the same x and y limits for each sub-plot based on global min and max
                h_axes[neuron].set_xlim(x_min, x_max)
                h_axes[neuron].set_ylim(y_min, y_max)

                # Show only the left and bottom spines
                h_axes[neuron].spines['top'].set_visible(False)
                h_axes[neuron].spines['right'].set_visible(False)
                h_axes[neuron].spines['left'].set_visible(True)
                h_axes[neuron].spines['bottom'].set_visible(True)

                # Enable the axes and show tick marks
                h_axes[neuron].set_frame_on(True)
                h_axes[neuron].tick_params(axis='both', which='both', length=5)

            else:
                h_axes[neuron] = None

        if mouse_click and connect_pick_event:
            kwargs['num1'] = x
            kwargs['num2'] = y
            fig.canvas.mpl_connect(
                'pick_event', lambda event: self.onpick(event, hexagons, hexagon_to_neuron, **kwargs)
            )

        return fig, ax, h_axes

    def component_positions(self, x):
        """
        Visualizes the positions of the components in a Self-Organizing Map (SOM) along with the input vectors.

        This method plots the trained SOM weight vectors as gray dots and the input vectors as green dots on a 2D plot.
        It also connects neighboring SOM neurons with red lines to represent the grid structure, illustrating the organization and clustering within the map.

        Parameters
        ----------
        x : array-like
            A 2D array or sequence of vectors, typically representing input data or test data that has been projected onto the SOM.
        """

        x = np.transpose(x)

        # Extract the trained weight vectors and the SOM grid dimensions
        weight_vectors = self.w
        grid_x, grid_y = self.dimensions

        # Plot the SOM weight vectors as gray dots
        plt.scatter(weight_vectors[:, 0], weight_vectors[:, 1], color='gray', s=50, label='Weight Vectors')
        # for i, vec in enumerate(weight_vectors):
        #     plt.annotate(str(i), (vec[0], vec[1]), textcoords="offset points", xytext=(0,5), ha='center')

        # Plot the Iris data points as blue dots
        plt.scatter(x[:, 0], x[:, 1], color='green', s=20, label='Input Vectors', alpha=0.5)

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
        """
        Visualizes the weight distribution across different features in a Self-Organizing Map (SOM) using a series of 2D plots.

        This method creates a grid of subplots where each subplot represents the weight distribution for a specific feature of the input data. The weight of each neuron for the given feature is represented in the plot by the color of a hexagonal cell, with darker colors indicating higher weights. This visualization helps in understanding the importance and distribution of each feature across the map.

        Parameters
        ----------
        X : array-like
            A 2D array of input data where each row represents a feature and each column represents a sample.
        """

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

    def weight_as_image(self, rows=None, mouse_click=False, connect_pick_event=True, **kwargs):
        """
        Visualizes the weights of a Self-Organizing Map (SOM) as images within a hexagonal grid layout.

        This method maps the weight vectors of each neuron onto a hexagonal cell and optionally enables interaction with each hexagon. The hexagons represent the neurons, and the colors within each hexagon represent the neuron's weight vector reshaped into either a specified or automatically determined matrix form. This visualization is useful for analyzing the learned patterns and feature representations within the SOM.

        Parameters
        ----------
        rows : int, optional
            The number of rows to reshape each neuron's weight vector into. If None, the weight vector is reshaped into a square matrix by default. If specified, the weight vector is reshaped into a matrix with the given number of rows, and the number of columns is determined automatically.
        mouse_click : bool, optional
            If True, enables the plot to respond to mouse clicks, allowing for interactive functionality such as querying or modifying neuron data, by default False.
        connect_pick_event : bool, optional
            If True, connects a pick event that triggers when a neuron is clicked, by default True.
        **kwargs : dict
            Arbitrary keyword arguments that can be passed to the event handler onpick when an interactive element is clicked. Common parameters could include data specific to the plot or visualization settings.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object containing the plot elements.
        patches : list
            A list of matplotlib.patches.Patch objects, each representing a hexagon in the plot.
        """

        w = self.w  # Weight matrix
        pos = self.pos  # Positions of the neurons
        numNeurons = self.numNeurons  # Number of neurons

        # Get the shape of a single hexagon
        shapex, shapey = get_hexagon_shape()

        # Create the figure and axis with a larger size to accommodate the decorations
        fig, ax = plt.subplots(figsize=(8, 6))

        # Set the aspect of the plot to be equal
        plt.axis('equal')

        # List to keep track of the hexagon patches
        patches = []

        for i in range(numNeurons):
            hex_center_x = pos[0, i]  # x coordinate of the ith position
            hex_center_y = pos[1, i]  # y coordinate of the ith position

            # Draw the hexagon
            temp, = ax.fill(hex_center_x + shapex, hex_center_y + shapey, facecolor='none', edgecolor='k', picker=True)
            patches.append(temp)

            # Assign the cluster number for each hexagon
            hexagon_to_neuron = {hex: neuron for neuron, hex in enumerate(patches)}

            # Transform the row of weights into a matrix if necessary
            if rows is None:
                weight_matrix = w[i].reshape(int(np.sqrt(w.shape[1])), -1)  # Default to square matrix for simplicity
            else:
                weight_matrix = w[i].reshape(rows, -1)

            # Calculate the size and position for the imshow plot
            # Find the radius of the hexagon, accounting for the scaling of the shape
            hex_radius = (np.max(shapex) - np.min(shapex)) / 2

            # Calculate the side length of the hexagon for a regular hexagon
            side_length = hex_radius * np.sqrt(3) / 2

            # Offset the inset_axes to be centered within the hexagon
            # The factor of sqrt(3)/2 is because in a regular hexagon, the distance from the center to a side is sqrt(3)/2 times the side length
            axins = ax.inset_axes([hex_center_x - side_length / 2,
                                   hex_center_y - side_length * (np.sqrt(3) / 2) / 2,
                                   side_length,
                                   side_length * (np.sqrt(3) / 2)], transform=ax.transData)

            # Ensure the imshow plot takes up the correct amount of space
            # Adjust aspect ratio if necessary, depending on the weight matrix shape
            aspect_ratio = weight_matrix.shape[0] / weight_matrix.shape[1]
            axins.imshow(weight_matrix, aspect='equal')
            axins.set_aspect(aspect_ratio * (side_length / (side_length * (np.sqrt(3) / 2))))

            # Turn off the axis
            axins.axis('off')

        # Connect the pick event for interactivity if required
        if mouse_click and connect_pick_event:
            fig.canvas.mpl_connect('pick_event', lambda event: self.onpick(event, patches, hexagon_to_neuron, **kwargs))

        # Adjust the layout to fit everything
        plt.tight_layout()

        # Display the plot
        plt.show()

        # Return the figure components
        return fig, ax, patches

    # Generic Plot Function
    def plot(self, plot_type, data_dict=None, ind=None, target_class=None, use_add_array=False,
             **kwargs):
        """
        Generic Plot Function.
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
            {"data", "target", "clust", "add_1d_array", "add_2d_array"}

        ind : int, str or array-like (optional)
            The indices of the data to be plotted.

        target_class: int (optional)
            The target class to be plotted.

        use_add_array: bool (optional)
            If true, the additional array to be used.

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
        elif plot_type in ['component_positions', 'component_planes']:
            # Error Handling if the data_dict have the scaled input data X
            validate_data_dict(["data"])
            # Data Preparation
            x = data_dict['data']
            x = self.normalize(x, self.norm_func)
            # Invoke Function
            return selected_plot(x)

        # =====================  Hit Histogram Family =====================
        elif plot_type in ['hit_hist', 'gray_hist', 'color_hist', 'complex_hist']:
            # Validate the data_dict have the scaled input data
            validate_data_dict(["data"])

            # Extract input data
            x = data_dict['data']

            if plot_type in ['hit_hist']:
                # Invoke the function
                return selected_plot(x, True, **kwargs)

            elif plot_type in ['gray_hist']:
                # Gray Hist with add_1d_array
                if use_add_array:
                    # Validate the data_dict have the additional 1D array
                    validate_data_dict(["add_1d_array"])

                    if len(data_dict['add_1d_array']) != self.numNeurons:
                        raise ValueError(
                            "The additional 1D array must have the same length as the clust data or original data.")

                    perc = data_dict['add_1d_array']

                # Gray hist with data
                else:
                    # Error Handling if the target data not provided
                    validate_data_dict(["clust"])

                    if target_class is None and ind is None:
                        raise ValueError("This plot requires either the target class or ind.")

                    elif target_class is not None and ind is not None:
                        raise ValueError("This plot requires only either the target class or ind.")

                    elif target_class is not None:
                        validate_data_dict(['target'])
                        perc = get_perc_cluster(data_dict['target'], target_class, data_dict['clust'])

                    elif ind is not None:
                        feature = x[:, ind]
                        perc = get_cluster_avg(feature, data_dict['clust'])

                # Invoke the gray hist function
                return selected_plot(x, perc, **kwargs)

            elif plot_type in ['color_hist']:
                # Color Hist with additional data
                if use_add_array:
                    validate_data_dict(['add_1d_array'])

                    if len(data_dict['add_1d_array']) != self.numNeurons:
                        raise ValueError("The additional 1D array must have the same length as the clust data or original data.")

                    avg = data_dict['add_1d_array']
                # Color Hist with input data
                else:
                    validate_data_dict(["clust"])

                    if target_class is None and ind is None:
                        raise ValueError("This plot requires either the target class or ind.")

                    elif target_class is not None and ind is not None:
                        raise ValueError("This plot requires only either the target class or ind.")

                    elif target_class is not None:
                        validate_data_dict(['target'])
                        avg = get_perc_cluster(data_dict['target'], target_class, data_dict['clust'])

                    elif ind is not None:
                        feature = x[:, ind]
                        avg = get_cluster_avg(feature, data_dict['clust'])

                return selected_plot(x, avg, **kwargs)

            elif plot_type in ['complex_hist']:
                if use_add_array:
                    # Validate the data_dict have the additional 2D array
                    validate_data_dict(["add_2d_array"])

                    add_2d_array = np.array(data_dict['add_2d_array'])

                    if add_2d_array.shape[0] != self.numNeurons:
                        raise ValueError(
                            "The additional 2D array must have the same length as the number of neurons.")

                    # Assuming add_2d_array is a list of lists
                    if add_2d_array.shape[1] != 3:
                        raise ValueError("Each inner list in the additional 2D array must have exactly 3 items. \
                        E.g. [numNeurons, [face_labels, edge_labels, edge_widths[0-20]]")

                    # Extract Data
                    face_labels = add_2d_array[:, 0]
                    edge_labels = add_2d_array[:, 1]
                    edge_widths = add_2d_array[:, 2]

                else:
                    raise ValueError("This plot requires an additional 2-D array in data_dict. "
                                     "The additional 2D array must have 3 features. "
                                     "E.g. [numNeurons, [face_labels, edge_labels, edge_widths[0-20]]")

                return selected_plot(x, face_labels, edge_labels, edge_widths, **kwargs)

        # ===================== Simple Grid =====================
        elif plot_type in ['simple_grid']:
            # Validate the data_dict have the original data
            validate_data_dict(["data"])
            data = data_dict['data']
            # Simple grid with addtional variable
            if use_add_array:
                # Error Handlig if the additional 2D arrays not privided
                validate_data_dict(["add_2d_array"])
                add_2d_array = np.asarray(data_dict["add_2d_array"], np.float32)
                # Validate Length
                if add_2d_array.shape[0] != self.numNeurons:
                    raise ValueError("The additional 2D array must have the same length as the number of cluster")
                # Validate number of items in each cluster
                if add_2d_array.shape[1] != 2:
                    raise ValueError("Each cluster must have only 2 items in the additional 2D array")

                avg = add_2d_array[:, 0]
                sizes = add_2d_array[:, 1]

            else:
                # Error Handling if the target and clust data not provided
                validate_data_dict(["clust", "target"])

                if ind is None:
                    raise ValueError("The indices is required for this plot type.")

                if target_class is None:
                    raise ValueError("The target class is required")

                clust = data_dict['clust']

                # Extract avg from the original data
                num_feature = data[:, ind]
                avg = get_cluster_avg(num_feature, clust)

                # Extract size from the target
                target = data_dict['target']
                sizes = get_perc_cluster(target, target_class, clust)

            return selected_plot(avg, sizes, **kwargs)

        # ===================== Basic Plot Family =====================
        elif plot_type in ['stem', 'pie']:

            # If the user want to plot addtional data
            if use_add_array:
                # Error Handling if the additional 2D array not provided
                validate_data_dict(["add_2d_array"])

                # Error Handling if additional vategorical variable has correct length
                if len(data_dict["add_2d_array"]) != self.numNeurons:
                    raise ValueError("The additional categorical data must have the same length as the clust data.")

                #  Get Additional Data
                sizes = data_dict['add_2d_array']

            else:
                # Error Handling if the clust data not provided
                validate_data_dict(["target", "clust"])

                # Extract Information
                clust = data_dict['clust']
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
                if use_add_array:
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
            validate_data_dict(["data", "clust"])

            # Extract the feature from the original data
            clust = data_dict['clust']
            feature = data_dict['data'][:, ind]

            x = get_cluster_array(feature, clust)

            return selected_plot(x, **kwargs)

        elif plot_type in ['scatter']:
            # Error Handling if the index not provided
            if ind is None:
                raise ValueError("The indices is required for this plot type.")

            if len(ind) != 2:
                raise ValueError("The indices must contain exactly two elements. Eg. [0, 1]")

            # Error Handling if the original data and clust not provided
            validate_data_dict(["data", "clust"])

            # Extract the feature from the original data
            x = data_dict['data'][:, ind[0]]
            y = data_dict['data'][:, ind[1]]
            clust = data_dict['clust']
            x = get_cluster_array(x, clust)
            y = get_cluster_array(y, clust)

            # Call the scatter plot function
            return selected_plot(x, y, **kwargs)

        elif plot_type in ['box', 'violin']:
            # Error Handling if the original data not provided
            validate_data_dict(["data", "clust"])

            # Extract the feature from the original data
            clust = data_dict['clust']

            if ind is None:  # Extract All data
                data = data_dict['data']
                x = get_cluster_data(data, clust)
            elif isinstance(ind, int):  # index just have 1 index
                data = data_dict['data'][:, ind]
                x = get_cluster_array(data, clust)
            elif isinstance(ind, (list, np.ndarray)):  # index have multiple indices
                data = data_dict['data'][:, ind]
                x = get_cluster_data(data, clust)

            # Call the box plot and violin function
            return selected_plot(x, **kwargs)

    # Interactive Functionality
    def onpick(self, event, hexagons, hexagon_to_neuron, **kwargs):
        """
        Interactive Plot Function

        Parameters
        ----------
        event: event
            a mouse click event
        hexagons: list
            a list of hexagons
        hexagon_to_neuron: dict
            a dictionary mapping hexagons to neurons
        **kwargs:
            a dictionary with input data
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
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(right=0.8)
        ax.set_aspect('equal')

        # Button Configuration
        button_types = self.determine_button_types(**kwargs)
        buttons = create_buttons(fig, button_types)

        # Store buttons in an attribute to maintain a reference
        self.buttons = buttons

        # Set up button click events
        for button_type, button in self.buttons.items():
            button.on_clicked(self.create_click_handler(button_type, ax, neuron_ind, **kwargs))

        # Show up the 2nd window
        plt.show()

    def create_click_handler(self, button_type, ax, neuron_ind, **kwargs):
        # Generates a custom event handler for button clicks in a plot.
        def handler(event):
            self.button_click_event(button_type, ax, neuron_ind, **kwargs)

        return handler

    def button_click_event(self, button_type, ax, neuron_ind, **kwargs):
        # Handle button click event by calling the appropriate plot function
        if button_type == 'pie':
            # Pre-process categorical variables
            sizes = kwargs['cat']
            sizes = sizes[neuron_ind][:kwargs['topn']]
            self.plot_pie(ax, sizes, neuron_ind)

        elif button_type == 'stem':
            sizes = kwargs['cat']
            # Generate the align array: 0, 1, 2, ..., number of unique item -1
            align = np.arange(len(sizes[0]))

            self.plot_stem(ax, align, sizes, neuron_ind)

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
            cluster_data = get_cluster_data(kwargs['data'], kwargs['clust'])
            sub_clust_data = cluster_data[neuron_ind]  # Get the data for the
            self.sub_clustering(sub_clust_data, neuron_ind)

        else:
            print(f"Unknown button type: {button_type}")

    def determine_button_types(self, **kwargs):
        # Determine the button type based on the contents of **kwargs
        button_types = []

        # Check for categorical data for pie charts
        if 'cat' in kwargs and kwargs['cat'] is not None:
            button_types.append('pie')
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
        if 'data' in kwargs:
            button_types.append('sub_cluster')

        return button_types

    # Helper function to create charts
    def plot_pie(self, ax, data, neuronNum):
        """
        Plots a pie chart on the specified matplotlib axes object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The matplotlib axes object where the pie chart will be plotted.
        data : array-like
            An array of numeric data which represents the portions of the pie chart.
        neuronNum : int
            The neuron number associated with the data, which is used to title the pie chart.
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
        """
        Plots a stem plot for a specific neuron's data on the given axes

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The matplotlib axes object where the stem plot will be drawn.
        align : array-like
            The x positions of the stems.
        height : array-like
            The y values for each stem, indexed by neuron number.
        neuronNum : int
            The index of the neuron for which the plot is being generated.
        """
        # Clear the axes
        ax.clear()
        # Stem plot
        ax.stem(align, height[neuronNum])  # x: cat, y: data
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
        """
        Generates a box plot for a specific neuron's data on the provided axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object on which the box plot will be drawn.
        data : array-like
            The data array for which the box plot is to be generated.
        neuronNum : int
            The neuron index that the data is associated with.
        """
        # Clear the axes
        ax.clear()
        # Box plot logic here
        ax.boxplot(data)
        ax.set_title("Box plot in the Cluster " + str(neuronNum))
        # Redraw the figure
        ax.figure.canvas.draw_idle()

    def plot_violin(self, ax, data, neuronNum):
        """
        Displays a violin plot for a specific neuron's data on the provided axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object where the violin plot will be drawn.
        data : array-like
            The data to be used for the violin plot.
        neuronNum : int
            The index of the neuron associated with the data.
        """
        # Clear the axes
        ax.clear()
        # Violin plot logic here
        ax.violinplot(data)
        ax.set_title('Violin Plot inside the Cluster ' + str(neuronNum))
        # Redraw the figure
        ax.figure.canvas.draw_idle()

    def plot_scatter(self, ax, num1, num2, neuronNum):
        """
        Plots a scatter plot for a specific neuron's data on the provided axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object on which the scatter plot will be drawn.
        num1 : array-like
            The x coordinates of the data points.
        num2 : array-like
            The y coordinates of the data points.
        neuronNum : int
            The index of the neuron for which the plot is being generated.
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
        Performs sub-clustering on the data associated with a specific neuron within the Self-Organizing Map (SOM).

        Parameters
        ----------
        data : array-like
            The dataset from which sub-clusters are to be derived. Typically, this is the subset of the overall dataset that
            has been mapped to the neuron specified by `neuron_ind`.
        neuron_ind : int
            The index of the neuron for which sub-clustering is to be performed. This index is used to refer to a specific neuron
            in the SOM's grid.

        Returns
        -------
        list of array-like
            A list of clusters, where each cluster is an array of data points that form a sub-group within the neuron's data.
        """
        if len(data) <= 1:
            print("There is no enough data to create sub-cluster")
            return

        if neuron_ind in self.sub_som:
            print('Sub clustering already done')
            sub_clust = self.sub_som[neuron_ind]
        else:
            # Training Sub Cluster
            sub_clust = SOMPlots((2, 2))
            sub_clust.init_w(data, norm_func=self.norm_func)
            sub_clust.train(data, 3, 500, 100, norm_func=self.norm_func)

            self.sub_som[neuron_ind] = sub_clust

        # Plot the sub cluster <- Can we h
        fig, ax, patches, text = sub_clust.hit_hist(data, True, connect_pick_event=False)

        plt.show()
