from .som import SOM
from .utils import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
        plt.axis('off')
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

    def hit_hist(self, x, textFlag):
        # Basic hit histogram
        # x contains the input data
        # If textFlag is true, the number of members of each cluster
        # is printed on the cluster.
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
        plt.axis('off')
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
        for i in range(numNeurons):
            plt.fill(pos[0, i] + shapex, pos[1, i] + shapey, facecolor=(0.4, 0.4, 0.6), edgecolor=(0.8, 0.8, 0.8))

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

        # Get rid of extra white space on sides
        fig.tight_layout()

        return fig, ax, patches

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

        # Get rid of extra white space on sides
        fig.tight_layout()

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
        for neuron in range(numNeurons):
            ax.fill(pos[0, neuron] + shapex, pos[1, neuron] + shapey,
                    facecolor=(1, 1, 1), edgecolor=(0.8, 0.8, 0.8))

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

        return fig, ax, h_axes

    def plt_dist(self, dist):
        # Plot distribution
        # Purpose:

        numNeurons = self.numNeurons

        # Setup figure, axes, and sub-axes
        fig, ax, h_axes = self.setup_axes()

        # Draw stem plot
        for neuron in range(numNeurons):
            # Make graph
            h_axes[neuron].stem(dist[neuron])

        title = 'dist plot'
        plt.suptitle(title, fontsize=16)

        return fig, ax, h_axes

    def plt_wgts(self):
        # Plot neuron weight as line
        # Purpose:

        numNeurons = self.numNeurons
        w =self.w

        # Setup figure, main axes, and sub-axes
        fig, ax, h_axes = self.setup_axes()

        # Draw line plots
        for neuron in range(numNeurons):
            # Make graph
            h_axes[neuron].plot(w[neuron])

        title = 'Cluster Centers as Lines'
        plt.suptitle(title, fontsize=16)

        return fig, ax, h_axes

    def plt_pie(self, title, perc, *argv):
        # Generate pie plot in the hexagon.
        # Purpose:

        pos = self.pos

        # Pull out the statistics (tp, fn, tn, fp) from the arguments
        numst = []
        for arg in argv:
            numst.append(arg)

        # If there are 4 arguments, it is for the PDB case
        pdb = False
        if len(numst) == 4:
            pdb = True

        # Find the locations and size for each neuron in the SOM
        w = self.w
        numNeurons = self.numNeurons

        # Assign the colors for the pie chart (tp, fn, tn, fp)
        if pdb:
            clrs = ['lawngreen', 'yellow', 'blue', 'red']
        else:
            # Only two colors for well docked bad binders (tn, fp)
            clrs = ['blue', 'red']

        # Setup figure, main axes, and sub-axes
        fig, ax, h_axes = self.setup_axes()

        # Draw pie plot in each neuron
        for neuron in range(numNeurons):
            # Scale the size of the pie chart according to the percent of PDB
            # data (or WD data) in that cell
            if pdb:
                scale = np.sqrt(perc[neuron] / 100)
            else:
                scale = np.sqrt((100 - perc[neuron]) / 100)

            if scale == 0:
                scale = 0.01
                # Set numbers (tp, fn, tn, fp) for pie chart
                if pdb:
                    nums = [numst[0][neuron], numst[1][neuron], numst[2][neuron], numst[3][neuron]]
                else:
                    nums = [numst[0][neuron], numst[1][neuron]]

            # Make pie chart
            if np.sum(nums) == 0:
                nums = [0.0, 1.0, 0.0, 0.0]
            h_axes[neuron].pie(nums, colors=clrs)

        plt.suptitle(title, fontsize=16)

        return fig, ax, h_axes

    def plt_histogram(self, data):
        # Create histogram.
        # Purpose:

        numNeurons = self.numNeurons

        # Setup figure, main axes, and sub-axes
        fig, ax, h_axes = self.setup_axes()

        # Draw histogram
        for neuron in range(numNeurons):
            # Make graph
            h_axes[neuron].hist(data[neuron])

        title = 'Cluster Centers as Lines'
        plt.suptitle(title, fontsize=16)

        return fig, ax, h_axes

    def plt_boxplot(self, data):
        # Create the box plot
        # Purpose:

        numNeurons = self.numNeurons

        # Setup figure, main axes, and sub-axes
        fig, ax, h_axes = self.setup_axes()

        for neuron in range(numNeurons):
            # Make graph
            h_axes[neuron].boxplot(data[neuron])

        title = 'Cluster Centers as BoxPlot'
        plt.suptitle(title, fontsize=16)

        return fig, ax, h_axes

    def plt_dispersion_fan_plot(self, data):
        # Create the dispersion fan plot
        # Purpose:
        numNeurons = self.numNeurons

        # Setup figure, main axes, and sub-axes
        fig, ax, h_axes = self.setup_axes()

        # Draw histogram in each neuron
        for neuron in range(numNeurons):
            # Make graph
            h_axes[neuron].hist(data[neuron])

        title = 'Dispersion fan plot'
        plt.suptitle(title, fontsize=16)

        return fig, ax, h_axes

    def plt_violin_plot(self, data):
        # Create violin plot.
        # Purpose: ...
        numNeurons = self.numNeurons

        # Setup figure, main axes, and sub-axes
        fig, ax, h_axes = self.setup_axes()

        for neuron in range(numNeurons):
            # Make graph
            h_axes[neuron].violin(data[neuron])

        title = 'violin plot'
        plt.suptitle(title, fontsize=16)

        return fig, ax, h_axes

    def multiplot(self, plot_type, *args):
        # Dictionary mapping plot types to corresponding plotting methods
        plot_functions = {
            'pie': self.plt_pie,
            'dist': self.plt_dist,
            'wgts': self.plt_wgts,
            'hist': self.plt_histogram,
            'boxplot': self.plt_boxplot,
            'fanchart': self.plt_dispersion_fan_plot,
            'violin': self.plt_violin_plot
        }

        selected_plot = plot_functions.get(plot_type)
        if selected_plot:
            return selected_plot(*args)
        else:
            raise ValueError("Invalid function type")


    def plt_scatter(self, x, indices, clust, reg_line=True):
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

        # Data preprocessing
        # This should be updated!!!!
        x1 = x[indices[0], :]
        x2 = x[indices[1], :]

        # Setup figure, main axes, and sub-axes
        fig, ax, h_axes = self.setup_axes()

        # Loop over each neuron for hexagons and scatter plots
        for neuron in range(numNeurons):
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

        return fig, ax, h_axes

    def plt_pos(self, inputs=None):
        # Extract necessary information from the SOM object
        weights = self.w
        grid_shape = self.dimensions

        # Plotting the SOM weights
        plt.figure(figsize=(8, 8))
        plt.title('SOM Classification')
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                plt.plot(weights[i * grid_shape[1] + j][0], weights[i * grid_shape[1] + j][1], 'o', color='gray',
                         markersize=10)

        # Plotting input data if provided
        if inputs is not None:
            outputs = self.sim_som(inputs)
            for i in range(len(inputs)):
                winner_neuron = np.argmax(outputs[:, i])
                plt.plot(weights[winner_neuron][0], weights[winner_neuron][1], 'go', markersize=5)

        # Connect neighboring neurons with red lines
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                if i > 0:
                    plt.plot([weights[i * grid_shape[1] + j][0], weights[(i - 1) * grid_shape[1] + j][0]],
                             [weights[i * grid_shape[1] + j][1], weights[(i - 1) * grid_shape[1] + j][1]], '-r')
                if j > 0:
                    plt.plot([weights[i * grid_shape[1] + j][0], weights[i * grid_shape[1] + j - 1][0]],
                             [weights[i * grid_shape[1] + j][1], weights[i * grid_shape[1] + j - 1][1]], '-r')

        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

        plt.show()

    def plt_mouse_click(self, config):
        """
        plt_mouse_click is a function designed to create an interactive plot for Self-Organizing Maps (SOM).
        It allows users to click on a SOM's neurons and visualize data associated with those neurons in various formats
        (pie chart, histogram, scatter plot).
        Args:
            config: example
            data_config = {"data": ,
            # "clust"",
            # "num_var1": ,
            # "num_var2": ,
            # "cat_var":,
            # "top": }

        Returns:
            None
        """
        # Helper functions
        # Helper function to create charts
        def plot_pie(ax, data, neuronNum):
            # Clear the axes
            ax.clear()
            # Pie chart plot logic here
            ax.pie(data)
            ax.set_title('Pie Chart inside the Cluster ' + str(neuronNum))
            # Redraw the figure
            ax.figure.canvas.draw_idle()

        def plot_hist(ax, data, neuronNum):
            # Clear the axes
            ax.clear()
            # Histogram plot logic here
            data.plot(kind='hist', bins=15, ax=ax)
            ax.set_xlabel(data.name)
            ax.set_title('Histogram inside the Cluster ' + str(neuronNum))
            # Redraw the figure
            ax.figure.canvas.draw_idle()

        def plot_scatter(ax, data, num1, num2, neuronNum):
            # Clear the axes
            ax.clear()
            # Scatter plot logic here
            ax.scatter(data.iloc[:, num1], data.iloc[:, num2])
            ax.set_title('Scatter Plot inside the Cluster ' + str(neuronNum))
            ax.set_xlabel(data.columns[num1])
            ax.set_ylabel(data.columns[num2])
            # Redraw the figure
            ax.figure.canvas.draw_idle()

        pos = self.pos
        numNeurons = self.numNeurons

        # Determmine the hexagon shape
        shapex, shapey = get_hexagon_shape()

        # Create the original figure main axes
        fig, ax = plt.subplots(figsize=(6, 6), frameon=False)
        xmin = np.min(pos[0, :]) + np.min(shapex)
        xmax = np.max(pos[0, :]) + np.max(shapex)
        ymin = np.min(pos[1, :]) + np.min(shapey)
        ymax = np.max(pos[1, :]) + np.max(shapey)

        ax.set_title('click on points')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.set_axis_off()

        hexagons = []  # Hexagon container to identify each hexagon
        for neuron in range(numNeurons):
            hex, = ax.fill(pos[0, neuron] + shapex,
                           pos[1, neuron] + shapey,
                           facecolor=(1, 1, 1),
                           edgecolor=(0.8, 0.8, 0.8),
                           linewidth=1,
                           picker=True)
            hexagons.append(hex)

        # Assign the number of clusters to each hexagon
        hexagon_to_neuron = {hex: neuron for neuron, hex in enumerate(hexagons)}

        def onpick(event):
            if event.artist not in hexagons:
                return

            # Detect the clicked hexagon
            thishex = event.artist
            neuron_ind = hexagon_to_neuron[thishex]

            # Show up the menu if the cluster has data
            if len(config['clust'][neuron_ind]) > 0:
                # Create 2nd Figure
                fig, ax1 = plt.subplots(figsize=(6, 6))
                fig.subplots_adjust(right=0.8)

                # Button Configuration
                button_types = ['pie', 'hist', 'scatter']
                num_buttons = len(button_types)
                button_ratio = 16 / 9

                # Button sizing and positioning
                sidebar_width = 0.2
                single_button_width = sidebar_width * 0.8
                single_button_height = single_button_width / button_ratio
                margin = 0.05

                total_buttons_height = num_buttons * single_button_height + (num_buttons - 1) * margin

                # Create the buttons with dynamic positioning
                buttons = {}
                for i, button_type in enumerate(button_types):
                    # Calculate y position from top to bottom
                    y_pos = (1 - total_buttons_height) / 2 + (num_buttons - 1 - i) * (single_button_height + margin)
                    # Calculate x position which is centered in the right side of 0.2 width space in figure
                    x_centered = 0.8 + (0.2 - single_button_width) / 2
                    ax_button = fig.add_axes([x_centered, y_pos, single_button_width, single_button_height])
                    buttons[button_type] = Button(ax_button, button_type.capitalize(), hovercolor='0.975')

                # Create new data frame with the inputs that in the cluster
                temp_df = config['data'].iloc[config['clust'][neuron_ind]]
                temp_cat_df = config['cat'][config['clust'][neuron_ind]]

                top5_temp_df = temp_df.head(config['topn'])
                top5_cat_df = temp_cat_df[:config['topn']]

                # On click event for buttons
                buttons['pie'].on_clicked(lambda event: plot_pie(ax1, top5_cat_df, neuron_ind))
                # First Item distribution
                buttons['hist'].on_clicked(
                    lambda event: plot_hist(ax1, top5_temp_df.iloc[:, config["num1"]], neuron_ind))
                # First two items scatter plot
                buttons['scatter'].on_clicked(
                    lambda event: plot_scatter(ax1, top5_temp_df, config["num1"], config["num2"], neuron_ind))

                plt.show()

            else:
                print('No data in this cluster')

        fig.canvas.mpl_connect('pick_event', onpick)
        plt.show()