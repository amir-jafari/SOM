from scipy.spatial.distance import cdist
from scipy import sparse
import numpy as np
import networkx as nx


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


