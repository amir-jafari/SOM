try:
    import cupy as cp
    from .som_gpu import SOMGpu
except ImportError:
    from .som import SOM

from .utils import *
from .plots import SOMPlots
