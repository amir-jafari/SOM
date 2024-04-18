try:
    import cupy
    from .som_gpu import SOMGpu
except ImportError:
    from .som import SOM

from .utils import *
from .plots import SOMPlots
