import numpy as np

def _onehot(a,M):
    b = np.zeros( (a.size, M), dtype='int')
    b[ np.arange(a.size),a] = 1
    return b.T
