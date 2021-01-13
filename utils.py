import numpy as np

def _onehot(a):
    b = np.zeros((a.size, a.max()+1),dtype='int')
    b[np.arange(a.size),a] = 1
    return b.T
