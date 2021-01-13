import numpy as np
from nayzakflow.utils import _onehot

def sigmoid(z):
    return (1/(1+np.exp(-1*z)))

def _diff_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))


def tanh(z):
    return np.tanh(z)

def _diff_tanh(z):
    return 1-np.square(tanh(z))

def relu(z):
    return np.maximum(0,z)

def _diff_relu(z):
    a= np.zeros_like(z,dtype='int')
    a[z>0] = 1
    return a

def leaky_relu(z):
    return np.maximum(z,0.1*z)

def _diff_leaky_relu(z):
    a= np.zeros_like(z,dtype='int')+0.1
    a[z>0] = 1
    return a

def identity(z):
    return z

def _diff_identity(z):
    return 1

def softmax(z):
    exp = np.exp(z)
    tot= exp.sum(axis=0)
    t= exp/tot
    return t

def _diff_softmax(z,y):
    yhat_r = softmax(z)
    onehotY = _onehot(y)
    one_yi = onehotY *-1*(1-yhat_r)
    z=(1-onehotY)*yhat_r
    return one_yi +z




