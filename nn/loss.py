import numpy as np
from nayzakflow.utils import _onehot

def MSE(y,yhat):
    return (1/(2*len(y)))*np.square(yhat-y).sum(axis=1, keepdims=True)


def MAE(y,yhat):
    return (1/(2*len(y)))*np.abs(yhat-y).sum(axis=1, keepdims=True)


def PerceptronCriteria(y,yhat):
    return (1/(len(y)))*np.max(0, -1*y*yhat).sum(axis=1,keepdims=True)


def HingeLoss(y,yhat):
    return (1/(len(y)))*np.max(0, 1-y*yhat).sum(axis=1,keepdims=True)

def SigmoidLogLikelihood(y,yhat):
    return -1*(1/(len(y)))*np.log(np.abs((y/2)-0.5+yhat)).sum(axis=1,keepdims=True)

def IdentityLogLikelihood(y,yhat):
    return (1/(len(y)))*np.log(1+np.exp(-1*y*yhat)).sum(axis=1,keepdims=True)

def SoftmaxLogLikelihood(y,yhat):
    onehotY= _onehot(y)
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)
    return (1/(len(y)))*-1*np.log(yhat_r).sum(axis=1)

def SoftmaxLogLikelihood_OneHotEncoded(y,yhat):
    onehotY= y
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)
    return (1/(len(y)))*-1*np.log(yhat_r).sum(axis=1)

def MultiClassPerceptronCriteria(y,yhat):
    onehotY= _onehot(y)
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)

    d= np.maximum(yhat_r-yhat,0)
    return (1/(len(y)))*np.max(d, axis=0).sum(axis=1)

def MultiClassHingeLoss(y,yhat):
    onehotY= _onehot(y)
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)
    z=1+(yhat_r-yhat)
    z= (1-onehotY)*z
    d= np.maximum(z,0)
    return (1/(len(y)))*np.sum(d, axis=0).sum(axis=1)

def MultiClassPerceptronCriteria_OneHotEncoded(y,yhat):
    onehotY= y
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)

    d= np.maximum(yhat_r-yhat,0)
    return (1/(len(y)))*np.max(d, axis=0).sum(axis=1)

def MultiClassHingeLoss_OneHotEncoded(y,yhat):
    onehotY= y
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)
    z=1+(yhat_r-yhat)
    z= (1-onehotY)*z
    d= np.maximum(z,0)
    return (1/(len(y)))*np.sum(d, axis=0).sum(axis=1)



    




