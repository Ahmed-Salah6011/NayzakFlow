import numpy as np
from nayzakflow.utils import _onehot

def MSE(y,yhat):
    return (1/(2*len(y)))* np.sum(np.square(yhat-y))

def _diff_MSE(y,yhat):
    return (yhat-y)


def MAE(y,yhat):
    return (1/(len(y)))* np.sum(np.abs(yhat-y))

def _diff_MAE(y,yhat):
    diff= np.array((yhat-y)>0 , dtype="int")
    diff= ((1-diff)*-1)+diff

    return diff


def PerceptronCriteria(y,yhat):
    return (1/(len(y)))* np.sum(np.max(0, -1*y*yhat))

def _diff_PerceptronCriteria(y,yhat):
    diff = np.array( (y*yhat)<0, dtype="int" )
    diff*= (-1*y)
    return diff


def HingeLoss(y,yhat):
    return (1/(len(y)))* np.sum(np.max(0, 1-y*yhat))

def _diff_HingeLoss(y,yhat):
    diff = np.array( (y*yhat)<1, dtype="int" )
    diff*= (-1*y)
    return diff


def SigmoidLogLikelihood(y,yhat):
    return -1*(1/(len(y)))*np.sum(np.log(np.abs((y/2)-0.5+yhat)))

def _diff_SigmoidLogLikelihood(y,yhat):
    return (-1*y)/(1+np.exp(y*yhat))


def IdentityLogLikelihood(y,yhat):
    return (1/(len(y)))*np.sum(np.log(1+np.exp(-1*y*yhat)))

def _diff_IdentityLogLikelihood(y,yhat):
    return (-1*y*np.exp(-1*y*yhat))/(1+np.exp(-1*y*yhat))


def SoftmaxLogLikelihood(y,yhat):
    onehotY= _onehot(y)
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)
    return (1/(len(y)))*-1*np.sum(np.log(yhat_r))

def SoftmaxLogLikelihood_OneHotEncoded(y,yhat):
    onehotY= y
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)
    return (1/(len(y)))*-1*np.sum(np.log(yhat_r))

def MultiClassPerceptronCriteria(y,yhat):
    onehotY= _onehot(y)
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)

    d= np.maximum(yhat_r-yhat,0)
    return (1/(len(y)))*np.sum(np.max(d, axis=0))

def _diff_MultiClassPerceptronCriteria(y,yhat):
    onehotY= _onehot(y)
    return (1-onehotY) + (-1* onehotY)

def MultiClassHingeLoss(y,yhat):
    onehotY= _onehot(y)
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)
    z=1+(yhat_r-yhat)
    z= (1-onehotY)*z
    d= np.maximum(z,0)
    return (1/(len(y)))*np.sum(d, axis=0)

def _diff_MultiClassHingeLoss(y,yhat):
    onehotY= _onehot(y)
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)
    z=1+(yhat_r-yhat)
    z= (1-onehotY)*z
    d= np.maximum(z,0)
    de = np.sum(d>0)
    out = np.zeros_like(yhat)
    out[d>0]=1
    out = out+ onehotY*(-de)
    return out

def MultiClassPerceptronCriteria_OneHotEncoded(y,yhat):
    onehotY= y
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)

    d= np.maximum(yhat_r-yhat,0)
    return (1/(len(y)))*np.sum(np.max(d, axis=0))

def MultiClassHingeLoss_OneHotEncoded(y,yhat):
    onehotY= y
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)
    z=1+(yhat_r-yhat)
    z= (1-onehotY)*z
    d= np.maximum(z,0)
    return (1/(len(y)))*np.sum(d)



    




