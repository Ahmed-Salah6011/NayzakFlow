import numpy as np
from nayzakflow.utils import _onehot

def MSE(y,yhat):
    a=np.square(yhat-y)
    a=np.sum(a)
    b= 1/(2*y.shape[1])
    return a*b

def _diff_MSE(y,yhat):
    return (yhat-y)


def MAE(y,yhat):
    return (1/(y.shape[1]))* np.sum(np.abs(yhat-y))

def _diff_MAE(y,yhat):
    diff= np.array((yhat-y)>0 , dtype="int")
    diff= ((1-diff)*-1)+diff

    return diff


def PerceptronCriteria(y,yhat):
    return (1/(y.shape[1]))* np.sum(np.maximum(0, -1*y*yhat))

def _diff_PerceptronCriteria(y,yhat):
    diff = np.array( (y*yhat)<=0, dtype="int" )
    diff*= (-1*y)
    return diff


def HingeLoss(y,yhat):
    return (1/(y.shape[1]))* np.sum(np.maximum(0, 1-y*yhat))

def _diff_HingeLoss(y,yhat):
    diff = np.array( (y*yhat)<=1, dtype="int" )
    diff*= (-1*y)
    return diff


def SigmoidLogLikelihood(y,yhat):
    return (1/(y.shape[1]))*np.sum(-1*np.log(np.abs((y/2)-0.5+yhat)))

def _diff_SigmoidLogLikelihood(y,yhat):
    return -1* y/ ( (((1+y)/2)*yhat) + (((1-y)/2)*(1-yhat))  )


def IdentityLogLikelihood(y,yhat):
    return (1/(y.shape[1]))*np.sum(np.log(1+np.exp(-1*y*yhat)))

def _diff_IdentityLogLikelihood(y,yhat):
    return (-1*y*np.exp(-1*y*yhat))/(1+np.exp(-1*y*yhat))


def SoftmaxLogLikelihood(y,yhat):
    onehotY= _onehot(y,yhat.shape[0])
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)
    return (1/(y.shape[1]))*-1*np.sum(np.log(yhat_r))

def SoftmaxLogLikelihood_OneHotEncoded(y,yhat):
    onehotY= y
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)
    return (1/(y.shape[1]))*-1*np.sum(np.log(yhat_r))

def MultiClassPerceptronCriteria(y,yhat):
    onehotY= _onehot(y,yhat.shape[0])
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)

    d= np.maximum(yhat-yhat_r,0)
    return (1/(y.shape[1]))*np.sum(np.max(d, axis=0))

def _diff_MultiClassPerceptronCriteria(y,yhat):
    onehotY= _onehot(y,yhat.shape[0])
    max_index=np.argmax(yhat,axis=0)
    temp=np.zeros_like(yhat)
    for i in range(temp.shape[1]): temp[max_index[i],i]=1 
    return  temp+ (-1* onehotY)

def MultiClassHingeLoss(y,yhat):
    onehotY= _onehot(y,yhat.shape[0])
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)
    z=1+(yhat-yhat_r)
    z= (1-onehotY)*z
    d= np.maximum(z,0)
    return (1/(y.shape[1]))*np.sum(np.sum(d, axis=0))

def _diff_MultiClassHingeLoss(y,yhat):
    onehotY= _onehot(y,yhat.shape[0])
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)
    z=1+(yhat-yhat_r)
    z= (1-onehotY)*z
    d= np.maximum(z,0)
    de = np.sum(d>0,axis=0)
    out = np.zeros_like(yhat)
    out[d>0]=1
    out = out+ onehotY*(-de)
    return out

def MultiClassPerceptronCriteria_OneHotEncoded(y,yhat):
    onehotY= y
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)

    d= np.maximum(yhat_r-yhat,0)
    return (1/(y.shape[1]))*np.sum(np.max(d, axis=0))

def MultiClassHingeLoss_OneHotEncoded(y,yhat):
    onehotY= y
    yhat_r = np.max(onehotY*yhat, axis=0,keepdims=True)
    z=1+(yhat_r-yhat)
    z= (1-onehotY)*z
    d= np.maximum(z,0)
    return (1/(y.shape[1]))*np.sum(d)



def get_diffs():
    return {MSE:_diff_MSE , MAE:_diff_MAE, PerceptronCriteria:_diff_PerceptronCriteria, HingeLoss:_diff_HingeLoss,
            SigmoidLogLikelihood: _diff_SigmoidLogLikelihood , IdentityLogLikelihood:_diff_IdentityLogLikelihood, 
            MultiClassPerceptronCriteria:_diff_MultiClassPerceptronCriteria , MultiClassHingeLoss:_diff_MultiClassHingeLoss}




