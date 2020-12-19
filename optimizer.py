import numpy as np
import nayzakflow as nf


class Vanilla_GD(object):
    def __init__(self, parameters,x,y,loss, alpha=0.1):
        self.parameters = parameters
        self.alpha = alpha
        self.x=x
        self.y=y
        self.loss= loss
    
    def zero(self):
        for p in self.parameters:
            # p= nf.Tensor(p)
            p.grad.data *= 0

    
    def update_param(self,zero):
        for p in self.parameters:
            # p=nf.Tensor(p)
            p.data -= p.grad.data * self.alpha
            if(zero):
                p.grad.data *= 0

    def step(self, zero=True):
        pred=self.x
        for param in self.parameters:
            # param=nf.Tensor(param)
            pred= pred.mm(param)
        

        loss = self.loss(pred,self.y).sum(0)
        loss.backward(nf.Tensor(np.ones_like(loss.data)))

        self.update_param(zero)
        print(loss)



class SGD(object):
    def __init__(self, parameters,x,y,loss, alpha=0.1):
        self.parameters = parameters
        self.alpha = alpha
        self.x=x
        self.y=y
        self.loss= loss
    
    def zero(self):
        for p in self.parameters:
            # p= nf.Tensor(p)
            p.grad.data *= 0

    
    def update_param(self,zero):
        for p in self.parameters:
            # p= nf.Tensor(p)
            p.data -= p.grad.data * self.alpha
            if(zero):
                p.grad.data *= 0

    def step(self, zero=True):
        i=0
        for d in self.x:
            d=nf.Tensor(d,autograd=True)
            pred=d.expand(1,1)
            for param in self.parameters:
                # param= nf.Tensor(param)
                pred= param.transpose().mm(pred)
            
            loss = self.loss(pred,self.y[i])
            loss.backward(nf.Tensor(np.ones_like(loss.data)))
            i+=1
            self.update_param(zero)
        print(loss)
        
        