import nayzakflow as nf
import numpy as np
import pandas as pd


class Linear():
    def __init__(self,n_output,n_input, activation="identity",name=None,parameter_initializer="normal",parameters=None,target=None):
        self.n_output= n_output
        self.n_input= n_input
        self.name= name
        self.activation = nf.nn.activation.get_activations()[activation]
        self.act_name=activation
        self.target=target

        if parameters:
            self.W= parameters['W']
            self.b= parameters['b']

        else:
            if parameter_initializer == "normal":
                self.W= np.random.normal(0,1,(self.n_output,self.n_input))
                self.b= np.random.normal(0,1,(self.n_output,1))
            elif parameter_initializer == "uniform":
                self.W= np.random.uniform(0,1,(self.n_output,self.n_input))
                self.b= np.random.uniform(0,1,(self.n_output,1))
        
        self.dW= np.zeros_like(self.W)
        self.db= np.zeros_like(self.b)

        self.Z=None
        self.X=None
        

    def _set_target(self, t):
        self.target=t

    def forward(self,input):
        x=input
        z = np.dot(self.W, x)+self.b
        A = self.activation(z)
        self.Z=z
        self.X= x
        return A
    
    def backward(self,input):
        if self.act_name == "softmax":
            f_dash = nf.nn.activation._diff_softmax(self.Z,self.target)
        
        else:
            f_dash = nf.nn.activation.get_activations_diff()[self.act_name](self.Z)

        bet= input * f_dash
        self.dW= np.dot(bet,self.X.T)
        self.db= bet

        return np.dot(self.W.T, bet)
    

class Model():
    def __init__(self):
        self.loss=None
        self.optimizer= None
        self.metrics= None

class Sequential(Model):
    def __init__(self,layers=None):
        super(Sequential).__init__(self)
        if layers == None:
            self.layers=None
        else:
            self.layers= list(layers)

    def add(self,layer):
        self.layers.append(layer)
    
    def summary(self):
        df = pd.DataFrame(columns=["No.","W_Shape","Activation","Name"])
        no = ["Layer{}".format(i+1) for i in range(len(self.layers))]
        wsh = [self.layers[i].W.shape for i in range(len(self.layers))]
        act = [self.layers[i].act_name for i in range(len(self.layers))]
        name = [self.layers[i].name for i in range(len(self.layers))]
        df["No."]= no
        df["W_Shape"]= wsh
        df["Activation"]= act
        df["Name"]= name
        print(df.head(len(df)))
    
    def get_layers(self):
        return self.layers

    def compile(self,loss,optimizer,metrics="accuracy"):
        self.loss = loss
        self.optimizer= optimizer
        # self.metrics= 

    




        
    


    


    
    
