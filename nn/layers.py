import nayzakflow as nf
import numpy as np
import pandas as pd
import random

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

    def compile(self,loss,optimizer,metrics=["accuracy"]):
        self.loss = loss
        self.optimizer= optimizer
        self.metrics= {metric:nf.nn.metrics.get_metrics()[metric] for metric in metrics}
    
    def forward(self,input):
        a=input
        for layer in self.layers:
            a = layer.forward(a)
        
        return a
    
    def backward(self,input):
        gd = input
        for layer in self.layers[::-1]:
            gd = layer.backward(gd)
    

    def _bathcing(self,x,y,bs):
        x= x.copy()
        y=y.copy()
        no_of_batches= np.ceil(len(x)/bs)
        # remaining_elements_no = (len(x)%bs)

        out_x=[]
        out_y=[]
        for _ in range(no_of_batches):
            if len(x)<bs:
                out_x.append(x)
                out_y.append(y)
                break
            indx=list(random.sample(range(len(x)),bs))
            out_x.append(x[indx])
            out_y.append(y[indx])
            x=np.delete(x,indx,axis=0)
            y=np.delete(y,indx,axis=0)
        
        return (np.array(out_x), np.array(out_y))
        

        



    def fit(self, train_data,validation_data=None, batch_size=32, epochs=5):
        x_train = train_data[0]
        y_train = train_data[1]
        no_of_batches_train= np.ceil(len(x_train)/batch_size)
        if validation_data:
            x_valid= validation_data[0]
            y_valid= validation_data[1]
            # x_valid, y_valid= self._bathcing(x_valid, y_valid,batch_size)
            # no_of_batches_valid= np.ceil(len(x_valid)/batch_size)
        
        x_train, y_train= self._bathcing(x_train, y_train,batch_size)

        for i in range(epochs):
            print("Epoch {}/{}....".format(i+1,epochs),end=" ")
            for j in range(no_of_batches_train):
                y_hat= self.forward(x_train[j])
                dl = nf.nn.loss.get_diffs()[self.loss](y_train[j],y_hat)
                self.backward(dl)

                if j == no_of_batches_train-1: #last batch
                    loss= self.loss(y_train[j],y_hat)
                    print("loss: {}....".format(loss),end=" ")
                    #calc metrics
                    for m in self.metrics.keys():
                        met = self.metrics[m](y_train[j],y_hat)
                        print("{}: {}...".format(m,met),end=" ")


                self.optimizer.update(self.layers)
            
            ###
            if validation_data:
                y_hat_val = self.forward(x_valid)
                loss_val= self.loss(y_valid,y_hat_val)
                print("val_loss: {}....".format(loss_val),end=" ")
                #calc metrics
                for m in self.metrics.keys():
                        met = self.metrics[m](y_valid,y_hat_val)
                        print("val_{}: {}...".format(m,met),end=" ")
            ###
                





        

        

        


    




        
    


    


    
    
