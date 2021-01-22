import nayzakflow as nf
import numpy as np
import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt

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
    
    def set_params(self,W,b):
        self.W= W
        self.b= b
    
    def get_params(self):
        return [self.W, self.b]


    def zeroing_delta(self):
        self.dW= np.zeros_like(self.W)
        self.db= np.zeros_like(self.b)

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

        e = np.ones((self.X.shape[1],1))
        bet= input * f_dash

        self.dW= self.dW+ np.dot(bet,self.X.T)
        self.db= self.db+ np.dot(bet,e)

        return np.dot(self.W.T, bet)
    

class Model():
    def __init__(self):
        self.loss=None
        self.optimizer= None
        self.metrics= None

class Sequential(Model):
    def __init__(self,layers=None):
        super().__init__()
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
    
    def zeroing(self):
        for layer in self.layers:
            layer.zeroing_delta()
        
        
    def _calc_metrics(self, y,y_hat,labels,val=False,eval=False,metrics=None):
        if self.loss != nf.nn.loss.MSE and self.loss != nf.nn.loss.MAE:
            if y_hat.shape[0]>1:
                yh = np.argmax(y_hat, axis=0)
            else:
                yh= np.zeros_like(y_hat)
                yh[y_hat>=0.5] = np.max((y[0]))
                yh[y_hat<0.5] = np.min((y[0]))
                yh=np.array(yh[0], dtype='int')
        
        else:
            yh= y_hat
            
        if eval:
            for m in self.metrics.keys():
                met = self.metrics[m](y[0],yh,labels)
                metrics.append(met)
            
            return metrics
            
        else:
            mms=[]
            for m in self.metrics.keys():
                met = self.metrics[m](y[0],yh,labels)
                mms.append(met)
                if val:
                    print("val_{}: {}   ".format(m,met),end=" ")
                else:
                    print("{}: {}   ".format(m,met),end=" ")

            return mms

    def batch(self,x,y,bs):
        x= x.copy()
        y=y.copy()
        # no_of_batches= int(np.ceil(len(x)/bs))
        rem= x.shape[0] % bs


        # out_x=[]
        # out_y=[]
        for i in range(0,x.shape[0],bs):
            # if len(x)<=bs:
            #     # out_x.append(x)
            #     # out_y.append(y)
            #     yield (x.copy(), y.copy())
            #     break
            # indx=list(random.sample(range(len(x)),bs))
            # curr_x= x[indx].copy()
            # curr_y= y[indx].copy()
            # # out_x.append(x[indx])
            # # out_y.append(y[indx])
            # x=np.delete(x,indx,axis=0)
            # y=np.delete(y,indx,axis=0)
            yield (x[i:i+bs],y[i:i+bs])
        
        if rem !=0:
            yield (x[x.shape[0]-rem:],y[x.shape[0]-rem:] )

        # return (np.array(out_x), np.array(out_y))

    def fit(self, train_data,validation_data=None, batch_size=32, epochs=5,plot=False):
        x_train = train_data[0]
        y_train = train_data[1]
        no_of_batches_train = np.ceil(x_train.shape[0]/batch_size)
        ###
        if self.loss != nf.nn.loss.MSE and self.loss != nf.nn.loss.MAE:
            self.labels = np.unique(y_train)
        else:
            self.labels = None
        ###
        if validation_data:
            x_valid= validation_data[0]
            y_valid= validation_data[1]

        ##############visualization
        if plot:
            nf.visualize.Visualize.setNumberOfPlots(1+len(self.metrics))
            plt.tight_layout()
            visual_metrics=[]
            if validation_data:
                loss_title =["loss","val_loss"]
                for met in self.metrics:
                    visual_metrics.append(nf.visualize.Visualize([met,met+"_val"]))
            else:
                loss_title =["loss"]
                for met in self.metrics:
                    visual_metrics.append(nf.visualize.Visualize([met]))

            visual_loss= nf.visualize.Visualize(loss_title)




        for i in range(epochs):
            print("Epoch {}/{}".format(i+1,epochs))
            v_l=[]
            j=0
            k=0
            data=self.batch(x_train,y_train,batch_size)
            for curr_x,curr_y in data:
                # print(curr_x.shape)
                curr_x =curr_x.T
                curr_y = curr_y.T
                y_hat= self.forward(curr_x)

                if self.loss ==nf.nn.loss.SoftmaxLogLikelihood:
                    self.layers[-1]._set_target(curr_y)
                    self.backward(1)
                
                else:
                    dl = nf.nn.loss.get_diffs()[self.loss](curr_y,y_hat)
                    self.backward(dl)
                
                if int(0.1*no_of_batches_train) == (k+1):
                    print("=",end="")
                    k=0
                # print(j)
                # print(no_of_batches_train)
                if j == no_of_batches_train-1: #last batch
                    loss= self.loss(curr_y,y_hat)
                    v_l.append(loss)
                    print()
                    print("loss: {}....".format(loss),end=" ")

                    ######
                    # calc metrics
                    mms=self._calc_metrics(curr_y,y_hat,self.labels)
                    ######
                if batch_size ==1:
                    N= train_data[0].shape[0]
                else:
                    N= curr_x.shape[-1]

                self.optimizer.update(self.layers,N)
                j+=1
                k+=1
            
            ###
            if validation_data:
                y_hat_val = self.forward(x_valid.T)
                loss_val= self.loss(y_valid.T,y_hat_val)
                v_l.append(loss_val)
                print("val_loss: {}....".format(loss_val),end=" ")
                ######
                #calc metrics
                mms_val=self._calc_metrics(y_valid,y_hat_val,self.labels,True)
                #####
            ###
            if plot:
                plt.clf()
                visual_loss.draw(v_l)
                i=0
                for vis_met in visual_metrics:
                    if validation_data:
                        vis_met.draw([mms[i], mms_val[i]])
                    else:
                        vis_met.draw([mms[i]])
                    
                    i+=1
                plt.pause(0.0001)

            print()
            self.zeroing()
        if plot:
            plt.show()

    def predict(self,data): #data dim is NxD .. N no of examples.. D no of dimension
        y_hat= self.forward(data.T)
        return y_hat.T
    
    def evaluate(self,x_test,y_test,draw_confusion_matrix=False):
        y_hat= self.forward(x_test.T)
        loss= self.loss(y_test.T,y_hat)

        metrics=[]
        #calc metric
        metrics= self._calc_metrics(y_test.T,y_hat,self.labels,eval=True,metrics=metrics)
        #########################################
        if draw_confusion_matrix:
            y=y_test.T
            if self.loss != nf.nn.loss.MSE and self.loss != nf.nn.loss.MAE:
                if y_hat.shape[0]>1:
                    yh = np.argmax(y_hat, axis=0)
                else:
                    yh= np.zeros_like(y_hat)
                    yh[y_hat>=0.5] = np.max((y[0]))
                    yh[y_hat<0.5] = np.min((y[0]))
                    yh=np.array(yh[0], dtype='int')
        
            else:
                yh= y_hat

            mat= nf.nn.metrics.confusion_matrix(y[0],yh,self.labels)
            print(mat)
            nf.visualize.Visualize.heatmap(mat)
        return (loss , metrics)
    
    def get_weights(self):
        params=[]
        for layer in self.layers:
            params.append(layer.get_params())
        
        return params

    def save_weights(self, path):
        #the path containing the file name
        params= []
        for layer in self.layers:
            layer_param= [layer.W , layer.b]
            params.append(layer_param)
        
        # params = np.array(params)
        file = open(path, 'wb')

        # dump information to that file
        pickle.dump(params, file)

        # close the file
        file.close()
    
    def load_weights(self, path):
        file = open(path, 'rb')

        # dump information to that file
        params = pickle.load(file)

        # close the file
        file.close()

        i=0
        for layer in self.layers:
            layer.set_params(params[i][0],params[i][1])
            i+=1
        



        
            










        

        

        


    




        
    


    


    
    
