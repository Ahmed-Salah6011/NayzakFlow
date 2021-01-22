import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from nayzakflow.utils import _onehot
import cv2

class CSVReader():
    def __init__(self,path,label_col_name,mode="regression",split=None,oneHotLabel=False):
        self.path=path
        self.split=split
        self.is_one_hot= oneHotLabel
        self.label_name= label_col_name
        self.mode= mode

    def read_data(self):
        ds = pd.read_csv(self.path)
        ds=ds.sample(frac=1).reset_index(drop=True)
        if self.mode=="classification":
            labels=set(ds[self.label_name])
            print("Labels are: ", labels)
            i=0
            for c in labels:
                ds.loc[ds[self.label_name] == c, self.label_name] = i
                i+=1
        
                
        if self.split:
            ds_valid = ds.sample(frac= self.split)
            ds_train = ds.drop(ds_valid.index)

            


            x_valid= ds_valid.copy()
            y_valid= np.expand_dims(x_valid.pop(self.label_name).to_numpy(), axis=1).astype('int')
            x_valid= x_valid.to_numpy()
            # x_valid= x_valid.T


            x_train= ds_train.copy()
            y_train= np.expand_dims(x_train.pop(self.label_name).to_numpy(), axis=1).astype('int')
            x_train= x_train.to_numpy()
            # x_train= x_train.T

            if self.is_one_hot: 
                y_train= _onehot(y_train,y_train.max()+1).T
                y_valid= _onehot(y_valid,y_valid.max()+1).T

            return (x_train,y_train), (x_valid,y_valid)
        else:
            x= ds.copy()
            y= np.expand_dims(x.pop(self.label_name).to_numpy() ,axis=1).astype('int')
            x= x.to_numpy()
            # x=x.T

            if self.is_one_hot: 
                y= _onehot(y,y.max()+1).T

            return (x,y)



class SparseDataReader():
    def __init__(self,folder_path,output_size=None,RGB=True,split=None,onehot=False):
        self.folder_path= folder_path
        self.split= split
        self.onehot= onehot
        self.output_size= output_size
        self.RGB=RGB

    
    def read_data(self):
        classes= os.listdir(self.folder_path)
        encoded_classes= {classes[c]:c for c in range(len(classes))}
        print("Found {} classes: ".format(len(classes)), encoded_classes)
        examples_list=[]
        labels_list=[]
        for clas in classes:
            path= os.path.join(self.folder_path,clas)
            examples= os.listdir(path)
            print("Found {} examples in class {}".format(len(examples),clas))

            for example in examples :
                if not self.RGB:
                    img= cv2.imread(os.path.join(path,example),cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(os.path.join(path,example))

                if self.output_size:
                    img = cv2.resize(img,self.output_size,interpolation=cv2.INTER_AREA)
                examples_list.append(img)
                labels_list.append(encoded_classes[clas])
        
        examples_list=np.array(examples_list)
        labels_list=np.array(labels_list)

        #shuffle 1
        shuffler = np.random.permutation(len(labels_list))
        x = examples_list[shuffler]
        y = labels_list[shuffler]

        #shuffle 2
        shuffler = np.random.permutation(len(labels_list))
        x = x[shuffler]
        y = np.expand_dims(y[shuffler],axis=1)

        if self.split:
            n_ele= int(self.split*len(labels_list))
            x_valid= x[:n_ele]
            y_valid= y[:n_ele]

            x_train= x[n_ele:]
            y_train= y[n_ele:]
            
            if self.onehot: 
                y_train= _onehot(y_train[:,0],y_train[:,0].max()+1).T
                y_valid= _onehot(y_valid[:,0],y_valid[:,0].max()+1).T
            

            return (x_train,y_train), (x_valid,y_valid)
        
        else:
            if self.onehot:  
                y= _onehot(y[:,0],y[:,0].max()+1).T


            return (x,y)
        



        




