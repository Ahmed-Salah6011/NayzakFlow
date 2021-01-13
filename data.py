import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from nayzakflow.utils import _onehot

class CSVReader():
    def __init__(self,path,label_col_name,mode,split=None,oneHotLabel=False):
        self.path=path
        self.split=split
        self.is_one_hot= oneHotLabel
        self.label_name= label_col_name
        self.mode= mode

    def read_data(self):
        ds = pd.read_csv(self.path)
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
            y_valid= np.expand_dims(x_valid.pop(self.label_name).to_numpy(), axis=0)
            x_valid= x_valid.to_numpy()
            x_valid= x_valid.T


            x_train= ds_train.copy()
            y_train= np.expand_dims(x_train.pop(self.label_name).to_numpy(), axis=0)
            x_train= x_train.to_numpy()
            x_train= x_train.T

            if self.is_one_hot: ##error here must be fixed
                y_train= _onehot(y_train)
                y_valid= _onehot(y_valid)

            return (x_train,y_train), (x_valid,y_valid)
        else:
            x= ds.copy()
            y= np.expand_dims(x.pop(self.label_name).to_numpy() ,axis=0)
            x= x.to_numpy()
            x=x.T

            if self.is_one_hot: ##error here must be fixed
                y= _onehot(y)

            return (x,y)



class SparseDataReader():
    def __init__(self,folder_path,split=None,onehot=False):
        self.folder_path= folder_path
        self.split= split
        self.onehot= onehot

    
    def read_data(self):
        classes= os.listdir(self.folder_path)
        print("Found {} classes: ".format(len(classes)), classes)
        examples_list=[]
        labels_list=[]
        for clas in classes:
            path= os.path.join(self.folder_path,clas)
            examples= os.listdir(path)
           
            for example in examples :
                img = plt.imread(os.path.join(path,example))
                examples_list.append(img)
                labels_list.append(clas)
        
        examples_list=np.array(examples_list)
        labels_list=np.array(labels_list)

        #shuffle 1
        shuffler = np.random.permutation(len(labels_list))
        x = examples_list[shuffler]
        y = labels_list[shuffler]

        #shuffle 2
        shuffler = np.random.permutation(len(labels_list))
        x = x[shuffler]
        y = y[shuffler]
        if self.split:
            n_ele= int(self.split*len(labels_list))
            x_train= x[:n_ele]
            y_train= y[:n_ele]

            x_valid= x[n_ele:]
            y_valid= y[n_ele:]
            
            if self.onehot: ##error here must be fixed
                y_train= _onehot(y_train)
                y_valid= _onehot(y_valid)
            

            return (x_train,y_train), (x_valid,y_valid)
        
        else:
            if self.onehot:  ##error here must be fixed
                y= _onehot(y)


            return (x,y)
        



        




