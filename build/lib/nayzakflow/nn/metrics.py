import numpy as np
import pandas as pd

def confusion_matrix(y,yhat,labels):
    y_actu = pd.Series(y, name='Actual')
    y_pred = pd.Series(yhat, name='Predicted')
    df_confusion = pd.crosstab(y_pred,y_actu,dropna=False)
    df_confusion = df_confusion.reindex(index=labels, columns= labels, fill_value=0)
    return df_confusion.to_numpy()

def calc(confusion_matrix):
    if(confusion_matrix.shape[0] == 2):
        TP = confusion_matrix[0][0]
        FP = confusion_matrix[0][1]
        FN = confusion_matrix[1][0]
        TN = confusion_matrix[1][1]
        return TP,FN,FP,TN
        
    length = np.sum(confusion_matrix)
    # print(length)
    I = np.eye(confusion_matrix.shape[0])
    TP = (I*confusion_matrix).sum(axis = 1).sum(axis = 0)
    FN = np.sum(np.multiply(1-I,confusion_matrix))
    FP = ((1-I)*confusion_matrix).sum(axis = 1).sum(axis = 0)
    TN = length*confusion_matrix.shape[0]-TP-FP-FN
    # print(TP.shape)
    return [TP,FN,FP,TN]

def accuracy(y,yhat,labels):
    matrix = confusion_matrix(y,yhat,labels)
    test = calc(matrix)
    # print(test)
    x = test[0] + test[3]
    z = test[0] +test[1] +test[2] +test[3] 
    return(x/z)

def percision(y,yhat,labels):
    matrix = confusion_matrix(y,yhat,labels)
    test = calc(matrix)
    # print(test[0]/(test[0]+test[1]))
    return test[0]/(test[0]+test[2])

def recall(y,yhat,labels):
    matrix = confusion_matrix(y,yhat,labels)
    test = calc(matrix)
    # print(test[0]/(test[0]+test[1]))
    return test[0]/(test[0]+test[1])

def f1_score(y,yhat,labels):
    R = recall(y,yhat,labels)
    P = percision(y,yhat,labels)
    return((2*P*R)/(P+R))

def mae(y,yhat,labels=None):
    return (1/len(y))*np.sum(np.abs(y-yhat))

def rmse(y,yhat,labels=None):
    return np.sqrt((1/len(y))*np.sum(np.square(y-yhat)))

def get_metrics():
    return {"accuracy":accuracy, "precision": percision, "recall":recall,"f1-score":f1_score, "mae":mae, "rmse": rmse}