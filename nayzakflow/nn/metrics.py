import numpy as np
import pandas as pd

def confusion_matrix(y,yhat,labels):
    y_actu = pd.Series(y, name='Actual')
    y_pred = pd.Series(yhat, name='Predicted')
    df_confusion = pd.crosstab(y_pred,y_actu,dropna=False)
    df_confusion = df_confusion.reindex(index=labels, columns= labels, fill_value=0)
    return df_confusion.to_numpy()

# def calc(confusion_matrix):
#     if(confusion_matrix.shape[0] == 2):
#         TP = confusion_matrix[0][0]
#         FP = confusion_matrix[0][1]
#         FN = confusion_matrix[1][0]
#         TN = confusion_matrix[1][1]
#         return TP,FN,FP,TN
#     length = np.sum(confusion_matrix)
#     # print(length)
#     I = np.eye(confusion_matrix.shape[0])
#     TP = (I*confusion_matrix).sum(axis = 1).sum(axis = 0)
#     FN = np.sum(np.multiply(1-I,confusion_matrix))
#     FP = ((1-I)*confusion_matrix).sum(axis = 1).sum(axis = 0)
#     TN = length*confusion_matrix.shape[0]-TP-FP-FN
#     # print(TP.shape)
#     return [TP,FN,FP,TN]

def new_recall(y,yhat,labels,matrix):
    new_list=list()
    for x in range(matrix.shape[0]):
        cur_colum=matrix[:,x]
        test=np.sum(cur_colum)
        if test == 0 :
            new_list.append(test)
        else :
            cur_recall=cur_colum[x]/np.sum(cur_colum)
            new_list.append(cur_recall)

    return np.array(new_list)

def new_recall_sum(y,yhat,labels):
    matrix = confusion_matrix(y,yhat,labels)
    if(matrix.shape[0] == 2):
        TP = matrix[0][0]
        FP = matrix[0][1]
        FN = matrix[1][0]
        TN = matrix[1][1]
        return TP / (TP + FN) if (TP + FN) else 0
    ls=new_recall(y,yhat,labels,matrix)
    return np.sum(ls) / labels.shape[0]

def new_precision(y,yhat,labels,matrix):
    matrix = confusion_matrix(y,yhat,labels)
    new_list=list()
    for x in range(matrix.shape[0]):
        cur_row=matrix[x,:]
        test=np.sum(cur_row)
        if test == 0 :
            new_list.append(test)
        else :
            cur_precision=cur_row[x]/np.sum(cur_row)
            new_list.append(cur_precision)

    return np.array(new_list)

def new_precision_sum(y,yhat,labels):
    matrix = confusion_matrix(y,yhat,labels)
    if(matrix.shape[0] == 2):
        TP = matrix[0][0]
        FP = matrix[0][1]
        FN = matrix[1][0]
        TN = matrix[1][1]
        return TP / (TP + FP) if (TP + FP) else 0
    ls=new_precision(y,yhat,labels,matrix)
    return np.sum(ls) / labels.shape[0]


def new_f1_score(y,yhat,labels,matrix):
    recall1=new_recall(y,yhat,labels,matrix)
    percision1=new_precision(y,yhat,labels,matrix)
    term1 =2 *  recall1 * percision1
    term2 =(recall1 + percision1)
    result = [term1[i] / term2[i] if term2[i] else 0 for i in range(term1.shape[0])]
    return np.array(result)

def new_f1_score_sum(y,yhat,labels):
    matrix = confusion_matrix(y,yhat,labels)
    if(matrix.shape[0] == 2):
        TP = matrix[0][0]
        FP = matrix[0][1]
        FN = matrix[1][0]
        TN = matrix[1][1]
        pre = TP / (TP + FP) if (TP + FP) else 0
        recal = TP / (TP + FN) if (TP + FN) else 0
        return (2 * pre * recal) / (pre + recal) if (pre + recal) else 0
    ls=new_f1_score(y,yhat,labels,matrix)
    return np.sum(ls) / labels.shape[0]


    
# def new_accuracy(y,yhat,labels,matrix):
#     ls=list()
#     for x in range(matrix.shape[0]):
#         mat=np.delete(matrix,x,axis=0)
#         mat=np.delete(mat,x,axis=1)
#         true_num=matrix[x,x]
#         mat_sum = np.sum(mat)
#         test = np.sum(matrix)
#         if test == 0 :
#             ls.append(0)
#         else :    
#             acc = (true_num) / (np.sum(matrix) - mat_sum)
#             ls.append(acc)

#     return np.array(ls)


def new_accuracy_sum(y,yhat,labels):
    matrix = confusion_matrix(y,yhat,labels)
    I=np.eye(matrix.shape[0])
    if(matrix.shape[0] == 2):
        TP = matrix[0][0]
        FP = matrix[0][1]
        FN = matrix[1][0]
        TN = matrix[1][1]
        return (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0
    total_True = np.sum(I*matrix)
    acc = total_True / np.sum(matrix) if np.sum(matrix) else 0
    # ls=new_accuracy(y,yhat,labels,matrix)
    #np.sum(ls) / labels.shape[0]
    return acc




# def accuracy(y,yhat,labels):
#     matrix = confusion_matrix(y,yhat,labels)
#     test = calc(matrix)
#     # print(test)
#     x = test[0] + test[3]
#     z = test[0] +test[1] +test[2] +test[3] 
#     return(x/z)

# def percision(y,yhat,labels):
#     matrix = confusion_matrix(y,yhat,labels)
#     test = calc(matrix)
#     # print(test[0]/(test[0]+test[1]))
#     return test[0]/(test[0]+test[2])

# def recall(y,yhat,labels):
#     matrix = confusion_matrix(y,yhat,labels)
#     test = calc(matrix)
#     # print(test[0]/(test[0]+test[1]))
#     return test[0]/(test[0]+test[1])

# def f1_score(y,yhat,labels):
#     R = recall(y,yhat,labels)
#     P = percision(y,yhat,labels)
#     return((2*P*R)/(P+R))

def mae(y,yhat,labels=None):
    return (1/len(y))*np.sum(np.abs(y-yhat))

def rmse(y,yhat,labels=None):
    return np.sqrt((1/len(y))*np.sum(np.square(y-yhat)))

def get_metrics():
    return {"accuracy":new_accuracy_sum, "precision": new_precision_sum, "recall":new_recall_sum,"f1-score":new_f1_score_sum, "mae":mae, "rmse": rmse}