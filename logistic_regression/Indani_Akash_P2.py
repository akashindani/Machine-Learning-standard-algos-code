# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 11:16:37 2020

@author: akash
"""

import numpy as np
import matplotlib.pyplot as plt


def convert_data_to_numpy(file):
    file_data=open(file,"r")
    line=file_data.readline()
    m=int(line.split('\t')[0].strip())
    n=int(line.split('\t')[1].strip())
    data=np.loadtxt(file, skiprows=1)

    return data,m,n

def split_input_output(data): 
    #adding bias x0 as 1 in x for generating x(input) and y(output) 
    x=np.hstack((np.ones((data.shape[0],1)),data[:, :-1]))
    y=data[:, -1].reshape((data.shape[0],1))
    #
    return x,y

def compute_hypothesis(x,w):
    global e
    xw=np.dot(x,w)
    return 1/(1+np.power(e, -1*xw))

def compute_j_with_regilization(y,h,m,n,reg_lambda,w):
    O_1xm=np.ones((1,m))
    ZO_1xn = np.ones((1,n+1))
    ZO_1xn[0][0]=0    
    cost = -1*(np.multiply(y,np.log(h))) - np.multiply((1-y), np.log(1-h))
    j= (1/m)* np.dot(O_1xm, cost) + (reg_lambda/(2*m)) * np.dot(ZO_1xn, np.power(w,2))
    
    return j

def compute_new_weights(train_alpha, reg_lambda, h, x, y, w,m):
    reduced_w = np.vstack((w[:1],(1- train_alpha * (reg_lambda/m)) * w[1:]))
    return reduced_w - np.transpose((train_alpha/m) * np.dot(np.transpose(h-y), x))

def plot_the_model(j_list,model):
    plt.figure()
    plt.plot(np.arange(len(j_list)), j_list)
    plt.xlabel("Time in epochs")
    plt.ylabel("cost Function (J)")
    plt.savefig(model+".png")
    plt.show()
    plt.close()


def predict(x_test, w):
    global e
    y_pred=1/ (1 + np.power(e,-1*np.dot(x_test,w)))
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0   
    return y_pred

def give_me_all_evaluaton_metrics(y_pred,y_test):
    TP=0
    FP=0
    FN=0
    TN=0
    for i in range(y_pred.shape[0]):
        if y_test[i][0] == 1.0 and y_pred[i][0] == 1.0:
            TP = TP +1
        elif y_test[i][0] == 0.0 and y_pred[i][0] == 0.0:
            TN = TN +1
        elif y_test[i][0] == 0.0 and y_pred[i][0] == 1.0:
            FP = FP +1
        elif y_test[i][0] == 1.0 and y_pred[i][0] == 0.0:
            FN = FN +1
    accuracy = (TP + TN)/ (TP + TN + FP + FN)
    precision = TP / (TP+FP)
    recall = TP / (TP + FN)
    f_score= 2 * (1/ ((1/precision) + (1/recall)))
    return FP,FN,TP,TN,accuracy, precision, recall, f_score
            
            
if __name__ == "__main__":
    e=2.718281
    train_alpha=0.05
    reg_lambda=2
    
    train_file = input("Enter your training file name: ") 
    data,m,n=convert_data_to_numpy(train_file)
    
    #split x(input) and y(output)
    x,y=split_input_output(data)
    
    #initial weights        
    weights=np.zeros((x.shape[1],1))
    
    #hypothesis computation
    H=compute_hypothesis(x,weights)

    
    #computing cost and j
    j=compute_j_with_regilization(y,H,m,n,reg_lambda,weights)
    
    new_w = compute_new_weights(train_alpha, reg_lambda, H, x, y, weights,m)
    weights=new_w
    
    j_new=np.zeros((1,1))
    
    j_list=[]
    j_list.append(j[0][0])
    
    for i in range(2500):
        j=j_new
        j_new = compute_j_with_regilization(y,H,m,n,reg_lambda,weights)

        new_w = compute_new_weights(train_alpha, reg_lambda, H, x, y, weights,m)
        weights=new_w
        H=compute_hypothesis(x,weights)
        j_list.append(j_new[0][0])
    
    plot_the_model(j_list,"pow_2")
    
    #test results
    
    test_file = input("Enter your test file name: ") 
    test_data,test_m,test_n=convert_data_to_numpy(test_file)
    
    #split x(input) and y(output)
    x_test,y_test=split_input_output(test_data)
    
    y_pred=predict(x_test, weights)
    
    H_test=compute_hypothesis(x_test,weights)
    print("Final J value for training dataset " + str(j_list[-1]))
    print("J value for test dataset " + str(compute_j_with_regilization(y_test,H_test,test_m,test_n,reg_lambda,weights)[0][0]))
    FP,FN,TP,TN,accuracy, precision, recall, f_score=give_me_all_evaluaton_metrics(y_pred,y_test)
    print("FP = " + str(FP) + "\nFN = " + str(FN) + "\nTP = " + str(TP) + "\nTN = " + str(TN))
    print("accuracy = " + str(accuracy) + "\nprecision = " + str(precision) + "\nrecall = " + str(recall) + "\nf_score = " +str(f_score) )
      
    
    
    
        
    