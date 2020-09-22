# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:32:48 2020

@author: akash
"""
"""
Using normal equaton method to solve this below multivariate 
regression problem

"""


import numpy as np

def convert_data_to_numpy(file):
    file_data=open(file,"r")
    line=file_data.readline()
    m=int(line.split('\t')[0].strip())
    n=int(line.split('\t')[1].strip())    
    data=np.loadtxt(file, delimiter='\t', skiprows=1)

    return data,m,n

def split_input_output(data): 
    #adding bias x0 as 1 in x for generating x(input) and y(output) 
    x=np.hstack((np.ones((data.shape[0],1)),data[:, :-1]))
    y=data[:, -1].reshape((data.shape[0],1))
    #
    return x,y
    
def compute_weights(x,y):
    return np.dot(np.linalg.pinv(np.dot(np.transpose(x),x)), np.dot (np.transpose(x), y))

def compute_J(x,y,w):
    m_1m_transpose=np.ones((1,x.shape[0]))
    return np.dot(m_1m_transpose, np.multiply(np.dot(x,w)-y,np.dot(x,w)-y)) / (2*x.shape[0])

def compute_adjusted_r_square(J,y, m, n):
    m_1m_transpose=np.ones((1,m))
    y_mean=np.mean(y)
    denominator= np.dot(m_1m_transpose, np.multiply(y-y_mean,y-y_mean))/(2*m)
    r_square=1- (J/denominator)
    adjusted_r_square = 1 - (((1-r_square)*(m-1))/(m-n-1))
    if adjusted_r_square < 0:
        return 0
    else:
        return adjusted_r_square

if __name__ == "__main__":
    
    train_file = input("Enter your training file name: ") 
    data,m,n=convert_data_to_numpy(train_file)
   
    #split x(input) and y(output)
    x,y=split_input_output(data)
    
    #Normal equation method
    
    #compute weights
    weights=compute_weights(x,y)
    print("Training weights are:")
    for cur_index,row in enumerate(weights):
        for value in row:
            print("w_"+str(cur_index)+" = "+str(value))
    
    #compute J
    J_train_model=compute_J(x,y,weights)
    print("J_value for training data = " + str(J_train_model[0][0]))
    
    #val or test file
    test_val_file = input("Enter your Validation or Test file name: ") 
    test_val_data,test_val_m,test_val_n=convert_data_to_numpy(test_val_file)
    
    #split x(input) and y(output)
    x_test_val,y_test_val=split_input_output(test_val_data)
    
    #compute J
    J_test_val_model=compute_J(x_test_val,y_test_val,weights)
    print("J_value for validation or testing data = " + str(J_test_val_model[0][0]))
    
    #compute adjusted R_square
    adjusted_r_square=compute_adjusted_r_square(J_test_val_model,y_test_val, test_val_m,test_val_n)
    print("Adjusted R2 value = " + str(adjusted_r_square[0][0]))
    
    
    
    
    
    
    
    
    