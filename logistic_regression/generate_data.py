# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:05:33 2020

@author: akash
"""

import numpy as np

def convert_data_to_numpy(file):
    file_data=open(file,"r")
    line=file_data.readline()
    m=int(line.split('\t')[0].strip())
    n=int(line.split('\t')[1].strip())
    data=np.loadtxt(file, skiprows=1)

    return data,m,n

def generate_model(data, m, n, filename):
    x1=data[:,0]
    x2=data[:,1]
    y=data[:,2]
    ans=np.array([], np.float32)
    thePower = 2
    #fout1=open(filename, "a+")
    num_features=0
    for j in range(thePower+1):
        for i in range(thePower+1):
                temp = (x1**i)*(x2**j)
                if (i != 0 or j !=0):
                    ans= np.append(ans, temp)
                    num_features = num_features +1
    ans= np.append(ans, y)
    
    return np.transpose(ans.reshape(num_features+1,m))

if __name__ == "__main__":
    train_file = input("Enter your training file name: ") 
    data,m,n=convert_data_to_numpy(train_file)
    
    ans=generate_model(data, m, n, "test_2.txt")
    
