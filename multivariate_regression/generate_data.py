# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:16:41 2020

@author: akash
"""
import numpy as np

def num_rows_in_file(file):
    data=open(file,"r")
    num_rows=0
    for line in data:
        num_rows=num_rows+1
    data.close()
    return num_rows-1 #because first line contains m and n, not data

def convert_data_to_numpy(file):
    num_rows=num_rows_in_file(file)
    file_data=open(file,"r")
    data=np.zeros([num_rows,7], dtype=float)
    
    for line_num,line in enumerate(file_data):
        if line_num ==0:
            m=int(line.split('\t')[0].strip())
            n=int(line.split('\t')[1].strip())
            continue
        
        data[line_num-1,0]=line.split('\t')[0].strip()
        data[line_num-1,1]=line.split('\t')[1].strip()
        data[line_num-1,2]=line.split('\t')[2].strip()
        data[line_num-1,3]=line.split('\t')[3].strip()
        data[line_num-1,4]=line.split('\t')[4].strip()
        data[line_num-1,5]=line.split('\t')[5].strip()
        data[line_num-1,6]=line.split('\t')[6].strip()
        
    file_data.close()  
    return data,m,n

def print_in_files(np_array,save_file):
    m=np_array.shape[0]
    n=np_array.shape[1] -1
    head=str(m)+'\t'+str(n)
    np.savetxt(save_file, np_array, delimiter='\t',header=head, comments='') 

def generate_files_for_models(data):
    train_mode1_1=data[:248,:]
    val_mode1_1=data[248:331,:]
    test_model_1=data[331:414,:]
    
    #model1
    print_in_files(train_mode1_1,'train_model_1.txt')
    print_in_files(val_mode1_1,'val_model_1.txt')
    print_in_files(test_model_1,'test_model_1.txt')
    
    #model2
    train_mode1_2 = np.hstack((np.square(train_mode1_1[:,:6]), train_mode1_1[:,6:7]))
    val_mode1_2 = np.hstack((np.square(val_mode1_1[:,:6]), val_mode1_1[:,6:7]))
    test_model_2 = np.hstack((np.square(test_model_1[:,:6]), test_model_1[:,6:7]))
    
    print_in_files(train_mode1_2,'train_model_2.txt')
    print_in_files(val_mode1_2,'val_model_2.txt')
    print_in_files(test_model_2,'test_model_2.txt')
    
    #model3
    train_mode1_3= np.hstack((train_mode1_1[:,:6], train_mode1_2))
    val_mode1_3= np.hstack((val_mode1_1[:,:6], val_mode1_2))
    test_model_3= np.hstack((test_model_1[:,:6], test_model_2))
    
    print_in_files(train_mode1_3,'train_model_3.txt')
    print_in_files(val_mode1_3,'val_model_3.txt')
    print_in_files(test_model_3,'test_model_3.txt')
    

if __name__ == "__main__":
    data,m,n=convert_data_to_numpy('REData.txt')
    print(m,n,data.shape)
    
    #shuffle the data
    np.random.shuffle(data)
    print(data.shape)
    
    generate_files_for_models(data)
    
    

