# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:03:11 2020

@author: akash
"""
import math

def cleantext(text):
    text = text.lower()
    text = text.strip()
    
    for letters in text:
        #punctuations = """[]!.,"-!_@;':#$%^&*()+/?"""
        punctuations = """!"#$%&'()*+,-./:;?@[\]^_`{|}~"""
        if letters in punctuations:
            text = text.replace(letters," ")
    
    return text

def countwords(text, is_spam, counted):
    for each_word in words:
        if each_word in counted:
            if is_spam == 1:
                counted[each_word][1]= counted[each_word][1] + 1
            else:
                counted[each_word][0]= counted[each_word][0] + 1
        else:
            if is_spam == 1:
                counted[each_word]=[0,1]
            else:
                counted[each_word]=[1,0]
    return counted

def make_percent_list(k, thecount, spams, hams):
    for each_key in thecount:
        thecount[each_key][0]=(thecount[each_key][0] + k)/(2*k + hams)
        thecount[each_key][1]=(thecount[each_key][1] + k)/(2*k + spams)
    return thecount


def stopwords_dict(file):
    stopwords=[]
    f= open(file, "r")
    stopwords = f.read().splitlines()
    f.close()
    
    return stopwords


def compute_prob(test_words, vocab, spam, ham):
    p_s1_s=0
    p_s1_not_s=0
    
    for k,v in vocab.items():
        if k in test_words:
            p_s1_s = p_s1_s + math.log(v[1])
            p_s1_not_s = p_s1_not_s + math.log(v[0])
        else:
            p_s1_s = p_s1_s + math.log(1-v[1])
            p_s1_not_s = p_s1_not_s + math.log(1-v[0])
    
    p_s1_s = math.exp(p_s1_s)
    p_s1_not_s = math.exp(p_s1_not_s)
    p_s= spam /(spam + ham)
    p_not_s= ham / (spam + ham)
    
    return p_s1_s, p_s1_not_s, p_s, p_not_s

def predict(test_words, vocab, spam, ham):
    p_s1_s, p_s1_not_s, p_s, p_not_s = compute_prob(test_words, vocab, spam, ham)
    
    term1= math.log(p_s1_not_s) + math.log(p_not_s)
    term2= math.log(p_s1_s) + math.log(p_s)
    term = term1 - term2
    
    ans = 1/(1 + math.exp(term))
    
    if ans >= 0.5:
        return 1
    else:
        return 0
    
def give_me_all_evaluaton_metrics(y_pred,y_test):
    TP=0
    FP=0
    FN=0
    TN=0
    for i in range(len(y_pred)):
        if y_test[i] == 1 and y_pred[i] == 1:
            TP = TP +1
        elif y_test[i] == 0 and y_pred[i] == 0:
            TN = TN +1
        elif y_test[i] == 0 and y_pred[i] == 1:
            FP = FP +1
        elif y_test[i] == 1 and y_pred[i] == 0:
            FN = FN +1
    accuracy = (TP + TN)/ (TP + TN + FP + FN)
    precision = TP / (TP+FP)
    recall = TP / (TP + FN)
    f_score= 2 * (1/ ((1/precision) + (1/recall)))
    return FP,FN,TP,TN,accuracy, precision, recall, f_score

if __name__ == "__main__":
    spam=0
    ham=0
    counted=dict()
    fname=input("Enter the Spam-ham training file name: ")
    stopwords_file = input("Enter the stopwords file name: ")   
    fin= open(fname, "r", encoding = 'unicode-escape')
    
    stopwords_list=stopwords_dict(stopwords_file)
    stopwords=set(stopwords_list)
    
    textline = fin.readline()
    
    while textline !="":
        is_spam = int(textline[:1])
        
        if is_spam == 1:
            spam = spam + 1
        else:
            ham = ham +1
        
        textline = cleantext(textline[1:])
        words = textline.split()
        words = set(words)
        words = words.difference(stopwords)        
        counted = countwords(words, is_spam, counted)
        
        textline = fin.readline()
    
    fin.close()
    #print (spam, ham)
    #print(counted)
    vocab = (make_percent_list(1, counted, spam, ham))
    #print(vocab)
    
    f_test=input("Enter the Spam-ham test file name: ")  
    fin= open(f_test, "r", encoding = 'unicode-escape')
    textline = fin.readline()
    
    prediction_results=[]
    test_results=[]
    
    while textline !="":
        test_results.append(int(textline[:1]))
        textline = cleantext(textline[1:])
        words = textline.split()
        words = set(words)
        words = words.difference(stopwords)
        prediction_results.append(predict(words, vocab, spam, ham))
        
        textline = fin.readline()
    
    fin.close()
    
    #print(prediction_results)
    #print(test_results)
    test_spam=0
    test_ham=0
    
    for label in test_results:
        if label == 0:
            test_ham=test_ham+1
        else:
            test_spam = test_spam +1
    
    print(" \nIn test file: No of Spam emails = {} and No of Ham emails = {}".format(test_spam,test_ham))
    
    FP,FN,TP,TN,accuracy, precision, recall, f_score=give_me_all_evaluaton_metrics(prediction_results, test_results)
    
    print("\nFP = " + str(FP) + "\nFN = " + str(FN) + "\nTP = " + str(TP) + "\nTN = " + str(TN))
    print("\naccuracy = " + str(accuracy) + "\nprecision = " + str(precision) + "\nrecall = " + str(recall) + "\nf_score = " +str(f_score) )

    
    '''
    test_subject="1 Funds Investment"
    label=int(test_subject[:1])
    test_subject=cleantext(test_subject[1:])
    print(test_subject)
    test_words=test_subject.split()
    
    test_words = set(test_words)
    test_words = test_words.difference(stopwords)
    print(test_words)
    
    ans = predict(test_words, vocab, spam, ham)
    
    print(ans)
    
    '''
    
