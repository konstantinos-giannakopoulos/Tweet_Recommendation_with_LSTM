#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 16:41:57 2018

@author: konstantinos
"""

import sys
#import xmlReader 
import HTLIC
import document2vector
import TopicEnhDoc
import InputPreparation
import LSTM
import numpy as np

#       numfeatures : parameter for setting the number of features
# 1. HTLIC
# 2. Doc2Vector
# 3. TopicEnhDoc
# 4. LSTM
def main(userstags, wordstags, tweets, topicWordsDoc, windowcounter, usertweetPairList, usersUsersTweetsPairList, usersetlist, numfeatures, numiterations, lstmposnegsamplingratio, tweetsWithHashtags, lstm_nb_epoch, lstm_batch_size, lstm_validation_split, lstm_optimizer): #positive_datapoints, negative_samples):
    
    #numfeatures = 10  # = 10
    
    # HTLIC.py :
    print("\n\nHTLIC: ... ") 
    U, W, V, n_iter = HTLIC.main(userstags,wordstags,numfeatures,numiterations)
    print("Shape of U: ", U.shape)
    print("Shape of W: ", W.shape)
    #HTLIC.evaluate(userstags,wordstags,U,W,V)

    
    # document2vector.py : 
    print("\n\nDoc2Vector: ")
    D = document2vector.main(tweets,numfeatures)
    print("Length of D: ",len(D))

    # TopicEnhDoc.py : 
    print("\n\nTopicEnhDoc: ")
    X = TopicEnhDoc.main(W,D, topicWordsDoc)
    print("Length of X: ",len(X))
    #TopicEnhDoc.main()
    
    # InputPreparation.py
    print("\n\nPrepare Input for LSTM")
    #X,y = InputPreparation.main(U,X,usertweetPairList,len(tweets), usersetlist)
    X_train, Y_train, X_test, Y_test = InputPreparation.main(U,X, usertweetPairList, usersUsersTweetsPairList, len(tweets), usersetlist, lstmposnegsamplingratio, tweetsWithHashtags)
    
    # LSTM.py : 
    print("\n\nLSTM: ")
    #LSTM.main(X,y)
    LSTM.main(X_train, Y_train, X_test, Y_test, numfeatures, windowcounter, lstm_nb_epoch, lstm_batch_size, lstm_validation_split, lstm_optimizer)
    
    #sys.exit(0)
    print("\n--- end of processing ---\n")
    






""" 
print("\n\nXMLReader: ... ")
#xmlReader.main()
userstags, wordstags, tweets, topicWordsDoc = xmlReader.main()
##print("\t[usersxtags]: ", userstags.shape)
##print("\t[wordsxtags]: ", wordstags.shape)
##print("size of dataset: ", len(tweets))


print("\n\nHTLIC: ... ")
U, W, V, n_iter = HTLIC.main(userstags,wordstags)
#HTLIC.evaluate(userstags,wordstags,U,W,V)

'''
print("\n\nDoc2Vector: ")
D = document2vector.main(tweets)

print("\n\nTopicEnhDoc: ")
TopicEnhDoc.main(W,D, topicWordsDoc)
#TopicEnhDoc.main()

#print("\n\nLSTM: ")
'''
""" 
