#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: konstantinos
"""

import numpy as np;

"""
W:[wordsxfeatures]
D:[documentsxfeatures]
"""

class TopicEnhDoc():

    #def load_data(self):
    #    W = np.matrix('1 2 3 4; 5 6 7 8')
    #    D = np.matrix('1 1 1 1; 1 1 1 1')  
    #    return W,D
    
    
    def estimate(self,W,D,topicWordsDoc):
        #print("D's length: ", len(D))
        #print("W's shape: ", np.shape(W))
        #print("Num of docs: ", len(topicWordsDoc))
        #numrows, numcolumns = np.shape(D)
        #print('rows', numrows)
        #print('columns', numcolumns)
        #for r in range(numrows): # foe each document
        X = []
        for docIndex in range (len(topicWordsDoc)):
            documentWords = topicWordsDoc[docIndex]
            dj = D[docIndex]
            #print("dj", dj)
            
            Nj = len(documentWords)
            #print("Nj", Nj)
            if(Nj == 0):
                xj = (1/2) * (dj)
            else:
                sumWk = 0;
                for wordIndex in documentWords:
                    wk = W[wordIndex]
                    #print("wk", wk)
                    sumWk = np.sum(wk)
                    #print("sumW", sumWk)
              
                sumWkNj = (1/Nj) * sumWk
                #print("sumWkNj",sumWkNj)
                xj = (1/2) * (dj + sumWkNj)

            X.append(xj)    
            #print("x[",docIndex,"]: ",xj)

        return X
    

def main(W,D, topicWordsDoc):
    topicdoc = TopicEnhDoc()
    #W,D = topicdoc.load_data()
    X = topicdoc.estimate(W,D,topicWordsDoc)
    return X