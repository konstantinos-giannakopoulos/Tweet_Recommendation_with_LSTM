#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 18:02:20 2018

@author: konstantinos
"""

import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

class InputPreparation():
    def __init__(self, datapairs,numtweets,numusers):
        self.datapairs = datapairs #<user_index,tweet_index>
        self.numTweets = numtweets
        self.numUsers = numusers
        #self.positive_n_clusters = 30
        #self.negative_n_clusters = 20
        
        
    def collectUnknownDatapairs(self,tweetsWithHashtags):
        unknownUsersTweetsPair = []
        for usertweetknownpair in self.datapairs:
            knownUserIndex = usertweetknownpair[0]
            knownTweetIndex = usertweetknownpair[1]
            if(knownTweetIndex not in tweetsWithHashtags):
                continue
            userIndex = 0
            while userIndex < self.numUsers:
                if (userIndex != knownUserIndex) :
                    pair = []
                    pair.append(userIndex)
                    pair.append(knownTweetIndex)  
                    unknownUsersTweetsPair.append(pair)    
                userIndex += 1

        #print(len(unknownUsersTweetsPair))
        #print(self.numUsers * self.numTweets)
        return unknownUsersTweetsPair
    
    def plot(self,D):     
        x,y = zip(*D)
        #print(x,y)
        plt.scatter(x, y)
        plt.show()
        
    def set_colors(self,labels, colors='rgbykcm'):
        colored_labels = []
        for label in labels:
            colored_labels.append(colors[label])
        return colored_labels
    
    
    def clustering(self,D,n_clusters):
        centers, labels = self.kmeans(D,n_clusters)
        return centers, labels
    
    def kmeans(self,D,n_clusters):
        kmeans = KMeans(n_clusters)
        X = np.array(D)
        print(X.shape)
        kmeans.fit(X)
        #x,y = zip(*X)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_ 
        #print(kmeans.cluster_centers_)
        #self.visualizekmeans(D,kmeans);
        return centers, labels
    
    '''     
    def visualizekmeans(self,D,kmeans):
        x,y = zip(*D)
        ##fig = plt.figure(figsize=(5, 5))
        plt.scatter(x,y, c=kmeans.labels_)
        centers = kmeans.cluster_centers_     
        #print(kmeans.labels_[:20])
        #print(centers)
        plt.scatter(centers[:,0], centers[:,1], c='black', s=100, alpha=0.5);
        #plt.xlim(0, 1750)
        #plt.ylim(0, 5700)
        plt.show()
    '''
    
    def collectFinalResults(self, D_known, D_unknown, positive_centers, positive_labels, negative_samples):
        self.positive_datapoints = D_known
        self.negative_datapoints = D_unknown
        self.positive_centers = positive_centers
        self.positive_labels = positive_labels
        self.negative_samples = negative_samples
        
    def getPositiveNegativeDatapointIndexes(self):
        return self.positive_datapoints, self.negative_samples
    
    
    # U:  [users x features]
    # X: [tweets x features]
    # positive_datapoints:   known <user_id,tweet_id>
    # negative_datapoints: unknown <user_id,tweet_id>
    def prepare(self,U,X,positive_datapoints,negative_samples):
        P = []
        N = []
        #print(U[3])
        #print(X[2])
        for p_datapoint in positive_datapoints:
            user_id_p = p_datapoint[0]
            tweet_id_p = p_datapoint[1]
            feature_p_datapoint = []
            user_feature_p = U[user_id_p]
            tweet_feature_p = X[tweet_id_p]
            feature_p_datapoint.append(user_feature_p)
            feature_p_datapoint.append(tweet_feature_p)
            P.append(feature_p_datapoint)
            
            
        for n_datapoint in negative_samples:
            user_id_n = n_datapoint[0]
            tweet_id_n = n_datapoint[1]
            feature_n_datapoint = []
            user_feature_n = U[user_id_n]
            tweet_feature_n = X[tweet_id_n]
            feature_n_datapoint.append(user_feature_n)
            feature_n_datapoint.append(tweet_feature_n)
            N.append(feature_n_datapoint)
        
        dim_user = len(U[0])
        dim_feature = len(X[0])
        
        P_1 = np.asarray(P)
        P_2 = P_1.reshape(len(P),dim_user+dim_feature)
        
        N_1 = np.asarray(N)
        N_2 = N_1.reshape(len(N),dim_user+dim_feature)
        
        print ("===P2 N2===")
        print (P_2.shape, N_2.shape)
                    
        return P_2,N_2
    
    
    def mergeDataset(self,P,N):
        dataset = []
        for item in P:   
            x_y = [list(item),1]
            dataset.append(x_y)
            
        for item in N:      
            x_y = [list(item),0]
            dataset.append(x_y)
        
        
             
        #print("--->>> ",dataset[0])
        ds = np.asarray(dataset)
        
        #print ("dataset shape: ", ds.shape)
        #print ("ds[0].shape: ", ds[0].shape)
        #print ("ds[0]: ", ds[0])
        #print ("ds[0][0]: ", ds[0][0])
        #print ("ds[0][1]: ", ds[0][1])   
    
        return ds

    def shuffleDataset(self,datasetArray):
        random.shuffle(datasetArray)
        return datasetArray
    
    def separate(self, shuffledMatrix):
        X = []
        y = []  
        for item in shuffledMatrix:
            X.append(item[0])
            y.append(item[1])
        return X,y
    
    
    
    def prep(self,P,N):
        P_new = []
        for item in P:   
            x_y = [list(item),1]
            P_new.append(x_y)
        N_new = [] 
        for item in N:      
            x_y = [list(item),0]
            N_new.append(x_y)
            
        #print ('====')
        #print (len(P_new), len(N_new))
        
        trainRatio = int(len(P_new) * 0.8)
        X_train_1 = P_new[:trainRatio]
        X_test_1  = P_new[trainRatio:]
        
        trainRatio = int(len(N_new) * 0.8)
        X_train_0 = N_new[:trainRatio]
        X_test_0  = N_new[trainRatio:]
        #print(len(X_train_1), len(X_test_1), len(X_train_0), len(X_test_0))
        #print ('====')
        
        dataset_train = []
        for item in X_train_1:   
            #x_y = [list(item),1]
            dataset_train.append(item)
            
        for item in X_train_0:      
            #x_y = [list(item),0]
            dataset_train.append(item)
            
        dataset_test = []
        for item in X_test_1:   
            #x_y = [list(item),1]
            dataset_test.append(item)
            
        for item in X_test_0:      
            #x_y = [list(item),0]
            dataset_test.append(item)
            
        ds_train = np.asarray(dataset_train)
        ds_test = np.asarray(dataset_test)
        
        #print("ds_train: ",ds_train.shape,ds_test.shape)
        #sffl_ds_train = random.shuffle(ds_train)
        #sffl_ds_test = random.shuffle(ds_test)
        #print(sffl_ds_train.shape, sffl_ds_test.shape)        
        #random.shuffle(ds_train)
        #random.shuffle(ds_test)
        
        #print(sffl_ds_train[0])
        X_train = []
        Y_train = [] 
        for item in ds_train:
            X_train.append(item[0])
            Y_train.append(item[1])
        #print(X_train[0])
        #print(Y_train[0])
 
        X_test = []
        Y_test = []
        for item in ds_test:
            X_test.append(item[0])
            Y_test.append(item[1])
    
        
        X_train_shuf = []
        Y_train_shuf = []
        index_shuf = list(range(len(X_train)))
        random.shuffle(index_shuf)
        for i in index_shuf:
            X_train_shuf.append(X_train[i])
            Y_train_shuf.append(Y_train[i])
        
        X_test_shuf = []
        Y_test_shuf = []
        index_shuf = list(range(len(X_test)))
        random.shuffle(index_shuf)
        for i in index_shuf:
            X_test_shuf.append(X_test[i])
            Y_test_shuf.append(Y_test[i])
        
        #sffl_ds_train = ds_train
        #sffl_ds_test = ds_test
        #print(sffl_ds_train.shape, sffl_ds_test.shape)    
        #print(len(X_train), len(Y_train), len(X_test), len(Y_test))
        #print ("train_positive:", Y_train_shuf.count(1))
        #print ("train_negative:", Y_train_shuf.count(0))
        #print ("test_positive:", Y_test_shuf.count(1))
        #print ("test_negative:", Y_test_shuf.count(0))
        
        return X_train_shuf, Y_train_shuf, X_test_shuf, Y_test_shuf
        

    def duplicateNegativeDatapairs(self,D_neg):   
        length = len(D_neg)
        pos = 0
        while pos < length:
            D_neg.append(D_neg[pos])  
            pos += 1
        return D_neg


def main(U,X,datapairs, negativeDatapairs, numtweets, usersetlist, lstmposnegsamplingratio, tweetsWithHashtags): #positive_datapoints,negative_samples):
    numusers = len(usersetlist)
    print("Number of tweets: ", numtweets)
    print("Number of users: ", numusers)
    inputPreparation = InputPreparation(datapairs,numtweets,numusers)
    # known <user_id,tweet_id>
    D_pos = inputPreparation.datapairs#[:50]  
    print("Positive datapoints length: " , len(D_pos))
    # unknown <user_id,tweet_id>
    #negative_datapoints = inputPreparation.collectUnknownDatapairs(tweetsWithHashtags)  
    D_neg = negativeDatapairs
    print("Negative datapoints length: " , len(D_neg))
    #inputPreparation.plot(D_pos);
    
    '''  
    # sampling
    if(len(D_neg) < len(D_pos)):
        print("Sampling of unknown (negative) <user,tweet> pairs:") 
        #num_negative_samples = int(len(D_pos) * lstmposnegsamplingratio)
        num_negative_samples = len(D_pos) - len(D_neg) #int(len(D_pos) * lstmposnegsamplingratio)
        print("numsamples: ", num_negative_samples)
        negative_samples = []
        index_samples = list(range(len(negative_datapoints)))
        negative_index_samples = random.sample(index_samples,num_negative_samples)
        for i in negative_index_samples:
            negative_samples.append(negative_datapoints[i])
            D_neg.append(negative_datapoints[i])
        print("Negative datapoints sampled length: " , len(negative_samples))
    #D_neg = negative_samples
    #inputPreparation.plot(negative_samples);
    '''
   
    D_pos = inputPreparation.duplicateNegativeDatapairs(D_pos)
    #if(len(D_neg) < int(len(D_pos)/2)):
    D_neg = inputPreparation.duplicateNegativeDatapairs(D_neg)
        
    
    # log file
    filename = "output/log.txt"
    #log_window_report  = "\nNumber of tweets: " + str(numtweets)
    log_window_report  = "\nNumber of users: " + str(numusers)
    log_window_report += "\nNumber of Positive datapoints: " + str(len(D_pos))
    log_window_report += "\nNumber of Negative datapoints: " + str(len(D_neg))
    #log_window_report += "\nNumber of Negative datapoints sampled: " + str(len(negative_samples))
    file = open(filename,'a')
    file.write(log_window_report) 
    file.flush()
    

    #W,D = topicdoc.load_data()
    P,N = inputPreparation.prepare(U,X, D_pos, D_neg)# D_neg) #(U,X, D_pos, negative_samples)
    
    D = inputPreparation.mergeDataset(P,N)
    #print(D.shape)
    shuffle_D = inputPreparation.shuffleDataset(D)#(ar)
    
    X,y = inputPreparation.separate(shuffle_D)
  
    # clustering
    #print("Clustering of known <user,tweet> pairs: ")
    #positive_centers, positive_labels = inputPreparation.clustering(P,inputPreparation.positive_n_clusters);
    
    #inputPreparation.collectFinalResults(D_pos, D_neg, positive_centers, positive_labels, negative_samples)
    
    X_train, Y_train, X_test, Y_test = inputPreparation.prep(P,N)
    
    
    
    # X = X_train, y = Y_train (=labels)
    #return X,y 
    # X_train, Y_train (=labels), X_test, Y_test (=labels)
    return X_train, Y_train, X_test, Y_test