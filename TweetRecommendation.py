#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:29:51 2018

@author: konstantinos
"""

import xmlReader



class TweetRecommendation():
    def __init__(self,numfeatures, numiterations, windowlength, lstmposnegsamplingratio, 
                 filterInputChoice, filteroutTweetsThreshold, 
                 lstm_nb_epoch, lstm_batch_size, lstm_validation_split, lstm_optimizer):
        self.numfeatures = numfeatures #<user_index,tweet_index>
        self.numiterations = numiterations
        self.windowlength = windowlength
        self.lstmposnegsamplingratio = lstmposnegsamplingratio
        self.filterInputChoice = filterInputChoice #boolean 1:yes, 0:no
        self.filteroutTweetsThreshold = filteroutTweetsThreshold
        self.lstm_nb_epoch = lstm_nb_epoch
        self.lstm_batch_size = lstm_batch_size
        self.lstm_validation_split = lstm_validation_split
        self.lstm_optimizer = lstm_optimizer
    
    def run(self):
        xmlReader.main(self.numfeatures, self.numiterations, self.windowlength, self.lstmposnegsamplingratio, 
                       self.filterInputChoice, self.filteroutTweetsThreshold, 
                       self.lstm_nb_epoch, self.lstm_batch_size, self.lstm_validation_split, self.lstm_optimizer)
    
#def main(numfeatures, numiterations, windowlength):
#    xmlReader.main(numfeatures, numiterations, windowlength)
    
    
# 
if __name__ == "__main__":
    numfeatures = 7
    numiterations = 10
    windowlength = 5 #5
    # not in use anymore - We do not sample for negative datapoints. We use the tweets of friends only
    lstmposnegsamplingratio = float(1/1) #float(1/5)
    #filter-out users with [filteroutTweetsThreshold] tweets and their tweets
    filterInputChoice = 1 #boolean 1:yes, 0:no 
    filteroutTweetsThreshold = 2 #3
    #LSTM
    lstm_nb_epoch = 10 #(overfitting problem)
    lstm_batch_size = 5
    lstm_validation_split = 0.2
    lstm_optimizer = 'adadelta' # adam, adadelta, adagrad, rmsprop, sgd, momentum
    
    tweetrecommendation = TweetRecommendation(numfeatures, numiterations, windowlength, lstmposnegsamplingratio, 
                                              filterInputChoice, filteroutTweetsThreshold, 
                                              lstm_nb_epoch, lstm_batch_size, lstm_validation_split, lstm_optimizer)
    tweetrecommendation.run()
    
    #main(numfeatures, numiterations, windowlength)
