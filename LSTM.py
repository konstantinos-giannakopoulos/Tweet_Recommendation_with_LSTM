# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:41:43 2016

@author: dsm
"""

import re
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, Activation, TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.metrics import classification_report


#class LSTM():
    
# X_train, Y_train (=labels), X_test, Y_test (=labels)
#def main(X_train,labels):
def main(X_train, Y_train, X_test, Y_test, numfeatures, windowcounter, lstm_nb_epoch, lstm_batch_size, lstm_validation_split, lstm_optimizer):
    numfeatures2 = 2 * numfeatures

    '''
    X_train = numpy.array(X_train).reshape(len(X_train),1,numfeatures2)
    Y_train = numpy.array(labels).reshape(len(labels),1)
    '''

    #print(len(X_train),len(X_test),len(Y_train),len(Y_test))
    X_train = numpy.array(X_train).reshape(len(X_train),1,numfeatures2)
    X_test = numpy.array(X_test).reshape(len(X_test),1,numfeatures2)
    #X_test = numpy.array(X_[1500:1800]).reshape(300,1,80)
    Y_train = numpy.array(Y_train).reshape(len(Y_train),1)
    Y_test = numpy.array(Y_test).reshape(len(Y_test),1)
    #Y_test = numpy.array(labels[1500:1800]).reshape(300,1)
    
    #print (Y_train[0:10])
    #print ("train_positive:", Y_train.flatten().tolist().count(1))
    #print ("train_negative:", Y_train.flatten().tolist().count(0))
    #print ("test_positive:", Y_test.flatten().tolist().count(1))
    #print ("test_negative:", Y_test.flatten().tolist().count(0))
    #print (X_train.shape, Y_train.shape)
    
    '''
    The batch size is a number of samples processed before the model is updated.
    The number of epochs is the number of complete passes through the training dataset.
    
    The size of a batch must be more than or equal to one and 
    less than or equal to the number of samples in the training dataset.

    '''
    
    # create the model
    model = Sequential()   
    model.add(Dense(32,input_shape = (1,numfeatures2))) # Dense(32
    model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu')) # filters=32, kernel_size=4,
    #model.add(MaxPooling1D(pool_size=(2,)))
    model.add(LSTM(128)) # 128
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=lstm_optimizer, metrics=['accuracy']) # optimizer='adam'
    print(model.summary())
    
    model.fit(X_train, Y_train, nb_epoch=lstm_nb_epoch, batch_size=lstm_batch_size, validation_split = lstm_validation_split) #, verbose=0) #nb_epoch=5, batch_size=32, validation_split = 0.1
      
    # Final evaluation of the model
    scores = model.evaluate(X_test, Y_test)#, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
        
    y_pred = model.predict(X_test).tolist()
    #y_p = [0 if x < 0.5 else 1 for x in y_pred]  
    y_p = []
    for x in y_pred:
        if x[0] < 0.5:
            val = 0
        else:
            val = 1
        y_p.append(val)
    #print(len(y_p))
    print(classification_report(Y_test, y_p))
    
    # log file
    log_window_report = "\nAccuracy: "+ str(scores[1]*100) +"%"
    log_window_report += "\n"+(classification_report(Y_test, y_p))
    filename = "output/log.txt"
    file = open(filename,'a') 
    file.write(log_window_report)  
    file.flush()
    #file.close() 
    #print(str(windowcounter))
    #print()

    '''
    # MLP
    model = Sequential()
    model.add(Dense(64, input_dim=80, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(X_[0:1500], labels[0:1500],
              epochs=3,
              batch_size=128)
    
    score = model.evaluate(X_[1500:1800], labels[1500:1800], verbose = 0)
    print("Accuracy: %.2f%%" % (score[1]*100))
    '''

#def main(X_train,labels):
#    lstm = LSTM()
#    lstm.run(X_train,labels)
    
#if __name__ == '__main__':
#    LSTM lstm = LSTM()
#    lstm.main()