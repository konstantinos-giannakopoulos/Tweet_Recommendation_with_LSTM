#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: konstantinos
"""

#Import all the dependencies
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize


#class DocumentToVector:
    
def main(tweets,numfeatures):
    #trainingFile = "document2vector/tweets.txt"
    #trainingFile = "document2vector/training.txt" # len = 4
    #with open(trainingFile) as f:
    #    content = f.readlines()
    #content = [x.strip() for x in content] 
    
    content = tweets
    #print("content: " , content)
    
    data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(content)]
    #print("number of documents: ", len(data))
    #print("data: ", data)
    model = doc2vec.Doc2Vec(data, vector_size = numfeatures, window=300, min_count=1, workers=1) 
    
    #doc_index = 3
    #docvec = model.docvecs[doc_index]  
    #print("vector representation of doc ", doc_index, ": ", docvec)
    return model.docvecs
    
    #model.save("document2vector/d2v.model")
    #print("Model Saved")