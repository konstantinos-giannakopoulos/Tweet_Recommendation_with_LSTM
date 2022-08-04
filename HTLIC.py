#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 17:17:25 2018

@author: dsm
"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from numpy import linalg as LA
from numpy import matmul


# load npz data file and transform 3-D array to a 2-D matrix

#def read_data(dataset_path):
def read_data(G,F):    
    # G: user(10000)-tag(50); F: word(500)-tag(50); V: tag(50)-feature(200)
    G = G#np.random.randint(5, size=(10000, 50))
    
    F = F#np.random.randint(10, size=(500, 50))
    
    return G, F
    

def norm(x):
    """Dot product-based Euclidean norm implementation
    """
    return sqrt(squared_norm(x))


def _initialize_nmf(G, F, n_components, random_state=None):
    
    #check_non_negative(X, "NMF initialization")
    n_users, n_tags = G.shape
    n_words = F.shape[0]

    # Random initialization

    rng = check_random_state(random_state)
    avg_G = np.sqrt(G.mean() / n_components)
    avg_F = np.sqrt(F.mean() / n_components)
    U = avg_G * rng.randn(n_users, n_components)
    W = avg_F * rng.randn(n_words, n_components)
    V = 0.5 * (avg_G+avg_F) * rng.randn(n_components, n_tags)
    
    # we do not write np.abs(H, out=H) to stay compatible with
    # numpy 1.5 and earlier where the 'out' keyword is not
    # supported as a kwarg on ufuncs
    np.abs(U, U)
    np.abs(W, W)
    np.abs(V, V)

    return U, W, V


def _fit_coordinate_descent(G, F, U, W, V, n_components, alpha, alpha_1, alpha_2, alpha_3, max_iter=20):

    n_users, n_tags = G.shape[0], G.shape[1]
    n_words = F.shape[0]


    step_size = 0.001

    for n_iter in range(max_iter):

        print (n_iter)

        # update U first
        for i in range(n_users):

            volation = np.zeros(n_components)

            for j in range(n_tags):

                volation = volation + alpha * (G[i,j]- np.dot(U[i],V[:,j])) * V[:,j]

            grad_u = alpha_1 * U[i] - volation

        U[i] = U[i]-grad_u*step_size

        #update V second
        for j in range(n_tags):

            volation_u = np.zeros(n_components)
            
            volation_w = np.zeros(n_components)

            for i in range(n_users):

                volation_u = volation_u + alpha * (G[i,j]- np.dot(U[i],V[:,j])) * U[i]
            
            for i in range(n_words):

                volation_w = volation_w + (1 - alpha) * (F[i,j]- np.dot(W[i],V[:,j])) * W[i]
            
            grad_v = alpha_3 * V[:,j] - volation_u - volation_w

        V[:,j] = V[:,j] - grad_v * step_size
        
        # update W third
        for i in range(n_words):

            volation = np.zeros(n_components)

            for j in range(n_tags):

                volation = volation + (1 - alpha) * (F[i,j]- np.dot(W[i],V[:,j])) * V[:,j]

            grad_u = alpha_2 * W[i] - volation

        W[i] = W[i] - grad_u*step_size

    #print X.shape, U.shape, V.shape, n_components, alpha_u, alpha_v, max_iter
    print ('======')

    return U, W, V, max_iter




def TNMF(G, F, W=None, U=None, V=None, n_components=None, max_iter=None,
                                       alpha=0., alpha_1=0., alpha_2=0., alpha_3=0.):

    """Compute  with Alternative Coordinate Descent
        The objective function is minimizing the sum of squared errors with quadratic
        regularization terms.
        Parameters
        ----------
        G : array-like, shape (n_users, n_tags)
            Constant matrix.
        F : array-like, shape (n_words, n_tags)
            Constant matrix.
        U : array-like, shape (n_users, n_components)
            Initial guess for the solution.
        U : array-like, shape (n_words, n_components)
            Initial guess for the solution.
        V : array-like, shape (n_components, n_tags)
            Initial guess for the solution.
        max_iter : integer, default: 2
            Maximum number of iterations before timing out.
        alpha, alpha_1, alpha_2, alpha_3 : float, default: 0.
        """

    #tol = 1e-4
    # n_samples, n_features = X.shape
    #print (alpha, alpha_1, alpha_2, alpha_3)

    G = check_array(G, accept_sparse=('csr', 'csc'), dtype=float)
    F = check_array(F, accept_sparse=('csr', 'csc'), dtype=float)

    # check W and H, or initialize them
    U, W, V = _initialize_nmf(G, F, n_components)

    #l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = _compute_regularization(alpha, l1_ratio, regularization)

    #print X.shape, U.shape, V.shape, n_components, alpha_u, alpha_v, max_iter

    U, W, V, n_iter = _fit_coordinate_descent(G, F, U, W, V, n_components, alpha, alpha_1, alpha_2, alpha_3, max_iter)

    '''
    if n_iter == max_iter and tol > 0:
        warnings.warn("Maximum number of iteration %d reached. Increase it to"
                      " improve convergence." % max_iter, ConvergenceWarning)
    '''

    return U, W, V, n_iter



class HTLIC(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=None,
                 max_iter=20,
                 alpha=0.,
                 alpha_1=0.,
                 alpha_2=0.,
                 alpha_3=0.):
        self.n_components = n_components
        self.alpha = alpha
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
    
    def fit_transform(self, X, y=None, U=None, W=None, V=None):
        
        U, W, V, n_iter_ = TNMF(G=X[1], F=X[2], U=U, W=W, V=V, n_components=self.n_components,
                                alpha=self.alpha, alpha_1=self.alpha_1, alpha_2=self.alpha_2, alpha_3=self.alpha_3)
        
        self.n_components_ = V.shape[0]
        self.components_ = V
        self.n_iter_ = n_iter_
        self.U = U
        self.W = W
   
    def fit(self, X, y=None, **params):
        '''learn a HTLIC model for the data X=[G,F]
        '''
        self.fit_transform(X, **params)
        return self
    
    def transform(self, X):
        """Transform the data X=[G,F] according to the fitted TNMF model
        """

        U, W, _, n_iter_ = TNMF(G=X[1], F=X[2], U=None, W=None, V=self.components_, n_components=self.n_components_, max_iter=20,
                                alpha=self.alpha, alpha_1=self.alpha_1, alpha_2=self.alpha_2, alpha_3=self.alpha_3)

        return U

    # inverse_transform is useless
    def inverse_transform(self, U):
        """Transform data back to its original space.
        """
        #check_is_fitted(self, 'n_components_')
        return np.dot(U, self.components_)
    
    # score is useless
    def score(self, X, y=None):
        """Returns the score of the model on the data X
        Parameters
        score : float
        """
        return -sqrt(mean_squared_error(X, self.inverse_transform(self.transform(X))))


def main(G,F,numfeatures,numiterations):
    X = np.concatenate([G,F])
    
    # n_components = n_features 
    U, W, V, n_iter = TNMF(G, F, n_components = numfeatures, max_iter=numiterations, alpha=0.5, alpha_1=0.33, alpha_2=0.33, alpha_3=0.33)

    return U, W, V, n_iter


def evaluate(G,F,U,W,V):
    new_G = matmul(U, V)
    
    new_F = matmul(W, V)
    
    loss = LA.norm((np.subtract(G, new_G)))+LA.norm((np.subtract(F, new_F)))
    
    print ("loss:", loss)
    print ("loss G:", LA.norm((np.subtract(G, new_G))))
    print ("loss F:", LA.norm((np.subtract(F, new_F))))


if __name__ == '__main__':

    
    np.random.seed(7)
    
    G = np.random.randint(5, size=(100, 20))
    #G = np.genfromtxt('/Users/konstantinos/python/HTLIC/userstags.csv', delimiter=',')
    F = np.random.randint(10, size=(50, 20))
    #F = np.genfromtxt('/Users/konstantinos/python/HTLIC/wordstags.csv', delimiter=',')
    
    X = np.concatenate([G,F])
    
    # n_components = n_features 
    U, W, V, n_iter = TNMF(G, F, n_components=20, max_iter=200, alpha=0.5, alpha_1=0.33, alpha_2=0.33, alpha_3=0.33)

    new_G = matmul(U, V)
    
    new_F = matmul(W, V)
    
    loss = LA.norm((np.subtract(G, new_G)))+LA.norm((np.subtract(F, new_F)))
    
    print ("loss:", loss)
    print ("loss G:", LA.norm((np.subtract(G, new_G))))
    print ("loss F:", LA.norm((np.subtract(F, new_F))))
    
    #np.save('HTLIC/result_U', U)
    #np.save('HTLIC/result_W', W)
    
    
    

    
    
    