#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:57:32 2017
@description: Non-negative matrix factorization for top-N recommendation tasks.
-------------------------------------------------------------------------------
"""
import time
import pandas as pd
import numpy as np
import math
#from matplotlib import rcParams
#rcParams['figure.figsize'] = (10, 6)
import matplotlib.pyplot as plt
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import os
import warnings
import sys
warnings.filterwarnings("ignore")


def get_precision(actual, pred):
    # Ignore nonzero terms.
    pred = np.array(pred[actual.nonzero()].flatten())
    actual = np.array(actual[actual.nonzero()].flatten())
    #pred = np.where(pred>0,1,0)
    #actual = np.where(actual>0,1,0)
    
    return precision_score(pred, actual)
    
def get_HR(pred, rating_index, hr_factor=0.6):
    # Ignore nonzero terms.
    pred = np.where(pred>0,1,0)
    hits = 0
    for i in range(len(pred)):
        content = list(pred[i][rating_index[i]])
        temp_hits = (sum(content))
        if temp_hits >= (size*hr_factor):
            hits += 1            
    hr = hits/len(pred)
    
    return hr
    
def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    #pred = np.where(pred>0,1,0)
    #actual = np.where(actual>0,1,0)
    
    return mean_squared_error(pred, actual)

class ExplicitMF():
    def __init__(self, 
                 ratings,
                 n_factors=20,
                 learning='sgd',
                 item_fact_reg=0.0, 
                 user_fact_reg=0.0,
                 item_bias_reg=0.0,
                 user_bias_reg=0.0,
                 verbose=False):
        """
        Train a matrix factorization model to predict empty 
        entries in a matrix. The terminology assumes a 
        ratings matrix which is ~ user x item
        
        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings
        
        n_factors : (int)
            Number of latent factors to use in matrix 
            factorization model
        learning : (str)
            Method of optimization. Options include 
            'sgd' or 'als'.
        
        item_fact_reg : (float)
            Regularization term for item latent factors
        
        user_fact_reg : (float)
            Regularization term for user latent factors
            
        item_bias_reg : (float)
            Regularization term for item biases
        
        user_bias_reg : (float)
            Regularization term for user biases
        
        verbose : (bool)
            Whether or not to printout training progress
        """
        
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_fact_reg = item_fact_reg
        self.user_fact_reg = user_fact_reg
        self.item_bias_reg = item_bias_reg
        self.user_bias_reg = user_bias_reg
        self.learning = learning
        if self.learning == 'sgd':
            self.sample_row, self.sample_col = self.ratings.nonzero()
            self.n_samples = len(self.sample_row)
        self._v = verbose

    def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 ratings,
                 _lambda,
                 type='user'):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        if type == 'user':
            # Precompute
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * _lambda

            for u in range(latent_vectors.shape[0]):
                latent_vectors[u, :] = solve((YTY + lambdaI), 
                                             ratings[u, :].dot(fixed_vecs))
        elif type == 'item':
            # Precompute
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * _lambda
            
            for i in range(latent_vectors.shape[0]):
                latent_vectors[i, :] = solve((XTX + lambdaI), 
                                             ratings[:, i].T.dot(fixed_vecs))
        return latent_vectors

    def train(self, n_iter=10, learning_rate=0.1):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors        
        self.user_vecs = np.random.normal(scale=1./self.n_factors,\
                                          size=(self.n_users, self.n_factors))
        self.item_vecs = np.random.normal(scale=1./self.n_factors,
                                          size=(self.n_items, self.n_factors))
        
        if self.learning == 'als':
            self.partial_train(n_iter)
        elif self.learning == 'sgd':
            self.learning_rate = learning_rate
            self.user_bias = np.zeros(self.n_users)
            self.item_bias = np.zeros(self.n_items)
            self.global_bias = np.mean(self.ratings[np.where(self.ratings != 0)])
            self.partial_train(n_iter)
    
    
    def partial_train(self, n_iter):
        """ 
        Train model for n_iter iterations. Can be 
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self._v:
                print ('\tcurrent iteration: {}'.format(ctr))
            if self.learning == 'als':
                self.user_vecs = self.als_step(self.user_vecs, 
                                               self.item_vecs, 
                                               self.ratings, 
                                               self.user_fact_reg, 
                                               type='user')
                self.item_vecs = self.als_step(self.item_vecs, 
                                               self.user_vecs, 
                                               self.ratings, 
                                               self.item_fact_reg, 
                                               type='item')
            elif self.learning == 'sgd':
                self.training_indices = np.arange(self.n_samples)
                np.random.shuffle(self.training_indices)
                self.sgd()
            ctr += 1

    def sgd(self):
        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]
            prediction = self.predict(u, i)
            e = (self.ratings[u,i] - prediction) # error
            eta = 0.000001
            beta = 0.000000001
            # Update biases
            self.user_bias[u] += self.learning_rate * \
                                (e - self.user_bias_reg * self.user_bias[u])
            self.item_bias[i] += self.learning_rate * \
                                (e - self.item_bias_reg * self.item_bias[i])
            
            #Update latent factors
            A = sum(self.item_vecs[i,:])
            B = sum(self.ratings[u, :])
            XTY = self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
            C = sum(self.user_vecs[u, :])
            
            self.user_vecs[u, :] = self.user_vecs[u, :] - eta * ( ((A*B)/XTY) + A - (beta * C) )
            #print("User:", self.user_vecs[u, :])
#            self.user_vecs[u, :] += self.learning_rate * \
#                                    (e * self.item_vecs[i, :] - \
#                                     self.user_fact_reg * self.user_vecs[u,:])
            D = sum(self.ratings[:, i])
            self.item_vecs[i, :] = self.item_vecs[i, :] - eta * ( ((C*D)/XTY) + C - (beta * A) )
            #print("Item:", self.item_vecs[i, :])
#            self.item_vecs[i, :] += self.learning_rate * \
#                                    (e * self.user_vecs[u, :] - \
#                                     self.item_fact_reg * self.item_vecs[i,:])
                                    
    def predict(self, u, i):
        """ Single user and item prediction."""
        
        if self.learning == 'als':
            return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
        elif self.learning == 'sgd':
            #prediction = self.global_bias + self.user_bias[u] + self.item_bias[i]
            #prediction += self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
            #print("Error here:", math.log((self.ratings[u,i]/XTY),2))
            prediction = self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
            #prediction += (self.ratings[u, i] * math.log((self.ratings[u,i]/XTY),2)) - self.ratings[u,i]
        
            if prediction > threshold:
                prediction = threshold
            if prediction < 0:
                prediction =0
            #print("Pred:", prediction)
            return prediction
    
    def predict_all(self):
        """ Predict ratings for every user and item."""
        predictions = np.zeros((self.user_vecs.shape[0], 
                                self.item_vecs.shape[0]))
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)
                
        return predictions
    
    def calculate_learning_curve(self, iter_array, test, rating_index, learning_rate=0.1):
        """
        Keep track of MSE as a function of training iterations.
        
        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of 
            the learning curve. e.g. [1, 5, 10, 20]
        test : (2D ndarray)
            Testing dataset (assumed to be user x item).
        
        The function creates two new class attributes:
        
        train_mse : (list)
            Training data MSE values for each value of iter_array
        test_mse : (list)
            Test data MSE values for each value of iter_array
        """
        print("In calculate learning curve")
        iter_array.sort()
        self.train_mse =[]
        self.test_mse = []
        self.test_hr = []
        self.test_precision = []

        iter_diff = 0
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print ('Iteration: {}'.format(n_iter))
            if i == 0:
                self.train(n_iter - iter_diff, learning_rate)
            else:
                self.partial_train(n_iter - iter_diff)

            predictions = self.predict_all()
            
            self.train_mse += [get_mse(predictions, self.ratings)]
            self.test_mse += [get_mse(predictions, test)]
            self.test_hr += [get_HR(predictions, rating_index)]
            self.test_precision += [get_precision(test, predictions)]
            if self._v:
                print ('Train mse: ' + str(self.train_mse[-1]))
                print ('Test mse: ' + str(self.test_mse[-1]))
                print ('Test hr: ' + str(self.test_hr[-1]))
                print ('Test precision: ' + str(self.test_precision[-1]))
            iter_diff = n_iter
"""
split data into training and test sets by removing 10 ratings per user 
from the training set and placing them in the test set
"""
def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    rating_index = [] # for hit rate computation
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=size, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        rating_index.append(test_ratings)
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test, rating_index
    
    
if __name__ == "__main__":
    
    print(__doc__)
    print("***********************SIMULATION for lastfm BEGINS**************************")
    start = time.clock()
    cd = os.getcwd()
    data_folder= cd+'/data/lastfm'
    output_folder = cd + '/output/'
    names = ['user_id', 'artist_id', 'rating']
    df = pd.read_csv(data_folder+'/user_artists.dat', sep='\t', names=names)
    df = df.iloc[1:]
    df.user_id = pd.to_numeric(df.user_id, errors='coerce').fillna(0).astype(np.int64)
    df.artist_id = pd.to_numeric(df.artist_id, errors='coerce').fillna(0).astype(np.int64)
    df.rating = pd.to_numeric(df.rating, errors='coerce').fillna(0).astype(np.int64)
    mn = min(df.rating)
    mx = max(df.rating)

    for i in range(len(df['rating'])):
        df['rating'].iloc[i] = (df['rating'].iloc[i] - mn) / (mx - mn)
        
    #df['rating'] = df['rating'].apply(lambda x: (df['rating'][x]-mn)/(mx-mn))
    
    #sys.exit()
    n_users = max(df.user_id)
    n_items = max(df.artist_id)
    ratings = np.zeros((n_users, n_items))

    for row in df.itertuples():
        ratings[row[1], row[2]] = row[3]

    global threshold
    threshold = 5
    global size
    size = 10
    
    sparsity = len(np.flatnonzero(ratings)) #this will do as well
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    print ('Sparsity: {:4.2f}%'.format(sparsity))
    sys.exit()
    train, test, rating_index = train_test_split(ratings) # missing indices are "rating_index"
    
    matrix_factors = [10,20,30,40,50]

    for k in matrix_factors:
        print("Working for factor: ", str(k))
        MF_SGD = ExplicitMF(train, k, learning='sgd', verbose=True)
        iter_array = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        MF_SGD.calculate_learning_curve(iter_array, test, rating_index, learning_rate=0.001)
        
        train_mse = np.array(MF_SGD.train_mse)
        test_mse = np.array(MF_SGD.test_mse)
        test_hr = np.array(MF_SGD.test_hr)
        test_precision = np.array(MF_SGD.test_precision)
        
        train_mse.dump(output_folder + 'train_mse_lastfm_factor_'+str(k)+'.dat')
        test_mse.dump(output_folder + 'test_mse_lastfm_factor_'+str(k)+'.dat')
        test_hr.dump(output_folder + 'test_hr_lastfm_factor_'+str(k)+'.dat')
        test_precision.dump(output_folder + 'test_precision_lastfm_factor_'+str(k)+'.dat')
    
    print("Done")
