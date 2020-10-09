#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:12:57 2017

@author: Md Enamul Haque
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    print(__doc__)
    print("***********************Yahoo! data processing**************************")
    cd = os.getcwd()
    data_folder = cd + '/data/yahoo/'
   
    
    names = ['user_id', 'movie_id', 'invalid_rating', 'rating', ]
    df = pd.read_csv(data_folder+'ydata.txt', sep='\t', names=names)
    
    df = df.drop(['invalid_rating'], axis=1)
    
    n_users = len(set(df.user_id))
    n_items = len(set(df.movie_id))
    ratings = np.zeros((n_users, n_items))

    temp_df = df.copy()
    
    arr_uid = sorted(set(temp_df.user_id))
    total_users = len(arr_uid)
    for i in range(len(arr_uid)):
        print(str(i)+" out of "+str(total_users)+" users are processed!")
        for j in range(len(temp_df.user_id)):
            if temp_df.user_id[j] == arr_uid[i]:
                temp_df.user_id[j] = i

    arr_mid=sorted(set(temp_df.movie_id))
    total_movies = len(arr_mid)
    
    for i in range(len(arr_mid)):
        print(str(i)+" out of "+str(total_movies)+" movies are processed!")
        for j in range(len(temp_df.movie_id)):
            if temp_df.movie_id [j] == arr_mid[i]:
                temp_df.movie_id[j]=i



    for row in temp_df.itertuples():
        ratings[row[1], row[2]] = row[3]
    print(ratings[0:].nonzero()[0])
    
    ratings.dump('yahoo_movie_ratings.dat')
    #np.load('yahoo_ratings.dat')

    sparsity = len(np.flatnonzero(ratings)) #this will do as well
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    print ('Sparsity: {:4.2f}%'.format(sparsity))