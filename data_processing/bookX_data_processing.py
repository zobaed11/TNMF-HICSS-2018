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
import sys
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    print(__doc__)
    print("***********************Book Crossing! data processing**************************")
    cd = os.getcwd()
    print (cd)
    data_folder = cd + '/data/bookX/'
   
    names = ['user_id', 'isbn', 'rating', ]
    df = pd.read_csv(data_folder+'/BX-Book-Ratings.csv', sep=';', names=names)


    
    df = df.head(30000)
    
    n_users = len(set(df.user_id))
    n_items = len(set(df.isbn))
    ratings = np.zeros((n_users, n_items))

    temp_df = df.copy()

    arr_uid = sorted(set(temp_df.user_id))
    total_users = len(arr_uid)
    for i in range(len(arr_uid)):
        print(str(i+1)+" out of "+str(total_users)+" users are processed!")
        for j in range(len(temp_df.user_id)):
            if temp_df.user_id[j] == arr_uid[i]:
                temp_df.user_id[j] = i

    arr_isbn=sorted(set(temp_df.isbn))
    total_isbn = len(arr_isbn)
    
    for i in range(len(arr_isbn)):
        print(str(i+1)+" out of "+str(total_isbn)+" books are processed!")
        for j in range(len(temp_df.isbn)):
            if temp_df.isbn[j] == arr_isbn[i]:
                temp_df.isbn[j] = i



    for row in temp_df.itertuples():
        ratings[row[1], row[2]] = row[3]
    print(ratings[0:].nonzero()[0])
    
    ratings.dump(data_folder+'bookX_ratings.dat')
    #np.load('yahoo_ratings.dat')

    sparsity = len(np.flatnonzero(ratings)) #this will do as well
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    print ('Sparsity: {:4.2f}%'.format(sparsity))