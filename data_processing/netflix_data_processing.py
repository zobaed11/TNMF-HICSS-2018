#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:27:48 2017

@author: nsrg
"""
import time
import pandas as pd
import numpy as np
import os
import warnings
import sys
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    print(__doc__)
    print("***********************Netflix data processing**************************")
    start = time.clock()
    cd = os.getcwd()

    
    data_folder= cd + '/data/netflix/'    
    processed_folder = cd + '/data/netflix_processed/'
    names = ['user_id', 'rating', 'timestamp', 'movie_id']
    
    file_list=os.listdir(data_folder)
    newDF = pd.DataFrame()
    counter =0
    
    """
    create raw rating format
    """
    for i in range(len(file_list)):
        counter += 1
        content = data_folder + file_list[i]
        oldDF = pd.read_csv(content, sep=',', names=names)
        oldDF['movie_id']= oldDF.user_id[0].strip(':')
        oldDF.drop(0, inplace=True)
        newDF = newDF.append(oldDF, ignore_index = True)
        if counter ==2000:
            break
    newDF.to_csv(processed_folder+'netflix_ratings_raw.txt', sep='\t', index=False)

    """
    convert the raw rating into ratings matrix
    where each row is a user and a column a movie
    """
    data_folder = cd+'/data/netflix_processed/'
    df = pd.read_csv(processed_folder + 'netflix_ratings_raw.txt', sep='\t')
    df = df.head(10000)
    sys.exit()
    n_users = len(set(df.user_id))
    n_items = len(set(df.movie_id))
    ratings = np.zeros((n_users, n_items))

    print("users:", n_users)
    print("movies:", n_items)
    sys.exit()
    temp_df = df.copy()

    arr_uid = sorted(set(temp_df.user_id)) # sorted unique user id's
    total_users = len(arr_uid)
    
    for i in range(len(arr_uid)):
        print(str(i+1)+" out of "+str(total_users)+" users are processed!")
        for j in range(len(temp_df.user_id)):
            if temp_df.user_id[j] == arr_uid[i]:
                temp_df.user_id[j] = i


    arr_mid = sorted(set(temp_df.movie_id)) # sorted unique movie id's
    total_movies = len(arr_mid)
    for i in range(len(arr_mid)):
        print(str(i+1)+" out of "+str(total_movies)+" movies are processed!")
        for j in range(len(temp_df.movie_id)):
            if temp_df.movie_id [j] == arr_mid[i]:
                temp_df.movie_id[j] = i



    for row in temp_df.itertuples():
        ratings[row[1], row[4]] = row[2]
    
    ratings.dump(processed_folder+'netflix_ratings.dat')

    
   