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
    print("***********************Yahoo! music data processing**************************")
    cd = os.getcwd()
    data_folder = cd + '/data/yahoo'
   
    names = ['user_id', 'music_id', 'Original_rating', ]
    df_main = pd.read_csv(data_folder+'/yahoo_music_ratings.dat', sep='\t', names=names)
    df_main.Original_rating[df_main.Original_rating==255] = 0
    df_main['rating']= df_main.Original_rating/20
    df = df_main.copy()
    df = df.drop(['Original_rating'],axis=1)
    #df = df.head(20000)
    print("min:", min(df['rating']))
    print("max:", max(df['rating']))
    
    
    
    sys.exit()
    n_users = len(set(df.user_id))
    n_items = len(set(df.music_id))
    ratings = np.zeros((n_users, n_items))

    temp_df = df.copy()
    
    arr_uid = sorted(set(temp_df.user_id))
    total_users = len(arr_uid)
    for i in range(len(arr_uid)):
        print(str(i+1)+" out of "+str(total_users)+" users are processed!")
        for j in range(len(temp_df.user_id)):
            if temp_df.user_id[j] == arr_uid[i]:
                temp_df.user_id[j] = i

    arr_mid=sorted(set(temp_df.music_id))
    total_music = len(arr_mid)
    
    for i in range(len(arr_mid)):
        print(str(i+1)+" out of "+str(total_music)+" musics are processed!")
        for j in range(len(temp_df.music_id)):
            if temp_df.music_id [j] == arr_mid[i]:
                temp_df.music_id[j]=i



    for row in temp_df.itertuples():
        ratings[row[1], row[2]] = row[3]
    print(ratings[0:].nonzero()[0])
    
    ratings.dump(data_folder+'yahoo_music_ratings.dat')
    #np.load('yahoo_ratings.dat')

    sparsity = len(np.flatnonzero(ratings)) #this will do as well
    sparsity /= (ratings.shape[0] * ratings.shape[1])
    sparsity *= 100
    print ('Sparsity: {:4.2f}%'.format(sparsity))