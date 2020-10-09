#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 00:40:17 2017

@author: nsrg
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys



if __name__ == "__main__":
    cd = os.getcwd()
    data_dir = cd + '/output'
    plot_dir = cd + '/plots'
    stat = cd + '/stats'
    #print(os.listdir(data_dir))
    bestK = 40
    dataset = ['yahooMovies', 'yahooMusic', 'netflix', 'ml-100k', 'jester', 'bookX']

    print("*******************************************************")
    for k in range(len(dataset)):
        iter_array = np.array([1, 2, 5, 10, 25, 50, 100, 200])
        factor = [10,20,30,40,50]
        for i in range(len(factor)):
            table = -np.sort(-np.load(data_dir+'/test_mse_'+str(dataset[k])+'_factor_'+str(factor[i])+'.dat'))
            if factor[i] == bestK:
                print("INFO: TNMF: Mean RMSE for "+str(dataset[k])+": "+ str(round(np.mean(table),3))+'+/-'+str(round(np.std(table),3)))
    print("*******************************************************")
    for i in range(len(dataset)):
        if dataset[i] == 'bookX':
            table = np.load(data_dir+'/pmf_'+str(dataset[i])+'.dat')
            table += 1
        else:
            table = np.load(data_dir+'/pmf_'+str(dataset[i])+'.dat')
        print("INFO: PMF: Mean RMSE for "+str(dataset[i])+": "+ str(round(np.mean(table),3))+'+/-'+str(round(np.std(table),3)))
        
    print("*******************************************************")       
    for i in range(len(dataset)):
        table = []
        with open(stat+'/als_'+str(dataset[i])+'.txt') as f:    
            for line in f:
                table.append(float(line.strip('\n')))
        print("INFO: ALS: Mean RMSE for "+str(dataset[i])+": "+ str(round(np.mean(table),3))+'+/-'+str(round(np.std(table),3)))
    print("*******************************************************")
    for i in range(len(dataset)):
        table = []
        with open(stat+'/bpmf_'+str(dataset[i])+'.txt') as f:    
            for line in f:
                table.append(float(line.strip('\n')))
        print("INFO: BPMF: Mean RMSE for "+str(dataset[i])+": "+ str(round(np.mean(table),3))+'+/-'+str(round(np.std(table),3)))
    print("*******************************************************")