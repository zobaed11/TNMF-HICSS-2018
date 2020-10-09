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


if __name__ == "__main__":
    cd = os.getcwd()
    data_dir = cd + '/output'
    plot_dir = cd + '/plots'
    stat = cd + '/stats'
    #print(os.listdir(data_dir))
    bestK = 40
    dataset = ['yahooMovies', 'yahooMusic', 'ml-100k', 'bookX', 'jester', 'netflix']

    for k in range(len(dataset)):
        iter_array = np.array([1, 2, 5, 10, 25, 50, 100, 200])
        factor = [10,20,30,40,50]
        fig, ax = plt.subplots()
        colors = ['r--','g-*','b-x','k:o','m^-']
        labels = [r'$\lambda$=10',r'$\lambda$=20',r'$\lambda$=30',r'$\lambda$=40',r'$\lambda$=50']

        for i in range(len(factor)):
            
            table = -np.sort(-np.load(data_dir+'/test_mse_'+str(dataset[k])+'_factor_'+str(factor[i])+'.dat'))
            ax.plot(range(table.shape[0]), table, str(colors[i]), linewidth=1)
            ax.legend(labels)
        ax.set_xticklabels(iter_array)
        ax.set_xlabel('Number of iterations')
        ax.set_ylabel('RMSE')
        ax.grid()
        fig.savefig(plot_dir+'/results_tnmf_'+str(dataset[k])+'.eps')
    
    for i in range(len(dataset)):
        fig, ax = plt.subplots()
        if dataset[i] == 'bookX':
            table = np.load(data_dir+'/pmf_'+str(dataset[i])+'.dat')
            table += 1
        else:
            table = np.load(data_dir+'/pmf_'+str(dataset[i])+'.dat')
        ax.plot(range(table.shape[0]), table, 'r-')
        ax.set_xlabel('Number of iterations')
        ax.set_ylabel('RMSE')
        ax.grid()
        fig.savefig(plot_dir+'/results_pmf_'+str(dataset[i])+'.eps')

    for i in range(len(dataset)):
        fig, ax = plt.subplots()
        table = pd.read_csv(stat+'/als_'+str(dataset[i])+'.txt')
        ax.plot(range(table.shape[0]), table, 'r-')
        #ax.legend(labels)
        #ax.set_xticklabels(dataset)
        ax.set_xlabel('Number of iterations')
        ax.set_ylabel('RMSE')
        ax.grid()
        fig.savefig(plot_dir+'/results_als_'+str(dataset[i])+'.eps')      
        
    for i in range(len(dataset)):
        fig, ax = plt.subplots()
        table = pd.read_csv(stat+'/bpmf_'+str(dataset[i])+'.txt')
        ax.plot(range(table.shape[0]), table, 'r-')
        ax.set_xlabel('Number of iterations')
        ax.set_ylabel('RMSE')
        ax.grid()
        fig.savefig(plot_dir+'/results_bpmf_'+str(dataset[i])+'.eps')
        
        
        