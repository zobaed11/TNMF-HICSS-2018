# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy import *
from LoadData import load_rating_data, spilt_rating_dat, load_our_data
from sklearn.metrics import precision_score
import os
import sys 



    
class PMF(object):
    def __init__(self, num_feat=10, epsilon=1, _lambda=0.1, momentum=0.8, maxepoch=200, num_batches=10,
                 batch_size=1000):
        self.num_feat = num_feat  # Number of latent features,
        self.epsilon = epsilon  # learning rate,
        self._lambda = _lambda  # L2 regularization,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)

        self.w_C = None  # Movie feature vectors
        self.w_I = None  # User feature vectors

        self.err_train = []
        self.err_val = []
        self.data = None
        self.train_data = None
        self.test_data = None
        self.train_rmse = []
        self.test_rmse = []
        self.test_hr = []
        self.test_precision = []
    # ***Fit the model with train_tuple and evaluate RMSE on both train and validation data.  ***********#
    # ***************** train_vec=TrainData, val_vec=TestData*************#
    def fit(self, train_vec, val_vec):
        # mean subtraction
        self.mean_inv = np.mean(train_vec[:, 2])  # 评分平均值
        #print("Test vec length:",len(val_vec[:, 2]))
        
        pairs_tr = train_vec.shape[0]  # traindata 中条目数
        pairs_va = val_vec.shape[0]  # testdata中条目数

        # 1-p-i, 2-m-c

        num_user = int(max(np.amax(train_vec[:, 0]), np.amax(val_vec[:, 0]))) + 1  # 第0列，user总数
        num_item = int(max(np.amax(train_vec[:, 1]), np.amax(val_vec[:, 1]))) + 1  # 第1列，movie总数

        incremental = False
        if ((not incremental) or (self.w_C is None)):
            # initialize
            self.epoch = 0
            self.w_C = 0.1 * np.random.randn(num_item, self.num_feat)  # numpy.random.randn 电影 M x D 正态分布矩阵
            self.w_I = 0.1 * np.random.randn(num_user, self.num_feat)  # numpy.random.randn 用户 N x D 正态分布矩阵

            self.w_C_inc = np.zeros((num_item, self.num_feat))  # 创建电影 M x D 0矩阵
            self.w_I_inc = np.zeros((num_user, self.num_feat))  # 创建用户 N x D 0矩阵

        while self.epoch < self.maxepoch:
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])  # 创建等差array
            np.random.shuffle(shuffled_order)  # 用于将一个列表中的元素打乱

            # Batch update
            for batch in range(self.num_batches):
                # print "epoch %d batch %d" % (self.epoch, batch+1)
                batch_idx = np.mod(np.arange(self.batch_size * batch, self.batch_size * (batch + 1)),
                                   shuffled_order.shape[0])  # 本次迭代要使用的索引下标
                batch_invID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_comID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

                # Compute Objective Function
                pred_out = np.sum(np.multiply(self.w_I[batch_invID, :], self.w_C[batch_comID, :]),
                                  axis=1)  # mean_inv subtracted
                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv

                # Compute gradients
                Ix_C = 2 * np.multiply(rawErr[:, np.newaxis], self.w_I[batch_invID, :]) + self._lambda * self.w_C[
                                                                                                         batch_comID, :]
                Ix_I = 2 * np.multiply(rawErr[:, np.newaxis], self.w_C[batch_comID, :]) + self._lambda * self.w_I[
                                                                                                         batch_invID, :]

                dw_C = np.zeros((num_item, self.num_feat))
                dw_I = np.zeros((num_user, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_C[batch_comID[i], :] += Ix_C[i, :]
                    dw_I[batch_invID[i], :] += Ix_I[i, :]

                # Update with momentum
                self.w_C_inc = self.momentum * self.w_C_inc + self.epsilon * dw_C / self.batch_size
                self.w_I_inc = self.momentum * self.w_I_inc + self.epsilon * dw_I / self.batch_size
                self.w_C = self.w_C - self.w_C_inc
                self.w_I = self.w_I - self.w_I_inc

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_I[np.array(train_vec[:, 0], dtype='int32'), :],
                                                  self.w_C[np.array(train_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - train_vec[:, 2] + self.mean_inv
                    obj = LA.norm(rawErr) ** 2 + 0.5 * self._lambda * (LA.norm(self.w_I) ** 2 + LA.norm(self.w_C) ** 2)
                    self.err_train.append(np.sqrt(obj / pairs_tr))

                # Compute validation error
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_I[np.array(val_vec[:, 0], dtype='int32'), :],
                                                  self.w_C[np.array(val_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - val_vec[:, 2] + self.mean_inv
                    self.err_val.append(LA.norm(rawErr) / np.sqrt(pairs_va))

                    # Print info
                if batch == self.num_batches - 1:
                    #print('Training RMSE: %f, Test RMSE %f' % (self.err_train[-1], self.err_val[-1]))
                    self.train_rmse.append(self.err_train[-1])
                    self.test_rmse.append(self.err_val[-1])
                    # ****************Predict rating of all movies for the given user. ***************#

    def predict(self, invID):
        return np.dot(self.w_C, self.w_I[int(invID), :]) + self.mean_inv  # numpy.dot 点乘

    # ****************Set parameters by providing a parameter dictionary.  ***********#
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)

    def topK(self, model, test_vec, k=10):  # model TrainDataSet, test_vec
    
        inv_lst = np.unique(test_vec[:, 0])
        pred = {}
        for inv in inv_lst:
            if pred.get(inv, None) is None:
                pred[inv] = np.argsort(self.predict(inv))[-k:]  
        

        intersection_cnt = {}
        for i in range(test_vec.shape[0]):
            if test_vec[i, 1] in pred[test_vec[i, 0]]:
                intersection_cnt[test_vec[i, 0]] = intersection_cnt.get(test_vec[i, 0], 0) + 1
        invPairs_cnt = np.bincount(np.array(test_vec[:, 0], dtype='int32'))
        

        
        precision_acc = 0.0
        recall_acc = 0.0
        for inv in inv_lst:
            precision_acc += intersection_cnt.get(inv, 0) / float(k)
            recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])

        return precision_acc / len(inv_lst), recall_acc / len(inv_lst)

    def get_precision(actual, pred):
        # Ignore nonzero terms.
        pred = np.array(pred[actual.nonzero()].flatten())
        actual = np.array(actual[actual.nonzero()].flatten())
        pred = np.where(pred>0,1,0)
        actual = np.where(actual>0,1,0)
        
        return precision_score(pred, actual)
    
    def get_HR(pred, rating_index, hr_factor=0.5):
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

if __name__ == "__main__":
    
    cd = os.getcwd()
    datalist = ["data/ml-100k/u.data", "data/yahoo/yahoo_movie_ratings_new.txt", "data/netflix_processed/netflix_ratings_new.txt",
                "data/jester/jester_ratings_new.txt", "data/yahoo/yahoo_music_ratings_new.txt", "data/bookX/bookX_ratings_new.txt"]
                
    names = ["ml-100k", "yahooMovie", "netflix", "jester", "yahooMusic", "bookX"]
    
    for k in range(len(datalist)):
        print("Working for: ", names[k])
        file_path = datalist[k]
        #file_path = "data/ml-100k/u.data"
        #file_path = "data/yahoo/yahooMovie_new.txt"
        #file_path = "data/netflix_processed/netflix_ratings_new_0.txt"
        #file_path = "data/jester/jester_ratings_new.txt"
        #file_path = "data/yahoo/yahoo_music_ratings_new.txt"
        #file_path = "data/bookX/bookX_ratings_new.txt"
        pmf = PMF()
        
        global size
        size = 10
        
        if file_path == datalist[0]:
            ratings = load_rating_data(file_path)
        else:    
            ratings = load_our_data(file_path)

        print(len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat)
        train, test = spilt_rating_dat(ratings)
        pmf.fit(train, test)
    
        

        # Check performance by plotting train and test errors
        print("Mean RMSE:", np.mean(pmf.test_rmse))
#        fig=plt.figure()
#        plt.plot(range(pmf.maxepoch), pmf.train_rmse, 'r--', linewidth=1, label='Training Data')
#        plt.plot(range(pmf.maxepoch), pmf.test_rmse, 'b:', linewidth=1, label='Test Data')
#        #plt.title('The '+str(names[k])+' Dataset Learning Curve')
#        plt.xlabel('Number of iterations')
#        plt.ylabel('RMSE')
#        plt.legend()
#        plt.grid()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
#        plt.show()
#        fig.savefig(cd+'/plots/pmf_'+str(names[k])+'.eps')
        
        rmse = np.array(pmf.test_rmse)
        rmse.dump(cd+'/stats/pmf_'+str(names[k])+'.dat')
        
        
        
        
        
        