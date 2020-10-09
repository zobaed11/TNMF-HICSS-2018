from __future__ import print_function

import os
import logging
import zipfile
import numpy as np
import sys
import pickle
from six.moves import urllib
from numpy.random import RandomState
from recommend.als import ALS
from recommend.utils.evaluation import RMSE
from recommend.utils.datasets import load_movielens_1m_ratings
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

ML_1M_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML_1M_FOLDER = "ml-1m"
ML_1M_ZIP_SIZE = 24594131

rand_state = RandomState(0)

def get_precision(actual, pred):
    # Ignore nonzero terms.
    pred = np.array(pred)
    actual = np.array(actual)
    pred = np.where(pred>1,1,0)
    actual = np.where(actual>1,1,0)
    
    return precision_score(pred, actual)




# download MovieLens 1M dataset if necessary
def ml_1m_download(folder, file_size):

    file_name = "ratings.dat"
    file_path = os.path.join(os.getcwd(), folder, file_name)
    if not os.path.exists(file_path):
        print("file %s not exists. downloading..." % file_path)
        zip_name, _ = urllib.request.urlretrieve(ML_1M_URL, "ml-1m.zip")
        with zipfile.ZipFile(zip_name, 'r') as zf:
            file_path = zf.extract('ml-1m/ratings.dat')

    # check file
    statinfo = os.stat(file_path)
    if statinfo.st_size == file_size:
        print('verify success: %s' % file_path)
    else:
        raise Exception('verify failed: %s' % file_path)
    return file_path


def my_custom(ratings_file):
    with open(ratings_file) as f:
        ratings_ch = []
    
        for line in f:
#            print(line)
            
            '''
            ........modify..........
            '''
    #            line = line.split(separator)[:2]
            '''
            line for ydata
            '''
#            line=line.split(":")[2:]
            '''
            line for yahoo_movie_ratings
            '''
            line=line.split(":")
            line.remove('')
            line.remove('')
            
            line[2]=line[2].strip('\n')
            
            
            
            '''
            /...../
            '''
            line[0] = int(line[0]) 
            line [1]= int(line[1])
            line[2]=int(np.ceil(float(line[2])))
#            line[2]=float(line[2])

    
#            line = [int (l) for l in line]
            ratings_ch.append(line)
            
    ratings_ch = np.array(ratings_ch)
    return ratings_ch

# load or download MovieLens 1M dataset
    

#ft=np.load('/Users/zobaed/Desktop/ir/code/yahoo_ratings.dat')
#print(ft)
#sys.exit()
    
'''
in code
'''
#rating_file = ml_1m_download(ML_1M_FOLDER, file_size=ML_1M_ZIP_SIZE)
#ratings = load_movielens_1m_ratings(rating_file)

'''
'''
#rating_file=np.load('/Users/zobaed/Desktop/ir/recommend/examples/ml-1m/ratings.dat')







#ratings=np.load('/Users/zobaed/Desktop/ir/code/jester_ratings.dat')

#arr=ratings[:,:1]
#print(ratings[:,1])
#sys.exit
#fil = open('testfile.txt', 'w')
#
#
#sys.exit()


#rating_file='/Users/zobaed/Desktop/ir/recommend/recommend/ml-1m/ratings.dat'
#
##ratings = my_custom(rating_file)
#ratings = load_movielens_1m_ratings(rating_file)


#i=1
#fa=[]
#ta=[] 
#ko=0
#
#for y in ratings:
#    j=1
#    for x in y:
#        if x==1.0:
#            x==1.0+1.0
##            print("ami 3: %d",ko)
#   
#        #for this code need 1
#        if x==0.0:
#            x=0.0+1
#        sa= str(i) + ("::") + str(j) + ("::") + str(x)
#        fa.append(sa)
#        j+=1
#    i+=1
#        
##    ta.append(fa)
##fil = open('testfile.txt', 'w')
#with open('/Users/zobaed/Desktop/ir/code/jester_ratings_1.txt', 'w') as file_handler:
#    for item in fa:
#        #print(item)
#        
#        file_handler.write("%s\n" % item)
#       
direc = os.getcwd()

rating_file=direc+'/data/bookX_new.txt'
print("Data loaded")
ratings = my_custom(rating_file)
print("Ratings converted!")
n_user = int(max(ratings[:, 0]))
n_item = int(max(ratings[:, 1]))

# shift user_id & movie_id by 1. let user_id & movie_id start from 0
ratings[:, (0, 1)] -= 1

# split data to training & testing
train_pct = 0.1
rand_state.shuffle(ratings)
train_size = int(train_pct * ratings.shape[0])
train = ratings[:train_size]
validation = ratings[train_size:]

# models settings
n_feature = 10
eval_iters = 200
print("n_user: %d, n_item: %d, n_feature: %d, training size: %d, validation size: %d" % (
    n_user, n_item, n_feature, train.shape[0], validation.shape[0]))
als = ALS(n_user=n_user, n_item=n_item, n_feature=n_feature,
          reg=5e-2, max_rating=5., min_rating=1, seed=0)

als.fit(train, n_iters=eval_iters)
train_preds = als.predict(train[:, :2])
train_rmse = RMSE(train_preds, train[:, 2])
val_preds = als.predict(validation[:, :2])
val_rmse = RMSE(val_preds, validation[:, 2])

#pr= get_precision(val_preds,validation[:, 2] )


mean_rating_ = np.mean(ratings.take(2, axis=1))
print("after %d iterations, train RMSE: %.6f, validation RMSE: %.6f" % \
      (eval_iters, train_rmse, val_rmse))
