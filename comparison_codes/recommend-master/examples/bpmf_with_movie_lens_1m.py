from __future__ import print_function

import os
import logging
import zipfile
from six.moves import urllib
from numpy.random import RandomState
from recommend.bpmf import BPMF
from recommend.utils.evaluation import RMSE
from recommend.utils.datasets import load_movielens_1m_ratings
import sys
import numpy as np
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

ML_1M_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML_1M_FOLDER = "ml-1m"
ML_1M_ZIP_SIZE = 24594131

rand_state = RandomState(0)




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

# load or download MovieLens 1M dataset
#rating_file = ml_1m_download(ML_1M_FOLDER, file_size=ML_1M_ZIP_SIZE)
#ratings = load_movielens_1m_ratings('/Users/zobaed/Desktop/ir/recommend/recommend/ml-1m/ratings.dat')
    

direc = os.getcwd()

rating_file=direc+'/data/bookX_ratings_1.txt'
ratings=my_custom(rating_file)






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

#rating_file='/Users/zobaed/Desktop/ir/code/jester_ratings_1.txt'
#
#ratings = my_custom(rating_file)


n_user = max(ratings[:, 0])
n_item = max(ratings[:, 1])

# shift user_id & movie_id by 1. let user_id & movie_id start from 0
ratings[:, (0, 1)] -= 1

# split data to training & testing
train_pct = 0.9

rand_state.shuffle(ratings)
train_size = int(train_pct * ratings.shape[0])
train = ratings[:train_size]
validation = ratings[train_size:]

# models settings
n_feature = 10
eval_iters = 200
print("n_user: %d, n_item: %d, n_feature: %d, training size: %d, validation size: %d" % (
    n_user, n_item, n_feature, train.shape[0], validation.shape[0]))
bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=n_feature,
            max_rating=5., min_rating=1., seed=0)

bpmf.fit(train, n_iters=eval_iters)

train_preds = bpmf.predict(train[:, :2])
train_rmse = RMSE(train_preds, train[:, 2])
val_preds = bpmf.predict(validation[:, :2])
val_rmse = RMSE(val_preds, validation[:, 2])
print("after %d iteration, train RMSE: %.6f, validation RMSE: %.6f" %
      (eval_iters, train_rmse, val_rmse))
