from numpy import *
import random
import numpy as np

def load_our_data(ratings_file):
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

    
#            line = [int (l) for l in line]
            ratings_ch.append(line)
            
    ratings_ch = np.array(ratings_ch)
    return ratings_ch
    
def load_rating_data(file_path='ml-100k/u.data'):
    """
    load movie lens 100k ratings from original rating file.
    need to download and put rating data in /data folder first.
    Source: http://www.grouplens.org/
    """
    prefer = []
    for line in open(file_path, 'r'):  # 打开指定文件
        (userid, movieid, rating, ts) = line.split('\t')  # 数据集中每行有4项
        uid = int(userid)
        mid = int(movieid)
        rat = float(rating)
        prefer.append([uid, mid, rat])
    data = array(prefer)
    return data


def spilt_rating_dat(data, size=0.2):
    train_data = []
    test_data = []
    for line in data:
        rand = random.random()
        if rand < size:
            test_data.append(line)
        else:
            train_data.append(line)
    train_data = array(train_data)
    test_data = array(test_data)
    return train_data, test_data
