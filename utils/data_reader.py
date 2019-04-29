"""
Method:
    save_to_pickle: Optional choice, can be saved to pickle for convenience
Features:
    1. Morning-Afternoon Split
    2. Normalization
    3. Choose Label
"""

#-*- coding:utf-8 -*-
import os
import pandas as pd
import numpy as np
import pickle                  # OPTIONAL, if not needed, ignore it
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import re

DATAROOT = "./data/data.csv"
COLS = ["midPrice", "LastPrice", "Volume", "LastVolume", "Turnover", "LastTurnover", 
        "AskPrice1", "BidPrice1", "AskVolume1", "BidVolume1", "OpenInterest", "UpperLimitPrice", "LowerLimitPrice"]

'''
TODO:
    1. 把训练数据换为108个indicator
    2. 做PCA降维
    3. 减少数据规模：增加一个跳步jump，每隔一个jump采集一个数据
'''

class Data():
    def __init__(self):
        self.whole_data = None
        self.whole_label = None

        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.train_length = None
        self.test_length = None
        self.train_batch_count = 0
        self.test_batch_count = 0

    def load_dataset(self, cols = COLS, seq_len = 10):
        '''
        1. Morning-Afternoon Split
        2. Slide through window
        3. Shuffle
        4. Train-Test Split: 60% training set + 40% testing set
        '''
        wholeDataFrame = pd.read_csv(DATAROOT)

        whole_dataset = wholeDataFrame.get(cols).values
        self.raw_scaled_dataset = preprocessing.scale(whole_dataset)

        # check the data is morning or afternoon: using re-match
        time_stamp = wholeDataFrame["UpdateTime"].values
        r = re.compile('.[901].*')
        vmatch = np.vectorize(lambda x:bool(r.match(x)))
        morning_selection = vmatch(time_stamp)

        # make new data set: 
        #   data: 10 timestamp data 
        #   label: 10-th predicted data
        self.whole_data = []
        self.whole_label = []
        for i in range(len(whole_dataset) - 2*seq_len):
            if sum(morning_selection[i:i+2*seq_len]) != 2*seq_len: #and sum(morning_selection[i:i+2*seq_len]) != 0:
                continue # check whether it is of the same half-day
            x, y = self._next_window(i, seq_len)
            self.whole_data.append(x)
            self.whole_label.append(y)
        self.whole_data = np.array(self.whole_data)
        self.whole_label = np.array(self.whole_label)
        self.whole_label = np.expand_dims(self.whole_label, axis=1)

    def train_test_split(self):
        self.whole_data = self.load_pickle("whole_scanned_data_morning")
        self.whole_label = self.load_pickle("whole_scanned_label_morning")

        shuffled_ix = np.random.permutation(np.arange(len(self.whole_data)))
        self.whole_data = self.whole_data[shuffled_ix]
        self.whole_label = self.whole_label[shuffled_ix]

        # Train-Test Split
        self.train_data = self.whole_data[:int(len(self.whole_data)*0.6)]
        self.train_label = self.whole_label[:int(len(self.whole_data)*0.6)]
        self.test_data = self.whole_data[int(len(self.whole_data)*0.6)+1:]
        self.test_label = self.whole_label[int(len(self.whole_data)*0.6)+1:]

        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)

        self.save_to_pickle(self.train_data, "trainX_morning")
        self.save_to_pickle(self.train_label, "trainY_morning")
        self.save_to_pickle(self.test_data, "testX_morning")
        self.save_to_pickle(self.test_label, "testY_morning")

    def load_pickle(self, name):  
        with open(name + ".pkl", "rb") as fw:
            return pickle.load(fw)

    def load_pickle_dataset(self):
        fr_train_data = open("./data/trainX_morning.pkl", 'rb')
        fr_train_label = open("./data/trainY_morning.pkl", 'rb')
        fr_test_data = open("./data/testX_morning.pkl", 'rb')
        fr_test_label = open("./data/testY_morning.pkl", 'rb')

        self.train_data = pickle.load(fr_train_data)
        self.train_label = pickle.load(fr_train_label)
        self.test_data = pickle.load(fr_test_data)
        self.test_label = pickle.load(fr_test_label)
        
        fr_train_data.close()
        fr_train_label.close()
        fr_test_data.close()
        fr_test_label.close()

        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)

    def get_next_train_batch(self, batch_num):
        if self.train_batch_count + batch_num > self.train_length:
            batch_x = self.train_data[self.train_batch_count:]
            batch_y = self.train_label[self.train_batch_count:]            
            self.train_batch_count = 0
            return batch_x, batch_y
        batch_x = self.train_data[self.train_batch_count:self.train_batch_count+batch_num]
        batch_y = self.train_label[self.train_batch_count:self.train_batch_count+batch_num]
        self.train_batch_count = self.train_batch_count + batch_num
        return batch_x, batch_y

    def get_next_test_batch(self, batch_num):
        if self.test_batch_count + batch_num > self.test_length:
            batch_x = self.test_data[self.test_batch_count:]
            batch_y = self.test_label[self.test_batch_count:]            
            self.test_batch_count = 0
            return batch_x, batch_y
        batch_x = self.test_data[self.test_batch_count:self.test_batch_count+batch_num]
        batch_y = self.test_label[self.test_batch_count:self.test_batch_count+batch_num]
        self.test_batch_count = self.test_batch_count + batch_num
        return batch_x, batch_y

    def _next_window(self, start, seq_len, decaying = 1):
        x = self.raw_scaled_dataset[start:start+seq_len]
        label_time_data = self.raw_scaled_dataset[start+2*seq_len]
        cur_time_data = self.raw_scaled_dataset[start+seq_len]
        y = (label_time_data[COLS.index("AskPrice1")] + label_time_data[COLS.index("BidPrice1")] - \
            cur_time_data[COLS.index("AskPrice1")] - cur_time_data[COLS.index("BidPrice1")])/2
        return x, y

    def save_to_pickle(self, data, name):
        fw = open(name + ".pkl", 'wb')
        pickle.dump(data, fw)
        fw.close()
        
        

if __name__ == '__main__':
    DataLoader = Data()
    DataLoader.train_test_split()