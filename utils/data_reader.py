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

class Data():
    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None

    def load_dataset(self, cols = COLS, seq_len = 10):
        '''
        1. Morning-Afternoon Split
        2. Slide through window
        3. Train-Test Split: 60% training set + 40% testing set
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
            if sum(morning_selection[i:i+2*seq_len]) != 0: #and sum(morning_selection[i:i+2*seq_len]) != 2*seq_len:
                continue # check whether it is of the same half-day
            x, y = self._next_window(i, seq_len)
            self.whole_data.append(x)
            self.whole_label.append(y)
        self.whole_data = np.array(self.whole_data)
        self.whole_label = np.array(self.whole_label)

        # Train-Test Split
        self.train_data = self.whole_data[:int(len(self.whole_data)*0.6)]
        self.train_label = self.whole_label[:int(len(self.whole_data)*0.6)]
        self.test_data = self.whole_data[int(len(self.whole_data)*0.6)+1:]
        self.test_label = self.whole_label[int(len(self.whole_data)*0.6)+1:]\
    
    def load_pickle(self):
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

    def get_next_train_batch(self):
        pass
    
    def get_next_test_batch(self):
        pass

    def _next_window(self, start, seq_len, decaying = 1):
        x = self.raw_scaled_dataset[start:start+seq_len]
        label_time_data = self.raw_scaled_dataset[start+2*seq_len]
        cur_time_data = self.raw_scaled_dataset[start+seq_len]
        y = (label_time_data[COLS.index("AskPrice1")] + label_time_data[COLS.index("BidPrice1")] - \
            cur_time_data[COLS.index("AskPrice1")] - cur_time_data[COLS.index("BidPrice1")])/2
        return x, y
    
    def normalise_window(self):
        pass

    def save_to_pickle(self):
        fw_trainX = open("trainX.pkl", 'wb')
        fw_trainY = open("trainY.pkl", 'wb')
        fw_testX = open("testX.pkl", 'wb')
        fw_testY = open("testY.pkl", 'wb')
        pickle.dump(self.train_data, fw_trainX)
        pickle.dump(self.train_label, fw_trainY)
        pickle.dump(self.test_data, fw_testX)
        pickle.dump(self.test_label, fw_testY)
        fw_trainX.close()
        fw_trainY.close()
        fw_testX.close()
        fw_testY.close()
        

if __name__ == '__main__':
    dataset = Data()
    trainX, testX, trainY, testY = dataset.Get_Train_Test()
    print(trainX.shape)
    print(testX.shape)
    print(trainY.shape)
    print(testY.shape)
