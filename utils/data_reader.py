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
from sklearn.decomposition import PCA
import re

DATAROOT = "./data/data.csv"
COLS = ["indicator1", "indicator2", "indicator3", "indicator4", "indicator5", "indicator6", 
        "indicator7", "indicator8", "indicator9", "indicator10", "indicator11", "indicator12",
        "indicator13", "indicator14", "indicator15", "indicator16", "indicator17", "indicator18",
        "indicator19", "indicator20", "indicator21", "indicator22", "indicator23", "indicator24",
        "indicator25", "indicator26", "indicator27", "indicator28", "indicator29", "indicator30", 
        "indicator31", "indicator32", "indicator33", "indicator34", "indicator35", "indicator36",
        "indicator37", "indicator38", "indicator39", "indicator40", "indicator41", "indicator42", 
        "indicator43", "indicator44", "indicator45", "indicator46", "indicator47", "indicator48",
        "indicator49", "indicator50", "indicator51", "indicator52", "indicator53", "indicator54",
        "indicator55", "indicator56", "indicator57", "indicator58", "indicator59", "indicator60",
        "indicator61", "indicator62", "indicator63", "indicator64", "indicator65", "indicator66", 
        "indicator67", "indicator68", "indicator69", "indicator70", "indicator71", "indicator72",
        "indicator73", "indicator74", "indicator75", "indicator76", "indicator77", "indicator78", 
        "indicator79", "indicator80", "indicator81", "indicator82", "indicator83", "indicator84",
        "indicator85", "indicator86", "indicator87", "indicator88", "indicator89", "indicator90",
        "indicator91", "indicator92", "indicator93", "indicator94", "indicator95", "indicator1",
        "indicator97", "indicator98", "indicator99", "indicator100", "indicator101", "indicator102", 
        "indicator103", "indicator104", "indicator105", "indicator106", "indicator107", "indicator108"]

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

        # Real Useful Data
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.train_length = None
        self.test_length = None

        # Batch Count
        self.train_batch_count = 0
        self.test_batch_count = 0

        # Data Columns
        self.data_cols = COLS
        self.label_cols = ["AskPrice1", "BidPrice1"]

    def load_dataset(self, seq_len = 10, jump = 10, predict_len = 90):
        '''
        1. Morning-Afternoon Split (suppressed)
        2. Slide through window
        3. Shuffle
        4. Train-Test Split: 60% training set + 40% testing set
        '''
        wholeDataFrame = pd.read_csv(DATAROOT)

        whole_dataset = wholeDataFrame.get(self.data_cols).values
        '''
        If PCA is needed
        '''
        pca = PCA(n_components=20)
        whole_dataset = pca.fit_transform(whole_dataset)

        self.raw_scaled_dataset = whole_dataset
        # self.raw_scaled_dataset = preprocessing.scale(whole_dataset)
        self.label_dataset = wholeDataFrame.get(self.label_cols).values

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
        for i in range(0, len(whole_dataset) - seq_len - predict_len, jump):
            if sum(morning_selection[i:i + seq_len + predict_len]) != (seq_len + predict_len) and \
               sum(morning_selection[i:i+ seq_len + predict_len]) != 0:
                continue # check whether it is of the same half-day
            x, y = self._next_window(i, seq_len, predict_len)
            self.whole_data.append(x)
            self.whole_label.append(y)
        self.whole_data = np.array(self.whole_data)
        self.whole_label = np.array(self.whole_label)
        self.whole_label = np.expand_dims(self.whole_label, axis=1)

        self.save_to_pickle(self.whole_data, "whole_data")
        self.save_to_pickle(self.whole_label, "whole_label")

    def train_test_split(self):
        self.whole_data = self.load_pickle("whole_data")
        self.whole_label = self.load_pickle("whole_label")

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

        self.save_to_pickle(self.train_data, "trainX")
        self.save_to_pickle(self.train_label, "trainY")
        self.save_to_pickle(self.test_data, "testX")
        self.save_to_pickle(self.test_label, "testY")

    # To be changed
    def load_pickle(self, name):  
        with open(name + ".pkl", "rb") as fw:
            return pickle.load(fw)

    # To be changed
    def load_pickle_dataset(self):
        fr_train_data = open("./data/trainX.pkl", 'rb')
        fr_train_label = open("./data/trainY.pkl", 'rb')
        fr_test_data = open("./data/testX.pkl", 'rb')
        fr_test_label = open("./data/testY.pkl", 'rb')

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

    def save_to_pickle(self, data, name):
        fw = open(name + ".pkl", 'wb')
        pickle.dump(data, fw)
        fw.close()
        
    '''
    Helper Functions
    '''
    def _next_window(self, start, seq_len, predict_len, decaying = 1):
        x = self.raw_scaled_dataset[start:start+seq_len]
        label_time_data = self.label_dataset[start + seq_len + predict_len]
        cur_time_data = self.label_dataset[start+seq_len]
        y = (label_time_data[self.label_cols.index("AskPrice1")] + label_time_data[self.label_cols.index("BidPrice1")] - \
            cur_time_data[self.label_cols.index("AskPrice1")] - cur_time_data[self.label_cols.index("BidPrice1")])/2
        return x, y

if __name__ == '__main__':
    DataLoader = Data()
    # DataLoader.load_dataset()
    DataLoader.train_test_split()
    print(DataLoader.train_data.shape)
    print(DataLoader.train_label.shape)
    print(DataLoader.test_label[:100])