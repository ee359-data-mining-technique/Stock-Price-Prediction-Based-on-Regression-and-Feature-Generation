"""
To Jerry:
    Change the ROOT when organizing the code

Method:
    Get_Train_Test: Get the separated training and testing dataset, with form of numpy.array [sample_num, dimensions]
                    60% train and 40% test ;
                    returned in order: trainX, testX, trainY, testY 
    save_to_pickle: Optional choice, can be saved to pickle for convenience

"""

#-*- coding:utf-8 -*-
import os
import pandas as pd
import numpy as np
import pickle                  # OPTIONAL, if not needed, ignore it
from sklearn.model_selection import train_test_split
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #CodeFramework替换成项目名

DATAROOT = os.path.join(ROOT, 'data', 'Animals_with_Attributes2-2', 'Features', 'ResNet101')#dataset替换成数据集名称



TRAINROOT = os.path.join(DATAROOT, 'train.json')
DEVROOT = os.path.join(DATAROOT, 'dev.json')
TESTROOT = os.path.join(DATAROOT, 'test.json')

class Data():#在其中拆分数据的属性并定义数据预处理的函数
    def __init__(self):
        self.FeatureRoot = os.path.join(DATAROOT, 'AwA2-features.txt')
        self.FilenameRoot = os.path.join(DATAROOT, 'AwA2-filenames.txt')
        self.LabelsRoot = os.path.join(DATAROOT, 'AwA2-labels.txt')
        self.features = pd.read_table(self.FeatureRoot, delim_whitespace=True, names=[x for x in range(2048)])
        self.filename = pd.read_table(self.FilenameRoot, delim_whitespace=True, names=["filename"])
        self.labels = pd.read_table(self.LabelsRoot, delim_whitespace=True, names=["labels"])
        self.data = pd.concat([self.features, self.filename, self.labels], axis=1)
        self.train_data_x = np.zeros(shape=[1, 2048])
        self.train_data_y = np.zeros(shape=[1])
        self.test_data_x = np.zeros(shape=[1, 2048])
        self.test_data_y = np.zeros(shape=[1])

    def Get_Train_Test(self):
        ## We are required to split the images in each category into 60% of training set and
        ## 40% of testing set
        ## Attention: filename is not deleted from the training and testing data
        for label in range(1, 51):
            tmp_data = self.data.loc[self.data["labels"]==label]
            tmp_label = tmp_data["labels"]
            tmp_feature = tmp_data.drop(["labels"], axis=1)
            tmp_label = np.array(tmp_label)
            tmp_feature = np.array(tmp_feature)
            tmp_train_x, tmp_test_x, tmp_train_y, tmp_test_y = train_test_split(tmp_feature, tmp_label, test_size=0.4, random_state=0)
            tmp_train_x = tmp_train_x[:, 0:2048]
            tmp_test_x = tmp_test_x[:, 0:2048]
            self.train_data_x = np.concatenate([self.train_data_x, tmp_train_x], axis=0)
            self.test_data_x = np.concatenate([self.test_data_x, tmp_test_x], axis=0)
            self.train_data_y = np.concatenate([self.train_data_y, tmp_train_y], axis=0)
            self.test_data_y = np.concatenate([self.test_data_y, tmp_test_y], axis=0)
            if (label % 10 == 0):
                print ("10 categories completed")
        self.train_data_x = self.train_data_x[1:]
        self.test_data_x = self.test_data_x[1:]
        self.train_data_y = self.train_data_y[1:]
        self.test_data_y = self.test_data_y[1:]

        # shuffle the data
        state = np.random.get_state()
        np.random.shuffle(self.train_data_x)
        np.random.set_state(state)
        np.random.shuffle(self.train_data_y)

        state = np.random.get_state()
        np.random.shuffle(self.test_data_x)
        np.random.set_state(state)
        np.random.shuffle(self.test_data_y)

        return self.train_data_x, self.test_data_x, self.train_data_y, self.test_data_y

    def save_to_pickle(self):
        fw_trainX = open("trainX.pkl", 'wb')
        fw_trainY = open("trainY.pkl", 'wb')
        fw_testX = open("testX.pkl", 'wb')
        fw_testY = open("testY.pkl", 'wb')
        pickle.dump(self.train_data_x, fw_trainX)
        pickle.dump(self.train_data_y, fw_trainY)
        pickle.dump(self.test_data_x, fw_testX)
        pickle.dump(self.test_data_y, fw_testY)
        fw_trainX.close()
        fw_trainY.close()
        fw_testX.close()
        fw_testY.close()
        # If want to read, use:
        # x = open("xxx", 'rb')
        # y = pickle.load(x)
        # x.close()

if __name__ == '__main__':
    dataset = Data()
    trainX, testX, trainY, testY = dataset.Get_Train_Test()
    print(trainX.shape)
    print(testX.shape)
    print(trainY.shape)
    print(testY.shape)
