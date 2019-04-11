#-*- coding:utf-8 -*-

import numpy as np
import time, os, gc
from sklearn.metrics import mean_squared_error

class Solver():

    def __init__(self, model, loss_function, optimizer, exp_path='', logger='', device='cpu'):
        super(Solver, self).__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.exp_path = exp_path
        self.logger = logger
        self.device = device

    def train_and_validate(self, *args, **kargs):
        raise NotImplementedError

    def test(self, *args, **kargs):
        raise NotImplementedError

'''
Benchmark for stock price prediction: choose current price as predicted price
'''
class SimplePrediction:
    def __init__(self):
        self.loss_function = mean_squared_error

    def train_and_validate(self, train_data = None, train_label = None):
        pass
    
    def test(self, test_data, test_label):
        y_pred = np.zeros([len(test_label),1])
        result_mse = self.loss_function(test_label, y_pred)
        return result_mse