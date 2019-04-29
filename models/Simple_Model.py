import numpy as np
import time, os, gc
from sklearn.metrics import mean_squared_error

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