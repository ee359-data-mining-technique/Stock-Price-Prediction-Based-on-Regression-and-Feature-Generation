from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import time, os, gc
from sklearn.metrics import mean_squared_error

class GradientBoostingModel:
    def __init__(self):
        self.loss_function = mean_squared_error
        self.regr = GradientBoostingRegressor()
        
    def train_and_validate(self, train_data = None, train_label = None):
        if len(train_data.shape) > 2:
            train_data = train_data.reshape(train_data.shape[0], -1)
        if len(train_label.shape) > 1:
            train_label = np.ndarray.flatten(train_label) 
        self.regr.fit(train_data, train_label)
    
    def test(self, test_data, test_label):
        if len(test_data.shape) > 2:
            test_data = test_data.reshape(test_data.shape[0], -1)
        if len(test_label.shape) > 1:
            test_label = np.ndarray.flatten(test_label) 
        y_pred = self.regr.predict(test_data)
        result_mse = self.loss_function(test_label, y_pred)
        return result_mse