import argparse, random, os, sys, time
install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)  # append root dir to sys.path

# Import necessary packages
from utils.data_reader import Data
import numpy as np
from models.LSTM_Model_v2 import Model
from sklearn.metrics import mean_squared_error

DataLoader = Data()
DataLoader.load_pickle_dataset()

lstm_model = Model()
lstm_model.build_model()

for i in range(1000):
    train_x, train_y = DataLoader.get_next_train_batch(500)
    mse_loss = lstm_model.train_model(train_x, train_y)

    if i%50 == 0:
        print("Training MSE Loss: %f" %mse_loss)

result_mse = 0
for i in range(100):
    test_x, test_y = DataLoader.get_next_test_batch(500)

    result_mse = result_mse + lstm_model.evaluate_model(test_x, test_y)
    print(result_mse/(i+1))
print(result_mse/100)