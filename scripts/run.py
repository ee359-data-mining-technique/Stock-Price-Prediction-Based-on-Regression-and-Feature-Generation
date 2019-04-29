import argparse, random, os, sys, time
install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)  # append root dir to sys.path

# Import necessary packages
from utils.data_reader import Data
import numpy as np
from models.LSTM_Model_v2 import Model
from models.RandomForestModel import RandomForestModel
from models.AdaBoostModel import AdaBoostModel
from models.GradientBoostingModel import GradientBoostingModel
from models.NeuralNetworkModel import VanillaNetworkModel
from sklearn.metrics import mean_squared_error

Learning_Model = "Vanilla Neural Network"

DataLoader = Data()
DataLoader.load_pickle_dataset()
test_x, test_y = DataLoader.test_data, DataLoader.test_label

if Learning_Model == "AdaBoost":
    model = AdaBoostModel()
    model.train_and_validate(DataLoader.train_data, DataLoader.train_label)
    mse_loss = model.test(test_x, test_y)
    print("Training MSE Loss: %f" %mse_loss)
if Learning_Model == "Gradient Boosting Decision Tree":
    model = GradientBoostingModel()
    model.train_and_validate(DataLoader.train_data, DataLoader.train_label)
    mse_loss = model.test(test_x, test_y)
    print("Training MSE Loss: %f" %mse_loss)
if Learning_Model == "Random Forest":
    random_forest_model = RandomForestModel()
    
    random_forest_model.train_and_validate(DataLoader.train_data, DataLoader.train_label)
    
    mse_loss = random_forest_model.test(test_x, test_y)
    print("Training MSE Loss: %f" %mse_loss)
if Learning_Model == "Vanilla Neural Network":
    model = VanillaNetworkModel()
    model.build_model()
    for i in range(1000):
        train_x, train_y = DataLoader.get_next_train_batch(500)
        model.train_model(train_x, train_y)
        
        if i%100 == 0:
            mse_loss = model.evaluate_model(test_x, test_y)
            print("Training MSE Loss: %f" %mse_loss)
            print(train_y[:10])
if Learning_Model == "LSTM":
    lstm_model = Model()
    lstm_model.build_model()

    for i in range(100):
        train_x, train_y = DataLoader.get_next_train_batch(500)
        lstm_model.train_model(train_x, train_y)

        if i%10 == 0:
            mse_loss = lstm_model.evaluate_model(test_x, test_y)
            print("Training MSE Loss: %f" %mse_loss)

# print("Testing MSE Loss: %f" %lstm_model.evaluate_model(DataLoader.test_data, DataLoader.test_label))
