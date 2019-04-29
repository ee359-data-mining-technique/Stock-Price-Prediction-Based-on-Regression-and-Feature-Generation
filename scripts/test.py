import argparse, random, os, sys, time

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)  # append root dir to sys.path
from utils.data_reader import Data
import numpy as np

from models.LSTM_Model_v2 import Model
from models.Simple_Model import SimplePrediction
from sklearn.metrics import mean_squared_error

DataLoader = Data()
DataLoader.load_pickle_dataset()

Model = SimplePrediction()

print(Model.test(DataLoader.test_data, DataLoader.test_label))
