import argparse, random, os, sys, time

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)  # append root dir to sys.path
from utils.data_reader import Data
from utils.solver import SimplePrediction

DataLoader = Data()
DataLoader.load_pickle()

predictor = SimplePrediction()
result_mse = predictor.test(DataLoader.test_data, DataLoader.test_label)
print(result_mse)