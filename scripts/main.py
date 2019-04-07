#-*- coding:utf-8 -*-
import argparse, random, os, sys, time


install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)  # append root dir to sys.path

from utils.data_reader import Data
from utils.solver import KNNSolver
from utils.util import MLearner



#################################################################################################
############################### Arguments parsing and Preparations ##############################
#################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--mltype', type=str, default='MMC', help='metric learn type')
parser.add_argument('--distance', type=str, default='euclidean', help='distance type for KNN')
parser.add_argument('--neighbors', type=int, default=50, help='number of neighbors for KNN')

# model paras
parser.add_argument('--dims', type=int, default=2, help='dims for NCA, LFDA, MLKR, RCA')
parser.add_argument('--k', type=int, default=50, help='k for LMNN, LFDA')
parser.add_argument('--gamma', type=float, default=0.5, help='gamma for ITML')
parser.add_argument('--bal', type = float, default=0.5, help='bal for SDML')
parser.add_argument('--spa', type=float, default=0.5, help='spa for SDML')

opt = parser.parse_args()
random.seed(999)


####################################################################################################
####################################### Data Reader ################################################
####################################################################################################

data_set = Data()
train_X, test_X, train_y, test_y = data_set.Get_Train_Test()

####################################################################################################
##################################### Model Construction ###########################################
####################################################################################################

metric_learner = MLearner(opt.mltype)
solver = KNNSolver(opt.distance, opt.neighbors)

####################################################################################################
###################################### Training and Decoding #######################################
####################################################################################################

train_X, test_X = metric_learner.learn(train_X, train_y, test_X, opt)
acc = solver.train_and_decode(train_X, test_X, train_y, test_y)
print('metric learn type:\t{}\nacc:\t{}'.format(opt.mltype, acc))