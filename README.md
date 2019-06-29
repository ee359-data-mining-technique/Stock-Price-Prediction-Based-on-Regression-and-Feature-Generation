# Task-1-Regression-Based-Stock-Price-Prediction

### Necessary Dependencies

We implement this task in python, you should install following packages:

- pandas
- numpy
- sklearn
- tensorflow
- pickle

### Data Preparation

To run this code, you should download corresponding dataset: **data.csv** and put it in the directory *./data/data.csv*. The directory of this whole project should be like:

```
-- data
---- data.csv
-- models
-- scripts
-- useless
-- utils
```

### How to Run

To see how to run, you should refer to **run.py** in *./scripts/run.py*:

```python
import argparse, random, os, sys, time
install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)  # append root dir to sys.path

# Import necessary packages
from utils.data_reader import Data
import numpy as np
from models.LSTM_Model_v2 import LSTM_Model
from models.RandomForestModel import RandomForestModel
from models.CNN_Model import CNN_Model
from models.AdaBoostModel import AdaBoostModel
from models.GradientBoostingModel import GradientBoostingModel
from models.NeuralNetworkModel import VanillaNetworkModel
from models.Simple_Model import SimplePrediction
from sklearn.metrics import mean_squared_error

LearningModel =  "Random Forest"

DataLoader = Data()
DataLoader.load_pickle_dataset()
test_x, test_y = DataLoader.test_data, DataLoader.test_label

if LearningModel == "Simple":
    Model = SimplePrediction()
    print(Model.test(DataLoader.test_data, DataLoader.test_label))  
if LearningModel == "AdaBoost":
    model = AdaBoostModel()
    model.train_and_validate(DataLoader.train_data, DataLoader.train_label)
    mse_loss = model.test(test_x, test_y)
    print("Training MSE Loss: %f" %mse_loss)
if LearningModel == "Gradient Boosting Decision Tree":
    model = GradientBoostingModel()
    model.train_and_validate(DataLoader.train_data, DataLoader.train_label)
    mse_loss = model.test(test_x, test_y)
    print("Training MSE Loss: %f" %mse_loss)
if LearningModel == "Random Forest":
    random_forest_model = RandomForestModel()
    random_forest_model.train_and_validate(DataLoader.train_data, DataLoader.train_label)    
    mse_loss = random_forest_model.test(test_x, test_y)
    print("Training MSE Loss: %f" %mse_loss)
if LearningModel == "Vanilla Neural Network":
    model = VanillaNetworkModel()
    model.build_model()
    for i in range(1000):
        train_x, train_y = DataLoader.get_next_train_batch(500)
        model.train_model(train_x, train_y)
        
        if i%100 == 0:
            mse_loss = model.evaluate_model(test_x, test_y)
            print("Training MSE Loss: %f" %mse_loss)
if LearningModel == "Convolutional Neural Network":
    model = CNN_Model()
    model.build_model()
    for i in range(1000):
        train_x, train_y = DataLoader.get_next_train_batch(500)
        model.train_model(train_x, train_y)
        
        if i%100 == 0:
            mse_loss = model.evaluate_model(test_x, test_y)
            print("Training MSE Loss: %f" %mse_loss)
if LearningModel == "LSTM":
    lstm_model = LSTM_Model()
    lstm_model.build_model()

    for i in range(100):
        train_x, train_y = DataLoader.get_next_train_batch(500)
        lstm_model.train_model(train_x, train_y)

        if i%10 == 0:
            mse_loss = lstm_model.evaluate_model(test_x, test_y)
            print("Training MSE Loss: %f" %mse_loss)

# print("Testing MSE Loss: %f" %lstm_model.evaluate_model(DataLoader.test_data, DataLoader.test_label))

```

### Task Details

This task involves supervise learning. The basic requirement is to learn a certain model and do regression
according to the features and labels in the dataset. 

**The label** equals (the future n-th tick's AskPrice1 + the future n-th tick's BidPrice1 - the current tick's AskPrice1 - the current tick's BidPrice1) / 2, the value of n can be customized. 

**Regression method** is not limited to least square method, other methods (such as ensemble learning) and other evaluation criteria can be used. In the end, you should **report your methods**' final results, with the predicted values on the testing set.

This project will be evaluated according to the difference between your predicted results and true values. Tail value will be assigned with higher weight (you should make accurate prediction in the long run). complex models not necessarily guarantee better results, because financial transaction problems depend on the problem itself and the size of the dataset.

Here are a few typical models and algorithms for reference:

1. Random Forest
2. Gradient Boosting Decision Tree
3. Adaboost
4. Deep Neural Network
5. Convolutional Neural Network
6. Long and Short-Term Memory 

### Sub-Tasks

- Process dataset
- Feature Engineering
  - PCA/ Sparse PCA
  - 特征的redundancy
  - 做差
- Test above methods
- Ensemble Learning

### Dateset

indicators 1~108	midPrice	**UpdateTime	UpdateMillisec**	LastPrice	Volume	LastVolume	Turnover	LastTurnover	AskPrice5	AskPrice4	AskPrice3	AskPrice2	AskPrice1	BidPrice1	BidPrice2	BidPrice3	BidPrice4	BidPrice5	AskVolume5	AskVolume4	AskVolume3	AskVolume2	AskVolume1	BidVolume1	BidVolume2	BidVolume3	BidVolume4	BidVolume5	OpenInterest	UpperLimitPrice	LowerLimitPrice

### TODO:

#### Update 04/07/2019 || Due: 04/10/2019

- Set up the coding architecture
- Write the data loader
- Write the baseline for this task

#### Update 04/11/2019

- Write LSTM with tensorflow
- Think about how to measure feature importance

# Task 2: Feature Generation

This task involves unsupervised learning to generate effective features using algorithms. Task 2 requires you to add generated features to the model of task 1, and test whether the model performance is improved in testing set.
Here are a few typical models and algorithms for reference:

1. Simple methods such as addition, subtraction, multiplication, and polynomial combinations.
  Mathematical tools in signal processing such as WT (wavelet transform).
2. [Deep Feature Synthesis.](https://www.featuretools.com/)
3. [Stacked Auto Encoder.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180944) A deep learning framework for financial time series using stacked auto-encoders
  and long-short term memory 

# Process

1. 数据处理过程
   - Normalize 整个数据的每一维
   - 把上午数据和下午数据分开
   - 数据为每10个一个框 + 往后第10个时间的  (the future 10-th tick's AskPrice1 + the future 10-th tick's BidPrice1 - the current tick's AskPrice1 - the current tick's BidPrice1) / 2
   - 分别存入上午/下午数据

# Results

- Benchmark: it just take the current value for prediction value. Thus, the predicted $y$ will be $0$.
- Simple LSTM: three layer LSTM with one dense layer (implemented by Keras)

### Measure

- MSE

|    Methods    | Mean Square Error |
| :-----------: | :---------------: |
|   Benchmark   |      0.8274       |
|     LSTM      |      0.8274       |
| Random Forest |      0.8220       |
|     GBDT      |      0.8222       |
|   Ada Boost   |      0.9029       |
|      CNN      |      0.8278       |
|      NN       |      0.8274       |
|               |                   |
|               |                   |

# 第一次实验总结

### 实验概要

本次实验旨在用一支股票的交易数据，进行预测股票将来的走向。主要问题为运用一些监督学习的方法，对股票短时间走势进行判断。

### 数据处理

#### 数据集描述

本次给出的数据集：

- 每一秒产生三个数据。股市数据分为上下午，中间在11:30到13:30停盘，停盘时间不产生数据。
- 数据包含股票的原始交易数据：交易价格，交易量，交易时间等等（详见数据集）
- 数据还包括给出的108维特征（feature）数据，此数据已经做出了标准化（Normalization），分别命名为indicator1到indicator108。

#### 数据集处理

本次争对此数据集做出如下处理：

- 将特征数据降维：我们运用PCA对特征数据进行了降维

- 训练数据：我们取降维的特征数据作为训练数据，每次取一定时间间隔的数据（即代码中的seq_len），作为一个训练数据。这里可以把它理解为数据窗，数据窗框入一段连续时间的数据作为一个训练数据。

- 训练标签：我们取给出的一段时间后的数据（即代码中的predict_len）到数据框现在时间对应数据的AskPrice1和BidPrice1的差值作为数据label：
  $$
  label = (AskPrice_{n+t} + BidPrice_{n+t} - AskPrice_t - BidPrice_t)/2
  $$

- 当遇到上午与下午间隔，隔天间隔的数据点，我们不取对应的训练数据。此外我们将上午和下午数据视为一致，放入同一个训练集（如果视为不同分布，其实可以放入不同训练集）。

- 以以上方法取出数据集后，我们对数据集进行洗牌（shuffle），然后分割60%为训练集，40%为测试集。

### 尝试方法

我们以Mean Square Error作为衡量预测准确率的标准，代码上传在[Github仓库](https://github.com/ee359-data-mining-technique/Stock-Price-Prediction-Based-on-Regression-and-Feature-Generation)中。

- Simple Prediction

  我们给出一个简单的预测方法，以作为此任务的benchmark：该方法用时间框的最后一个数据直接当作预测数据，这样预测的值（即Price的插值），由于相同，全都为0。这样算出来的MSE，我们作为Benchmark研究其他算法的表现。

其余我们还实现了：

- Random Forest (by Sci-Learn)
- Gradient Boosting Decision Tree (by Sci-Learn)
- Ada Boost (by Sci-Learn)
- Neural Network (by Tensorflow)
- Convolutional Neural Network (by Tensorflow)
- LSTM Neural Network (by Tensorflow)

### 实验结果

实验结果如下

|    Methods    | Mean Square Error |
| :-----------: | :---------------: |
|   Benchmark   |      0.8274       |
|     LSTM      |      0.8274       |
| Random Forest |      0.8220       |
|     GBDT      |      0.8222       |
|   Ada Boost   |      0.9029       |
|      CNN      |      0.8278       |
|      NN       |      0.8274       |

由此实验结果，我们有如下观察：

- 相对来说，Random Forest方法效果最好，其次为GBDT方法。
- 神经网络的方法在此数据上表现得不好，效果并没有超过Benchmark，是一个相当差的结果

### 待解决的困难

- ISSUE 1: 

  在训练过程中我们发现，LSTM模型和NN模型最后收敛到对于所有输入的输出都是0，也就是和Benchmark没有区别。为什么是0呢，我们认为和数据集的制作有关。在数据集的制作中，label的选取为Price的差值，而此差值在大部分情况下都是0（因为时间差很小，导致价格变动不大）。

  对于此情况，我们取了predict length更大的数据，使差值变得不那么多0，但LSTM和NN的表现仍旧。所以此问题需要解决：

  - 大概率是数据集制作的原因，因为label中很多0，导致最后学到的知识就是产生0，所以导致算法无效
  - 也有可能是问题的原因，我们当作一个回归问题来做，而label实际上多是整数值，而且大多在[-5, 5]的区间内，也就是此问题还可以当作分类问题做

  下一步我们将解决这个问题。

- ISSUE 2：

  尝试不同地seq_len, predict_len, jump等参数对数据集的影响

  尝试不同地衡量效果的方法

  尝试Random Forest和Boost模型的参数，求取最优值