# Task-1-Regression-Based-Stock-Price-Prediction
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
|  Simple LSTM  |      0.8109       |
| Random Forest |      0.8063       |
|     GBDT      |      0.8222       |
|   AdaBoost    |      0.9029       |
|      CNN      |                   |
|      NN       |      0.8274       |
|               |                   |
|               |                   |

