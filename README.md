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
- Test above methods
- Ensemble Learning

### TODO:

#### Update 04/07/2019 || Due: 04/10/2019

- Set up the coding architecture

- Write the data loader
- Write the baseline for this task

# Task 2: Feature Generation

This task involves unsupervised learning to generate effective features using algorithms. Task 2 requires you to add generated features to the model of task 1, and test whether the model performance is improved in testing set.
Here are a few typical models and algorithms for reference:

1. Simple methods such as addition, subtraction, multiplication, and polynomial combinations.
  Mathematical tools in signal processing such as WT (wavelet transform).
2. [Deep Feature Synthesis.](https://www.featuretools.com/)
3. [Stacked Auto Encoder.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180944) A deep learning framework for financial time series using stacked auto-encoders
  and long-short term memory 