import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")


# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
# For time stamps
from datetime import datetime

from Models import LinearModel, LSTMModel, TwoLSTMsModel, ResLSTMsModel, LinLSTMModel
from dataset import create_dataset
from Train_evaluate import train, evaluate, visualize_result
    
#Data downloading    
df = yf.download('AAPL', start='2012-01-01', end=datetime.now())

#Vizualization
def vizualize(df):
    plt.figure(figsize=(16,6))
    plt.title('Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.show() 
    
    
    
#Data preporation
# Create a new dataframe with only the 'Close column 
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
data_len = int(len(dataset))


# train-test split for time series
train_size = int(data_len * 0.80)
test_size = data_len - train_size
train, test = dataset[:train_size], dataset[train_size:]

lookback = 4
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#Model

model = LinLSTMModel

#Training

model = train(model, X_train, X_test, y_train, y_test)

#Evaluation
evaluate(model, X_train, X_test, y_train, y_test)

#Visialisation
visualize_result(dataset, X_train, X_test, model)