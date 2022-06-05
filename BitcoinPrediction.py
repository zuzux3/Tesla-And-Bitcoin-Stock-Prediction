import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

def btcPredict():
    btc = 'bitcoin.csv'
    dataset = pd.read_csv(btc, index_col='Date')
    dataset = dataset.loc[:, dataset.columns!='Name']
    dataset = dataset.loc[:, dataset.columns!='Symbol']
    dataset = dataset.loc[:, dataset.columns!='SNo']
    trainingSet = dataset.iloc[:, 1:2].values

    scaler = MinMaxScaler(feature_range=(0,1))
    trainingSetGetScaled = scaler.fit_transform(trainingSet)    
    
    xTrain = []
    yTrain = []
    xTest = []

    for i in range (60, 2991):
        xTrain.append(trainingSetGetScaled[i - 60:i , 0])
        yTrain.append(trainingSetGetScaled[i, 0])

    xTrain, yTrain = np.array(xTrain), np.array(yTrain)
    xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(xTrain.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(xTrain, yTrain, epochs=100, batch_size=32)

    btcTest = 'bitcointest.csv'
    testData = pd.read_csv(btcTest, index_col='Date')
    testData = testData[-20:]
    realStockPrice = testData.iloc[:, 1:2].values

    totalData = pd.concat((dataset['Open'], testData['Open']), axis = 0)
    inputs = totalData[len(totalData) - len(testData) - 60:].values
    inputs = inputs.reshape(-1, 1) 
    inputs = scaler.transform(inputs)

    for i in range (60, 76):
        xTest.append(inputs[i-60:i, 0])
    
    xTest = np.array(xTest)
    xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))
    predictedStockPrice = model.predict(xTest)
    predictedStockPrice = scaler.inverse_transform(predictedStockPrice)

    plt.plot(realStockPrice, color = 'Blue', label = "Bitcoin Stock Price")
    plt.plot(predictedStockPrice, color='Red', label="Predicted Bitcoin Stock Price")
    plt.title("Bitcoin Stock Price Prediction")
    plt.xlabel('Time')
    plt.ylabel('BTC Stock Price')
    plt.legend()
    plt.show()