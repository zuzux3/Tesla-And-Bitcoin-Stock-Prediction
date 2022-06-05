import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

def teslaPredict():
    tesla = 'tesla.csv'
    dataset = pd.read_csv(tesla, index_col='Date')
    trainingSet = dataset.iloc[:, 1:2].values

    ##print(dataset.head())

    scaler = MinMaxScaler()
    trainingSetGetScaled = scaler.fit_transform(trainingSet)
    
    xTrain = []
    yTrain = []
    xTest = []

    for i in range (60, 2416):
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

    teslaTest = 'teslatest.csv'
    testData = pd.read_csv(teslaTest, index_col='Date')
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
    
    plt.plot(realStockPrice, color = 'Blue', label = "Tesla Stock Price")
    plt.plot(predictedStockPrice, color='Red', label="Predicted Tesla Stock Price")
    plt.title("Tesla Stock Price Prediction")
    plt.xlabel('Time')
    plt.ylabel('Tesla Stock Price')
    plt.legend()
    plt.show()        