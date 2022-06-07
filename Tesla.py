import warnings
import math
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def predict():
    dataFrame = pd.read_csv('teslatest.csv')
    
    stockCloseData = dataFrame.filter(['Close'])
    stockCloseDataSet = stockCloseData.values
    trainingDataLength = math.ceil(len(stockCloseDataSet) * 0.8)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaledData = scaler.fit_transform(stockCloseDataSet)

    stockTrainData = scaledData[0:trainingDataLength, :]
    xTrain = []
    yTrain = []
    xTest = []
    yTest = []

    for i in range (60, len(stockTrainData)):
        xTrain.append(stockTrainData[i-60:i, 0])
        yTrain.append(stockTrainData[i, 0])

    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)

    xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

    model = Sequential()
    neurons = 60
    model.add(LSTM(neurons, return_sequences=True, input_shape = (xTrain.shape[1], 1)))
    model.add(LSTM(neurons, return_sequences=False))
    model.add(Dense(30))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    historyData = model.fit(xTrain, yTrain, batch_size=50, epochs=200, verbose=2, validation_split=0.2)

    plt.figure(figsize=(20,10))
    plt.title('Training Validation Loss')
    plt.plot(historyData.history['loss'])
    plt.plot(historyData.history['val_loss'])
    plt.xlabel('epochs')
    plt.ylabel('Training loss')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    testingData = scaledData[trainingDataLength - 60:, :]
    YTest = stockCloseDataSet[trainingDataLength:, :]

    for i in range(60, len(testingData)):
        xTest.append(testingData[i-60:i, 0])
    xTest = np.array(xTest)

    xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))
    predictions = model.predict(xTest)
    predictions = scaler.inverse_transform(predictions)

    training = stockCloseData[:trainingDataLength]
    validation = stockCloseData[trainingDataLength:]
    validation["Predictions"] = predictions

    plt.figure(figsize=(20,10))
    plt.title("TESLA - Trained Model Accuracy")
    plt.xticks(range(0, dataFrame.shape[0], 500), dataFrame['Date'].loc[::500], rotation=45)
    plt.xlabel('Date', fontsize=21)
    plt.ylabel('Close Stock Price $ (USD)', fontsize=21)
    plt.plot(training['Close'])
    plt.plot(validation[['Close', 'Predictions']])
    plt.legend(['Training', 'Validation', 'Predictions'], loc='lower right')
    plt.show()