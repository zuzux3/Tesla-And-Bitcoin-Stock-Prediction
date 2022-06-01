import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
##keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

def btcPredict():
    btc = 'bitcoin.csv'
    dataset = pd.read_csv(btc)

    trainingSet = dataset.iloc[:, 1:2].values

    ##to delete before end
    '''head = dataset.head()
    print(head)'''

    scaler = MinMaxScaler(feature_range=(0,1))
    trainingSetGetScaled = scaler.fit_transform(trainingSet)    
    
    xTrain = []
    yTrain = []
    xTest = []
    yTest = []

    for i in range (60, 2991):
        xTrain.append(trainingSetGetScaled[i - 60:i , 0])
        yTrain.append(trainingSetGetScaled[i, 0])

        xTrain, yTrain = np.array(xTrain), np.array(yTrain)
        xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
