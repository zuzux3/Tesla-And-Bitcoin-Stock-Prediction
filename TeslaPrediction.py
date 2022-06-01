import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def teslaPredict():
    tesla = 'tesla.csv'
    dataset = pd.read_csv(tesla)

    trainingSet = dataset.iloc[:, 1:2].values
    dataset.head()