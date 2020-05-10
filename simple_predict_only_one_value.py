# %% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import math
import datetime

# %%
stock_history = pd.read_csv("./data/dummy_data_incremental_up_only.csv")

# %% prepare training data
TRAINING_LENGTH = math.ceil(0.7 * len(stock_history))
training_data = stock_history.iloc[:TRAINING_LENGTH, :]

# %%
x_train, y_train = [], []
BATCH_SIZE = 3

for i in range(BATCH_SIZE, len(training_data) - 2):
    x_train.append(training_data.iloc[i - BATCH_SIZE : i, 1:3].to_numpy().tolist())
    y_train.append(training_data.iloc[i, 3:4].to_numpy().tolist())

x_train, y_train = np.array(x_train), np.array(y_train)
x_train.shape, y_train.shape

# %% Build model
regressor = Sequential()
regressor.add(
    LSTM(
        units=50,
        activation="relu",
        return_sequences=True,
        input_shape=(x_train.shape[1], x_train.shape[2]),
    )
)
regressor.add(Dropout(0.3))
regressor.add(Dense(units=1))

regressor.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

# %% Train model
regressor.fit(x_train, y_train, epochs=10, batch_size=BATCH_SIZE)

# %% Prepare Test data and Run Prediction
test_data = stock_history.iloc[TRAINING_LENGTH:, :]
x_test, y_test = [], []

for i in range(BATCH_SIZE, len(test_data)):
    x_test.append(training_data.iloc[i - BATCH_SIZE : i, 1:3].to_numpy().tolist())
    y_test.append(training_data.iloc[i, 3:4].to_numpy().tolist())

_, accuracy = regressor.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
y_predict = regressor.predict(x_test)

print(accuracy)
print(y_predict)
