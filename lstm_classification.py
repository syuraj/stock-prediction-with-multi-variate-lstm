# %% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import math
import datetime

# %% Import data
stock_history = pd.read_csv("./data/dummy_data_incremental_up_only.csv")

# %% Prepare training data
TRAINING_LENGTH = math.ceil(0.7 * len(stock_history))
training_data = stock_history.iloc[:TRAINING_LENGTH, :]

# scaler = MinMaxScaler()
# training_data_scaled = scaler.fit_transform(training_data.iloc[:,1:3])

x_train, y_train = [], []
BATCH_SIZE = 2

for i in range(BATCH_SIZE, len(training_data) - 2):
    x_train.append(training_data.iloc[i - BATCH_SIZE : i, 1:3].to_numpy().tolist())
    y_train.append(training_data.iloc[i : i + 1, 3:4].iloc[0, :].to_numpy().tolist())

x_train, y_train = np.array(x_train), np.array(y_train)

# %% Build & Train Model
regressor = Sequential()
regressor.add(
    LSTM(
        units=50,
        activation="relu",
        return_sequences=True,
        input_shape=(x_train.shape[1], x_train.shape[2]),
    )
)
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=120, activation="relu", return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=120, activation="relu"))
regressor.add(Dropout(0.3))
regressor.add(Dense(units=1))
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
regressor.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])

# %% Train the model

regressor.fit(x_train, y_train, epochs=10, batch_size=BATCH_SIZE)

# %% Prepare Test Data & Run Prediction
test_data = stock_history.iloc[TRAINING_LENGTH:, :]
x_test = []

for i in range(BATCH_SIZE, len(test_data)):
    x_test.append(test_data.iloc[i - BATCH_SIZE : i, 1:3].to_numpy().tolist())

y_predict = regressor.predict(x_test)

y_predict
