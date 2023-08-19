# %% Import Libraries
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from scipy.stats import pearsonr
import datetime

# %% Import data
# stock = yf.Ticker("NFLX")
# stock_history = stock.history(period="max")
stock_history = pd.read_csv("./data/dummy-data.csv", index_col=0)
# stock_history = stock_history.drop(["Dividends", "Stock Splits"], axis=1)

# %% Prepare Training Data
TRAINING_LENGTH = math.ceil(0.7 * len(stock_history))
training_data = stock_history.iloc[:TRAINING_LENGTH, :]

scaler = MinMaxScaler()
training_data_scaled = scaler.fit_transform(training_data)

x_train, y_train = [], []
BATCH_SIZE = 20

for i in range(BATCH_SIZE, len(training_data_scaled) - 10):
    x_train.append(training_data_scaled[i - BATCH_SIZE : i])
    y_train.append(training_data_scaled[i : i + 10, 1:2])

# training_data_scaled.iloc[0:1, 1:2]

x_train = np.array(x_train)
x_train, y_train = np.array(x_train), np.array(y_train)

# %% Build & Train Model
regressor = Sequential()
regressor.add(
    LSTM(
        units=50,
        activation="relu",
        return_sequences=True,
        input_shape=(x_train.shape[1], 2),
    )
)
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=120, activation="relu", return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=120, activation="relu"))
regressor.add(Dropout(0.3))
regressor.add(Dense(units=10))
regressor.compile(optimizer="adam", loss="mean_squared_error")

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

regressor.fit(
    x_train, y_train, epochs=10, batch_size=BATCH_SIZE, callbacks=[tensorboard_callback]
)

# %% Prepare Test Data
test_data = stock_history.iloc[TRAINING_LENGTH:, :]

test_scaler = MinMaxScaler(feature_range=(0, 1))
test_data_scaled = test_scaler.fit_transform(test_data)
test_data = test_data.reset_index()

# %% Prepare Test data & Run Prediction
x_test = []

for i in range(BATCH_SIZE, len(test_data_scaled)):
    x_test.append(test_data_scaled[i - BATCH_SIZE : i])

x_test = np.array(x_test)
x_test.shape
y_predict = regressor.predict(x_test)

# %% Calculate Error (diff with moving average)
from sklearn.metrics import mean_squared_error

y_predict = test_scaler.data_min_[1] + y_predict / test_scaler.scale_[1]

test_data_ma = test_data["Price"].rolling(window=BATCH_SIZE).mean()
mse = mean_squared_error(test_data_ma.tail(10), y_predict[0])

# %% Calculate Correlation between prediction and test_data
correlation_ = pearsonr(test_data_ma.tail(10), y_predict[0])

# %% Plot the Test & Prediction
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=test_data["Date"].tail(10), y=test_data_ma.tail(10), mode="lines", name="Moving Average")
)
fig.add_trace(
    go.Scatter(
        x=test_data["Date"].tail(10), y=test_data["Price"].tail(10), mode="lines", name="Actual Price"
    )
)
fig.add_trace(
    go.Scatter(
        x=test_data["Date"].tail(10),
        y=y_predict[0],
        mode="lines",
        name="Prediction",
    )
)
title = "NFLX Prediction: MSE error: {mse:.2f}".format(mse=mse)
fig.update_layout(title=title)
fig.show()
