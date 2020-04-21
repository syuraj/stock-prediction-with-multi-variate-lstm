# %% Import Libraries
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# %% Import data
# stock = yf.Ticker("NFLX")
# stock_history = stock.history(period="max")
stock_history = pd.read_csv("./data/nflx.csv", index_col=0)
stock_history = stock_history.drop(["Dividends", "Stock Splits"], axis=1)

# %% Prepare Training Data
TRAINING_LENGTH = math.ceil(0.7 * len(stock_history))
training_data = stock_history.iloc[:TRAINING_LENGTH, :]

scaler = MinMaxScaler()
training_data_scaled = scaler.fit_transform(training_data)

x_train, y_train = [], []

for i in range(30, len(training_data_scaled)):
    x_train.append(training_data_scaled[i - 30 : i])
    y_train.append(training_data_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# %% Build & Train Model
regressor = Sequential()
regressor.add(
    LSTM(
        units=50,
        activation="relu",
        return_sequences=True,
        input_shape=(x_train.shape[1], 5),
    )
)
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=120, activation="relu"))
regressor.add(Dropout(0.3))
regressor.add(Dense(units=1))
regressor.compile(optimizer="adam", loss="mean_squared_error")
regressor.fit(x_train, y_train, epochs=2, batch_size=10)

# %% Prepare Test Data
test_data = stock_history.iloc[TRAINING_LENGTH:, :]

test_scaler = MinMaxScaler(feature_range=(0, 1))
test_data_scaled = test_scaler.fit_transform(test_data)
test_data = test_data.reset_index()

# %% Run Prediction
x_test, y_test = [], []

for i in range(30, len(test_data_scaled)):
    x_test.append(test_data_scaled[i - 30 : i])
    y_test.append(test_data_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predict = regressor.predict(x_test)

# %% Re-Scale the Prediction
y_predict = test_scaler.data_min_[0] + y_predict / test_scaler.scale_[0]


# %% Calculate Error (diff with moving average)
from sklearn.metrics import mean_squared_error

test_data_ma = test_data["Open"].rolling(window=20).mean()
mse = mean_squared_error(test_data_ma.tail(1000), y_predict.flatten()[-1000:])

# %% Plot the Test & Prediction
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=test_data["Date"], y=test_data["Open"], mode="lines", name="Actual Price"
    )
)
fig.add_trace(
    go.Scatter(
        x=test_data["Date"].tail(1322),
        y=y_predict.flatten(),
        mode="lines",
        name="Prediction",
    )
)
title = "NFLX Prediction: MSE error: {mse:.2f}".format(mse = mse)
fig.update_layout(title=title)
fig.show(title="sss")
