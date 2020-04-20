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

# %% Run Prediction
x_test, y_test = [], []

for i in range(30, len(test_data_scaled)):
    x_test.append(test_data_scaled[i - 30 : i])
    y_test.append(test_data_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predict = regressor.predict(x_test)

# %% Re-Scale the Prediction
y_predict = test_scaler.data_min_[0] + y_predict / test_scaler.scale_[0]

# %% Plot the Test & Prediction

plt.figure(figsize=(14, 5))
plt.plot(test_data.iloc[-50:, 0], color="red")
plt.plot(y_predict[-50:, 0], color="orange")
plt.show()
