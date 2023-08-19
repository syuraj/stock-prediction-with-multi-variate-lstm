# %% Import Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from autokeras import StructuredDataRegressor
from ta import add_all_ta_features

# Load your stock price data (assuming it's in a CSV file)
data = pd.read_csv('./data/nflx.csv', index_col=0)

# Calculate technical indicators using 'ta' library
data = add_all_ta_features(data, 'Open', 'High', 'Low', 'Close', 'Volume', fillna=True)

# Select the columns containing calculated indicators
indicator_columns = ['Close', 'volume_adi', 'trend_macd', 'momentum_rsi', 'trend_ema_fast']

# Preprocess the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[indicator_columns])

# Separate scaler for the target variable ("Close" column)
# target_scaler = MinMaxScaler()
# data_scaled[:, 0] = target_scaler.fit_transform(data_scaled[:, 0].reshape(-1, 1)).flatten()


# Define window size for creating sequences
window_size = 10

X, y = [], []
for i in range(len(data_scaled) - window_size):
    X.append(data_scaled[i:i+window_size])
    y.append(data_scaled[i+window_size][0])  # Predicting the 'Close' price

X = np.array(X)
y = np.array(y)

# Flatten the input sequences
X_flattened = X.reshape(X.shape[0], -1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.2, random_state=42)

# Initialize AutoKeras StructuredDataRegressor
reg = StructuredDataRegressor(max_trials=10, objective="val_mean_squared_error")

# Search for the best model architecture and hyperparameters
reg.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)

# Evaluate the model
loss = reg.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# %% Make predictions
predictions = reg.predict(X_test)

# predictions = scaler.data_min_[0] + predictions / scaler.scale_[0]

# Create a new DataFrame with shape (900, 5) and fill with zeros
num_new_columns = 4
empty_data = np.zeros((predictions.shape[0], num_new_columns))
predictions_reshaped_data = np.concatenate((predictions, empty_data), axis=1)


# predictions = scaler.inverse_transform(np.tile(predictions, 5))  # Inverse transform all columns

predictions = scaler.inverse_transform(predictions_reshaped_data)  # Inverse transform all columns


# Plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(predictions[ :, 0], label='Predicted')
plt.plot(data['Close'].tail(len(predictions)), label='Actual')
plt.legend()
plt.show()

# %% plot the original data

plt.figure(figsize=(12, 6))
plt.plot(data['Close'].values, label='Actual')
plt.legend()
plt.show()
data[indicator_columns].iloc[:, 0].values.shape
