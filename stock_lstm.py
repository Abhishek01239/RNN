import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers

LOOKBACK = 60
EPOCHS = 20
BATCH_SIZE = 32
LSTM_UNITS = 50

print("Loading dataset...")

df = pd.read_csv("APPL.CSV")

data = df["Close"].values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(data)

X = []
y = []

for i in range(LOOKBACK, len(scaled_data)):
    X.append(scaled_data[i-LOOKBACK:i,0])
    y.append(scaled_data[i,0])

X = np.array(X)
y = np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1],1))

print("Training samples:", X.shape)

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

model = tf.keras.Sequential([
    layers.LSTM(LSTM_UNITS, return_sequences =True, input_shape=(X.shape[1],1)),

    layers.LSTM(LSTM_UNITS),

    layers.Dense(25, activation = "relu"),

    layers.Dense(1)
])

model.compile(
    optimizer = "adam",
    loss = "mean_squared_error"
)

model.summary()

history = model.fit(
    X_train, y_train, epochs = EPOCHS,
    batch_size = BATCH_SIZE
)

prediction = model.predict(X_test)

prediction  = scaler.inverse_transform(prediction)

y_test_real = scaler.inverse_transform(y_test.reshape(-1,1))

plt.figure(figsize = (10,6))
plt.plot(y_test_real, label = "Actual Price")
plt.plot(prediction, label = "Predicted Price")
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

last_60_days = scaled_data[-LOOKBACK:]
last_60_days = np.reshape(last_60_days, (1, LOOKBACK,1))
next_price = model.predict(last_60_days)
print("Predicted next day price:", next_price[0][0])
