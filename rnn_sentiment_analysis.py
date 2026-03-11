"""
RNN Sentiment Analysis - Industry Style
Author: Abhishek
Framework: TensorFlow
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

# ==============================
# Configuration
# ==============================

MAX_WORDS = 20000
MAX_LEN = 200
EMBEDDING_DIM = 128
RNN_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 5
MODEL_PATH = "sentiment_rnn.h5"

# ==============================
# Load Dataset
# ==============================

print("Loading dataset...")

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)

print("Training samples:", len(x_train))
print("Test samples:", len(x_test))

# ==============================
# Feature Engineering
# ==============================

print("Padding sequences...")

x_train = pad_sequences(x_train, maxlen=MAX_LEN)
x_test = pad_sequences(x_test, maxlen=MAX_LEN)

# ==============================
# Model Selection
# ==============================

print("Building RNN model...")

model = tf.keras.Sequential([

    layers.Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LEN
    ),

    layers.SimpleRNN(RNN_UNITS),

    layers.Dense(64, activation="relu"),

    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")

])

model.compile(

    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]

)

model.summary()

# ==============================
# Training
# ==============================

print("Training model...")

history = model.fit(

    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2

)

# ==============================
# Save Model
# ==============================

model.save(MODEL_PATH)

print("Model saved!")

# ==============================
# Evaluation
# ==============================

print("Evaluating model...")

loss, acc = model.evaluate(x_test, y_test)

print("Test Accuracy:", acc)

# ==============================
# Plot Training Results
# ==============================

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["train", "validation"])
plt.show()

# ==============================
# Prediction Function
# ==============================

word_index = imdb.get_word_index()

def encode_review(text):

    tokens = text.lower().split()

    encoded = []

    for word in tokens:
        encoded.append(word_index.get(word, 2))

    return pad_sequences([encoded], maxlen=MAX_LEN)


def predict_review(text):

    encoded = encode_review(text)

    prediction = model.predict(encoded)[0][0]

    if prediction > 0.5:
        print("Positive Review 😀")
    else:
        print("Negative Review 😡")


# ==============================
# Test Prediction
# ==============================

predict_review("This movie was absolutely amazing and fantastic")
predict_review("This was the worst movie I have ever seen")