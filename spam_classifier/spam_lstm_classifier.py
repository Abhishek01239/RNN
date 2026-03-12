import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

VOCAB_SIZE = 10000
MAX_LEN = 120
EMBEDDING_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 5

MODEL_PATH = "spam_lstm_model.h5"

print("Loading dataset...")

data = pd.read_csv("SMSSpamCollection", sep="\t", header=None)

data.columns = ["label", "message"]

print(data.head())

print("Cleaning labels...")

data["label"] = data["label"].map({
    "ham":0,
    "spam":1
})

texts = data["message"].values
labels = data["label"].values

print("Tokenizing text...")

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

padded_sequences = pad_sequences(
    sequences,
    maxlen=MAX_LEN,
    padding="post"
)

x_train, x_test, y_train, y_test = train_test_split(

    padded_sequences,
    labels,
    test_size=0.2,
    random_state=42

)

print("Building model...")

model = tf.keras.Sequential([

    layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),

    layers.Bidirectional(layers.LSTM(LSTM_UNITS)),

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

print("Training model...")

history = model.fit(

    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2

)

print("Evaluating model...")

loss, acc = model.evaluate(x_test, y_test)

print("Test Accuracy:", acc)

pred = model.predict(x_test)

pred = (pred > 0.5)

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))

model.save(MODEL_PATH)

print("Model saved!")

def predict_message(text):

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    prediction = model.predict(padded)[0][0]

    if prediction > 0.5:
        print("SPAM MESSAGE 🚨")
    else:
        print("NOT SPAM ✅")

predict_message("Congratulations you won a free lottery ticket")
predict_message("Hey are we still meeting tomorrow?")