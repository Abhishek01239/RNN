import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
LSTM_UNITS = 150
EPOCHS = 20
BATCH_SIZE = 64

print("Loading dataset...")

with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = text.lower()

print("Dataset length:", len(text))

print("Tokenizing dataset...")

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

print("Generating training sequences...")

input_sequences = []

for line in text.split("\n"):

    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):

        n_gram_sequence = token_list[:i+1]

        input_sequences.append(n_gram_sequence)

max_seq_len = max(len(seq) for seq in input_sequences)

input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_seq_len, padding="pre")
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = tf.keras.utils.to_categorical(y, num_classes=total_words)

print("Training samples:", X.shape)

print("Building model...")

model = tf.keras.Sequential([

    layers.Embedding(
        input_dim=total_words,
        output_dim=EMBEDDING_DIM,
        input_length=max_seq_len-1
    ),

    layers.LSTM(LSTM_UNITS),

    layers.Dense(150, activation="relu"),

    layers.Dense(total_words, activation="softmax")

])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("Training model...")

history = model.fit(
    X,
    y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

def generate_text(seed_text, next_words=10):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_seq_len-1,
            padding="pre"
        )

        predicted = model.predict(token_list, verbose=0)

        predicted_word_index = np.argmax(predicted)

        output_word = ""

        for word, index in tokenizer.word_index.items():

            if index == predicted_word_index:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text

print("\nGenerated Text:\n")

print(generate_text("alice was beginning to", 10))