import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

NUM_SAMPLES = 10000
LATENT_DIM = 256
BATCH_SIZE = 64
EPOCHS = 20

data = pd.read_csv("dataset.csv")

input_texts = []
target_texts = []

for i in range(min(NUM_SAMPLES, len(data))):
    eng = str(data.iloc[i,0])
    fra = str(data.iloc[i,1])
    
    input_texts.append(eng)

    target_texts.append("\t" + fra + "\n")

print("Total samples:", len(input_texts))

input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts)

target_tokenizer = Tokenizer()
target_tokenizer.fit_on_texts(target_texts)

input_sequences = input_tokenizer.texts_to_sequences(input_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)

max_encoder_len = max(len(seq) for seq in input_sequences)
max_decoder_len = max(len(seq) for seq in target_sequences)

num_encoder_tokens = len(input_tokenizer.word_index) + 1
num_decoder_tokens = len(target_tokenizer.word_index) + 1

encoder_input_data = pad_sequences(
    input_sequences,
    maxlen = max_encoder_len,
    padding = "post"
)

decoder_input_data= pad_sequences(
    target_sequences,
    maxlen = max_decoder_len,
    padding = "post"
)

decoder_target_data = np.zeros(
    (len(input_texts),max_decoder_len, num_decoder_tokens),
    dtype="float32"
)

for i, seq in enumerate(target_sequences):
    for t, word in enumerate(seq[1:]):
        decoder_target_data[i,t,word] = 1.0

encoder_inputs = tf.keras.Input(shape=(None,))

enc_emb = layers.Embedding(
    num_encoder_tokens,
    LATENT_DIM
)(encoder_inputs)

encoder_lstm = layers.LSTM(
    LATENT_DIM,
    return_state = True
)

encoder_output, state_h, state_c = encoder_lstm(enc_emb)

encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.Input(shape=(None,))

dec_emb_layer = layers.Embedding(
    num_decoder_tokens,
    LATENT_DIM
)

dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = layers.LSTM(
    LATENT_DIM,
    return_sequences = True,
    return_state = True
)

decoder_outputs, _, _ = decoder_lstm(
    dec_emb,
    initial_state = encoder_states
)

decoder_dense = layers.Dense(
    num_decoder_tokens,
    activation="softmax"
)

decoder_outputs = decoder_dense(decoder_outputs)

model = tf.keras.Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs
)

model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)
model.summary()

model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=BATCH_SIZE,
    epochs = EPOCHS,
    validation_split = 0.2
)

print("Training completed")