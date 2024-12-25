import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical

# Sample text data
text = "hello world"
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Hyperparameters
seq_length = 4
vocab_size = len(chars)

# Prepare the dataset
X = []
y = []

for i in range(len(text) - seq_length):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    X.append([char_to_idx[char] for char in seq_in])
    y.append(char_to_idx[seq_out])

X = np.array(X)
y = np.array(y)

# One-hot encode the labels
y = to_categorical(y, num_classes=vocab_size)

# Build the LSTM model
model = Sequential([
    Embedding(vocab_size, 8, input_length=seq_length),
    LSTM(50, return_sequences=False),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, batch_size=1, verbose=1)

# Generate text
seed = "hell"
seed_idx = [char_to_idx[char] for char in seed]

for _ in range(10):
    input_seq = np.array(seed_idx[-seq_length:]).reshape(1, seq_length)
    predicted_idx = np.argmax(model.predict(input_seq, verbose=0))
    seed += idx_to_char[predicted_idx]
    seed_idx.append(predicted_idx)

print("Generated Text:", seed)
