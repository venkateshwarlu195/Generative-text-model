import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense

# -----------------------------
# Load Dataset
# -----------------------------

with open("dataset.txt", "r", encoding="utf-8") as file:
    data = file.read().lower()

lines = data.split("\n")

# -----------------------------
# Tokenization
# -----------------------------

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)

total_words = len(tokenizer.word_index) + 1

# -----------------------------
# Create Sequences
# -----------------------------

input_sequences = []

for line in lines:

    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# -----------------------------
# Padding
# -----------------------------

max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
)

# -----------------------------
# Split X and y
# -----------------------------

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# -----------------------------
# Build Model
# -----------------------------

model = Sequential()

model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# -----------------------------
# Train or Load Model
# -----------------------------

model_path = "trained_model.h5"

if os.path.exists(model_path):

    print("Loading trained model...")
    model = load_model(model_path)

else:

    print("Training model...")
    model.fit(X, y, epochs=100, verbose=1)

    model.save(model_path)
    print("Model saved as trained_model.h5")

# -----------------------------
# Text Generation Function
# -----------------------------

def generate_text(seed_text, next_words=10):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len-1,
            padding='pre'
        )

        predicted = np.argmax(model.predict(token_list), axis=-1)

        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text


# -----------------------------
# Run Text Generation
# -----------------------------

prompt = input("Enter prompt: ")

generated_text = generate_text(prompt, 10)

print("\nGenerated Text:")
print(generated_text)