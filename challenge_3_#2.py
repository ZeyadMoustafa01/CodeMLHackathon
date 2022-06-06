import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

sentences = list(df['Body'])
labels = list(df['Subreddit'])

training_sentences = sentences[:150000]
testing_sentences = sentences[150000:]
final_training_sentences = []
final_testing_sentences = []


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

for entry in training_sentences:
    final_training_sentences.append(str(entry))
    
for entry in testing_sentences:
    final_testing_sentences.append(str(entry))

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(final_training_sentences)
word_index = tokenizer.word_index


sequences = tokenizer.texts_to_sequences(final_training_sentences)
padded_training = pad_sequences(sequences, padding='post', maxlen=100, truncating='post')

testing_sequences = tokenizer.texts_to_sequences(final_testing_sentences)
padded_testing = pad_sequences(testing_sequences, padding='post', maxlen=100, truncating='post')

num_labels = []
for entry in labels:
    if entry == 'Republican':
        num_labels.append(0)
    else:
        num_labels.append(1)

training_labels = np.array(num_labels[:150000])
testing_labels = np.array(num_labels[150000:])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Embedding, Dense, GlobalAveragePooling1D

model = Sequential()
model.add(Embedding(10000, 32, input_length=100))
model.add(Conv1D(64, 5, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(padded_training, training_labels, validation_data=(padded_testing, testing_labels), epochs=5, verbose=1)

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')

plt.figure()
plt.show()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.figure()
plt.show()

model.save('challenge_3_model.h5')

from tensorflow.keras.models import load_model

new_model = load_model('challenge_3_model.h5')

dataframe = pd.read_csv('submission.csv')

performance = list(dataframe['Body'])
final_performance = []
for entry in performance:
    final_performance.append(str(entry))

validation_sequence = tokenizer.texts_to_sequences(final_performance)
validation_padded = pad_sequences(validation_sequence, padding='post', maxlen=100, truncating='post')

predictions = model.predict_classes(validation_padded)

predictions = pd.DataFrame(predictions)
predictions.columns = ['Subreddit']

predictions.to_csv('predictions7.csv')