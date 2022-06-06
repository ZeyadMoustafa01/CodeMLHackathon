import numpy as np

data = open('train.txt').read()
sentences = data.lower().split('\n')

labels = open('train_labels.txt').read().lower().split('\n')
num_labels = []
for reveiw in labels:
    if reveiw == 'positive ':
        num_labels.append(1)
    else:
        num_labels.append(0)

training_sentences = sentences[:12500]
testing_sentences = sentences[12500:]
training_labels = np.array(num_labels[:12500])
testing_labels = np.array(num_labels[12500:])

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
padded_training = pad_sequences(training_sequences, padding='post', maxlen=240, truncating='post')

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
padded_testing = pad_sequences(testing_sequences, padding='post', maxlen=240, truncating='post')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalAveragePooling1D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, LSTM, Bidirectional

model = Sequential()
model.add(Embedding(5000, 100, input_length=240))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(padded_training, training_labels, validation_data=(padded_testing, testing_labels), epochs=15, verbose=1)

validation = open('test.txt').read().lower().split('\n')
validation.pop()
validation_sequences = tokenizer.texts_to_sequences(validation)
validation_padded = pad_sequences(validation_sequences, padding='post', maxlen=240, truncating='post')

predictions = model.predict_classes(validation_padded)

import pandas as pd

dataframe = pd.DataFrame(predictions)
dataframe.columns = ['testLabels']

dataframe.to_csv('predictions.csv')

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

