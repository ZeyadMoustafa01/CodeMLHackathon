import numpy as np
from PIL import Image
import os
import glob
images = []


for image in new_list:
    im = Image.open(image, 'r')
    width, height = im.size
    pix_val = list(im.getdata())
    arr = np.array(pix_val).reshape(width, height, 3)
    images.append(arr)

images = np.array(images)

encoding = {'airplane': 0, 'automobile':1,'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

labels = open('train_label.txt').read().split('\n')
labels.pop()

num_labels = []

for label in labels:
    num_labels.append(encoding[label])

num_labels = np.array(num_labels)

training_images = images[:40000]
testing_images = images[40000:]
training_labels = num_labels[:40000]
testing_labels = num_labels[40000:]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop

model = Sequential()


'''
model.add(Conv2D(6, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(AveragePooling2D())
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
''' 
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=0.001, momentum=0.6), metrics=['accuracy'])
model.summary()

training_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

history = model.fit(train_datagen.flow(training_images, training_labels, batch_size=100), 
          validation_data=validation_datagen.flow(testing_images, testing_labels, batch_size=100), epochs=10, verbose=1)

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

performance_images = []
model_performace_list = os.listdir('test_imagess')
new_performance_list = []
for i in range(1, 10001):
    if 'Image_{}.png'.format(i) in model_performace_list:
        new_performance_list.append('Image_{}.png'.format(i))

for image in new_performance_list:
    im = Image.open(image, 'r')
    width, height = im.size
    pix_val = list(im.getdata())
    arr = np.array(pix_val).reshape(width, height, 3)
    performance_images.append(arr)

performance_images = np.array(performance_images)
from tensorflow.keras.models import load_model

model = load_model('0.6accuracy.h5') 
predictions = model.predict_classes(performance_images)

import pandas as pd

dataframe = pd.DataFrame(predictions)

dataframe = dataframe.reset_index()
dataframe.rename(columns={'0': 'classes'}, inplace=True)

dataframe.columns = ['classes']

dataframe.to_csv('predictions2.csv')
    