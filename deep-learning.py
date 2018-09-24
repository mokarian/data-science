
# coding: utf-8
rootDir = '/home/maysam/notebooks/challenges/files/images_reza'
batch_size = 32
num_classes =  int(12)
epochs = 25
input_shape = (128,128)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 12, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(rootDir + '/train_data',
                                                 target_size = input_shape,
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(rootDir + '/test_data',
                                            target_size = input_shape,
                                            batch_size = batch_size,
                                            class_mode = 'categorical')

model.fit_generator(training_set,
                         steps_per_epoch = 100,
                         epochs = epochs,
                         validation_data = test_set,
                         validation_steps = 2000)
