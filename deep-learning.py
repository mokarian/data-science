
# coding: utf-8

# In[1]:


import keras
from keras.models import Sequential

model = Sequential()

# In[16]:

import os
from os.path import basename
from sklearn.preprocessing import StandardScaler
from PIL import Image
from sklearn.model_selection import train_test_split

import numpy as np
import os
from os.path import basename

rootDir = '/home/maysam.mokarian/notebooks/files/converted_images_normalized'
imageArray = []
labelArray = []
counter = 0
npImageArray = np.array({})
npLabelArray = np.array({})
for dirName, subdirList, fileList in os.walk(rootDir):
    # for subdir in subdirList:
    for fname in fileList:
        img = Image.open(os.path.join(dirName, fname))
        imageArray.append(np.ravel(img))
        img.close()
        labelArray.append(os.path.basename(dirName))

print(len(imageArray))
print(len(labelArray))
labelArray = np.array(labelArray)
imageArray = np.array(imageArray)

X_train, X_test, y_train, y_test = train_test_split(
    imageArray, labelArray, test_size=0.15, random_state=23)

X_train.shape

X_test.shape

# In[12]:

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes =  int(10)
epochs = 30 # 12

# input image dimensions
img_rows, img_cols = 128, 128 # 128, 128
input_shape = X_test[0].shape # (img_rows, img_cols, 1)

x_train = X_train.astype('float32')
x_test = X_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print(y_train.shape[0], 'train classes')
print(y_test.shape[0], 'test classes')
print("num_classes =", num_classes)


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

feature_size = (4, 4)  # (IMAGE_WIDTH//20, IMAGE_HEIGHT//20)  # (3, 3)

model = Sequential()

# original layers

model.add(Conv2D(32, feature_size,  # was 32
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, feature_size, activation='relu'))  # was 64
model.add(MaxPooling2D(pool_size=(2, 2)))  # was (2, 2)
model.add(Dropout(0.25))  # was 0.25
model.add(Flatten())
model.add(Dense(64, activation='relu'))   # was Dense(128, ...)
model.add(Dropout(0.25))  # was 0.5
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer="rmsprop",
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
loss = score[0]
accuracy = score[1] * 100.0
print('Test loss: %2.1f' % loss)
print('Test accuracy:  %2.1f%%' % accuracy)
