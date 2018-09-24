
# coding: utf-8
import os
from os.path import basename
from sklearn.preprocessing import StandardScaler
from PIL import Image
from sklearn.model_selection import train_test_split

import numpy as np
import os
from os.path import basename

rootDir = '/home/maysam/notebooks/challenges/files/converted_images_normalized'
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

print(X_train.shape)
# In[9]:

from PIL import Image
from sklearn import svm
import numpy as np



image = Image.open("/home/maysam/notebooks/challenges/files/test/test.png")

image = np.array(image)
img = image.flatten()


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train) 
print("The gear image is in the category of:" + str(knn.predict([img])))
y_pred= knn.predict(X_test)
print("Accurary of the test data is:"+str(accuracy_score(y_pred,y_test))+"%")

