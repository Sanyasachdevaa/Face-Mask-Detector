# =============================packages=========================

import cv2
from mechanize import ImageControl
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D , Dense , MaxPooling2D , Activation, Dropout, Flatten
from keras.optimizers import Adam


# ===========================Global variables==========================

path = 'images'
images = []
classNo = []
testRatio = 0.2
valRatio = 0.2
imgDimension = (32,32,3)

# =========================Converting Images in numpy array==================

myList = os.listdir(path)
numOfclasses = len(myList)

print("Importing classes.....")

for x in range(0, numOfclasses):
    pictureList = os.listdir(path + "/" + str(x))

    for y in pictureList:
        curImage = cv2.imread(path + "/" + str(x) + "/" + y)
        curImage = cv2.resize(curImage,(imgDimension[0],imgDimension[1]))
        images.append(curImage)
        classNo.append(x)
    print(x)

images = np.array(images)
classNo = np.array(classNo)


# ========================Splitting Data=========================

x_train , x_test , y_train , y_test = train_test_split(images, classNo, test_size=testRatio)
x_train , x_validation , y_train , y_validation = train_test_split(x_train, y_train, test_size=valRatio)

numOfSample = []

for x in range(0, numOfclasses):
    numOfSample.append(len(np.where(y_train==x)[0]))

def process(img):
    img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

x_train = np.array(list(map(process, x_train)))
x_test = np.array(list(map(process, x_test)))
x_validation = np.array(list(map(process, x_validation)))


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation = x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)

dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)

dataGen.fit(x_train)

y_train = to_categorical(y_train, numOfclasses)
y_test = to_categorical(y_test, numOfclasses)
y_validation = to_categorical(y_validation, numOfclasses)

# ==============================Training CNN==================================

def Model():
    sizeFilter1 = (3,3)
    sizeFilter2 = (3,3)
    sizePool = (2,2)

    model = Sequential()
    model.add((Conv2D(32, sizeFilter1, input_shape=(imgDimension[0],imgDimension[1],1),activation='relu')))
    model.add((Conv2D(32, sizeFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizePool))

    model.add((Conv2D(64,sizeFilter2,activation='relu')))
    model.add(MaxPooling2D(pool_size=sizePool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numOfclasses, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

mymodel = Model()
print(mymodel.summary())

history = mymodel.fit_generator(dataGen.flow(x_train, y_train, batch_size=50),
steps_per_epoch=100,
epochs=2,
validation_data=(x_validation, y_validation),
shuffle=1
)

mymodel.save("TrainingModel.h5")