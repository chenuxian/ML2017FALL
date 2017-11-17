import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import codecs
import pandas
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

# do not show scientific notation
np.set_printoptions(suppress=True)

# read training data
data = pandas.read_csv(sys.argv[1]).values.tolist()
data_len = len(data)
y_train = [[]] * data_len
x_train = [[]] * data_len
for i in range(0, len(data)):
    temp_y = [0] * 7
    temp_y[data[i][0]] = 1
    y_train[i] = temp_y
    x_train[i] = data[i][1].split(' ')
    for j in range(0, len(x_train[i])):
        x_train[i][j] = float(x_train[i][j]) / 255;
x_train = np.array(x_train)
x_train = np.reshape(x_train, [-1, 48, 48, 1])
y_train = np.array(y_train)

# -------------build model------------------------------------------
# convolution
input_img = Input(shape=(48, 48, 1))
block1 = Conv2D(32, (3, 3), padding='same', activation='relu')(input_img)

block2 = Conv2D(32, (3, 3), activation='relu')(block1)
block2 = MaxPooling2D(pool_size=(2, 2))(block2)
block2 = Dropout(0.2)(block2)

block3 = Conv2D(64, (3, 3), padding='same', activation='relu')(block2)

block4 = Conv2D(64, (3, 3), activation='relu')(block3)
block4 = MaxPooling2D(pool_size=(2, 2))(block4)
block4 = Dropout(0.2)(block4)

block5 = Conv2D(128, (3, 3), padding='same', activation='relu')(block4)

block6 = Conv2D(128, (3, 3), activation='relu')(block5)
block6 = MaxPooling2D(pool_size=(2, 2))(block6)
block6 = Dropout(0.2)(block6)

block6 = Flatten()(block6)

# NN
fc1 = Dense(512, activation='relu')(block6)
fc1 = Dropout(0.5)(fc1)

predict = Dense(7)(fc1)
predict = Activation('softmax')(predict)

model = Model(inputs=input_img, outputs=predict)

lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()

def lr_schedule(epoch):
    return lr*(0.1**int(epoch/10))
# -------------build model------------------------------------------

# training
model.fit(x_train, y_train, 
          batch_size=32, 
          epochs=150, 
          shuffle=True,
          callbacks=[LearningRateScheduler(lr_schedule)]
         )

# save model
model.save("./model.h5")
