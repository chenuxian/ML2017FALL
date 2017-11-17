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

# read testing data
t_data = pandas.read_csv(sys.argv[1]).values.tolist()
x_test = [[]] * len(t_data)
for i in range(0, len(t_data)):
    x_test[i] = t_data[i][1].split(' ')
x_test = np.array(x_test)
x_test = np.reshape(x_test, [-1, 48, 48, 1])

model = load_model("./model.h5")
opt = Adam(lr = 1e-8)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# model.summary()

# testing
classes = model.predict(x_test, batch_size=32)

# output result
index = 0
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    spamwriter = csv.writer(f, delimiter=',')
    spamwriter.writerow(['id', 'label'])
    for i in range(0, len(classes)):
        max_i = 0
        max_value = 0
        for j in range(0, 7):
            if classes[i][j] > max_value:
                max_value = classes[i][j];
                max_i = j
        spamwriter.writerow([str(index), max_i])
        index += 1
