import sys
import csv
import math
import numpy
import pandas

# use [feature] to select what you want from train data
# features below use this type
# 'age':0, 'fnlwgt':0, 'education_num':0, 'sex':0, 'capital_gain':0, 'capital_loss':0, 'hours_per_week':0
dic = {'age':0, 'sex':0, 'capital_gain':0, 'capital_loss':0, 'hours_per_week':0}
# other features choosen by number
feature = [3, 6, 8, 13]
row_name = pandas.read_csv("./data/train.csv", sep=',' , header = -1, nrows = 1).values.tolist()
row_name = row_name[0]
# start select features
for i in range(0, len(feature)):
    item = pandas.read_csv("./data/train.csv", sep=',' , usecols=[feature[i]]).values.tolist()
    for j in range(0, len(item)):
        if item[j][0] == " ?":
                dic['?_'+row_name[feature[i]]] = '0'
        else:
                dic[item[j][0]] = '0'


# start to load feature from X_train data
row_name = pandas.read_csv("./data/X_train.csv", sep=',' , header = -1, nrows = 1).values.tolist()
row_name = row_name[0]
x_name = []
x_item = []
y_item = []
# sort the data order
for i in row_name:
    if i in dic:
        item = pandas.read_csv("./data/X_train.csv", sep=',' , usecols=[i]).values.tolist()
        x_name.append(i)
        y_item.append(item)

print(x_name)
# data normalization
row_name = [0, 2, 3, 4]
for i in row_name:
    tmp = numpy.array(y_item[i])
    print(numpy.amax(tmp))
    tmp = tmp / numpy.amax(tmp)
    y_item[i] = tmp.tolist()
	
# square item
row_name = [0, 2, 4]
for i in row_name:
    tmp = numpy.array(y_item[i])
    tmp = tmp ** 2
    y_item.append(tmp.tolist())

# 3rd power
row_name = [0]
for i in row_name:
    tmp = numpy.array(y_item[i])
    tmp = tmp ** 3
    y_item.append(tmp.tolist())

# append data into x_item and y_item
row_name = len(y_item[0])
for i in range(0, row_name):
    tmp = []
    for j in range(0, len(y_item)):
        tmp += y_item[j][i]
    tmp += [1]
    x_item.append(tmp)
y_item = []
y_item = pandas.read_csv("./data/Y_train.csv", sep=',').values.tolist()
x_item = numpy.array(x_item)
y_item = numpy.array(y_item)

# start training
training_time = 78900
learning_rate = 0.0001
w = [0] * len(x_item[0])
w = numpy.array(w)[numpy.newaxis]
w = w.transpose()
x_item_t = x_item.transpose()

for i in range(0, training_time):
    inner = numpy.dot(x_item, w)
    inner = 1 / (numpy.exp(-1 * inner)+1)
    # use cross entropy to find loss
    #loss = -1 * (y_item * numpy.log(inner) + (1 - y_item) * numpy.log(1 - inner))
    #print(numpy.sum(loss))
    inner = inner - y_item
    inner = numpy.dot(x_item_t, inner)
    w = w - learning_rate * inner

print(w.transpose().tolist()[0])
