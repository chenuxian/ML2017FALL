import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import codecs
import pandas

# do not show scientific notation
np.set_printoptions(suppress=True)

repeat = 10000000
l_rate = 0.00001

# Model: amb_temp + O3 + PM10 + PM2.5 + RH + PM10^2 + PM2.5^2 + bias
w = [0] * 64
w = np.array(w)

all_feature = [[]] * 18
# get all data into feature list
data = pandas.read_csv(r"train.csv").values.tolist()
for i in range(0, len(data)):
    all_feature[i%18] = all_feature[i%18] + data[i][3:27]

# only need amb_temp, O3, PM10, PM2.5, RH
temp_feature = []
temp_feature.append(all_feature[0])
temp_feature.append(all_feature[7])
temp_feature.append(all_feature[8])
temp_feature.append(all_feature[9])
temp_feature.append(all_feature[11])

# split each feature to month's feature
# one feature has 5760 data, and one feature / one month has continuously 480 data
feature = [[]] * 5
len_feature = len(temp_feature)
len_each_feature = len(temp_feature[0])
for f_index in range(0, len_feature):
    # becuase [[]] * X is copy reference so use "append" will append to all lists
    # but if use "assign", it will not assign to all lists(new memory address)
    feature[f_index] = [[],[],[],[],[],[],[],[],[],[],[],[]]
    for i in range(0, len_each_feature):
        month = i // 480
        feature[f_index][month].append(temp_feature[f_index][i])

# get every 9 data into x and get 10th PM2.5 into y
# x will look like this [1, amb_temp, O3, PM10, PM2.5, RH, PM10^2, PM2.5^2]
x = []
y = []
for month in range(0, 12):
    for i in range(0, 471):
        temp_x = []
        for f_index in range(0, 5):
            temp_x += feature[f_index][month][i:i+9]

        for j in range(2, 4):
            for k in range(i, i+9):
                temp_x.append(feature[j][month][k] ** 2)
        #temp_x += [1]
        x.append(temp_x)
        y.append(feature[3][month][i+9])

x = np.array(x)
y = np.array(y)
# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
    # wx
    hypo = np.dot(x,w)
    # y - wx
    loss = hypo - y

    # cost
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    
    # x(y - wx)
    gra = np.dot(x_t,loss)
    
    # adagrad
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    
    w = w - l_rate * gra/ada

print_w = []
for m in range(64):
    print_w.append(w[m])
print(print_w)

"""
# start training w
len_y = len(y)
for i in range(0, training_time):
    total = [0] * 64
    cost= 0
    for j in range(0, len_y):
        # wx
        inner_product = 0
        for k in range(0, 64):
            inner_product = inner_product + w[k] * float(x[j][k])

        # get y - wx
        inner_product = inner_product - float(y[j])

	# get total cost
        cost = cost + inner_product ** 2
        
        # get Σ(y - wx)x 
        for k in range(0,64):
            total[k] = total[k] + learning_rate * inner_product * float(x[j][k])
    
    print(cost/(2*len_y))
    # update w
    for k in range(0,64):
        w[k] = w[k] - (total[k] / len_y)
        
print(w)"""
