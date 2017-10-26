import sys
import csv
import math
import numpy
import pandas

features = ['age', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', ' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving', '?_occupation', ' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White', ' Cambodia', ' Canada', ' China', ' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador', ' El-Salvador', ' England', ' France', ' Germany', ' Greece', ' Guatemala', ' Haiti', ' Holand-Netherlands', ' Honduras', ' Hong', ' Hungary', ' India', ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan', ' Laos', ' Mexico', ' Nicaragua', ' Outlying-US(Guam-USVI-etc)', ' Peru', ' Philippines', ' Poland', ' Portugal', ' Puerto-Rico', ' Scotland', ' South', ' Taiwan', ' Thailand', ' Trinadad&Tobago', ' United-States', ' Vietnam', ' Yugoslavia', '?_native_country']

#0.83230
w = [19.37884325345297, 1.176017977266435, 31.78927385317544, 2.8597234723746, 6.232254259603168, -0.8127820530670926, -0.7580904787344264, -0.4345176917886594, -1.3787943064799864, -1.0443485507463215, -1.155802310810396, -0.9496116346301241, 0.36754943099444687, 0.42701313689564047, 0.894227255991637, 1.8190762578773225, -0.05066966935870326, 1.0962283076965496, -8.751132678670894, 1.7264356456903112, 0.24428804946022267, -0.2331020894292164, -1.045955489669171, -0.21743055214441354, 0.5499179833986952, -1.2415056518916248, -0.9811718144319769, -0.4986243757424808, -1.1447241075513033, -4.147634366070021, 0.21166793630715253, 0.28985433694481216, 0.06786519244254657, 0.33286430860673843, -0.33789569007169884, -0.3650569103784836, -2.204611961503411, -1.464472853132917, -1.8360890212974694, -1.8534455156131966, -1.402311938133584, 1.0535954553926075, 0.34891502273993263, -0.3097447420630359, -2.1486526662895913, 0.2918248694200899, -1.879510825959006, 0.22884709098054315, -0.38275616826455877, 0.21323940947362183, 0.3279799809320599, 0.3799835220502294, -0.8915757109714427, -0.4335435014615134, 0.18570218497934315, -0.7886170687888875, -1.113072094336058, 0.6655025570643978, -0.02997397612687009, -0.14713671683286786, -0.05542955467881717, 0.2546584331845129, 0.810611579846941, 0.12404660800742272, 0.29442271806765935, -0.1345624240908826, -0.5021742415245934, -0.42630180478140534, -2.913975051531967, -0.6589761263615853, 0.5234959346214983, -0.05342851820080335, -0.004685506446987174, -0.40141476872936516, -0.1337905105170614, -0.6948997758460657, 0.39118529506532174, -0.07937790542681092, -0.34035957010259577, 0.09268957392745178, -1.243958462510259, 1.0011867489129351, -0.18090058250421864, -9.450392445628824, 1.5219430974457429, -4.150200433393307, -8.073480153216966, -8.760931289680391]

x_item = []
temp_item = []
y_item = []
# sort the data order
for i in features:
    x_item.append(pandas.read_csv("./data/X_test.csv", sep=',' , usecols=[i]).values.tolist())

# data normalization
tmp = numpy.array(x_item[0])
tmp = tmp / numpy.amax(tmp)
x_item[0] = tmp.tolist()
tmp = numpy.array(x_item[2])
tmp = tmp / numpy.amax(tmp)
x_item[2] = tmp.tolist()
tmp = numpy.array(x_item[3])
tmp = tmp / numpy.amax(tmp)
x_item[3] = tmp.tolist()
tmp = numpy.array(x_item[4])
tmp = tmp / numpy.amax(tmp)
x_item[4] = tmp.tolist()

# square item
row_name = [0, 2, 4]
#row_name = [2, 3]
for i in row_name:
    tmp = numpy.array(x_item[i])
    tmp = tmp ** 2
    x_item.append(tmp.tolist())

# 3rd power
row_name = [0]
for i in row_name:
    tmp = numpy.array(x_item[i])
    tmp = tmp ** 3
    x_item.append(tmp.tolist())

# append data into x_item and y_item
for i in range(0, len(x_item[0])):
    tmp = []
    for j in range(0, len(x_item)):
        tmp += x_item[j][i]
    tmp += [1]
    temp_item.append(tmp)

x_item = numpy.array(temp_item)
y_item = numpy.dot(x_item, w)
y_item = 1 / (numpy.exp(-1 * y_item)+1)

index = 1
with open('output.csv', 'w', encoding='utf-8') as f:
    spamwriter = csv.writer(f, delimiter=',')
    spamwriter.writerow(['id', 'label'])
    for i in y_item:
        if i < 0.5:
            spamwriter.writerow([str(index), 0])
        else:
            spamwriter.writerow([str(index), 1])
        index += 1
