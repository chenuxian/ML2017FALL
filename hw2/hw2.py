import sys
import csv
import math
import numpy
import pandas

#0.85810
features = ['age', 'fnlwgt', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', ' Federal-gov', ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay', '?_workclass', ' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college', ' Divorced', ' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving', '?_occupation', ' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife', ' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White', ' Cambodia', ' Canada', ' China', ' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador', ' El-Salvador', ' England', ' France', ' Germany', ' Greece', ' Guatemala', ' Haiti', ' Holand-Netherlands', ' Honduras', ' Hong', ' Hungary', ' India', ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan', ' Laos', ' Mexico', ' Nicaragua', ' Outlying-US(Guam-USVI-etc)', ' Peru', ' Philippines', ' Poland', ' Portugal', ' Puerto-Rico', ' Scotland', ' South', ' Taiwan', ' Thailand', ' Trinadad&Tobago', ' United-States', ' Vietnam', ' Yugoslavia', '?_native_country']
w = [10.175022337003373, 1.2053912269773648, 0.866701654923052, 24.29436504143158, 2.730848291036242, 2.466370740819122, 0.02663793692166096, -0.6208234213568908, -0.2357181785245513, -0.40006035545793067, -0.21154759416458332, -0.8674538949357053, -0.7458645430825964, -1.5022318673813908, -0.3641943502124113, -1.0423861458748347, -0.9253987268305508, -0.5950113621522957, -1.5713359403606881, -1.2686850520841197, -1.454061930698394, -1.2484581983655116, 0.19879426270914916, 0.22306170953110277, 0.8146458391503592, 1.8667189158819009, -0.2878300595264757, 1.124251130373592, -2.498587590298447, 1.6935227510698025, 0.04950412928089755, -1.4558562759407216, 1.4522848457841457, 0.7498518517079328, -1.408496480738696, -1.7121809933220635, -1.5591692665635155, -0.9876899491215104, -0.13054914827030936, -0.5544558113314341, -0.0992854560935373, 0.6457745449199479, -1.074758916100407, -0.7940472774867446, -0.4462395396253782, -0.9565468151235204, -2.1744057936313657, 0.38067321893569267, 0.4831867496369176, 0.16527078519115423, 0.4999626511478978, -0.2659229316263561, -0.5999125287369578, -1.0503872453489973, -0.5088908108930158, -1.3676231423676384, -1.5787871799777295, -0.7088958089931794, 0.2933279193861394, -1.3143163728445248, -0.6788498888856394, -0.9825272317871995, -1.166977281680851, -0.7785854929962476, 1.0983860339028508, 0.41325194826612055, -0.6448413159555576, -1.7188800502546466, 0.4538217396463491, -1.4474525291895513, 0.0570236301341023, -0.5101215183624934, 0.48454669380120935, 0.6733312699686497, 0.5265549811104624, -0.8054811506619444, -0.1506686559661474, -0.10852445888886583, -0.019546011284571407, -0.2935158176473804, -0.019550106782881323, 0.07593424827136852, -0.3234450349174528, 0.05678923394553972, 0.6575634813445349, 0.8232439560836574, 0.07424214107779459, 0.37218505410154024, -0.40284505599499326, -0.5343890068993407, -0.6819737800477625, -0.8782529786814677, -0.6684416753448417, 0.4471740255164582, 0.17095340208104357, 0.021721933924325323, -0.32269172164298454, 0.033821570984352546, -0.9952213510466961, 0.05837435766095967, -0.41719832842883503, -0.3521511288939421, 0.27524878564455624, -0.9485742739725831, 0.6781912164364621, -0.12985002123190417, -0.8977811590252648, -0.38076243624630957, -8.682973651598243, -4.921256268194406]

x_item = []
temp_item = []
y_item = []
# sort the data order
for i in features:
    x_item.append(pandas.read_csv(sys.argv[1], sep=',' , usecols=[i]).values.tolist())

# data normalization
tmp = numpy.array(x_item[0])
tmp = tmp / 90
x_item[0] = tmp.tolist()
tmp = numpy.array(x_item[1])
tmp = tmp / 1484705
x_item[1] = tmp.tolist()
tmp = numpy.array(x_item[3])
tmp = tmp / 99999
x_item[3] = tmp.tolist()
tmp = numpy.array(x_item[4])
tmp = tmp / 4356
x_item[4] = tmp.tolist()
tmp = numpy.array(x_item[5])
tmp = tmp / 99
x_item[5] = tmp.tolist()

# square item
row_name = [0, 1]
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
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    spamwriter = csv.writer(f, delimiter=',')
    spamwriter.writerow(['id', 'label'])
    for i in y_item:
        if i < 0.5:
            spamwriter.writerow([str(index), 0])
        else:
            spamwriter.writerow([str(index), 1])
        index += 1
