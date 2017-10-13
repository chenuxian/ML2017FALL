import csv
import sys
import codecs
import pandas

w = [0.78540992597061843, -0.00023977257799605723, 0.0029146200715993754, -0.024342980838507833, 0.030086069575661522, -0.014914030098146219, -0.026586918935869272, 0.02584639723590159, -0.015856682150996251, 0.081190455643082166, -0.034467826633749275, -0.0095848457483306317, 0.21058236696447102, -0.23949523781865853, -0.032912777631804412, 0.50406453964928977, -0.57127594991965869, 0.0031285853003381629, 1.0167586987682236]


index = 0
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    spamwriter = csv.writer(f, delimiter=',')
    spamwriter.writerow(['id', 'value'])
    data = pandas.read_csv(sys.argv[1], header = -1).values.tolist()
    len_data = len(data)
    # start testing x_vector to get y
    for i in range(0, len_data, 18):
        x = []
        x += data[i+8][2:11]
        x += data[i+9][2:11]

        # add bias
        inner_product = w[0]
        # do inner product
        for m in range(1, 19):
            inner_product += w[m] * float(x[m-1])
        spamwriter.writerow(['id_'+str(index), inner_product])
        index += 1
