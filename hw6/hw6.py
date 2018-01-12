import sys, pandas, csv, numpy, tensorflow, pickle, keras.backend.tensorflow_backend
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, Input, Flatten, Concatenate
from sklearn.cluster import KMeans
from keras.initializers import RandomNormal

epochs = 100
gpu_fraction = 1.0
        
keras.backend.tensorflow_backend.set_session(tensorflow.Session(config=tensorflow.ConfigProto(gpu_options=tensorflow.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction))))

input_img = Input(shape=(784,))
encoded = Dense(128, activation = 'relu', kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.1, seed = None), bias_initializer = 'zeros')(input_img)
encoded = Dense(64, activation = 'relu', kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.1, seed = None), bias_initializer = 'zeros')(encoded)
encoded = Dense(32, activation = 'relu', kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.1, seed = None), bias_initializer = 'zeros')(encoded)

decoded = Dense(64, activation = 'relu', kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.1, seed = None), bias_initializer = 'zeros')(encoded)
decoded = Dense(128, activation = 'relu', kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.1, seed = None), bias_initializer = 'zeros')(decoded)
decoded = Dense(784, activation = 'relu', kernel_initializer = RandomNormal(mean = 0.0, stddev = 0.1, seed = None), bias_initializer = 'zeros')(decoded)

encoder = Model(input = input_img, output = encoded)
autoencoder = Model(input = input_img, output = decoded)
autoencoder.compile(optimizer = 'adadelta', loss = 'mse')
autoencoder.summary()

# training
if sys.argv[1] == "train":
    input_data = numpy.load("image.npy")    
    autoencoder.fit(input_data, input_data, epochs = epochs)
    with open('map', 'wb') as f:
        obj = KMeans(2, n_init = 200).fit(encoder.predict(input_data, verbose = 1)).labels_
        pickle.dump(obj, f, True)

# testing
if sys.argv[1] == "test":
    with open('map', 'rb') as f:
        cluster = pickle.load(f)

    input_data = pandas.read_csv(sys.argv[2], sep = ',', encoding = 'utf-8', usecols = ['image1_index', 'image2_index'])
    input_data_1 = input_data['image1_index'].values
    input_data_2 = input_data['image2_index'].values

    with open(sys.argv[3], 'w', encoding='utf-8') as f:
        spamwriter = csv.writer(f, delimiter=',')
        spamwriter.writerow(['ID', 'Ans'])
        for i in range(len(input_data_1)): 
            if cluster[input_data_1[i]] == cluster[input_data_2[i]]:
                spamwriter.writerow([str(i), str(1)])
            else:
                spamwriter.writerow([str(i), str(0)])
