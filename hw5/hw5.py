import sys, csv, numpy,tensorflow, pandas, keras.backend.tensorflow_backend
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, Reshape, Merge, Input, Flatten, Dot, Add

input_data = pandas.read_csv(sys.argv[1], sep = ',', encoding='utf-8', usecols=['UserID', 'MovieID'])
users = input_data['UserID'].values - 1
movies = input_data['MovieID'].values - 1
max_user = input_data['UserID'].max()
max_movie = input_data['MovieID'].max()

def getmodel(n_users, n_items, latent_dim):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_users, latent_dim)(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items, latent_dim)(item_input)
    item_vec = Flatten()(item_vec)
    user_bias = Embedding(n_users, 1)(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_items, 1)(item_input)
    item_bias = Flatten()(item_bias)
    r_hat = Dot(axes=1)([user_vec, item_vec])
    r_hat = Add()([r_hat, user_bias, item_bias])
    model = Model([user_input, item_input], r_hat)
    model.compile(loss='mse', optimizer='adamax')
    return model

model = getmodel(max_user, max_movie, 120)
model.summary()
model.load_weights('model.h5')
output = model.predict([users, movies])

with open(sys.argv[2], 'w', encoding='utf-8') as f:
    spamwriter = csv.writer(f, delimiter=',')
    spamwriter.writerow(['TestDataID', 'Rating'])
    for i, j in enumerate(output):  
        spamwriter.writerow([str(i+1), str(j[0])])
