from keras import layers, models
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from sklearn import preprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prompt_toolkit import input

#######################################################################
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.,1.))

df = pd.read_csv('ZZ13.csv', header=0)
df.columns = ['age','CUST_NEW_DT','CARD_BSS_GRCD','card_acept_bal','ilsi_acept_bal',
              'halbu_acept_bal','card_bal','ilsi_aver','check_count','for_exc','pay_count',
              'atm_count','auto_count','tele_count','MBBK_UZ_DVCD','INBN_UZ_DVCD',
              'yogubul_aver','guchi_aver','loan_aver','loan_down_int2','loan_down_int',
              'loan_up_int','loan_up_int2','loan_zeo','loan_new','depo_down_int2','depo_up_int',
              'depo_up_int2','depo_new','profit_down_int2','profit_up_int','profit_up_int2',
              'profit_zeo','profit_new','auto_down_int','auto_up_int','auto_up_int2','auto_out','outyn']
X = df[df.outyn == 0]
Y = df[df.outyn == 1]

print(X)
print(Y)
X = X[df.columns[:-1]]
X = np.array(X)
X = min_max_scaler.fit_transform(X)

Y = Y[df.columns[:-1]]
Y = np.array(Y)
Y = min_max_scaler.fit_transform(Y)
print(X)
print(Y)

x_train = X
print(X.shape)
print(Y.shape)

####################################################################### 
# this is the size of our encoded representations
encoding_dim1 = 18
encoding_dim2 = 8  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
# intput size
input_dim = 38
# this is our input placeholder
input_img = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded1 = Dense(encoding_dim1, activation='relu')(input_img)
encoded2 = Dense(encoding_dim2, activation='relu')(encoded1)

# "decoded" is the lossy reconstruction of the input
decoded2 = Dense(encoding_dim1, activation='relu')(encoded2)
decoded1 = Dense(input_dim, activation='sigmoid')(decoded2)
# decoded = Dense(input_dim, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded1)
print(autoencoder.summary())

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded2)
print(encoder.summary())
# create a placeholder for an encoded (32-dimensional) input01
#encoded_input = Input(shape=(encoding_dim1,))
# retrieve the last layer of the autoencoder model
#decoder_layer = autoencoder.layers[-1]
# create the decoder model

output_img = Input(shape=(encoding_dim2,))

decoder_layers = Sequential()
for layer in autoencoder.layers[3:]:
    print(layer)
    decoder_layers.add(layer)
    
decoder = Model(output_img, decoder_layers(output_img))
print(decoder.summary())
# compile
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

## train
history = autoencoder.fit(x_train, x_train, epochs=100, batch_size=1000, shuffle=True)



encoded_imgs = encoder.predict(Y)
decoded_imgs = decoder.predict(encoded_imgs)
encoded_imgs1 = encoder.predict(X)
decoded_imgs1 = decoder.predict(encoded_imgs1)

diff=[]
diff1=[]
for i in range(38):
    diff.append(Y[0][i] - decoded_imgs[0][i])
    diff1.append(X[0][i] - decoded_imgs1[0][i])

# print(Y[0])
# print(decoded_imgs[0])
print(diff)
print(diff1)
# print(encoded_imgs)
# print(decoded_imgs)

              
