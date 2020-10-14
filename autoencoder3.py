from keras import layers, models
from keras.models import load_model
from sklearn import preprocessing

import numpy as np
import pandas as pd
from prompt_toolkit import input
 
class AE(models.Model):
  def __init__(self, x_nodes=38, z_dim=8):
    x = layers.Input(shape=(x_nodes,))
    z = layers.Dense(z_dim, activation='relu')(x)
    y = layers.Dense(x_nodes, activation='sigmoid')(z)
    
    super().__init__(x, y)
    
    self.x = x
    self.z = z
    self.z_dim = z_dim
    
    self.compile(optimizer='adadelta', loss='binary_crossentropy',
                metrics=['accuracy'])
    
  def get_encoder(self):
    return models.Model(self.x, self.z)
  
  def get_decoder(self):
    z = layers.Input(shape=(self.z_dim,))
    y = self.layers[-1](z)
    return models.Model(z, y)

#######################################################################
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

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
 
autoencoder = load_model('AE.h5')
 
input = autoencoder.encoder(Y[0])
output = autoencoder.decoder(Y[0])

print(input)
print(output)


  
  

