# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:58:50 2020
Published on Fri Dec 18

@main_contributor: Cazeneuve Dorian (Deep Learning)
@contributor: Breau-Trenit Antoine (Wavelets)
@contributor : Johansson Rodolphe (CSV)

@project: mano-robot - Au513 
"""


#%%
#
## Import database
#

# we concatenate the signals

import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data

liste=[]
Y=np.zeros((len(liste),4))
T = 10 #Threshold for low-pass filter in wavelets 

import csv
with open('dataBase.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    rownumber = 0
    for row in spamreader:
        #print(', '.join(row))
        if (rownumber/2)==int(rownumber/2) :
            liste.append(row)
        rownumber += 1
    Y_train=np.zeros((len(liste),4))
    
    for i in range(0,len(liste),1):
        for j in range(0,len(liste[0])-4,1):
            
            liste[i][j]=liste[i][j].split(',')
            Y_train[i][0]=int(liste[i][-4])
            Y_train[i][1]=int(liste[i][-3])
            Y_train[i][2]=int(liste[i][-2])
            Y_train[i][3]=int(liste[i][-1])

        
        
    mini=[]
    for i in range(len(liste)):
        mini.append(len(liste[i]))         
    minimu=min(mini)

    for i in range(0,len(liste)):
        if len(liste[i])!=minimu:
            liste[i]=liste[i][0:minimu]
    
    X_train_noise=np.zeros((len(liste),len(liste[0])-4))
    X_train=np.zeros((len(liste),len(liste[0])-4))
    
    for j in range(0,len(liste)):
        for i in range(0,len(liste[0])-5):
            # liste[i][j]=int(str(liste[0][0])[2:-2])
            # print('A :' ,str(liste[j][i])[2:-2])
            try:
                coeffs2_tresh=[]
                X_train_noise[j][i]=int(str(liste[j][i])[2:-2])
                db2 = pywt.Wavelet('db2')
                coeffs2=pywt.wavedec(X_train_noise[j],db2,mode="zero",level=3)
                for i in range(0,np.size(coeffs2)):
                    coeffs2_tresh.append(pywt.threshold(coeffs2[i],T,mode='soft'))
                X_train[j]=pywt.waverec(coeffs2_tresh,db2)
                
            except:
                print("error")
        

    # X_train.reshape(-1,1)
    

    # X_train = np.delete(A,0,1)
    # X_train= np.delete(X_train, 6, 1)
    
    # Y_train = np.delete(A, np.s_[:7], 1)
    
#%%
#
## Neural Network
#
    

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, MaxPooling1D , Flatten
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.python import keras

import tensorflow as tf
# for the droupout rate
rate=0.10
# for the optimizer
tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, 
    name='Adam', clipnorm=0.001
)

model=Sequential()

model.add(Dense(len(X_train[0]),input_dim=len(X_train[0]), kernel_initializer='normal', activation='linear'))
model.add(Dense(200,activation='linear'))
# model.add(keras.layers.Dropout(rate, noise_shape=None, seed=None))
model.add(Dense(100,activation='linear'))
# model.add(keras.layers.Dropout(rate, noise_shape=None, seed=None))
model.add(Dense(50,activation='linear'))
# model.add(Dense(100,activation='relu'))
# model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='linear'))
model.add(Dense(4,activation='linear'))
model.summary()
model.compile(loss='mse', optimizer='Adam', metrics = [ 'mse', 'mae'])

#%%
#
## fit
#

history= model.fit(X_train,Y_train, epochs = 800, batch_size = 60, verbose=1, validation_split=0.2)



#%%
#
## predict 
#
# print("The true value is " , '\n'*3,Y_train[-10:-1],'\n'*3)
x_test=(X_train[-10:-1])
y_test=model.predict(x_test)
for i in range(0,len(y_test)):
    for j in range(0,len(y_test[0])):
        if y_test[i][j]>0.5:
            y_test[i][j]=1
        else:
            y_test[i][j]=0
print("The model error")
print(y_test-Y_train[-10:-1])
