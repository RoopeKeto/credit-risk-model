# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 00:18:19 2018

@author: roope
"""

# Luottoriskimalli hyödyntäen "neural network" -mallia kolmella layerilla

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:39:07 2018

@author: roope
"""

import keras
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model


import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

#-------------------------------------------------

# Ladataan tiedostosta training data muuttujaan "training_data"
from numpy import genfromtxt
training_data = genfromtxt("E:\\KAGGLE\\give_me_some_credit\\training_data.csv", delimiter=',')

# luodaan X_train ja Y_train, 
X_train = training_data[1:120271, 2:12]
Y_train = training_data[1:120271, 1]
Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))

# ensimmäiselle hidden layerille 5 hidden unitia
luottoriskimalli = Sequential()
luottoriskimalli.add(Dense(5, input_shape=(10,)))

luottoriskimalli.add(Activation('relu'))

# 2 hidden layer
luottoriskimalli.add(Dense(5))
luottoriskimalli.add(Activation('relu'))

# 3 hidden layer
luottoriskimalli.add(Dense(5))
luottoriskimalli.add(Activation('relu'))

# lopuksi sigmoid funkito
luottoriskimalli.add(Dense(1))
luottoriskimalli.add(Activation('sigmoid'))

# compile model
luottoriskimalli.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# mallin treenaus
luottoriskimalli.fit(x = X_train, y = Y_train, epochs = 10, batch_size = 50 )

# seuraavaksi ennustukset
X_test_orig = genfromtxt("E:\\KAGGLE\\give_me_some_credit\\testi_data.csv", delimiter=',')
X_test = X_test_orig[1:101504, 1:11]
classes = luottoriskimalli.predict(X_test)

# kirjoitetaan ennustukset csv tiedostoon, formattiin, jonka Kaggle hyväksyy.
np.savetxt("predictions4.csv", classes, delimiter=",")
