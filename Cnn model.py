# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:02:44 2020

@author: Oloyede John
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cv2
import os
import tensorflow as tf

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import datasets, metrics, svm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D




## **************************** functions *************************************
def get_image_path(folder_path,image_type,healthy="yes"):
    '''
      folder_path = 'the full folder path'
      image_type = 'wave' or 'spiral'
      healthy = 'yes' or 'no' i.e. yes is healthy, no is parkinson 
      return = create full file path
    '''
    if healthy.lower()  == "yes":
        healthy_or_parkinson = 'healthy'
    else:
        healthy_or_parkinson = 'parkinson'
        
    file_path = os.path.join(folder_path,image_type,healthy_or_parkinson)
    
    return file_path

def retreive_images(img_folder_path):
    '''
      return list of images  
    '''
    img_file_list = os.listdir(img_folder_path)
    image_list = []
    
    for img_file in img_file_list:
        img_path = os.path.join(img_folder_path, img_file)
        img = cv2.imread(img_path)
        image_list.append(img)
    return image_list

#transformed file location
tx_wave_path = 'revised/wave'
tx_spiral_path = 'revised/spiral'

# wave images
wv_train_healthy = retreive_images(get_image_path(tx_wave_path, 'training', 'yes'))
wv_train_park  = retreive_images(get_image_path(tx_wave_path, 'training', 'no'))

# spiral images
sp_train_healthy = retreive_images(get_image_path(tx_spiral_path, 'training', 'yes'))
sp_train_park = retreive_images(get_image_path(tx_spiral_path, 'training', 'no'))

#Merge data, define X and Y
# Wave
wv_train_X = wv_train_healthy.copy()
wv_train_X.extend(wv_train_park)
wv_X = np.array(wv_train_X)


# target Y is equals to 0 when healthy or 1 when it is parkison
# healthy label = 0
# parkisons label = 1
wv_train_Y_healthy = np.zeros(len(wv_train_healthy))
wv_train_Y_park = np.ones(len(wv_train_park))
wv_Y = np.concatenate((wv_train_Y_healthy, wv_train_Y_park))

# Spiral
sp_train_X = sp_train_healthy.copy()
sp_train_X.extend(sp_train_park)
sp_X = np.array(sp_train_X)

sp_train_Y_healthy = np.zeros(len(sp_train_healthy))
sp_train_Y_park = np.ones(len(sp_train_park))
sp_Y = np.concatenate((sp_train_Y_healthy, sp_train_Y_park))


## Commencing Data Preparation for Modeling of Wave **************************
wv_X_train, wv_X_test, wv_y_train, wv_y_test = train_test_split(wv_X, wv_Y, 
                                                    train_size = 0.80,
                                                  test_size = 0.20,
                                                    stratify=wv_Y)

## Commencing Data Preparation for Modeling of Spiral **************************
sp_X_train, sp_X_test, sp_y_train, sp_y_test = train_test_split(sp_X, sp_Y, 
                                                    train_size = 0.80,
                                                    test_size = 0.20,
                                                    stratify=sp_Y)





# ********************* *** WAVE *******************************************
# Convolutional Neural Network

wv_X_train = wv_X_train / 255.0
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = wv_X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = "binary_crossentropy",
              optimizer = "adam",
              metrics = ['accuracy'])

model.fit(wv_X_train, wv_y_train, batch_size = 8, epochs =5, validation_split=0.1)



# ********************* *** SPIRAL *******************************************
# Convolutional Neural Network

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = sp_X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = "binary_crossentropy",
              optimizer = "adam",
              metrics = ['accuracy'])

model.fit(sp_X_train, sp_y_train, batch_size = 8, epochs =10, validation_split=0.1)























