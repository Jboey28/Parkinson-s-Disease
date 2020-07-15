# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:47:54 2020

@author: Oloyede John
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cv2
import os
from shutil import copyfile

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# helper functions 
## **************************** functions *****************************************
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
        image_list.append(img_path)
    return image_list


def copy_images_to_dl_folders(images_full_file_path_list, image_label_list, image_folder_path, dl_folder, parkinson, healthy):
    ''' 
    copy files in the different dl folder for the dl modeling
    '''
    for counter in range(len(image_label_list)):
        
        healthy_dest_folder = os.path.join(dl_folder, healthy)
        park_dest_folder = os.path.join(dl_folder, parkinson)
        
        healthy_folder_path = os.path.join(image_folder_path, healthy)
        healthy_img_list = os.listdir(healthy_folder_path)
        
        image_file_name = os.path.basename(images_full_file_path_list[counter])
       
        # copy to the appropriate folder
        if image_label_list[counter] == 0:   #healthy          
            copyfile(images_full_file_path_list[counter],\
                     os.path.join(healthy_dest_folder, image_file_name))
            
        elif image_label_list[counter] == 1: #parkinson
            copyfile(images_full_file_path_list[counter],\
                     os.path.join(park_dest_folder, image_file_name))
        
        park_dest_files = os.listdir(park_dest_folder)
        healthy_dest_files = os.listdir(healthy_dest_folder)
    print ("Total number of images in source: {}".format(len(images_full_file_path_list)))
    print ("Parkison images : {}".format(len(park_dest_files)))
    print ("Healthy images : {}".format(len(healthy_dest_files)))
    print ("Total copied images : {}".format(len(healthy_dest_files) + len(park_dest_files)))
# ##################################################################################

# #################   Define the image folders for the DL model ####################
"/dlimages/spiral/"
"/dlimages/spiral/training"
"/dlimages/spiral/validation"
"/dlimages/spiral/test"
"/dlimages/spiral/training/parkinson"
"/dlimages/spiral/training/healthy"
"/dlimages/spiral/validation/parkinson"
"/dlimages/spiral/validation/healthy"
"/dlimages/spiral/test/parkinson"
"/dlimages/spiral/test/healthy"

"/dlimages/wave/"
"/dlimages/wave/training"
"/dlimages/wave/validation"
"/dlimages/wave/test"
"/dlimages/wave/training/parkinson"
"/dlimages/wave/training/healthy"
"/dlimages/wave/validation/parkinson"
"/dlimages/wave/validation/healthy"
"/dlimages/wave/test/parkinson"
"/dlimages/wave/test/healthy"

parent_folder = "dlimages"
spiral = "dlimages/spiral/"
spiral_training = "dlimages/spiral/training"
spiral_validation =  "dlimages/spiral/validation"
spiral_test =  "dlimages/spiral/test"

wave = "dlimages/wave/"
wave_training = "dlimages/wave/training"
wave_validation =  "dlimages/wave/validation"
wave_test =  "dlimages/wave/test"

parkinson = "parkinson"
healthy = "healthy"


# Create folder path to hold images
os.chdir(os.getcwd())


if not os.path.exists(parent_folder):
    os.mkdir(parent_folder)
    
    os.mkdir(spiral)
    os.mkdir(spiral_training)
    os.mkdir(spiral_validation)
    os.mkdir(spiral_test)
    
    os.mkdir(wave)
    os.mkdir(wave_training)
    os.mkdir(wave_validation)
    os.mkdir(wave_test)
    
    # create parkison and healthy folder in each folder 
    os.mkdir(os.path.join(spiral_training, parkinson))
    os.mkdir(os.path.join(spiral_training, healthy))
    os.mkdir(os.path.join(spiral_validation, parkinson))
    os.mkdir(os.path.join(spiral_validation, healthy))
    os.mkdir(os.path.join(spiral_test, parkinson))
    os.mkdir(os.path.join(spiral_test, healthy))
    
    os.mkdir(os.path.join(wave_training, parkinson))
    os.mkdir(os.path.join(wave_training, healthy))
    os.mkdir(os.path.join(wave_validation, parkinson))
    os.mkdir(os.path.join(wave_validation, healthy))
    os.mkdir(os.path.join(wave_test, parkinson))
    os.mkdir(os.path.join(wave_test, healthy))

###############################################################################


# ############ Extract Images into Training, Validation and Test ##############

''' NOTE
    split into training, test ensures that the same set of dataset are used for
    both the training and test as in the classical ML algorithms
    Training is further split into 90% - training and 10%-validation
'''

#set random state to ensure reproducibility
set_random_state = 12   # ensure it is the same for all the models.


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



## ############### Commencing Data Preparation for Modeling of Wave ############
wv_X_trainp, wv_X_test, wv_y_trainp, wv_y_test = train_test_split(wv_X, wv_Y, 
                                                    train_size = 0.80,
                                                  test_size = 0.20,
                                                    stratify=wv_Y,
                                                    random_state= set_random_state)

## Commencing Data Preparation for Modeling of Spiral **************************
sp_X_trainp, sp_X_test, sp_y_trainp, sp_y_test = train_test_split(sp_X, sp_Y, 
                                                    train_size = 0.80,
                                                    test_size = 0.20,
                                                    stratify=sp_Y,
                                                    random_state= set_random_state)

# create validation split 
wv_X_train, wv_X_validation, wv_y_train, wv_y_validation = train_test_split(wv_X_trainp, wv_y_trainp, 
                                                    train_size = 0.80,
                                                    test_size = 0.20,
                                                    stratify= wv_y_trainp,
                                                    random_state= set_random_state)

sp_X_train, sp_X_validation, sp_y_train, sp_y_validation = train_test_split(sp_X_trainp, sp_y_trainp, 
                                                    train_size = 0.80,
                                                    test_size = 0.20,
                                                    stratify= sp_y_trainp,
                                                    random_state= set_random_state)

################################################################################



##########  Upload images into appropriate folder path fo the DL model #########
print ('\nwave_training')
tx_wave_training_path = os.path.join(tx_wave_path, "training")
copy_images_to_dl_folders(wv_X_train, wv_y_train, tx_wave_training_path, wave_training, parkinson, healthy)

print ('\nwave_validation')
tx_wave_validation_path = os.path.join(tx_wave_path, "training")
copy_images_to_dl_folders(wv_X_validation, wv_y_validation, tx_wave_validation_path, wave_validation, parkinson, healthy)

print ('wave_training')
tx_wave_test_path = os.path.join(tx_wave_path, "training")
copy_images_to_dl_folders(wv_X_test, wv_y_test, tx_wave_test_path, wave_test, parkinson, healthy)

print ('\nspiral_training')
tx_spiral_training_path = os.path.join(tx_spiral_path, "training")
copy_images_to_dl_folders(sp_X_train, sp_y_train, tx_spiral_training_path, spiral_training, parkinson, healthy)

print ('\nspiral_validation')
tx_spiral_validation_path = os.path.join(tx_spiral_path, "training")
copy_images_to_dl_folders(sp_X_validation, sp_y_validation, tx_spiral_validation_path, spiral_validation, parkinson, healthy)

tx_spiral_test_path = os.path.join(tx_spiral_path, "training")
print ('\nspiral_test')
copy_images_to_dl_folders(sp_X_test, sp_y_test, tx_spiral_test_path, spiral_test, parkinson, healthy)
##################################################################################


# ################ WAVE  MODEL #################################################
#Define Deep Learning model - CNN
model_wave = tf.keras.models.Sequential([
# YOUR CODE HERE
    tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=(258, 512, 3)),
    tf.keras.layers.MaxPool2D((2,2)),
    
    tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
        
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    
    tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
        
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#    RMSprop(lr=0.001)
model_wave.compile(optimizer=RMSprop(lr=0.03), loss='binary_crossentropy', metrics=['acc'])

TRAINING_DIR = wave_training
VALIDATION_DIR = wave_validation
train_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range =0.3,
                                    fill_mode='nearest',
                                   rotation_range = 90,
                                  horizontal_flip = True,
                                  vertical_flip = True,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2
                                  )
train_generator = train_datagen.flow_from_directory(
                    TRAINING_DIR,
                    target_size = (258, 512,),
                    batch_size =10,
                    class_mode = 'binary',  
                    seed =set_random_state
                     )

validation_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range =0.3,
                                   fill_mode='nearest',
                                  rotation_range = 90,
                                  horizontal_flip = True,
                                  vertical_flip = True,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2
                                  )
validation_generator = train_datagen.flow_from_directory(
                    VALIDATION_DIR,
                    target_size = (258, 512,),                 
                    batch_size =20,
                    class_mode = 'binary',  
                    seed = set_random_state
                     )


# Define a Callback class that stops training once accuracy reaches 97.0%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.97):
      print("\nReached 97.0% accuracy so cancelling training!")
      self.model.stop_training = True

#Run model
      
callback = myCallback()
history_wave = model_wave.fit_generator(train_generator,
                              epochs=15,
                              verbose=1,
                              validation_data=validation_generator,
                              callbacks= [callback])

# PLOT LOSS AND ACCURACY
acc = history_wave.history['acc']
val_acc = history_wave.history['val_acc']
loss = history_wave.history['loss']
val_loss = history_wave.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()


#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history_wave.history['acc']
val_acc=history_wave.history['val_acc']
loss=history_wave.history['loss']
val_loss=history_wave.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')


###############################################################################

#######################  SPIRAL MODEL #########################################

#Define Deep Learning model - CNN
model_spiral = tf.keras.models.Sequential([
# YOUR CODE HERE
    tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=(256,256, 3)),
    tf.keras.layers.MaxPool2D((2,2)),
    
    tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
#    
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
        
    tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    
    tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
        
    tf.keras.layers.Flatten(),
#    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# RMSprop(lr=0.003)
model_spiral.compile(optimizer=RMSprop(lr=0.03), loss='binary_crossentropy', metrics=['acc'])

TRAINING_DIR_SP = spiral_training
VALIDATION_DIR_SP = spiral_validation
train_datagen = ImageDataGenerator(rescale=1./255,
#                                  shear_range =0.3,
#                                    fill_mode='nearest',
                                   rotation_range = 20,
                                  horizontal_flip = True,
                                  vertical_flip = True,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2
                                  )
train_generator_sp = train_datagen.flow_from_directory(
                    TRAINING_DIR_SP,
                    target_size = (256,256),
                    batch_size =5,
                    class_mode = 'binary',  
                    seed =set_random_state
                     )

validation_datagen_sp = ImageDataGenerator(rescale=1./255,
#                                       shear_range =0.3,
#                                    fill_mode='nearest',
                                   rotation_range = 20,
                                  horizontal_flip = True,
                                  vertical_flip = True,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2
                                  )
validation_generator_sp = train_datagen.flow_from_directory(
                    VALIDATION_DIR_SP,
                    target_size = (256,256),                 
                    batch_size =10,
                    class_mode = 'binary',  
                    seed = set_random_state
                     )


# Define a Callback class that stops training once accuracy reaches 97.0%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.97):
      print("\nReached 97.0% accuracy so cancelling training!")
      self.model.stop_training = True

#Run model
      
callback = myCallback()
history_spiral = model_spiral.fit_generator(train_generator_sp,
                              epochs=50,
                              verbose=1,
                              validation_data=validation_generator_sp,
                              callbacks= [callback])

# PLOT LOSS AND ACCURACY
import matplotlib.pyplot as plt
acc = history_spiral.history['acc']
val_acc = history_spiral.history['val_acc']
loss = history_spiral.history['loss']
val_loss = history_spiral.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy spiral')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy spiral')
plt.title('Training and validation accuracy spiral')
plt.legend(loc=0)
plt.figure()


plt.show()
# Desired output. Charts with training and validation metrics. No crash :)


#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history_spiral.history['acc']
val_acc=history_spiral.history['val_acc']
loss=history_spiral.history['loss']
val_loss=history_spiral.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy - Spiral")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy - Spiral")
plt.title('Training and validation accuracy - Spiral')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')



# Review results