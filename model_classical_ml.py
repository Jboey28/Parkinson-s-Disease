# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:29:30 2020

@author: Oloyede John
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cv2
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import datasets, metrics, svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.dummy import DummyClassifier
import joblib


## ################################ Helper Functins ###########################
def transform_image(img_folder_path, tx_img_height, tx_img_width):
    '''
     wave_height, wave_width = 258, 512
     spiral_height, spiral_width = 256, 256  
    '''
    img_file_list = os.listdir(img_folder_path)
    image_list = []
    
    for img_file in img_file_list:
        img_path = os.path.join(img_folder_path, img_file)
        img = cv2.imread(img_path)       
        #resize image
        img2 = cv2.resize(img,(tx_img_width, tx_img_height))            
        # change image to gray scale
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)       
        # flatten image
        img_flattened = img2.flatten()       
        #normalise image 
        img_flattened = img_flattened/255.0       
        image_list.append(img_flattened)        
    return image_list

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
        # flatten image
        img_flattened = img.flatten()
        
        #normalise image 
        img_flattened = img_flattened/255.0
        
        image_list.append(img_flattened)
    return image_list
# #############################################################################
    
##transformed file location
#tx_wave_path = 'revised/wave'
#tx_spiral_path = 'revised/spiral'
#
## wave images
#wv_train_healthy = retreive_images(get_image_path(tx_wave_path, 'training', 'yes'))
#wv_train_park  = retreive_images(get_image_path(tx_wave_path, 'training', 'no'))

tx_wave_path = 'parkinsons-drawings\\wave'
tx_spiral_path = 'parkinsons-drawings\\spiral'

# wave images
wv_train_healthy = transform_image(get_image_path(tx_wave_path, 'training', 'yes'), 258, 512)
wv_train_park  = transform_image(get_image_path(tx_wave_path, 'training', 'no'), 258, 512)

# spiral images
sp_train_healthy = transform_image(get_image_path(tx_spiral_path, 'training', 'yes'), 256,256)
sp_train_park = transform_image(get_image_path(tx_spiral_path, 'training', 'no'), 256,256)



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

#set random state to ensure reproducibility
set_random_state = 12

## Commencing Data Preparation for Modeling of Wave **************************
wv_X_train, wv_X_test, wv_y_train, wv_y_test = train_test_split(wv_X, wv_Y, 
                                                    train_size = 0.80,
                                                  test_size = 0.20,
                                                    stratify=wv_Y,
                                                    random_state= set_random_state)

## Commencing Data Preparation for Modeling of Spiral **************************
sp_X_train, sp_X_test, sp_y_train, sp_y_test = train_test_split(sp_X, sp_Y, 
                                                    train_size = 0.80,
                                                    test_size = 0.20,
                                                    stratify=sp_Y,
                                                    random_state= set_random_state)

print(" healthy label = 0" )
print(" parkisons label = 1" )

# *********************     MODELING ******************************************
# Dummy Classifier
# k-Nearest Neighbor
# Logistic regression
# Support Vector
# Decision tree Classifier
# Random Forest Classifier

# ***************  Dummy Classifier  *****************************************

# Wave
dummy_model_wv = DummyClassifier(strategy='most_frequent').fit(wv_X_train, wv_y_train)
dummy_model_wv_train = dummy_model_wv.predict(wv_X_train)
dummy_model_wv_test = dummy_model_wv.predict(wv_X_test)

print(classification_report(wv_y_test, dummy_model_wv_test, target_names=['healthy', 'parkison']))


# Spiral
dummy_model_sp = DummyClassifier(strategy='most_frequent').fit(sp_X_train, sp_y_train)
dummy_model_sp_test = dummy_model_sp.predict(sp_X_test)
print(classification_report(sp_y_test, dummy_model_sp_test, target_names=['healthy', 'parkison']))


# ***************  KNN Classifier  *****************************************
# ******* Wave ************
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(wv_X_train, wv_y_train)
Knn_wv_matrix = clf.predict(wv_X_test)
print(classification_report(wv_y_test, Knn_wv_matrix, target_names=['healthy', 'parkison']))

# **** SPIRAL ************
clf_sp = KNeighborsClassifier(n_neighbors=5)
clf_sp.fit(sp_X_train, sp_y_train)
Knn_sp_matrix = clf_sp.predict(sp_X_test)
print(classification_report(sp_y_test, Knn_sp_matrix, target_names=['healthy', 'parkison']))


# ************       Logistic Regression  *************************************

# **** WAVE *******

logreg = LogisticRegression(C=0.1).fit(wv_X_train, wv_y_train)
logreg_wv_matrix = logreg.predict(wv_X_test)
print(classification_report(wv_y_test, logreg_wv_matrix, target_names=['healthy', 'parkison']))


#  ***  SPIRAL ******
# Logistic Regression

logreg_sp = LogisticRegression().fit(sp_X_train, sp_y_train)
logreg_sp_matrix = logreg_sp.predict(sp_X_test)
print(classification_report(sp_y_test, logreg_sp_matrix, target_names=['healthy', 'parkison']))


# ************************ Support Vector Classifier ******************************

#  *** WAVE ****
svc_wv = SVC(kernel = 'linear', C=10, gamma = 0.1)
svc_wv.fit(wv_X_train, wv_y_train)
svc_wv_matrix = svc_wv.predict(wv_X_test)
print(classification_report(wv_y_test, svc_wv_matrix, target_names=['healthy', 'parkison']))

# ***  SPIRAL ****
svc_sp = SVC(kernel = 'linear', C=10, gamma = 0.1)
svc_sp.fit(sp_X_train, sp_y_train)
svc_matrix = svc_sp.predict(sp_X_test)

print(classification_report(sp_y_test, svc_matrix, target_names=['healthy', 'parkison']))


# *******************   Decision tree Classifier ******************************

# *** WAVE *****
tree_wv = DecisionTreeClassifier(max_depth=2, max_features=10,\
                                 random_state=set_random_state)
tree_wv.fit(wv_X_train, wv_y_train)
tree_wv_matrix = tree_wv.predict(wv_X_test)
print(classification_report(wv_y_test, tree_wv_matrix, target_names=['healthy', 'parkison']))

#  *** SPIRAL ******
tree_sp = DecisionTreeClassifier(max_depth=3, \
                                 max_features=100,\
                                 random_state=set_random_state)
tree_sp.fit(sp_X_train, sp_y_train)
tree_sp_matrix = tree_sp.predict(sp_X_test)
print(classification_report(sp_y_test, tree_sp_matrix, target_names=['healthy', 'parkison']))



# ******   Random Forest Classifier  ******************************************
# *** Wave ****

forest_wv = RandomForestClassifier(n_estimators=85, max_features = 250,\
                                   min_samples_split =10,  random_state = set_random_state)
forest_wv.fit(wv_X_train, wv_y_train)
forest_wv_matrix = forest_wv.predict(wv_X_test)

print(classification_report(wv_y_test, forest_wv_matrix, target_names=['healthy', 'parkison']))

# ********************* *** Spiral *******************************************
# Random Forest Classifier

forest_sp = RandomForestClassifier(n_estimators=200, max_features = 300, \
                                  random_state = set_random_state)
forest_sp.fit(sp_X_train, sp_y_train)
forest_sp_matrix = forest_sp.predict(sp_X_test)
print(classification_report(wv_y_test, forest_sp_matrix, target_names=['healthy', 'parkison']))

#################################################################################


##################     Model: Best Two Performers using Training  Test Data  #########################

# Best Performing models in terms of accuracy, precision and recall in terms of training and test data
# 1. Random Forest
# 2. Decision Tree

# **********  Save Best Performaing  Model ************************************

# ******** Uncomment this session to overwrite the models already created ****
#model_random_forest_wave = "saved_model\\random_forest_model_wave.pkl"
#model_random_forest_spiral  = "saved_model\\random_forest_model_spiral.pkl"
#
#
#model_tree_wave = "saved_model\\decision_tree_model_wave.pkl"
#model_tree_spiral  = "saved_model\\decision_tree_model_spiral.pkl"
#
## *** Save Model ******************
#joblib.dump(forest_wv, model_random_forest_wave)
#joblib.dump(forest_sp, model_random_forest_spiral)    
#joblib.dump(tree_wv, model_tree_wave)    
#joblib.dump(tree_sp, model_tree_spiral)

# *************** End *********************************************************\
    
## ** Load Saved Model from File *********************
# Load from file - from saved_model folder

from_file_forest_wv, from_file_forest_sp = None, None
from_file_tree_wv, from_file_tree_sp = None, None

from_file_forest_wv= joblib.load(model_random_forest_wave)
from_file_forest_sp = joblib.load(model_random_forest_spiral)    

from_file_tree_wv = joblib.load(model_tree_wave)    
from_file_tree_sp = joblib.load(model_tree_spiral)  


###  *************** TEST MODELS WITH NEW DATA *********************************   
# Actual Test Data - How well does model perform on New Data That was never exposed to model
wv_newdata_path = "parkinsons-drawings\\wave"
sp_newdata_path = "parkinsons-drawings\\spiral"

# standardize images
# - wave_height, wave_width = 258, 512
# - spiral_height, spiral_width = 256, 256  
     
# wave images
nw_wv_train_healthy = transform_image(get_image_path(wv_newdata_path, 'testing', 'yes'), 258, 512)
nw_wv_train_park  = transform_image(get_image_path(wv_newdata_path, 'testing', 'no'),258, 512)
nw_wv_train_healthy = np.array(nw_wv_train_healthy)
nw_wv_train_park = np.array(nw_wv_train_park)


# spiral images
nw_sp_train_healthy = transform_image(get_image_path(sp_newdata_path, 'testing', 'yes'), 256, 256)
nw_sp_train_park = transform_image(get_image_path(sp_newdata_path, 'testing', 'no'), 256, 256)
nw_sp_train_healthy = np.array(nw_sp_train_healthy)
nw_sp_train_park = np.array(nw_sp_train_park)

new_data_y1_wv = np.repeat(0, len(nw_wv_train_healthy))
new_data_y2_wv = np.repeat(1, len(nw_wv_train_park))
new_data_y_wv = np.append(new_data_y1_wv, new_data_y2_wv)

new_data_y1_sp = np.repeat(0, len(nw_sp_train_healthy))
new_data_y2_sp = np.repeat(1, len(nw_sp_train_park))
new_data_y_sp = np.append(new_data_y1_sp, new_data_y2_sp)



# Ramdom Forest
wv_hlthy_forest_predict =  from_file_forest_wv.predict(nw_wv_train_healthy)
wv_park_forest_predict =  from_file_forest_wv.predict(nw_wv_train_park)

new_data_prediction = np.append(wv_hlthy_forest_predict, wv_park_forest_predict)
print ("\nClassification Report for New Data(Wave) - Random Forest")
print(classification_report(new_data_y_wv, new_data_prediction, target_names=['healthy', 'parkison']))

# Support Vector Machine
wv_hlthy_tree_predict =  from_file_tree_wv.predict(nw_wv_train_healthy)
wv_park_tree_predict =  from_file_tree_wv.predict(nw_wv_train_park)
new_data_prediction_svm = np.append(wv_hlthy_tree_predict, wv_park_tree_predict)
print ("\nClassification Report for New Data (Wave) - Decision Treee")
print(classification_report(new_data_y_wv, new_data_prediction_svm, target_names=['healthy', 'parkison']))


sp_hlthy_forest_predict =  from_file_forest_sp.predict(nw_sp_train_healthy)
sp_park_forest_predict =  from_file_forest_sp.predict(nw_sp_train_park)
new_data_prediction_sp_forest = np.append(sp_hlthy_forest_predict, sp_park_forest_predict)
print ("\nClassification Report for New Data(Spiral) - Random Forest")
print(classification_report(new_data_y_sp, new_data_prediction_sp_forest, target_names=['healthy', 'parkison']))

sp_hlthy_tree_predict =  from_file_tree_sp.predict(nw_sp_train_healthy)
sp_park_tree_predict =  from_file_tree_sp.predict(nw_sp_train_park)
new_data_prediction_sp_tree = np.append(sp_hlthy_tree_predict, sp_park_tree_predict)
print ("\nClassification Report for New Data(Spiral) - Decision Tree")
print(classification_report(new_data_y_sp, new_data_prediction_sp_tree, target_names=['healthy', 'parkison']))
























































































































































































