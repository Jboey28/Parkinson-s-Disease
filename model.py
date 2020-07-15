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
import tensorflow as tf


from random import shuffle
from tqdm import tqdm
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
        img_flattened = img.flatten()
        image_list.append(img_flattened)
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

# ********************* ***  WAVE *******************************************
# k-Nearest Neighbor

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10.
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(wv_X_train, wv_y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(wv_X_train, wv_y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(wv_X_test, wv_y_test))
    
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.title('WAVE - K-NEAREST NEIGHBOR')
plt.legend()
plt.savefig('plots\WAVE - K-NEAREST NEIGHBOR.png')
plt.show()



clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(wv_X_train, wv_y_train)
wv_pred = clf.predict(wv_X_train)
Knn_wv_matrix = clf.predict(wv_X_test)
wv_metrics = (metrics.accuracy_score(wv_y_train, wv_pred))
print(wv_metrics)
confusion = confusion_matrix(wv_y_test, Knn_wv_matrix)

print("Confusion matrix for KNN on wave:\n{}".format(confusion))

print(classification_report(wv_y_test, Knn_wv_matrix, target_names=['healthy', 'parkison']))

# ********************* ***  SPIRAL *******************************************
# k-Nearest Neighbor

training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10.
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(sp_X_train, sp_y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(sp_X_train, sp_y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(sp_X_test, sp_y_test))
    
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.title('SPIRAL - K-NEAREST NEIGHBOR')
plt.legend()
plt.savefig('plots\SPIRAL - K-NEAREST NEIGHBOR.png')
plt.show()


clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(sp_X_train, sp_y_train)
sp_pred = clf.predict(sp_X_train)
Knn_sp_matrix = clf.predict(sp_X_test)
sp_metrics = metrics.accuracy_score(sp_y_train, sp_pred)
print(sp_metrics)

confusion = confusion_matrix(sp_y_test, Knn_sp_matrix)
print("Confusion matrix for KNN on spiral:\n{}".format(confusion))

print(classification_report(sp_y_test, Knn_sp_matrix, target_names=['healthy', 'parkison']))







# ********************* ***  WAVE *******************************************
# Logistic Regression


logreg = LogisticRegression(C=10).fit(wv_X_train, wv_y_train)
logreg_wv_matrix = logreg.predict(wv_X_test)
print("Training set score for Logistic Regression wave: {:.3f}".format(logreg.score(wv_X_train, wv_y_train)))
print("Test set score for Logistic Regression wave: {:.3f}".format(logreg.score(wv_X_test, wv_y_test)))
confusion = confusion_matrix(wv_y_test, logreg_wv_matrix)
print("Confusion matrix for Logistic Regression on wave:\n{}".format(confusion))

print(classification_report(wv_y_test, logreg_wv_matrix, target_names=['healthy', 'parkison']))


#tranformed(scaled)  X value
scaler= MinMaxScaler()
scaler.fit(wv_X_train)

wv_X_train_transformed = scaler.transform(wv_X_train)
wv_X_test_transformed = scaler.transform(wv_X_test)

logreg_scaled = LogisticRegression(C=0.1,).fit(wv_X_train_transformed, wv_y_train)
logreg_wv_scaled_matrix = logreg_scaled.predict(wv_X_test_transformed)
print("Training set score for Transformed Logistic Regression wave: {:.3f}".format(logreg.score(wv_X_train_transformed, wv_y_train)))
print("Test set score for Transformed Logistic Regression wave: {:.3f}".format(logreg.score(wv_X_test_transformed, wv_y_test)))
confusion = confusion_matrix(wv_y_test, logreg_wv_matrix)
print("Confusion matrix for Transformed Logistic Regression on wave:\n{}".format(confusion))

print(classification_report(wv_y_test, logreg_wv_scaled_matrix, target_names=['healthy', 'parkison']))


# ********************* ***  SPIRAL *******************************************
# Logistic Regression

logreg_sp = LogisticRegression().fit(sp_X_train, sp_y_train)
logreg_sp_matrix = logreg_sp.predict(sp_X_test)
print("Training set score for Logistic Regression spiral: {:.3f}".format(\
      logreg_sp.score(sp_X_train, sp_y_train)))
print("Test set score for Logistic Regression spiral: {:.3f}".format(\
      logreg_sp.score(sp_X_test, sp_y_test)))
confusion = confusion_matrix(sp_y_test, logreg_sp_matrix)
print("Confusion matrix for Logistic Regression on spiral:\n{}".format(confusion))

print(classification_report(sp_y_test, logreg_sp_matrix, target_names=['healthy', 'parkison']))

#tranformed(scaled)  X value
scaler_sp= MinMaxScaler()
scaler_sp.fit(sp_X_train)

sp_X_train_transformed = scaler_sp.transform(sp_X_train)
sp_X_test_transformed = scaler_sp.transform(sp_X_test)

logreg_sp = LogisticRegression().fit(sp_X_train_transformed, sp_y_train)
logreg_sp_scaled_matrix = logreg_sp.predict(sp_X_test_transformed)

print("Training set score for Transformed Logistic Regression spiral: {:.3f}".format(\
      logreg_sp.score(sp_X_train_transformed, sp_y_train)))
print("Test set score for Transformed Logistic Regression spiral: {:.3f}".format(\
      logreg_sp.score(sp_X_test_transformed, sp_y_test)))
confusion = confusion_matrix(sp_y_test, logreg_sp_scaled_matrix)
print("Confusion matrix for Transformed Logistic Regression on spiral:\n{}".format(confusion))

print(classification_report(sp_y_test, logreg_sp_scaled_matrix, target_names=['healthy', 'parkison']))




# ********************* *** WAVE *******************************************
# Support Vector Classifier

svc = SVC(kernel = 'linear', C=10, gamma = 0.1)
svc.fit(wv_X_train, wv_y_train)
svc_wv_matrix = svc.predict(wv_X_test)
print("Accuracy on training set for Support Vector Classifier wave: {:.2f}".format(svc.score(wv_X_train, wv_y_train)))
print("Accuracy on test set for Support Vector Classifier wave: {:.2f}".format(svc.score(wv_X_test, wv_y_test)))
confusion = confusion_matrix(wv_y_test, svc_wv_matrix)
print("Confusion matrix for Support Vector Classifier on wave:\n{}".format(confusion))

print(classification_report(wv_y_test, svc_wv_matrix, target_names=['healthy', 'parkison']))



# transformed(scaled X)
svc = SVC(kernel = 'linear', C=10, gamma = 0.1)
svc.fit(wv_X_train_transformed, wv_y_train)
svc_wv_matrix_transformed = svc.predict(wv_X_test_transformed)
print("Accuracy on training set for Transformed Support Vector Classifier wave: {:.2f}".format(\
      svc.score(wv_X_train_transformed, wv_y_train)))
print("Accuracy on test set for Transformed Support Vector Classifier wave: {:.2f}".format(\
      svc.score(wv_X_test_transformed, wv_y_test)))
confusion = confusion_matrix(wv_y_test, svc_wv_matrix_transformed)
print("Confusion matrix for Transformed Support Vector Classifier on wave:\n{}".format(confusion))

print(classification_report(wv_y_test, svc_wv_matrix_transformed, target_names=['healthy', 'parkison']))




# ********************* ***  SPIRAL *******************************************
# Support Vector Classifier

svc = SVC(kernel = 'linear', C=10, gamma = 0.1)
svc.fit(sp_X_train, sp_y_train)
svc_matrix = svc.predict(sp_X_test)
print("Accuracy on training set for Support Vector Classifier spiral: {:.2f}".format(svc.score(sp_X_train, sp_y_train)))
print("Accuracy on test set for Support Vector Classifier spiral: {:.2f}".format(svc.score(sp_X_test, sp_y_test)))
confusion = confusion_matrix(sp_y_test, svc_matrix)
print("Confusion matrix for Support Vector Classifier on spiral:\n{}".format(confusion))

print(classification_report(sp_y_test, svc_matrix, target_names=['healthy', 'parkison']))

# transformed(scaled X)
svc.fit(sp_X_train_transformed, sp_y_train)
svc_matrix_transformed = svc.predict(sp_X_test_transformed)
print("Accuracy on training set for Transformed Support Vector Classifier spiral: {:.2f}".format(\
      svc.score(sp_X_train_transformed, sp_y_train)))
print("Accuracy on test set for Transformed Support Vector Classifier spiral: {:.2f}".format(\
      svc.score(sp_X_test_transformed, sp_y_test)))
confusion = confusion_matrix(sp_y_test, svc_matrix_transformed)
print("Confusion matrix for Transformed Support Vector Classifier on spiral:\n{}".format(confusion))

print(classification_report(sp_y_test, svc_matrix_transformed, target_names=['healthy', 'parkison']))




# ********************* *** WAVE *******************************************
# Decision tree Classifier


tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(wv_X_train, wv_y_train)
tree_wv_matrix = tree.predict(wv_X_test)
print("Accuracy on training set for Decision Tree Classifier wave: {:.3f}".format(tree.score(wv_X_train, wv_y_train)))
print("Accuracy on test set for Decision Tree Classifier wave: {:.3f}".format(tree.score(wv_X_test, wv_y_test)))
confusion = confusion_matrix(wv_y_test, tree_wv_matrix)
print("Confusion matrix for Decision Tree Classifier wave:\n{}".format(confusion))

print(classification_report(wv_y_test, tree_wv_matrix, target_names=['healthy', 'parkison']))



# ********************* *** SPIRAL *******************************************
# Decision tree Classifier


tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(sp_X_train, sp_y_train)
tree_sp_matrix = tree.predict(sp_X_test)
print("Accuracy on training set for Decision Tree Classifier spiral: {:.3f}".format(tree.score(sp_X_train, sp_y_train)))
print("Accuracy on test set for Decision Tree Classifier spiral: {:.3f}".format(tree.score(sp_X_test, sp_y_test)))
confusion = confusion_matrix(sp_y_test, tree_sp_matrix)
print("Confusion matrix for Decision Tree Classifier spiral:\n{}".format(confusion))

print(classification_report(sp_y_test, tree_sp_matrix, target_names=['healthy', 'parkison']))






# ********************* *** WAVE *******************************************
# Random Forest Classifier

forest = RandomForestClassifier()
forest.fit(wv_X_train, wv_y_train)
forest_wv_matrix = forest.predict(wv_X_test)
print("Accuracy on training set for Random Forest Classifier wave: {:.3f}".format(forest.score(wv_X_train, wv_y_train)))
print("Accuracy on test set for Random Forest Classifier wave: {:.3f}".format(forest.score(wv_X_test, wv_y_test)))
confusion = confusion_matrix(wv_y_test, forest_wv_matrix)
print("Confusion matrix for Random Forest Classifier wave:\n{}".format(confusion))

print(classification_report(wv_y_test, forest_wv_matrix, target_names=['healthy', 'parkison']))

# ********************* *** Spiral *******************************************
# Random Forest Classifier

forest = RandomForestClassifier()
forest.fit(sp_X_train, sp_y_train)
forest_sp_matrix = forest.predict(sp_X_test)
print("Accuracy on training set for Random Forest Classifier spiral: {:.3f}".format(forest.score(sp_X_train, sp_y_train)))
print("Accuracy on test set for Random Forest Classifier spiral: {:.3f}".format(forest.score(sp_X_test, sp_y_test)))
confusion = confusion_matrix(sp_y_test, forest_sp_matrix)
print("Confusion matrix for Random Forest Classifier spiral:\n{}".format(confusion))

print(classification_report(wv_y_test, forest_sp_matrix, target_names=['healthy', 'parkison']))


# ********************* *** WAVE *******************************************
# Convolutional Neural Network

model = Sequential()
model.add = (Conv2D(64, (3,3), input_shape = wv_X_train.shape[1:]))





















































































































































































