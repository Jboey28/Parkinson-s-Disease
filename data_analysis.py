# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:38:31 2020

@author: john oloyede
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cv2
import os


# **** functions **************************************************************
def transform_image(img_name_list,file_path,new_file_path,tx_img_height, tx_img_width):
    '''
      img_name_list = list containing image names
      file_path = original image file path 
      new_file_path = new file path to store transformed image
      tx_img_height = image_height
      tx_img_width = image_width
      return = dataframe with each column containing one image
    '''    
    flattened_img_list = []
    for img_file in img_name_list:
        img_path = os.path.join(file_path, img_file)
        img = cv2.imread(img_path)
        
        #resize image
        img2 = cv2.resize(img,(tx_img_width, tx_img_height))
            
        # change image to gray scale
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        img2_path = os.path.join(new_file_path, img_file)
        cv2.imwrite(img2_path, img2)
        
        # flatten image and append to list
        img2_flattened = img2.flatten()
        flattened_img_list.append(img2_flattened)
    df_imgs = pd.DataFrame(flattened_img_list)
    df_imgs = df_imgs.T
    return df_imgs


def retrieve_images_dimension(img_file_list,image_file_path):
    '''
      return the dimension of the images  
    '''
    image_dimension_list =[]
    for img_file in img_file_list:
        img_path = os.path.join(image_file_path, img_file)
        img = cv2.imread(img_path)
        image_dimension_list.append(img.shape)
    return image_dimension_list
        
        
        
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


def create_file_path(file_path):
    if not os.path.exists(file_path):
       os.makedirs(file_path)
# *****************************************************************************
    
# folder path
       
#original file path
wave_path = 'parkinsons-drawings/wave'
spiral_path = 'parkinsons-drawings/spiral'


#transformed file location
tx_wave_path = 'revised/wave'
tx_spiral_path = 'revised/spiral'


# retrieve wave images
wave_healthy_path = get_image_path(wave_path, 'training', 'yes')
wave_parkinson_path = get_image_path(wave_path, 'training', 'no')

# retrieve spiral images
spiral_healthy_path = get_image_path(spiral_path, 'training', 'yes')
spiral_parkinson_path = get_image_path(spiral_path, 'training', 'no')

# retrieve image names
wave_healthy = os.listdir(wave_healthy_path)
wave_parkinson = os.listdir(wave_parkinson_path)

spiral_healthy = os.listdir(spiral_healthy_path)
spiral_parkinson = os.listdir(spiral_parkinson_path)



# check the image dimensions
healthy_wave_img_dim_list = retrieve_images_dimension(wave_healthy, wave_healthy_path)
parkinson_wave_img_dim_list = retrieve_images_dimension(wave_parkinson, wave_parkinson_path)
print ('Wave Image Dimensions:')
print ('Healthy')
print(healthy_wave_img_dim_list)
print()
print ('Parkinson')
print(parkinson_wave_img_dim_list)

healthy_spiral_img__dim_list = retrieve_images_dimension(spiral_healthy, spiral_healthy_path)
parkinson_spiral_img__dim_list = retrieve_images_dimension(spiral_parkinson, spiral_parkinson_path)
print ('Spiral Image Dimensions:')
print ('Healthy')
print(healthy_spiral_img__dim_list)
print()
print ('Parkinson')
print(parkinson_spiral_img__dim_list)   


# Transform images
## wave images come in different dimension hence standardizing the dimension
wave_height, wave_width = 258, 512
spiral_height, spiral_width = 256, 256

## create new folder to house transformed images
tx_wave_healthy_path = get_image_path(tx_wave_path, 'training', 'yes')
tx_wave_parkinson_path = get_image_path(tx_wave_path, 'training', 'no')

tx_spiral_healthy_path = get_image_path(tx_spiral_path, 'training', 'yes')
tx_spiral_parkinson_path = get_image_path(tx_spiral_path, 'training', 'no')

create_file_path(tx_wave_healthy_path)
create_file_path(tx_wave_parkinson_path)

create_file_path(tx_spiral_healthy_path)
create_file_path(tx_spiral_parkinson_path)

# *** Resize all the images, change into grayscale  and save in resize/  ******     
healthy_wave_img_grey_list_by_col_df = transform_image(wave_healthy, \
                                                  wave_healthy_path, \
                                                  tx_wave_healthy_path, \
                                                  wave_height,\
                                                  wave_width)
parkinson_wave_img_grey_list_by_col_df = transform_image(wave_parkinson, \
                                                  wave_parkinson_path, \
                                                  tx_wave_parkinson_path, \
                                                  wave_height,\
                                                  wave_width)

healthy_spiral_img_grey_list_by_col_df = transform_image(spiral_healthy, \
                                                  spiral_healthy_path, \
                                                  tx_spiral_healthy_path, \
                                                  spiral_height,\
                                                  spiral_width)
parkinson_spiral_img_grey_list_by_col_df = transform_image(spiral_parkinson, \
                                                  spiral_parkinson_path, \
                                                  tx_spiral_parkinson_path, \
                                                  spiral_height,\
                                                  spiral_width)

# transformed dataset
# 1. healthy_wave_img_grey_list_by_col
# 2. parkinson_wave_img_grey_list_by_col
# 3. healthy_spiral_img_grey_list_by_col
# 4. parkinson_spiral_img_grey_list_by_col


# Visualization 

# Box plot 
healthy_wave_box_plot = healthy_wave_img_grey_list_by_col_df.plot(kind='box',\
                                          figsize=(20,15),\
                                          title='Box Plot of Healthy Wave Images',\
                                          grid = True,\
                                          fontsize = 20)
plt.savefig('plots\healthy_wave_box_plot.png')
plt.show()


parkinson_wave_img_grey_list_by_col_df.plot(kind='box',\
                                           figsize=(20,15),\
                                          title='Box Plot of Parkinson Wave Images',\
                                          grid = True,\
                                          fontsize = 20)
plt.savefig('plots\parkinson_wave_box_plot.png')
plt.show()

healthy_spiral_img_grey_list_by_col_df.plot(kind='box',\
                                           figsize=(20,15),\
                                          title='Box Plot of Healthy Spiral Images',\
                                          grid = True,\
                                          fontsize = 20)
plt.savefig('plots\healthy_spiral_box_plot.png')
plt.show()



parkinson_spiral_img_grey_list_by_col_df.plot(kind='box',\
                                            figsize=(20,15),\
                                          title='Box Plot of Parkinson Sprial Images',\
                                          grid = True,\
                                          fontsize = 20)
plt.savefig('plots\parkinson_spiral_box_plot.png')
plt.show()


# Histogram
healthy_wave_img_grey_list_by_col_df.plot(kind='hist',\
                                          facecolor='k',\
                                          figsize=(20,15),\
                                          ax=None,\
                                          sharex=False,\
                                          sharey=True,\
                                          grid = True,\
                                          subplots=True,\
                                          layout= (8,5),\
                                          title='Histogram of Healthy Wave Images')
plt.savefig('plots\healthy_wave_histogram.png')
plt.show()

parkinson_wave_img_grey_list_by_col_df.plot(kind='hist',\
                                          facecolor='k',\
                                          figsize=(20,15),\
                                          ax=None,\
                                          sharex=False,\
                                          sharey=True,\
                                          grid = True,\
                                          subplots=True,\
                                          layout= (8,5),\
                                          title='Histogram of Parkinson Wave Images')
plt.savefig('plots\parkinson_wave_histogram.png')
plt.show()

healthy_spiral_img_grey_list_by_col_df.plot(kind='hist',\
                                          facecolor='k',\
                                          figsize=(20,15),\
                                          ax=None,\
                                          sharex=False,\
                                          sharey=True,\
                                          grid = True,\
                                          subplots=True,\
                                          layout= (8,5),\
                                          title='Histogram of Healthy Spiral Images')
plt.savefig('plots\healthy_spiral_histogram.png')
plt.show()


parkinson_spiral_img_grey_list_by_col_df.plot(kind='hist',\
                                          facecolor='k',\
                                          figsize=(20,15),\
                                          ax=None,\
                                          sharex=False,\
                                          sharey=True,\
                                          grid = True,\
                                          subplots=True,\
                                          layout= (8,5),\
                                          title='Histogram of Parkinson Sprial Images')
plt.savefig('plots\parkinson_spiral_histogram.png')
plt.show()



# compare the box plot of healthy wave versus parkinson wave
healthy_wave  = healthy_wave_img_grey_list_by_col_df.stack()
parkinson_wave = parkinson_wave_img_grey_list_by_col_df.stack()

df_wave = pd.DataFrame({'healthy':healthy_wave, 'parkinson':parkinson_wave})
df_wave_plot = df_wave.plot(kind='box',\
                                          figsize=(20,15),\
                                          title='Box Plot of Healthy versus Parkinson Wave Images',\
                                          grid = True,\
                                          fontsize = 15)
plt.savefig('plots\Healthy_vs_Parkinson_Wave_Box_Plot.png')
plt.show()


# compare the box plot of healthy spiral images versus parkinson wave images
healthy_spiral  = healthy_spiral_img_grey_list_by_col_df.stack()
parkinson_spiral = parkinson_spiral_img_grey_list_by_col_df.stack()

df_wave = pd.DataFrame({'healthy':healthy_spiral, 'parkinson':parkinson_spiral})
df_wave_plot = df_wave.plot(kind='box',\
                                          figsize=(20,15),\
                                          title='Box Plot of Healthy versus Parkinson Spiral Images',\
                                          grid = True,\
                                          fontsize = 20)
plt.savefig('plots\Healthy_vs_Parkinson_Spiral_Box_Plot.png')
plt.show()