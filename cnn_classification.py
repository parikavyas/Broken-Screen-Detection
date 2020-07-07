#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 10:28:14 2019

@author: Parika
"""

class BrokenPhoneDetection:
   #Importing all the required libraries .i.e.whose functions will be called
    import os 
    from skimage import io
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.utils import np_utils
    from keras.layers import Flatten
    from keras.layers.convolutional import Convolution2D
    from keras.layers.convolutional import MaxPooling2D
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    
    #Load all the data
    #First Importing all the broken images.
    root_dir=os.getcwd() #get current directory
    #joining the current directory path and path of folder containing 
    #images so that images can be accessed
    img_dir=os.path.join(root_dir,"/Users/Parika/Desktop/MICROTEK/icici1/DATA_0/")
    max_r=0 #variable for storing maximum no. of rows (height)
    max_c=0 #variable for storing maximum no. of columns(width)
    img0=[]#list declared for loading images
    img0_tag=[] #list declared for storing the tags (class) of the individual images
    for img_file in sorted(os.listdir(img_dir)):
        #loop through the files in the folder specified by the joint path. 
        #Looping through images expecting name of file to be integer and hence
        #looping in a order low to high using sorted 
        if not img_file.startswith('.'):
            #dealing with backup files (automatically generated in mac.)
            im=io.imread(os.path.join(img_dir,img_file))
            #reading image with name img_file in img_dir 
            if max_r<im.shape[0]:
                max_r=im.shape[0]
                #checking for max no. of rows
            if max_c<im.shape[1]:
                max_c=im.shape[1]
                    #checking for max no. of columns
            img0.append(im)
                    #appending image array to the list
            img0_tag.append(0)
                    #appending image ag tarray to the list
    print("max size in img 0",max_r,max_c)
    print("size of data 0",len(img0))
    
    #similarly for not broken images.
    img_dir=os.path.join(root_dir,"/Users/Parika/Desktop/MICROTEK/icici1/DATA_1/")
    img1=[]
    img1_tag=[]
    for img_file in sorted(os.listdir(img_dir)):
        if not img_file.startswith('.'):
            im=io.imread(os.path.join(img_dir,img_file))
            if max_r<im.shape[0]:
                max_r=im.shape[0]
            if max_c<im.shape[1]:
                max_c=im.shape[1]
            img1.append(im)
            img1_tag.append(1)
    print("max size in img 1",max_r,max_c)
    print("size of data 0",len(img1))
    
    #making a common list i.e. array of images by extending list for both images and their tags.
    img_data = img0
    img_data.extend(img1)
    
    img_tags = img0_tag
    img_tags.extend(img1_tag)
    np.array(img_tags)
    
    print("length of image data",len(img_data))
    print("length of image tags",len(img_tags))
    
    # shape to be kept [samples][channels][width][height]
    img_data_reshape=np.empty([0,max_r,max_c,3])
    for x in img_data:
        #    print("**",x.shape)
        p_r = max_r - x.shape[0]
        p_c = max_c - x.shape[1]
        #padding zeros or white areas around the image so that all 
        #images can be loaded into a numpy ndarray without worrying about
        #different dimensions.
        x=np.pad(x, [(p_r,0),(p_c,0),(0,0)], 'constant', constant_values=(0))
        #    print("**",x.shape)
        x =x.reshape(1,x.shape[0],x.shape[1],x.shape[2])
        img_data_reshape =np.append(img_data_reshape,x,axis=0)
    print("reshaped array size", img_data_reshape.shape)

#define a train and test set for the model.
    X_train=img_data_reshape
    X_test=img_data_reshape[[1,3,5,7,15,20,24,28,32,38,41,46,51,55,60,63,69,72],:]
    y_train=img_tags
    y_test=list( img_tags[i] for i in [1,3,5,7,15,20,24,28,32,38,41,46,51,55,60,63,69,72] )

#creating a train and test set in a way model will read and predict. 
#The model will need variables as many as categories.
    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)    
    
    def baseline_model():
    # create model with the different layers
        model = Sequential()
        
        model.add(Convolution2D(32, (7, 7), input_shape=(max_r,max_c,3),activation='relu'))
        model.add(Convolution2D(32, (5, 5), activation='relu',data_format='channels_last'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Convolution2D(16, (7, 7),activation='relu',data_format='channels_last'))
        model.add(Convolution2D(16, (5, 5), activation='relu',data_format='channels_last'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Convolution2D(32, (3, 3),activation='relu',data_format='channels_last'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Convolution2D(16, (3, 3), activation='relu',data_format='channels_last'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(84, activation='relu'))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        # Compile model and defining the methods for calculating loss, optimising model and calculating metrics for returning
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # build the model
    model = baseline_model()
    # Fit the model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=20,verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    
import pickle

#serializing our model to a file called model.pkl
pickle.dump(regr, open("model.pkl","wb"))

#loading a model from a file called model.pkl
model = pickle.load(open("model.pkl","r"))  
    
    