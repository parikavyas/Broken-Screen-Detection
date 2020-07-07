#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:58:18 2020

@author: Parika
"""
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
#import h5py
#import os 
#from skimage import io
#from PIL import Image 
import numpy as np
import cv2
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
sklearn.model_selection.train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score








num_trees = 100
test_size = 0.10
seed      = 9

images = {}
labels = []
bins = 8

img_dir=("/Users/Parika/Desktop/MICROTEK/Data/Broken")
for img_file in os.listdir(img_dir):
    img = cv2.imread(os.path.join(img_dir,img_file))
    image_col = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image_col)).flatten()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    img_col = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([img_col], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    final_hist = hist.flatten()
    global_feature = feature.tolist()+haralick.tolist()+final_hist.tolist()
    images[img_file.split('.')[0]] = global_feature
    labels.append(0)
    
img_dir=("/Users/Parika/Desktop/MICROTEK/Data/Normal")
for img_file in os.listdir(img_dir):
    img = cv2.imread(os.path.join(img_dir,img_file))
    image_col = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image_col)).flatten()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    img_col = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([img_col], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    final_hist = hist.flatten()
    global_feature = feature.tolist()+haralick.tolist()+final_hist.tolist()
    images[img_file.split('.')[0]] = global_feature
    labels.append(1)
    
x = pd.DataFrame.from_dict(images, orient = 'index')
x_global=np.array(x)
y_global=np.array(labels) 

# split the training and testing data
#(trainData, testData, trainLabels, testLabels) = train_test_split(x_global, y_global,test_size=test_size,random_state=seed)
scoring = "accuracy"
models = []
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=seed)))

results = {}
x_train, x_test, y_train, y_test = train_test_split(x_global, y_global, test_size=0.2, random_state=42)

for name, model in models:
    #kfold = KFold(n_splits=10, random_state=seed)
    #cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    clf = model.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    clf_results = [precision_score(y_test, y_pred),recall_score(y_test, y_pred),f1_score(y_test, y_pred),accuracy_score(y_test, y_pred)]
    results[name] = clf_results
    
model_metrics = pd.DataFrame.from_dict(results, orient='index')
model_metrics = model_metrics.rename(columns = {0:'Precision', 1:'Recall', 2:'F1-score', 3:'Accuracy'})
model_metrics.to_csv("/Users/Parika/Desktop/MICROTEK/Model_Metrics.csv")


