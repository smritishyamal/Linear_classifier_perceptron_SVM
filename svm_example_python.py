#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  12 23:45:00 2017

@author: Smriti
"""
from __future__ import division
from numpy import *
from sklearn import svm
import time


rawTrainingData = loadtxt('abalone.data.txt', delimiter = ',')
# Seven attributes, 1323 examples
X = rawTrainingData[:,0:7]
# Two classes, dennoted by 1 and -1
y = rawTrainingData[:,7]

# We have used linear support vector machine for classification problem
svmtrain = svm.SVC(C=1, kernel = 'linear', shrinking = False)

startTime = time.time()
# Use the first 801 data points to classify
svmtrain.fit(X[:801,:],y[:801])

elapsedTime = time.time() - startTime

print svmtrain.support_vectors_

# We predict on the test data [rest of the data points]
ypredict = svmtrain.predict(X[801:,:])
yactual = y[801:]

# Confusion matrix
# initialize the matrix
cm = [[0,0],[0,0]]
cm_p = [[0,0],[0,0]]

for i in range(len(ypredict)):
    if yactual[i] == 1:
        if ypredict[i] == 1:
            cm[0][1] = cm[0][1] + 1
        else:
            cm[1][1] = cm[1][1] + 1
    else:
        if ypredict[i] == -1:
            cm[1][0] = cm[1][0] + 1
        else:
            cm[0][0] = cm[0][0] + 1

cm_p[0][0] = ((cm[0][0])/len(ypredict))*100 
cm_p[0][1] = ((cm[0][1])/len(ypredict))*100 
cm_p[1][0] = ((cm[1][0])/len(ypredict))*100 
cm_p[1][1] = ((cm[1][1])/len(ypredict))*100

print 'Comfusion matrix (in percent)'
print cm_p
print 'Time to train'
print elapsedTime
print 'Done'