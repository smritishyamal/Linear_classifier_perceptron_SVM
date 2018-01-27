#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 22:26:00 2017

@author: Smriti
"""
from __future__ import division
from numpy import *
import time

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        print self.w_
        print 'initial printed'
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            print self.w_
        print 'perceptron done'    
        return self

    def net_input(self, X):
        """Calculate net input"""
        return dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return where(self.net_input(X) >= 0.0, 1, -1)

rawTrainingData = loadtxt('abalone.data.txt', delimiter = ',')
# Seven attributes, 1323 examples
X = rawTrainingData[:,0:7]
# Two classes, dennoted by 1 and -1
y = rawTrainingData[:,7]

per1 = Perceptron(eta=0.1, n_iter=3)

startTime = time.time()
# Use the first 801 data points to train
trainedP = per1.fit(X[:801,:],y[:801])
elapsedTime = time.time() - startTime

# We predict on the test data [rest of the data points]
yactual = y[801:]
ypredict = per1.predict(X[801:,:])

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