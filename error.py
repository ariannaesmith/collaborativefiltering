#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 00:56:33 2019

@author: ariannasmith
"""
import model as m
import math 
import functions as fx

def mae(prediction, actual):
    n = 0
    summation = 0
    for i in range(prediction.shape[0]):    
        for j in range(1, prediction.shape[1]):
            if actual[i, j] != 0:
                summation += abs(actual[i, j] - prediction[i, j])
                n += 1
    error = round((1/n) * summation, 3)
    print("Mean average error:", error)

    
def rmse(prediction, actual):
    n = 0
    summation = 0
    for i in range(prediction.shape[0]):
        for j in range(1, prediction.shape[1]):
            if actual[i, j] != 0:
                summation += (actual[i, j] - prediction[i, j])**2
                n += 1
    error = round(math.sqrt((1/n) * summation), 3)
    print("Root mean squared error:", error)



movies = "movie_titles.txt"
trainingData = "TrainingRatings.txt"
testingData = "TestingRatings.txt"

print("building training")
train = m.buildM(movies, trainingData, moviesList = [])
trainM = train[0]
moviesL = train[1]
userDict = train[2]

print("building testing")
testM = m.buildM(movies, testingData, moviesList = moviesL)[0]


print("making mean list for training")
meanTrain = fx.meanVote(trainM)

print("predicting")
prediction = fx.predictVote(meanTrain, trainM, testM, userDict)

print("performing mae")
mae(prediction, testM)
print("performing rmse")
rmse(prediction, testM)