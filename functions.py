#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 21:31:44 2019

@author: ariannasmith
"""
import numpy as np
import math 
import model as m

def meanVote(matrix):
    meanList = np.zeros(matrix.shape[0])
    
    for user in range(matrix.shape[0]):
        voteSum = sum(matrix[user, 1:])
        Ii = np.count_nonzero(matrix[user, 1:])
        meanUser = float(voteSum) / float(Ii)
        meanList[user] = meanUser

    return meanList
    

def predictVote(avgTrain, trainM, testM, userDict):
    predictM = np.zeros((testM.shape[0], testM.shape[1]))
    
    # Create vote location, vote value, and  num interaction count arrays
    allVoteLocations = np.full((trainM.shape[0], 2000), -1)
    allVoteValues = np.zeros((trainM.shape[0], 2000))
    maxInteractions = np.zeros((trainM.shape[0]))    

    for a in range(trainM.shape[0]):
        
        aRow = trainM[a, :]
        
        aVoteLocation = np.nonzero(aRow)
        aVotes = aRow[np.nonzero(aRow)]
                
        flatLocation = [val for sublist in aVoteLocation for val in sublist]
    
        for x in range(len(flatLocation)):
            allVoteLocations[a, x] = flatLocation[x]
            allVoteValues[a, x] = aVotes[x]
    
    
    print("done making value and location matrices")

    
    # Make n x n size matrix for weights, initialized to a value that 
    # won't be calculated
    weightsM = np.full((trainM.shape[0], trainM.shape[0]), -100.0)
    normWeights = 0  
    
    predictM[:, 0] = testM[:, 0]
    # For every testing example
    for a in range(testM.shape[0]):
        print("on row", a, "of testing")
        
        userID = int(testM[a, 0])
        userIndex = userDict[userID][0]
        
        # List of features to predict
        aRow = testM[a, :]
        aVotesL = np.nonzero(aRow)  
        aAvg = avgTrain[userIndex]
        
        # This row is the userID being predicted in the training set
        
        thisVotes = allVoteValues[userIndex, 1:]
        thisUserLocations = allVoteLocations[userIndex, 1:]

        # For every feature
        for k in aVotesL:
            for j in range(1, len(k)):
                j = k[j]
                actual = aRow[j]
                
                summation = 0.0
                normWeights = 0.0
                
                for i in range(trainM.shape[0]):
                    if trainM[i, j] != 0:
                        # If weight hasn't been computed yet
                        w = weightsM[userIndex, i]
                        iAvg = avgTrain[i]
                        
                        if  w == -100.0:
        
                            thatVotes = allVoteValues[i, 1:]
                            thatUserLocations = allVoteLocations[i, 1:]
                            
                            intersect = np.intersect1d(thisUserLocations, thatUserLocations)
                            
                            if intersect.size > 1:
                                w = weights(iAvg, aAvg, thisVotes[intersect], thatVotes[intersect])
                            else:
                                w = 0.0
                            weightsM[userIndex, i] = weightsM[i, userIndex] = w
                            
        
                        # If weight isn't 0
                        elif w != 0.0:
                            normWeights += abs(w)
                            summation += (w * (trainM[i, j] - iAvg))
                    
                # If all votes of one user were the same
                if normWeights == 0.0:
                    normWeights = 0.000001
                
                predict = (aAvg + summation / normWeights)
                predictM[a, j] = predict

    
    return predictM

def weights(iAvg, aAvg, aRow, iRow):
    # Calculate numerator
    num = 0
    den1 = 0
    den2 = 0
    
    # Skip first as it will be -1
    for j in range(1, len(aRow)):
        aj = aRow[j]
        ij = iRow[j]
        num += (aj - aAvg) * (ij - iAvg)
        den1 += (aj - aAvg)**2 
        den2 += (ij - iAvg)**2
            
    den = math.sqrt(den1 * den2)
    # If either user had the same value for all ratings
    if den == 0.0:
        weight = 0
    else:
        weight = num / den
    
    return weight

