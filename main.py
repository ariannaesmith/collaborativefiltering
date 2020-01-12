#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:58:03 2019

@author: ariannasmith
"""

import numpy as np




def buildM(movies, data, moviesList = []):    
    
    # Import movie titles
    with open(movies, 'r', errors = 'ignore') as f:
        lines = f.readlines()
    f.close()
    
    actualMovies = moviesList
    # Build dictionary of each user ID and values of movie ID, rating
    userDict = {}
    
    with open(data, 'r', errors = 'ignore') as f2:
        linesTrain = f2.readlines()
        userCount = 0
        
        for lines2 in linesTrain:
            s = lines2.split(',')
            movieID = int(s[0]) - 1
            
            if data == "TrainingRatings.txt":
                if movieID not in actualMovies:
                    actualMovies.append(movieID)
                    
            userID = int(s[1])
            rating = float(s[2].rstrip('\n'))
            
            if userID not in userDict.keys():
                userDict.setdefault(userID, [])
                userDict[userID].append(userCount)
                userCount += 1
                
            userDict[userID].append((movieID, rating))
        
    f2.close()
    

    # Initialize matrix of zeros of size user ID count x movie count + 1 
    matrix = np.zeros((len(userDict), len(lines) + 1))

    # Update matrix to reflect user ratings of their rated movies
    # For each key
    keyCount = 0
    for key in userDict.keys():
        matrix[keyCount][0] = key
        # For each value
        for v in range(1, len(userDict[key])):
            mID = userDict[key][v][0]
            rtg = userDict[key][v][1]
            matrix[keyCount, (mID)] = rtg           
        keyCount += 1
        

    # Resize matrix so that it only contains movies in training set
    newMatrix = np.zeros((len(userDict), len(actualMovies) + 1))
    
    newMatrix[:, 0] = matrix[:, 0]
    featureCount = 1
    for j in range(1, matrix.shape[1]):    
        if j in actualMovies:
            newMatrix[:, featureCount] = matrix[:, j]
            featureCount += 1
 

    
    
    
    return newMatrix, actualMovies, userDict
