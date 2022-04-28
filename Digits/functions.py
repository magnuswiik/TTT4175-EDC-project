import numpy as np
from math import sqrt
from statistics import mode
from tensorflow import keras
from keras.datasets import mnist

# Clustering
def countEachDigit(target):
    numberOfEach = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(target)):
        numberOfEach[target[i]] += 1
    return numberOfEach

def euclideanDistance(row1, row2):
    distance = 0.0
    for i in range (len(row1)-1):
        distance += (row1[i] - row2[i])**2 # TODO: use matrix multiplication instead for faster runtime
    return sqrt(distance)

def nearestNeighbors(train, train_targets, test_pic, num_neighbors):
    distances = list()
    for j in range(train.shape[0]):
        distance = 0
        for i in range(train[j].shape[0]):
            distance_row = euclideanDistance(train[j][i], test_pic[i])
            distance += distance_row
        
        distances.append((train_targets[j], distance))
        distances.sort(key=lambda tup: tup[1])
    return distances[0:num_neighbors]

def KNN(train, train_targets, test_pic, num_neighbors):
    neighbors = nearestNeighbors(train, train_targets, test_pic, num_neighbors)
    neighbors_id = [row[0] for row in neighbors]
    prediction = mode(neighbors_id)
    return prediction

def confusionMatrix(predictions, targets):
    confusionMatrix = np.zeros((10,10))
    count = len(predictions)
    for i in range(count):
        pred = int(predictions[i])
        targ = int(targets[i])
        confusionMatrix[targ][pred] += 1

    return confusionMatrix

def errorRate(confusionMatrix, count):
    fails = 0
    for i in range(confusionMatrix.shape[0]):
        for j in range(confusionMatrix.shape[1]):
            if i != j:
                fails += confusionMatrix[i][j]
    return np.round((fails/count)*100,2)
