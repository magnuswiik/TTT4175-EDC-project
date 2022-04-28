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

def euc(img1, img2 ):
    img1 = np.reshape(img1, (img1.size,1))
    img2 = np.reshape(img2, (img2.size,1))
    dif = np.array(img1-img2)
    distance = np.matmul(dif.T, dif)
    return distance

def nearestNeighbors(train, train_targets, test_pic, num_neighbors):
    distances = [[0.]*2 for i in range(train.shape[0])]
    for j in range(train.shape[0]): 
        #flatImg = flattenImg(train[j])
        #flatTestImg = flattenImg(test_pic)
        
        distances[j][0] = train_targets[j]
        #distances[j][1] = np.linalg.norm(flatImg-flatTestImg)
        distances[j][1] = euc(train[j], test_pic) #np.linalg.norm(train[j]-test_pic)

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
