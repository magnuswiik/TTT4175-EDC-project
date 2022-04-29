import numpy as np
from math import sqrt
from tensorflow import keras
from keras.datasets import mnist
import functions as fnc
import datetime as dt

# Load data
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Choose test size
test_size = 1000
test_collection = test_x[0:test_size]
test_targets = test_y[0:test_size]

# Choose appropriate memory size
memory_size = 60000
memory_data = train_x[0:memory_size]
memory_targets = train_y[0:memory_size]
predictions = np.zeros(test_size)

# Number of neighbors
num_neighbors = 10


def main():
    # Make predictions
    iter = 0
    startTime = dt.datetime.now().replace(microsecond=0)

    guesses = np.zeros(test_size)

    clusters = fnc.clustering(memory_data, memory_targets, 64)
    for i in range(test_size-1):
        prediction_cluster = fnc.KNNClustering(clusters, test_collection[i], 64)
        guesses[i] = prediction_cluster

    failures = np.zeros((test_size,len(train_x[0][0]),len(train_x[0][0])))
    failed_predictions = np.zeros(test_size)
    failed_targets = np.zeros(test_size)
    contenders = np.zeros((test_size,num_neighbors))


    fail = 0
    for i in range(test_size-1):  
        test_pic = test_collection[i]
        test_target = test_targets[i]
        neighbors = fnc.nearestNeighbors(memory_data, memory_targets, test_pic, num_neighbors)
        prediction = fnc.KNN(neighbors)
        predictions[iter] += prediction
        if prediction != test_target:
            failures[fail] = test_pic
            failed_predictions[fail] = prediction
            failed_targets[fail] = test_target
            contenders[fail] = neighbors
            fail += 1
            print("fuck ",fail)
        iter += 1



    # Make confusion matrix and calculate error rate
    confusionMatrix = fnc.confusionMatrix(predictions, test_targets)
    errorRate = fnc.errorRate(confusionMatrix, test_size)

    confusionMatrix = fnc.confusionMatrix(predictions, test_targets)
    errorRate = fnc.errorRate(confusionMatrix, test_size)

    endTime = dt.datetime.now().replace(microsecond=0)
    # Print results
    print("Confusion matrix: \n", confusionMatrix)
    print("Error rate: ", errorRate)
    print("Classification time: ", endTime-startTime)

    fnc.plotImg(failures, failed_predictions, failed_targets, contenders)

main()

print("Digits:)")