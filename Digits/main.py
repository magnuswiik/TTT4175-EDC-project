import numpy as np
from math import sqrt
from tensorflow import keras
from keras.datasets import mnist
import functions as fnc
import datetime as dt

# Load data
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Choose test size
test_size = 100
test_collection = test_x[0:test_size]
targets = test_y[0:test_size]

# Choose appropriate memory size
memory_size = 5000
memory_data = train_x[0:memory_size]
memory_target = train_y[0:memory_size]
predictions = np.zeros(test_size)


def main():
    # Make predictions
    iter = 0
    startTime = dt.datetime.now().replace(microsecond=0)
    for test_pic in test_collection:
        prediction = fnc.KNN(memory_data, memory_target, test_pic, 10)
        predictions[iter] += prediction
        iter += 1

    # Make confusion matrix and calculate error rate
    confusionMatrix = fnc.confusionMatrix(predictions, targets)
    errorRate = fnc.errorRate(confusionMatrix, test_size)

    endTime = dt.datetime.now().replace(microsecond=0)
    # Print results
    print("Confusion matrix: \n", confusionMatrix)
    print("Error rate: ", errorRate)
    print("Time used ",endTime-startTime)

main()

print("Digits:)")