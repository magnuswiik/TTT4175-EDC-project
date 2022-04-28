import numpy as np
from math import sqrt
from tensorflow import keras
from keras.datasets import mnist
import functions

# Load data
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Choose test size
test_size = 10
test_collection = test_x[0:test_size]
targets = test_y[0:test_size]

# Choose appropriate memory size
memory_size = 1000
memory_data = train_x[0:memory_size]
memory_target = train_y[0:memory_size]
predictions = np.zeros(test_size)


def main():
    # Make predictions
    iter = 0
    for test_pic in test_collection:
        prediction = functions.KNN(memory_data, memory_target, test_pic, 10)
        predictions[iter] += prediction
        iter += 1

    # Make confusion matrix and calculate error rate
    confusionMatrix = functions.confusionMatrix(predictions, targets)
    errorRate = functions.errorRate(confusionMatrix, test_size)

    # Print results
    print("Confusion matrix: \n", confusionMatrix)
    print("Error rate: ", errorRate)

main()

print("Digits:)")