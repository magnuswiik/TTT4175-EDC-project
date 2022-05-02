import numpy as np
import functions as fnc
import datetime as dt
        
def main():

    # Load data
    num_train_chunks = 1
    train_chunk = 0
    num_test_img = 100
    train_data, train_targets, test_data, test_targets = fnc.loadData(num_train_chunks, train_chunk, num_test_img)

    test_size = len(test_targets)
    predictions = np.zeros(test_size)

    # Number of neighbors
    k = 7

    startTime = dt.datetime.now().replace(microsecond=0)

    predictions = fnc.KNN(train_data, train_targets, test_data, k)




    # Make confusion matrix and calculate error rate
    confusionMatrix = fnc.confusionMatrix(predictions, test_targets)
    errorRate = fnc.errorRate(confusionMatrix, test_size)

    endTime = dt.datetime.now().replace(microsecond=0)
    # Print results
    print("Confusion matrix: \n", confusionMatrix)
    print("Error rate: ", errorRate)
    print("Classification time: ", endTime-startTime)


main()

print("Digits:)")