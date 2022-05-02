import numpy as np
import functions as fnc
import datetime as dt

class Failed(object):
    def __init__(self,preAllocLen,imgSize=28,nearestNeighbors=7):
        self.len = 0
        self.prediction = np.empty(preAllocLen)
        self.target = np.empty(preAllocLen)
        self.image = np.empty((preAllocLen,imgSize*imgSize))
        self.contenders = np.empty((preAllocLen,nearestNeighbors))
        
    def Append(self, type, obj,index):
        match type:
            case "prediction":
                self.prediction[index]=obj
                self.len=self.len+1
            case "target":
                self.target[index]=obj
            case "image":
                self.image[index]=obj
            case "contenders":
                self.contenders[index]=obj
        
def main():

    # Load data
    num_train_chunks = 10
    train_chunk = 0
    train_data, train_targets, test_data, test_targets = fnc.loadData(num_train_chunks, train_chunk)

    test_size = len(test_targets)

    predictions = np.zeros(test_size)

    # Number of neighbors
    num_neighbors = 7

    #guesses = np.zeros(test_size)

    #clusters = fnc.clustering(train_data, train_targets, 64)
    #for i in range(test_size-1):
    #    prediction_cluster = fnc.KNNClustering(clusters, test_data[i], 64)
    #    guesses[i] = prediction_cluster


    failed = Failed(5000)

    contenders = np.zeros((test_size,num_neighbors))

    startTime = dt.datetime.now().replace(microsecond=0)

    fail = 0
    for i in range(test_size-1):  
        test_pic = test_data[i]
        test_target = test_targets[i]

        #neighbors = fnc.nearestNeighbors(train_data, train_targets, test_pic, num_neighbors)
        neighbors = fnc.NN(train_data, train_targets, test_pic, num_neighbors)
        prediction = fnc.KNN(neighbors)

        predictions[i] = prediction
        if prediction != test_target:
            failed.Append("prediction",prediction,fail)
            failed.Append("target",test_target,fail)
            failed.Append("image",test_pic,fail)
            failed.Append("contenders",neighbors,fail)
            #failures[fail] = test_pic
            #failed_predictions[fail] = prediction
            #failed_targets[fail] = test_target
            #contenders[fail] = neighbors
            fail += 1
            print("fuck ",fail)



    # Make confusion matrix and calculate error rate
    confusionMatrix = fnc.confusionMatrix(predictions, test_targets)
    errorRate = fnc.errorRate(confusionMatrix, test_size)

    endTime = dt.datetime.now().replace(microsecond=0)
    # Print results
    print("Confusion matrix: \n", confusionMatrix)
    print("Error rate: ", errorRate)
    print("Classification time: ", endTime-startTime)

    fnc.plotImg(failed)

main()

print("Digits:)")