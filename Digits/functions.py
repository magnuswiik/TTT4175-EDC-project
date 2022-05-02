import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from statistics import mode
from sklearn.cluster import KMeans
from tensorflow import keras
from keras.datasets import mnist
import datetime as dt


def loadData(num_train_chunks, train_chunk):
    (train_data, train_targets), (test_data, test_targets) = mnist.load_data()

    train_data = np.asarray(train_data)
    train_targets = np.asarray(train_targets)
    test_data = np.asarray(test_data)
    test_targets = np.asarray(test_targets)

    num_train, dim_x, dim_y = train_data.shape
    train_setx = train_data.reshape(num_train,dim_x*dim_x)

    num_test, dim_testx, dim_testy = test_data.shape
    test_data_reshaped = test_data.reshape(num_test,dim_testx*dim_testy)

    train_data_split = np.asarray(np.split(train_setx,num_train_chunks))
    train_targets_split = np.asarray(np.split(train_targets,num_train_chunks))

    return train_data_split[train_chunk], train_targets_split[train_chunk], test_data_reshaped, test_targets

# Clustering
def sortClasses(memory_data, targets):
    numb_classes = np.zeros(10)
    sorted_memory_data = np.empty_like(memory_data)
    sorted_targets = np.argsort(targets)

    for i in range(len(targets)):
        numb_classes[targets[i]] += 1
        sorted_memory_data[i] = memory_data[sorted_targets[i]]

    return np.asarray(sorted_memory_data), numb_classes

def clustering(memory_data, targets, M):
    time_start = dt.datetime.now().replace(microsecond=0)
    
    sorted_memory_data, numb_classes = sortClasses(memory_data, targets)
    flattened_sorted_memory_data = sorted_memory_data.flatten().reshape(memory_data.shape[0], 28*28)
    clusters = np.empty((len(numb_classes), M, 28*28))
    start = 0
    end = 0

    for count, i in enumerate(numb_classes):
        end += i
        cluster = KMeans(n_clusters=M,random_state=0).fit(flattened_sorted_memory_data[int(start):int(end)]).cluster_centers_
        start = end
        clusters[count] = cluster

    time_end = dt.datetime.now().replace(microsecond=0)
    print("Clustering time: ", time_end-time_start)
    return clusters.flatten().reshape(len(numb_classes)*64, 28*28)

def eucDist(img1, img2):
    distance = np.matmul((img1-img2).T, (img1-img2))
    return distance

def KNNClustering(clusters, test_pic, M):
    flattened_pic = test_pic.flatten().reshape(1, 28*28)
    distances = np.zeros(len(clusters))

    for i in range(len(clusters)):
        distances[i] = np.linalg.norm(flattened_pic - clusters[i])
    nearest_neighbor_index = np.argmin(distances)

    return nearest_neighbor_index // 64

def nearestNeighbors(train_data, train_targets, test_pic, num_neighbors):
    distances = np.zeros((train_data.shape[0], 2))
    for j in range(train_data.shape[0]):
        distances[j][0] = int(train_targets[j])
        #distances[j][1] = np.linalg.norm(test_pic-train_data[j],2) #euclidian distance
        distances[j][1] = eucDist(train_data[j], test_pic)

    #distances.sort(key=lambda tup: tup[1]) 
    distances = distances[distances[:, 1].argsort()]
    nearest_neighbors = [distances[i][0] for i in range(num_neighbors)]
    return nearest_neighbors

def KNN(nearest_neighbors):
    prediction = mode(nearest_neighbors)
    #prediction = np.argmin(nearest_neighbors)

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

def plotImg(images, predictions, targets, contenders):
    for i in range(len(images)):
        title = "Prediction: " + str(predictions[i]) + " But it was: " + str(targets[i])
        plt.title(title)
        plt.xlabel(str(contenders[i]))
        plt.imshow(images[i])
        plt.show()
        input("Press for next pic")