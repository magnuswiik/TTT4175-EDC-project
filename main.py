from operator import matmul
from sklearn.datasets import load_iris
import numpy as np
from numpy import linalg as la
import stat_helper as sh
import typing as tp
import datetime as dt
import matplotlib.pyplot as plt


iris = load_iris()

iris_classes =np.array(iris['target_names'])
iris_data = np.array(iris['data'])
iris_feature = np.array(iris['feature_names'])
iris_target = np.array(iris['target'])

def makePredictionMatrix(W, bias, x):
    numberOfVectors = x.shape[0]
    predictionMatrix = np.array([sh.sigmoid((np.matmul(W,x[i])+bias)) for i in range(numberOfVectors)])
    return predictionMatrix

def trainWMatrix(W, bias, TTS, alpha):
    n_classes = len(TTS.trainingData)
    gradW_MSE = 0
    for c in range(n_classes):
        x = TTS.trainingData[c]
        t = TTS.trainingTarget[c]
        g = makePredictionMatrix(W, bias, x)

        gradW_MSE += sh.calculate_grad_W_MSE(g,t,x)

    W-=alpha*gradW_MSE

    return W, bias

def trainUntilSatisfactory(W,bias,TTS, alpha, itt):
    for i in range(itt):
        W, bias = trainWMatrix(W, bias, TTS, alpha)
    return W, bias

def makeConfusionMatricies(W, bias, TTS):
    n_classes = len(TTS.testData)
    confusionMatrixTestSet = np.zeros((n_classes, n_classes))
    confusionMatrixTrainingSet = np.zeros((n_classes, n_classes))
    for c in range(n_classes):
        predictionTestSet = makePredictionMatrix(W, bias, TTS.testData[c])
        predictionTrainingSet = makePredictionMatrix(W, bias, TTS.trainingData[c])
        answearTestSet = TTS.testTarget[c]
        answearTrainingSet = TTS.trainingTarget[c]
        for i in range(len(TTS.testTarget[0])):
            confusionMatrixTestSet[np.argmax(answearTestSet[i])][np.argmax(predictionTestSet[i])] += 1
        for i in range(len(TTS.trainingTarget[0])):
            confusionMatrixTrainingSet[np.argmax(answearTrainingSet[i])][np.argmax(predictionTrainingSet[i])] += 1
    return confusionMatrixTestSet, confusionMatrixTrainingSet

def makePercentErrorRate(confusionMatrix):
    errorPercent = 0
    n_classes = len(confusionMatrix[0])
    n_pred = np.sum(confusionMatrix)
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                errorPercent += confusionMatrix[i][j]/n_pred
    return np.around(errorPercent*100,2)

class TrainingAndTestStruct(object):
    def __init__(self):
        self.trainingData = []
        self.trainingTarget = []
        self.testData = []
        self.testTarget = []
    def AddToTTS(self, type, obj):
        match type:
            case "trainingData":
                self.trainingData.append(obj)
            case "trainingTarget":
                self.trainingTarget.append(obj)
            case "testData":
                self.testData.append(obj)
            case "testTarget":
                self.testTarget.append(obj)

def makeTrainingAndTestDataClass(data, target, trainingStart, trainingStop, classStart, classLength, n_classes):
    n_trainingData = trainingStop - trainingStart
    n_testData = classLength - n_trainingData
    n_trainingOffset = trainingStart-classStart
    TTS = TrainingAndTestStruct()
    for c in range(n_classes):
        c_off = c*classLength
        trainingTarget = [target[c_off+trainingStart + i] for i in range(n_trainingData)]
        testTarget = [target[c_off+classStart + i] if i < n_trainingOffset else target[c_off+trainingStop-n_trainingOffset +i] for i in range(n_testData)]

        TTS.AddToTTS("trainingData",np.array([data[c_off+trainingStart + i] for i in range(n_trainingData)]))
        TTS.AddToTTS("trainingTarget",np.array(sh.fix_target(trainingTarget,n_classes)))
        TTS.AddToTTS("testData",np.array([data[c_off+classStart + i] if i < n_trainingOffset else data[c_off+trainingStop-n_trainingOffset + i] for i in range(n_testData)]))
        TTS.AddToTTS("testTarget",np.array(sh.fix_target(testTarget,n_classes)))
        
    return TTS

def runIrisTask(alphaStart, n_alphas, itt, TTS, file=0):
    alpErrList = np.array([[0.]*3 for i in range(n_alphas)])
    startTime = dt.datetime.now().replace(microsecond=0)
    n_classes = len(TTS.trainingData)
    n_features = len(TTS.trainingData[0][0])
    for i in range(n_alphas):
        if n_alphas > 1:
            alphaStart+=1*10**(-4)
        W = np.zeros((n_classes, n_features))
        bias = np.zeros((n_classes,))
    
        
        W, bias = trainUntilSatisfactory(W, bias, TTS, alphaStart, itt)

        confusionMatrixTestSet, confusionMatrixTrainingSet = makeConfusionMatricies(W, bias, TTS)
        errorPercentTestSet = makePercentErrorRate(confusionMatrixTestSet)
        errorPercentTrainingSet = makePercentErrorRate(confusionMatrixTrainingSet)

        alpErrList[i][0] = alphaStart
        alpErrList[i][1] = errorPercentTestSet
        alpErrList[i][2] = errorPercentTrainingSet

        if i%100==0:
            print("Itterations ", i)
            print("Time taken: ", dt.datetime.now().replace(microsecond=0)-startTime)
    print("ConfusionMatrixTestSet: \n", confusionMatrixTestSet)
    print("ConfusionMatrixTrainingSet: \n", confusionMatrixTrainingSet)
    min = 100
    minitt= 0
    for i in range (n_alphas):
        if alpErrList[i][1] < min:
            min = alpErrList[i][1]
            minitt = i
    print("Best Alpha and ErrorMargin, with ErrorRates was: ", alpErrList[minitt])
    stopTime = dt.datetime.now().replace(microsecond=0)
    print("Time taken: ", stopTime-startTime)
    if file != 0:
        np.savetxt(file, alpErrList, delimiter=",")

def hist(data, features, classes, file = 0):
    histData = data.T
    fig = plt.figure()
    plt.suptitle('Histograms of the different features and classes', fontweight='bold')
    
    featuresLeftToPlot = len(features)
    for f in range(len(features)):
        if featuresLeftToPlot>=2:
            xdir = 2
        else:
            xdir = 1
        plt.subplot(int(np.ceil(len(features)/2)), xdir, f+1)
        for c in range(3):
            plt.hist(histData[f][c*50:(c+1)*50], bins=30, alpha=0.5, label=classes[c])
    
        plt.title(features[f])
        plt.legend(loc='upper right')

    # Adding a plot in the figure which will encapsulate all the subplots with axis showing only
    fig.add_subplot(1, 1, 1, frame_on=False)

    # Hiding the axis ticks and tick labels of the bigger plot
    plt.tick_params(labelcolor="none", bottom=False, left=False)

    #Make common x- and y-labels
    plt.xlabel('Length (cm)', fontweight='bold')
    plt.ylabel('Occurrences', fontweight='bold')

    if file != 0:
        plt.savefig(file)

    plt.show()
    return

def _removeFeature(data, features, featureToBeRemoved):
    n_features = len(features)
    newFeature = np.array([]) ##hard to preallocate string as you need to know the size, bad for efficiancy but whatever
    n_data = len(data)
    newData = np.array([[0.]*(n_features-1) for j in range(n_data)])
    
    j = 0 #ugly hack to get correct index for newFeature
    for i in range(n_features):
        if i%n_features != featureToBeRemoved:
            newFeature = np.append(newFeature, features[i])
            j+=1

    for i in range(n_data):
        k = 0
        for j in range(n_features):
            if j%n_features != featureToBeRemoved:
                newData[i][k] = data[i][j]
                k+=1

    return newData, newFeature

def removeListOfFeatures(data, feature, featureRemoveList):
    newData = data
    newFeature = feature
    for i in range(len(featureRemoveList)):
        newIndex = np.where(newFeature == feature[i])[0][0]
        newData, newFeature = _removeFeature(newData, newFeature, newIndex)

    return newData, newFeature


alpha = 0.0075
n_a = 1
itt = 1000
newData, newFeatures = removeListOfFeatures(iris_data, iris_feature, []) #leave list empty to include all
#hist(newData,newFeatures,iris_classes,"RemovedWorsedOneHist.png") #add a file as last input if you want to save
TTS = makeTrainingAndTestDataClass(newData, iris_target, 0, 30, 0, 50, 3)
runIrisTask(alpha,n_a, itt, TTS)

#newData, newFeatures = removeListOfFeatures(iris_data, iris_feature, [0]) #leave list empty to include all
##hist(newData,newFeatures,iris_classes,"RemovedWorsedOneHist.png") #add a file as last input if you want to save
#TTS = makeTrainingAndTestDataClass(newData, iris_target, 0, 30, 0, 50, 3)
#runIrisTask(alpha,n_a, errorMargin,n_e, TTS,"ErrorMargin1.csv")
#
#newData, newFeatures = removeListOfFeatures(iris_data, iris_feature, [0,1]) #leave list empty to include all
##hist(newData,newFeatures,iris_classes,"RemovedWorsedOneHist.png") #add a file as last input if you want to save
#TTS = makeTrainingAndTestDataClass(newData, iris_target, 0, 30, 0, 50, 3)
#runIrisTask(alpha,n_a, errorMargin,n_e, TTS,"ErrorMargin2.csv")
#
#newData, newFeatures = removeListOfFeatures(iris_data, iris_feature, [0,1,2]) #leave list empty to include all
##hist(newData,newFeatures,iris_classes,"RemovedWorsedOneHist.png") #add a file as last input if you want to save
#TTS = makeTrainingAndTestDataClass(newData, iris_target, 0, 30, 0, 50, 3)
#runIrisTask(alpha,n_a, errorMargin,n_e, TTS,"ErrorMargin3.csv")
## god alpha 0.011, test 50 000 ganger
# alpha_errorTest_errorTraining_30_20.csv has tested 1000 alphas trained 1000 itterations each
#Best Alpha and Error was:  [0.0061 3.33   3.33  ]