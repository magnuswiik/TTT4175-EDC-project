from operator import matmul
from sklearn.datasets import load_iris
import numpy as np
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
    MSE = 0
    gradW_MSE = 0
    for c in range(n_classes):
        x = TTS.trainingData[c]
        t = TTS.trainingTarget[c]
        g = makePredictionMatrix(W, bias, x)

        MSE += sh.calculate_MSE(g,t)
        gradW_MSE += sh.calculate_grad_W_MSE(g,t,x)

    W-=alpha*gradW_MSE

    return W, bias, MSE

def trainUntilSatisfactory(W,bias,TTS, alpha,itt):
    for i in range(itt):
        W, bias, MSE = trainWMatrix(W, bias, TTS, alpha)
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

def runIrisTask(alphaStart, n_alphas, n_trainingItterations, TTS, file):
    alpErrList = np.zeros((n_alphas,3))
    startTime = dt.datetime.now().replace(microsecond=0)
    for i in range(n_alphas):
        W = np.zeros((len(iris_classes), len(iris_feature)))
        bias = np.zeros((len(iris_classes),))
        if n_alphas > 1:
            alphaStart+=1*10**(-4)

        
        W, bias = trainUntilSatisfactory(W, bias, TTS, alphaStart, n_trainingItterations)

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
    print("Best Alpha and Error was: ", alpErrList[minitt])
    stopTime = dt.datetime.now().replace(microsecond=0)
    print("Time taken: ", stopTime-startTime)
    np.savetxt(file, alpErrList, delimiter=",")

def histogram(TTS, features): #kunne droppet feature og brukt len(TTS.trainingData[0][0]) for antall
    n_features = len(features)
    n_classes = len(TTS.testData)
    n_training = len(TTS.trainingData[0])
    n_test = len(TTS.testData[0])
    featureList = np.zeros((n_features,n_classes*(n_training+n_test)))
    for f in range(n_features):
        for c in range(n_classes):
            for i in range(n_training):
                featureList[f][c*n_training + i] = TTS.trainingData[c][f]
            for i in range(n_test):
                featureList[f][c*n_training + i] = TTS.trainingData[c][f]

    return n_features

def hist(data, features):
    histData = data.T
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Kuk og balle')
    a = 0
    b = 0
    for f in range(len(features)):
        for c in range(3):
            classStr = "Class " + str(c)
            axs[a, b].hist(histData[f][c*50:(c+1)*50], alpha=0.5, label=classStr)
        axs[a, b].set_title(features[f])
        if b == 1:
            a = 1
            b = 0
        else:
            b=1
        #plt.xlabel('Length (cm)')
        #plt.ylabel('Number of cases')
        #plt.title(features[f])
        ##plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        ##plt.xlim(3, 8)
        ##plt.ylim(0, 25)
        #plt.grid(True)
    plt.show()
    

    return

TTS = makeTrainingAndTestDataClass(iris_data, iris_target, 0, 30, 0, 50, 3)
alpha = 0.0321

hist(iris_data,iris_feature)
#runIrisTask(alpha,1, 1000, TTS,"dump.csv")
## god alpha 0.011, test 50 000 ganger
# alpha_errorTest_errorTraining_30_20.csv has tested 1000 alphas trained 1000 itterations each
#Best Alpha and Error was:  [0.0061 3.33   3.33  ]