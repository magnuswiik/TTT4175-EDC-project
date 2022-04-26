from email.policy import default
from nis import match
from sklearn.datasets import load_iris
import numpy as np
import stat_helper as sh

iris = load_iris()

iris_classes = np.reshape(np.array(iris['target_names']),(1,3))
iris_data = np.array(iris['data'])
iris_feature = np.reshape(np.array(iris['feature_names']),(1,4))
iris_target_predict = np.array(iris['target'])
iris_target_train = sh.fix_target(np.array(iris['target']),iris_classes)

def iris_train(data,target, start, stop, classes, features, error_margin):

    #Training variables
    W = np.ones((classes.size,features.size +1)) #+1 for bias 
    MSE = 10000
    prevMSE = 0
    grad_W_MSE = 0
    alpha = 0.5#make a better alpha
    count = 0
    
    while abs(MSE-prevMSE) > error_margin*0.0000001:
        count +=1
        prevMSE = MSE
        MSE = 0
        #Dataset acces variables
        itt = 0
        skip_len = int(data.shape[0]/classes.size - (stop-start))
       
        for i in range (0,classes.size*(stop-start)):
            if i%(stop-start) == 0 and i!=0:
                itt+=1
            index = itt*skip_len + start + i
            
            xk = np.reshape(data[index],(data[index].size,1))
            xk = np.append(xk,1)
            xk = np.reshape(xk,(xk.size,1))
            gk = sh.sigmoid(np.matmul(W,xk))
            tk = np.reshape(target[index],(target[index].size,1))

            MSE += sh.calculate_MSE(gk,tk)
            grad_W_MSE += sh.calculate_grad_W_MSE(gk,tk,xk) 
        #update W
        W -= alpha*grad_W_MSE
        #print(MSE)
    print(W)
    #print(count)
    return W

def iris_predict(data, W, start, stop, classes, target):
    #Dataset acces variables
    itt = 0
    skip_len = int(data.shape[0]/classes.size - (stop-start))

    confusion_matrix = np.zeros((classes.size,classes.size))
    for i in range (0,classes.size*(stop-start)):

        if i%(stop-start) == 0 and i!=0:
            itt+=1
        index = itt*skip_len + start+ i

        xk = np.reshape(data[index],(data[index].size,1))
        xk = np.append(xk,1)
        xk = np.reshape(xk,(xk.size,1))

        gk = np.matmul(W,xk)

        gk, guess = sh.make_percentage_guess(gk)
        confusion_matrix[target[index]][guess]+=1
    return confusion_matrix

W = iris_train(iris_data,iris_target_train, 0,30,iris_classes,iris_feature,0.01)
confusion = iris_predict(iris_data,W,30,50,iris_classes,iris_target_predict)
print(confusion)