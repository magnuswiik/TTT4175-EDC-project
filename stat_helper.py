import numpy as np

def fix_target(target,n_classes):
    n_target = len(target)
    new_target = np.zeros((n_target,n_classes))
    for i in range(n_target):
        vector_target = np.zeros((1,n_classes))[0]
        vector_target[target[i]]+=1
        new_target[i]=vector_target
    new_target = np.reshape(new_target,(len(target),n_classes))
    return new_target

def sigmoid(z):
    return 1/(1+ np.exp(-z))

def calculate_MSE(gk,tk):
    return 0.5*np.matmul((gk-tk).T,(gk-tk))

def calculate_grad_W_MSE(g, t, x):
    return np.matmul(((g-t)*g*(1-g)).T,x)
