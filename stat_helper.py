import numpy as np

def fix_target(target,classes):
    new_target = np.zeros((1,len(classes)))
    new_target = np.tile(new_target,(int(len(target)),classes.size))

    print()

    for i in range (len(target)):
        new_target[i][target[i]] = 1
    return new_target

def sigmoid(x):
    y = x
    for i in range (len(x)):
        x[i] = 1/(1+ np.exp(-x[i]))
    x = np.reshape(x,(x.size,1))
    return x

def grad_gk_mse(gk, tk):
    grad = gk-tk
    #Muligens heller slik:
    #grad = np.multiply((gk-tk,gk))
    return grad

def grad_zk_g(gk):
    grad = np.multiply((1-gk),gk)
    return grad

#def grad_w_zk(x):
def calculate_MSE(gk,tk):
    return 0.5*np.matmul((gk-tk).T,(gk-tk))

def calculate_grad_W_MSE(gk, tk, xk):
    return np.matmul((gk-tk)*gk*(1-gk),xk.T)

def make_percentage_guess(gk):
    total = np.sum(abs(gk))
    largest = 0
    for i in range (0,gk.size):
        gk[i] = np.around(0.5*(1 - abs(gk[i])/total),2) # gk[i]/np.sum(gk) si how sure it is of not being correct
        if gk[i] > gk[largest]:
            largest = i
    return gk, largest