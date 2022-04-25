import numpy as np
import pandas as pd

def Sigmoid(x):
    return np.array(1/(1+ np.exp(-x)))

def grad_gk_mse(gk, tk):
    grad = gk-tk
    #Muligens heller slik:
    #grad = np.multiply((gk-tk,gk))
    return grad

def grad_zk_g(gk):
    grad = np.multiply((1-gk),gk)
    return grad

def grad_w_zk(x):
    x_trans = 

def w_grad_mse(grad_gk_mse, grad_zk_g, grad_w_zk):
