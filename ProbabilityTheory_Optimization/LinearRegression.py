from pylab import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def linear_regression(pred_values, target_values, lr, No_Itr):
    N = len(pred_values)
    noCol = len(pred_values.columns)
    arr = np.ndarray((1,noCol), dtype = float)
    for i in range(noCol):
        arr[0]=-1
    a = np.array(arr, dtype = float)
    b = np.array([-1], dtype = float)
    L = np.array([])
    out = np.array(target_values, dtype = float).reshape(N,1)
    for iter in range(No_Itr):
        paramsDotX = np.array((np.dot(pred_values, a[-1].T)), dtype = float).reshape(N,1) 
        y = paramsDotX + b[-1]
        diffY = y-out
        diffYY = np.dot(diffY.T,pred_values)
        a = np.append(a,[a[-1] - lr*(1/N)*np.sum(diffYY)], 0)
        b = np.append(b,[b[-1] - lr*(1/N)*np.sum(diffY)],0)
        L = np.append(L, (1/N)*np.sum(diffY**2))
    return (a,b,L)
