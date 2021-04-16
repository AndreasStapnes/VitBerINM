import numpy as np
from stepMethods import *

a = np.array([
[[  0,      0,      0,      0],
 [  0,      0,      0,      0],
 [  0,      0,      0,      0],
 [  0,      0,      0,      0],],

[[  0,      0,      0,      0],
 [  0,      0,      0,      0],
 [  0,      0,      0,      0],
 [  0,      0,      0,      0],],

[[  0,     -2,      0,      0],
 [  0,      0,      0,      0],
 [  0,      0,      0,      0],
 [  0,      0,      0,      0],],

[[ -1,      0,      0,      0],
 [  0,      1,      0,      0],
 [  0,      0,      0,      0],
 [  0,      0,      0,      0],]])*1.0

b = np.array(
   [[0, 0, 1, 0],
    [0, 0, 0, 1],
    [-1, 0, 0, 0],
    [0, -1, 0, 0]])*1.0

c = np.array([0,0,0,0])*1.0

H0 = 1/12
q01 = 0; q02 = 0.45
p02 = 0
p01 = np.sqrt(2*H0-p02**2-(q01**2+q02**2)-2*q01**2*q02 +2/3*q02**3)
qp0 = np.array([q01,q02,p01, p02])


def U(y):
    dim = np.shape(y)[1] // 2
    q = y[:, :dim]
    qSep = q.T
    return 1/2*np.sum(q*q, axis=1) + qSep[0]**2*qSep[1]-1/3*qSep[1]**3

def K(y):
    dim = np.shape(y)[1]//2
    p=y[:,dim:]
    return 1/2*np.sum(p*p,axis=1)

def H(y):
    return K(y)+ U(y)


