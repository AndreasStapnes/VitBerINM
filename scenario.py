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


def U(q):
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

pl = plane(np.array([0,0,0,0]), normal=np.array([1,0,0,0]))

rout = routine(f=(a,b,c), tInit=0,tFinal=10000, y0=qp0, ordinaryStepLen=0.003,method="Kah",timeline=True, savePlaneCuts=True)
rout.planes.append(pl)
y, times,cuts = rout.run()


q1,q2,p1,p2=y.T
q=np.array([q1,q2]).T
p=np.array([p1,p2]).T

import matplotlib.pyplot as plt
plt.plot(q1,q2, linewidth=0.03)
plt.show()

#plt.plot(times, K(p),label="K")
#plt.plot(times, U(q),label="U")
plt.plot(times, H(y),label="H")
plt.legend()
plt.show()



cuts = np.array(cuts[0]).T
plt.figure()
plt.plot(cuts[0], cuts[1], "ko", markersize=0.4)
plt.show()
plt.figure()
plt.plot(cuts[0], cuts[2], "ko", markersize=0.4)
plt.show()