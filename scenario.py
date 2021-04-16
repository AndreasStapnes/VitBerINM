import numpy as np
from stepMethods import *


H0 = 1/12
q01 = 0; q02 = 0.45
p02 = 0
p01 = np.sqrt(2*H0-p02**2-(q01**2+q02**2)-2*q01**2*q02 +2/3*q02**3)
qp0 = np.array([q01,q02,p01, p02])


print(ndimBasis(np.array([0,1,1])))

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


rout = routine(f=(a,b,c), tInit=0,tFinal=3000, y0=qp0, ordinaryStepLen=0.01,method="Kah",timeline=True)
y, times = rout.run()
energies = H(y)

q1,q2,p1,p2=y.T
q=np.array([q1,q2]).T
p=np.array([p1,p2]).T

import matplotlib.pyplot as plt
plt.plot(q1,q2, linewidth=0.1)
plt.show()

#plt.plot(times, K(p),label="K")
#plt.plot(times, U(q),label="U")
plt.plot(times, H(y),label="H")
plt.legend()
plt.show()