from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from stepMethods import *

@jit(nopython=True)
def f(t, y, F):
    F[:] = np.array([t ** 3 / y[0] ** 2])


rout = routine(f=f, tInit=0, tFinal=1, y0=np.array([1]), steps=4, method="Mid", timeline=True)
ys, t = rout.run()



ys = ys.T
plt.plot(t, ys[0])
print(ys[0])
plt.show()