
from stepMethods import *
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def fun(t, y):  return np.array([y[2], y[3], -y[0]*(1+2*y[1]), -(y[1]+y[0]**2-y[1]**2)])
from scipy import integrate


diffdata = []
hdata = np.logspace(-4,-0.5,40)
gamma = integrate.solve_ivp(fun, [0,1], np.array([4,4,4,4]), method=integrate.RK45, rtol=1e-12)
gammaend = gamma.y[:,-1]

reference = 1 * hdata**4



for h in hdata:
    rout = routine(f=fun, y0=np.array([4.0,4.0,4.0,4.0]), ordinaryStepLen=h, method="StV")
    yend = rout.run()
    print("*",end="")
    diffdata.append([*(yend-gammaend)])


import matplotlib.pyplot as plt
diffdata = np.array(diffdata)
fig, ax = plt.subplots(1,1)

for enum,diffs in enumerate(diffdata.T):
    ax.plot(hdata, np.abs(diffs), label="y_{}".format(enum))
for i in range(-12,8):
    ax.plot(hdata, reference*10**-i, "k--", alpha=0.5)
ax.set_xscale("log", base=10)
ax.set_yscale("log", base=10)
ax.set_xlim([1e-3,10**-0.5])
ax.set_ylim([1e-10,1e1])
ax.legend()
plt.show()


#print(t, "\n", y)

