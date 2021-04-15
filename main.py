
from numba import jit
from numba.typed import List
import math
import numpy as np


class stepMethods:
    @jit(nopython=True)
    def RK4(function, yValues, t, h):
        F1 = function(t, yValues)
        F2 = function(t + h / 2, yValues + h / 2 * F1)
        F3 = function(t + h / 2, yValues + h / 2 * F2)
        F4 = function(t + h, yValues + h * F3)
        yNext = yValues + h / 6 * (F1 + 2 * F2 + 2 * F3 + F4)
        return yNext

    @jit(nopython=True)
    def BS3(function, yValues, t, h):
        F1 = function(t, yValues)
        F2 = function(t + h / 2, yValues + h / 2 * F1)
        F3 = function(t + 3 * h / 4, yValues + h * 3 / 4 * F2)
        yNext = yValues + h / 9 * (F1 * 2 + F2 * 3 + F3 * 4)
        return yNext


class routine:
    def __init__(self, **kwargs):
        '''
        :param kwargs:
            (float) tInit, tFinal                               | default: tInit=0, tFinal=1
            (float) ordinaryStepLen eller (int) steps           | default: ordinaryStepLen=1
            (string) method {RK4, BS3, Ka2, SV2}                | default: RK4
            (ndim float) y0                                     | default: 0-vec
            (f : t,y->(ndim float)) f                           | default: 0-function
        '''
        self.tInit = kwargs.get("tInit", 0)
        self.tFinal = kwargs.get("tFinal", 1)

        self.stepAmtBased = "steps" in kwargs
        if  (self.stepAmtBased):  self.steps=kwargs.get("steps")
        else                   :  self.ordinaryStepLen = kwargs.get("ordinaryStepLen", 1)

        self.methodName = kwargs.get("method", "RK4")
        self.function = kwargs.get("f")
        self.y0 = kwargs.get("y0")

        self.method = None
        self._load()

    def run(self):
        return self.method(self.function)

    def _load(self):
        tInterval = self.tFinal-self.tInit
        tInit = self.tInit
        normalSteps = int(np.floor(tInterval/self.ordinaryStepLen)) if not self.stepAmtBased else self.steps
        endStep = not self.stepAmtBased
        totalSteps = normalSteps + endStep
        totalPoints = totalSteps + 1

        stepLen = float(self.ordinaryStepLen if not self.stepAmtBased else tInterval/self.steps)

        methodYInitial = self.y0
        n = np.shape(self.y0)[0]


        method = stepMethods.__dict__[self.methodName]

        @jit(nopython=True)
        def execute(function):
            values = np.zeros((totalPoints,n))  *0.0 #Cast til float
            times = np.zeros(totalPoints)       *0.0 #Cast til float for typesafety i jit
            index = 0
            values[0,:]=methodYInitial
            times[0] = tInit


            while(index < normalSteps):
                index += 1
                values[index,:] = method(function, values[index-1,:], tInit+(index-1)*stepLen, stepLen)
                times[index] = times[index-1] + stepLen
            if(endStep):
                endStepLen = tInterval-normalSteps*stepLen
                values[totalSteps,:] = method(function, values[totalSteps-1, :], tInit+(totalSteps-1)*stepLen, endStepLen)
                times[totalSteps] = times[totalSteps-1] + endStepLen
            return times, values
        self.method = execute


@jit(nopython=True)
def fun(t, y):
    return np.array([y[2], y[3], -y[0]*(1+2*y[1]), -(y[1]+y[0]**2-y[1]**2)])
from scipy import integrate


diffdata = []
hdata = np.logspace(-2,-0.5,40)
gamma = integrate.solve_ivp(fun, [0,1], np.array([1,1,1,1]), method=integrate.RK45, rtol=1e-12)
gammaend = gamma.y[:,-1]

reference = 1 * hdata**4

for h in hdata:
    rout = routine(f=fun, y0=np.array([1.0,1.0,1.0,1.0]), ordinaryStepLen=h)
    t, y = rout.run()
    yend = y[-1]
    print("*",end="")
    diffdata.append([*(yend-gammaend)])


import matplotlib.pyplot as plt
diffdata = np.array(diffdata)
fig, ax = plt.subplots(1,1)

for diffs in diffdata.T:
    ax.plot(hdata, diffs)
for i in range(-5,8):
    ax.plot(hdata, reference*10**-i, "k--", alpha=0.5)
ax.set_xscale("log", base=10)
ax.set_yscale("log", base=10)
ax.set_xlim([1e-2,10**-0.5])
ax.set_ylim([1e-10,1e-2])
plt.show()


#print(t, "\n", y)

