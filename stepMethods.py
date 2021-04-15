from numba import jit
from numba.typed import List
import math
import numpy as np

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

    @jit(nopython=False)
    def Kah(matrixRepresentation, yvalues, t, h):
        a,b,c = matrixRepresentation
        n = len(a)
        at = np.transpose(a, (0, 2, 1))
        N = np.sum(at * yvalues, axis=2)
        #N = np.einsum('ijk,j->ik',a,yvalues)
        M = np.sum(a  * yvalues, axis=2)
        #M = np.einsum('ijk,k->ij',a,yvalues)
        A = np.identity(n) - 1/2*h*((M+N) + b)
        B = np.identity(n) + 1/2*h*b
        return np.linalg.solve(A, B@yvalues + c)






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

    def run(self):
        self._load()
        return self.method(self.function)

    def _load(self, timeline=False):
        tInterval = self.tFinal-self.tInit
        tInit = self.tInit
        normalSteps = int(np.floor(tInterval/self.ordinaryStepLen)) if not self.stepAmtBased else self.steps
        endStep = not self.stepAmtBased
        totalSteps = normalSteps + endStep
        totalPoints = totalSteps + 1

        stepLen = float(self.ordinaryStepLen if not self.stepAmtBased else tInterval/self.steps)

        methodYInitial = self.y0
        n = np.shape(self.y0)[0]

        if(callable(self.methodName)):method=self.methodName
        elif(self.methodName in stepMethods.__dict__): method = stepMethods.__dict__[self.methodName]
        else: raise Exception("No applicable procedure found for method")

        if(timeline):
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
        else:
            @jit(nopython=True)
            def execute(function):
                y = methodYInitial
                index = 0
                while(index < normalSteps):
                    index += 1;
                    y = method(function, y, tInit+(index-1)*stepLen, stepLen)
                if(endStep):
                    endStepLen = tInterval - normalSteps * stepLen
                    y = method(function, y, tInit + (totalSteps - 1) * stepLen, endStepLen)
                return y
        self.method = execute