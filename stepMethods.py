from numba import jit
from numba.experimental import jitclass
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

    @jit(nopython=True)
    def Kah(matrixRepresentation, yvalues, t, h):
        a,b,c = matrixRepresentation
        n = len(a)
        at = np.transpose(a, (0, 2, 1))
        N = np.sum(at * yvalues, axis=2)
        M = np.sum(a  * yvalues, axis=2)
        A = np.identity(n) - 1/2*h*((M+N) + b)
        B = np.identity(n) + 1/2*h*b
        return np.linalg.solve(A, B@yvalues + c)

    @jit(nopython=True)
    def StV(function, yValues, t, h, rollover):
        n = len(yValues)//2
        q = yValues[:n]; p=yValues[n:];
        F1P = rollover[n:]
        phalf = p + 1/2*h*F1P
        qNext = q + h*phalf
        F2 = function(t, np.concatenate((qNext, p)))
        F2P = F2[n:]
        pNext = phalf+1/2*h*F2P
        yNext = np.concatenate((qNext, pNext))
        return yNext, F2
    def StVInitialRollover(function, yValues, t, h):
        return function(t, yValues)

    hasRollover = {"RK4":False, "BS3":False, "Kag":False, "StV":True}


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
        return self.method()

    def _load(self, timeline=False):
        tInterval = self.tFinal-self.tInit
        tInit = self.tInit
        normalSteps = int(np.floor(tInterval/self.ordinaryStepLen)) if not self.stepAmtBased else self.steps
        endStep = not self.stepAmtBased
        totalSteps = normalSteps + endStep
        stepLen = float(self.ordinaryStepLen if not self.stepAmtBased else tInterval/self.steps)
        methodYInitial = self.y0

        if(self.methodName in stepMethods.__dict__): method = stepMethods.__dict__[self.methodName]
        else: raise Exception("No applicable procedure found for method")

        function = self.function

        methodHasRollover = stepMethods.hasRollover.get(self.methodName)
        rolloverInit = stepMethods.__dict__[self.methodName + "InitialRollover"](function, methodYInitial, tInit, stepLen) if methodHasRollover else None
        #@jit(nopython=True)
        def execute():
            nonlocal methodHasRollover, rolloverInit
            y = methodYInitial
            index = 0
            rollover = rolloverInit
            while(index < normalSteps):
                index += 1;
                if(methodHasRollover) : y, rollover = method(function, y, tInit+(index-1)*stepLen, stepLen, rollover)
                else : y = method(function, y, tInit+(index-1)*stepLen, stepLen)
            if(endStep):
                endStepLen = tInterval - normalSteps * stepLen
                if(methodHasRollover): y, rollover = method(function, y, tInit + (totalSteps - 1) * stepLen, endStepLen, rollover)
                else : y = method(function, y, tInit + (totalSteps - 1) * stepLen, endStepLen)
            return y
        self.method = execute