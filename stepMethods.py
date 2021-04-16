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

def emptyDecorator(f):
    return f

def normalize(zeta): return zeta/siz(zeta)
def siz(zeta): return np.sqrt(np.sum(zeta**2))
def orthoProj(a,b): return a-b*np.dot(a,b)/siz(b)**2
def ortoHyperProj(a,B):
    for b in B:
        a = orthoProj(a,b)
    return a

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

def ndimBasis(zeta):
    zeta = normalize(zeta)
    n = len(zeta)
    stdBasis = np.identity(n).tolist()
    Basis = [zeta]
    while(len(Basis)!=n):
        candidates = [ortoHyperProj(ei, Basis) for ei in stdBasis]
        sizes = [*map(siz, candidates)]
        maxArg = np.argmax(sizes)
        Basis.append(normalize(candidates[maxArg]))
    return Basis


class plane:
    def __init__(self, planePosition, **kwargs):
        if('basis' in kwargs): self.basis = kwargs.get('basis')
        elif('normal' in kwargs): self.basis = ndimBasis(kwargs.get("normal"))
        else: raise Exception("Specify basis or normal")
        self.normal = self.basis[0]
        self.position = planePosition
        if not (len(self.position) == len(self.normal)): raise Exception("Position and normal not in same vector-space")


    def halfplane(self, position):
        dot = np.dot(position-self.position, self.normal)
        return 1 if dot > 0 else (0 if dot==0 else -1)



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
        self.saveTimeline=kwargs.get('timeline', False)

        self.method = None
        self.nopython=kwargs.get('nopythonExe')

    def run(self):
        self._load()
        return self.method()

    def _load(self, timeline=False):
        tInterval = self.tFinal-self.tInit
        tInit = self.tInit
        normalSteps = int(np.floor(tInterval/self.ordinaryStepLen)) if not self.stepAmtBased else self.steps
        endStep = not self.stepAmtBased
        totalSteps = normalSteps + endStep
        totalPoints = totalSteps + 1
        stepLen = float(self.ordinaryStepLen if not self.stepAmtBased else tInterval/self.steps)
        methodYInitial = self.y0

        if(self.methodName in stepMethods.__dict__): method = stepMethods.__dict__[self.methodName]
        else: raise Exception("No applicable procedure found for method")

        function = self.function
        saveTimeline = self.saveTimeline


        methodHasRollover = stepMethods.hasRollover.get(self.methodName)
        rolloverInit = stepMethods.__dict__[self.methodName + "InitialRollover"](function, methodYInitial, tInit, stepLen) if methodHasRollover else None


        decorator = jit(nopython=True) if self.nopython else emptyDecorator
        @decorator
        def execute():
            nonlocal methodHasRollover, rolloverInit, saveTimeline, methodYInitial, totalPoints
            y = methodYInitial
            arr = times = None
            if(saveTimeline) :
                arr = np.zeros((totalPoints,len(y),)) * 0.0;
                arr[0,:] = y
                times = np.zeros((totalPoints))    * 0.0;
                times[0] = tInit

            index = 0
            rollover = rolloverInit
            while(index < normalSteps):

                time = tInit + index*stepLen
                if(methodHasRollover) : y, rollover = method(function, y, time, stepLen, rollover)
                else : y = method(function, y, time, stepLen)
                index += 1;
                if (saveTimeline): times[index] = time+stepLen; arr[index, :] = y
            if(endStep):
                endStepLen = tInterval - normalSteps * stepLen
                time = tInit + normalSteps*stepLen
                if(methodHasRollover): y, rollover = method(function, y, time, endStepLen, rollover)
                else : y = method(function, y, time, endStepLen)
                index += 1
                if (saveTimeline): times[index] = time+endStepLen; arr[index, :] = y
            return (arr, times) if saveTimeline else y
        self.method = execute