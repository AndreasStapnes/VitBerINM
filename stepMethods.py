from numba import jit
import numba
from numba.typed import List
from vectorAlgebra import *
import numpy as np


class stepMethods:
    @jit(nopython=True)
    def RK4(function, yValues, t, h):
        dim = len(yValues)
        Fs = np.zeros((4,dim))*1.0
        function(t, yValues, Fs[0,:])
        function(t + h / 2, yValues + h / 2 * Fs[0,:], Fs[1,:])
        function(t + h / 2, yValues + h / 2 * Fs[1,:],Fs[2,:])
        function(t + h, yValues + h * Fs[2,:], Fs[3,:])
        yNext = yValues + h / 6 * (Fs[0,:] + 2 * Fs[1,:] + 2 * Fs[2,:] + Fs[3,:])
        return yNext

    @jit(nopython=True)
    def BS3(function, yValues, t, h):
        dim = len(yValues)
        Fs = np.zeros((3, dim))*1.0
        function(t, yValues, Fs[0,:])
        function(t + h / 2, yValues + h / 2 * Fs[0,:], Fs[1,:])
        function(t + 3 * h / 4, yValues + h * 3 / 4 * Fs[1,:], Fs[2,:])
        yNext = yValues + h / 9 * (Fs[0,:] * 2 + Fs[1,:] * 3 + Fs[2,:] * 4)
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
        F2 = np.zeros(len(yValues))*1.0
        function(t, np.concatenate((qNext, p)), F2[:])
        F2P = F2[n:]
        pNext = phalf+1/2*h*F2P
        yNext = np.concatenate((qNext, pNext))
        return yNext, F2
    def StVInitialRollover(function, yValues, t, h):
        dim = len(yValues)
        F = np.zeros(dim)*1.0
        function(t, yValues, F[:])
        return F

    hasRollover = {"RK4":False, "BS3":False, "Kah":False, "StV":True}




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


        self.y0 = kwargs.get("y0")
        self.saveTimeline=kwargs.get('timeline', False)
        self.timelineJumps=kwargs.get('timelineJumps', 1)

        self.method = None
        self.nopython=kwargs.get('nopythonExe', False)

        self.savePlaneCuts = kwargs.get("savePlaneCuts", False)
        self.planes = []

        self.methodName = kwargs.get("method", "RK4")
        self.function = kwargs.get("f")

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
        else: raise Exception("No applicable method found for gived method-name")

        function = self.function


        saveTimeline = self.saveTimeline
        savePlaneCuts = self.savePlaneCuts


        methodHasRollover = stepMethods.hasRollover.get(self.methodName)
        rolloverInit = stepMethods.__dict__[self.methodName + "InitialRollover"](function, methodYInitial, tInit, stepLen) if methodHasRollover else None

        dimension = len(methodYInitial)

        planesAmt = len(self.planes)
        planes = self.planes
        planesLocations = np.array([pl.position for pl in planes])
        planesNormals = np.array([pl.normal for pl in planes])
        planesBasis = np.array([pl.basis for pl in planes])

        jumpLen = self.timelineJumps
        totTimelinePts = totalPoints // jumpLen

        global normalize, siz, dot
        jitNormalize = jit(nopython=True)(normalize)
        jitSiz = jit(nopython=True)(siz)
        jitDot = jit(nopython=True)(dot)

        @jit(nopython=True)
        def jitRegisterCuts(inputPosition):
            nonlocal planesLocations, planesAmt, planesNormals
            def jitHalfPlane(inputPosition, planePosition, normal):
                return np.sum((inputPosition - planePosition) * normal) >= 0
            newCutState = [jitHalfPlane(inputPosition, planesLocations[enum], planesNormals[enum])
                           for enum in range(planesAmt)]
            return np.array(newCutState)

        @jit(nopython=True)
        def planeCutCoordinates(direction, inputPosition, basis, planePosition):
            nonlocal dimension, methodHasRollover, function
            planeNormal = basis[0]
            planeDistance = jitDot((inputPosition-planePosition),planeNormal)
            requiredTravel = -planeDistance/jitDot(planeNormal,direction)
            position = inputPosition + requiredTravel*direction
            crd = np.zeros(dimension-1)
            for i in range(dimension-1):
                crd[i] = np.sum(position*basis[i+1])
            return crd


        decorator = jit(nopython=True) if self.nopython else emptyDecorator
        @decorator
        def execute():
            nonlocal methodHasRollover, rolloverInit, saveTimeline, methodYInitial, totalPoints, planesBasis
            y = methodYInitial
            prevstepY = np.zeros_like(y)*1.0
            arr = times = None
            h = 0.0
            time = 0.0
            index = 0

            if(saveTimeline) :
                arr = np.zeros((totTimelinePts,len(y),)) * 0.0;
                arr[0,:] = y
                times = np.zeros((totTimelinePts))    * 0.0;
                times[0] = tInit
            rollover = rolloverInit

            cutpoints = [[np.array([0.0])][:0]] * planesAmt
                                    #Jeg har ærlig talt ingen peiling på hvorfor dette fungerer
                                    #dette er et resultat av timer med prøving og feiling.
                                    #Koden som burde ha stått der et [[]]*planesAmt for å få
                                    # en liste av planesAmt antall lister.
                                    #Likevel har jeg en hypotese: Se for deg Jit er som en hund
                                    # og numpy-float-arrays er en skinkebit. anta cutPoints er
                                    # hundens mage. hundens mage vil senere bli fylt med skinkebiter
                                    # men for at dette skal skje må hunden vite at den skal lete etter
                                    # skinkebiter. Derfor må man gi en skinkebit til hunden og
                                    # umiddelbart nappe den bort, altså sette inn [np.array([0.0])]
                                    # i cutPoints, for deretter å fjerne denne med [:0]-indeksering

            cutState = jitRegisterCuts(y)


            def checkCuts():
                nonlocal cutState, cutpoints, function
                newCutState = jitRegisterCuts(y)
                if(planesAmt and savePlaneCuts):
                    for i in range(planesAmt):
                        if newCutState[i]!=cutState[i]:
                            cutState[i]=newCutState[i]
                            direction = y - prevstepY
                            cutpt = planeCutCoordinates(direction, y, planesBasis[i], planesLocations[i])
                            cutpoints[i].append(cutpt)


            while(index < totalSteps):
                time = tInit + index*stepLen
                h = stepLen if index!=normalSteps else tInterval-normalSteps*stepLen
                prevstepY = y;
                if(methodHasRollover) : y, rollover = method(function, y, time, h, rollover)
                else : y = method(function, y, time, h)
                index += 1;
                if (saveTimeline and index%jumpLen==0): times[index//jumpLen] = time+h; arr[index//jumpLen, :] = y
                checkCuts()
            return ((arr, times, cutpoints) if savePlaneCuts else (arr, times)) \
                                       if saveTimeline else \
                            ((y, cutpoints) if savePlaneCuts else y)
        self.method = execute