import numpy as np
from stepMethods import *
#Her spesifiseres kun en rekke variabler spesifikke for vårt problem
#, ikke for prosedyren definert i stepMethods og routine.

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
#a-4x4x4-matrisen slik angitt i oppgaveteksten

b = np.array(
   [[0, 0, 1, 0],
    [0, 0, 0, 1],
    [-1, 0, 0, 0],
    [0, -1, 0, 0]])*1.0
#b-4x4-matrisen slik angitt i oppgaveteksten

c = np.array([0,0,0,0])*1.0
#c-4-matrisen slik angitt i oppgaveteksten

H0 = 1/12
q01 = 0; q02 = 0.45
p02 = 0
p01 = np.sqrt(2*H0-p02**2-(q01**2+q02**2)-2*q01**2*q02 +2/3*q02**3)
qp0 = np.array([q01,q02,p01, p02])
#Angir initial-conditions slik som i oppgaveteksten


def U(y):
    dim = np.shape(y)[1]//2
    q = y[:, :dim]
    qSep = q.T
    return 1/2*np.sum(q*q, axis=1) + qSep[0]**2*qSep[1]-1/3*qSep[1]**3
#Definerer potensiell-energi-funksjon

def K(y):
    dim = np.shape(y)[1]//2
    p=y[:,dim:]
    return 1/2*np.sum(p*p,axis=1)
#Definerer kinetisk-energi-funksjon

def H(y):
    return K(y)+ U(y)
#Definerer hamilton-operator

@jit(nopython=True)
def fun(t, y):  return np.array([y[2], y[3], -y[0]*(1+2*y[1]), -(y[1]+y[0]**2-y[1]**2)])

@jit(nopython=True)
def goof(t, y, F): #Dette er den jit-akselererte funksjonen hvilket angis i oppgaven. Den endrer på input-parameter-referansen F
    F[0] = y[2]
    F[1] = y[3]
    F[2] = -y[0]*(1+2*y[1])
    F[3] = -(y[1]+y[0]**2-y[1]**2)
    #F[:] = np.array([y[2], y[3], -y[0]*(1+2*y[1]), -(y[1]+y[0]**2-y[1]**2)])
