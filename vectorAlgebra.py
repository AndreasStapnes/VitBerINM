import numpy as np
from numba import jit
from numba.typed import List

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


def emptyDecorator(f):
    return f

def normalize(zeta): return zeta/siz(zeta)
def siz(zeta): return np.sqrt(np.sum(zeta**2))
def orthoProj(a,b): return a-b*np.dot(a,b)/siz(b)**2
def ortoHyperProj(a,B):
    for b in B:
        a = orthoProj(a,b)
    return a


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
        return 1 if dot >= 0 else 0




