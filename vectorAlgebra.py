import numpy as np


#Implementerer her en metode for å lage en n-dim ortogonal enhetsbasis hvor første element er
#input-variabelen zeta. Dette er brukbart når en ønsker en basis for et hyperplan med en gitt normalvektor
#Jeg går ikke inn på detaljer for denne implementasjonen
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

#dekorator-funksjon hvilket ikke endrer den dekorerte funksjonen. (Brukes i stedet for jit
# når man ikke ønsker jit-akselerasjon)
def emptyDecorator(f):
    return f

def normalize(zeta): return zeta/siz(zeta) #Normaliserings-funksjon
def siz(zeta): return np.sqrt(np.sum(zeta**2))  #Størrelses-funksjon i 2-norm
def orthoProj(a,b): return a-b*np.dot(a,b)/siz(b)**2 #funksjon hvilket finner ortogonal projeksjon av a på b
def ortoHyperProj(a,B): #funksjon hvilket finner ortogonal projeksjon av a på alle (antatt ortogonale) kolonnevektorer i B
    for b in B:
        a = orthoProj(a,b)
    return a
def cross(A): #ndimensjonelt kryssprodukt, dvs, finner vektoren ortogonal på n-1 kryssede n-dim-vektorer.
              #Definisjonen er som i 3d-kryss prod, altså determinant av en rad i,j,k,... av enhetsvektorer,
              #etterfulgt av koordinater for n-1 vektorer på rad-form.
    sh = np.shape(A)
    dim = sh[1]
    if(not sh[0]+1 == sh[1]): raise Exception("Inconsistent cross product")
    resultant = np.zeros(dim)
    for i in range(dim):
        ei = np.zeros(dim); ei[i]=1; ei = np.array([ei,])
        resultant[i] = np.linalg.det(np.concatenate([ei, A],axis=0))
    return resultant
def dot(a,b):
    return np.sum(a*b)

class plane:    #Klasse for å definere et hyperplan. Består av en punkt som planet krysser og en normal (eller alternativt en basis.
                # Dersom ingen basis spesifisert, vil en genereres fra normal med ndim-basis)
    def __init__(self, planePosition, **kwargs):
        if('basis' in kwargs): self.basis = kwargs.get('basis')
        elif('normal' in kwargs): self.basis = ndimBasis(kwargs.get("normal"))
        else: raise Exception("Specify basis or normal")
        self.normal = self.basis[0]
        self.position = planePosition
        self.dim = len(self.normal)
        if not (len(self.position) == len(self.normal)): raise Exception("Position and normal not in same vector-space")

    def halfspace(self, position):
        dot = np.dot(position-self.position, self.normal) #Finner halvrommet av et plan et gitt punkt ligger i
        return 1 if dot >= 0 else 0
    def projectedHyperplaneCoordinates(self, position): #Selvforklarende
        relpos = position - self.position
        return np.array([np.dot(self.basis[i+1], relpos) for i in range(self.dim-1)])



