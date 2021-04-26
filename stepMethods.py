from numba import jit
import numba
from numba.typed import List
from vectorAlgebra import *
import numpy as np


class stepMethods:
    '''
        Dette er en klasse hvilket i grunnen opptrer som et namespace.
        Instansiering av stepMethods-objekter her vil skape all verdens problemer, men dette ignoreres
        instanser skapes aldri

        Implementerer her de numeriske metodene
        på en systematisk metode for å forenkle bruk
        i rutine-klassen. Alle metodene er jit-akselerert
        og vil videre anvendes i routine sin execute-prosedyre
        (hvilket også vil jit-akselereres avhengig av om man spesifiserer dette ved instansiering)
        '''

    '''
    Alle metodene er implementert på samme form. Dette vil si at de tar inn
    en diff-liknings-funksjon f : y'=f(t,y),
    start-y-verdi (y_n), et start-tidspunkt (t_n), og en steglengde h
    Metodene returnerer neste y-verdi (y_(n+1)).

    (Selv om parameteren t_n strengt talt ikke er nødvendig, er dette implementert
    for å generalisere koden)

    Ett unntak fra denne regelen er at Kahans metode ikke tar inn en funksjon, men
    en tuppel matrixRepresentation hvilket består av matrisene a,b,c og formen (a,b,c) 
    slik angitt i oppgaveteksten. Disse representerer funksjonen f, og vil brukes for å løse 
    for y_(n+1). Når man bruker Kahans metode vil man ved instansiering av route bruke
    (a,b,c) i stedet for function. Ingen andre steder enn i disse stepMethods-metodene vil
    innholdet av denne parameteren ha noe å si (den passeres rundt med den tolking at den
    er et objekt hvilket kan brukes til å beregne y_n+1 fra y_n)

    En annen ting som er verd å merke er at dette namespacet inneholder en dictionary
    hvilket beskriver om hver metode baserer seg på en 'rollover', dvs, om et
    tidssteg baserer seg på data fra det forrige. Alle metodene som har rollover-verdi
    satt til True kreves det at metodene tar inn et argument 'rollover' hvilket
    inneholder disse data. Dessuten må de ha en 'metodenavn'InitialRollover-metode
    for å beregne rollover-data for starten (oftest hvor man gjør en ekstra funk-
    sjons-evaluering, som i StV). Metoder med spesifisiert rollover må også returnere en ekstra 
    rollover variabel til bruk i neste tidssteg
    
    Alle metodene er definert som motpart til det oppgavens forfatter kaller 'bad-f'
    det vil si, det antas funksjonene de bruker endrer verdiene til en input-referanse Fs
    I denne oppgaven brukes funksjonen 'goof' definert i scenario.py
    '''

    '''
               RK4 implementerer den 'berømte' runge-kutta-metoden
               hvilket har orden 4. Fremgangsmåten er slik beskrevet i oppgaven
       '''
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

    '''
                BS3 implementerer en Bogacki-Shampine-metode av orden 3. 
                Fremgangsmåten er slik beskrevet i oppgaven
    '''
    @jit(nopython=True)
    def BS3(function, yValues, t, h):
        dim = len(yValues)
        Fs = np.zeros((3, dim))*1.0
        function(t, yValues, Fs[0,:])
        function(t + h / 2, yValues + h / 2 * Fs[0,:], Fs[1,:])
        function(t + 3 * h / 4, yValues + h * 3 / 4 * Fs[1,:], Fs[2,:])
        yNext = yValues + h / 9 * (Fs[0,:] * 2 + Fs[1,:] * 3 + Fs[2,:] * 4)
        return yNext

    '''
                Kah implementerer Kahans metode med orden 2
                slik spesifisert i oppgaven. 
    '''
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

    '''
                StV implementerer en Störmer-Verlet-metode
                av orden 2. Merk at denne også tar inn rollover.
                Under er det også definert en initialRollover-metode
                hvilket produserer den første rollover-verdien til bruk
                i første funksjons-kjøring.
    '''
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

    @jit(nopython=True)
    def Eul(function, yValues, t, h):
        F = np.zeros(len(yValues))
        function(t, yValues, F[:])
        return yValues + h*F;

    @jit(nopython=True)
    def Trp(function, yValues, t, h):
        F1 = np.zeros(len(yValues))
        function(t, yValues, F1[:])
        F2 = np.zeros(len(yValues))
        function(t+h, yValues+h*F1, F2[:])
        return yValues + h*(F1+F2)/2

    @jit(nopython=True)
    def Mid(function, yValues, t, h):
        F1 = np.zeros(len(yValues))
        function(t, yValues, F1[:])
        F2 = np.zeros(len(yValues))
        function(t+h/2, yValues+h/2*F1, F2[:])
        return yValues + h*F2


    hasRollover = {"RK4":False, "BS3":False, "Kah":False, "StV":True,
                   "Eul":False, "Trp":False, "Mid":False}




class routine:
    def __init__(self, **kwargs):
        ''''
        :param kwargs:
            (float) tInit, tFinal                               | default: tInit=0, tFinal=1
            (float) ordinaryStepLen eller (int) steps           | default: ordinaryStepLen=1
            (string) method {RK4, BS3, Ka2, SV2}                | default: RK4
            (ndim float) y0                                     | default: 0-vec
            (objekt hvilket representerer f:y'=f) f             | default: 0-function

            funksjonalitets-parametre:
            (bool) timeline
                Hvorvidt hver y-verdi i kjøring skal lagres
            (int) timelineJumps
                Antall verdier i timeline man skipper (+1) før man lagrer en ny verdi
            (bool) nopythonExe
                Hvorvidt execution-delen skal kompileres med jit-nopython (i tillegg til funksjonsevalueringer)
            (bool) savePlaneCuts
                Hvordvidt alle poincare-kutt med de spesifiserte plan for instansen (lagret i instans.planes)
                skal lagres i en liste, hvilket omsider returneres ved slutt av instans.run()

        '''

        #I de kommende linjene lagres alle instans-variablene slik angitt ved initialisering
        self.tInit = kwargs.get("tInit", 0)
        self.tFinal = kwargs.get("tFinal", 1)

        #Avhengig om stepAmtBased/step-Amount-based eller step-length-based vil det lagres antall steps eller steg-lengde
        #Dette valget vedvarer i execution-metoden
        self.stepAmtBased = "steps" in kwargs
        if  (self.stepAmtBased):  self.steps=kwargs.get("steps")
        else                   :  self.ordinaryStepLen = kwargs.get("ordinaryStepLen", 1)


        self.y0 = kwargs.get("y0") #Henter den n-dimensjonale startverdien y
        self.saveTimeline=kwargs.get('timeline', False) #Angir om en rekke punkter skal lagres for tilstander under integrasjonen
        self.timelineJumps=kwargs.get('timelineJumps', 1) #Angir hopp mellom beregnede y-verdier som lagres (1 lagrer alle, 2 lager annen-hver osv.)

        self.execution = None #Dette er metoden hvilket kalles i instans.run. Den defineres opprinnelig som none

        self.nopython=kwargs.get('nopythonExe', False) #Angir hvorvidt execute skal jit-kompileres

        self.savePlaneCuts = kwargs.get("savePlaneCuts", False) #Angir hvorvidt poincare-cuts skal lagres for plan i instans.planes
        self.planes = [] #Array av hyperplan aktuelle for poincare-kutt.

        self.methodName = kwargs.get("method", "RK4") #
        self.function = kwargs.get("f")

    #run er funksjonen en bruker når en ønsker å kjøre rutinen spesifisert av input-parametrene
    #Den kaller på _load hvilket forbereder kjøring. Deretter utfører den execution assosiert med instansen.
    #Resultatet fra denne blir returnert av run.
    def run(self):
        self._load()
        return self.execution()

    #Funksjon som ikke skal kalles utenfra. _load forbereder kjøring av prosedyren, deriblant med å (avhengig av valg) jit-kompilere prosedyren
    def _load(self, timeline=False):
        tInterval = self.tFinal-self.tInit          #Definerer lengde på t-intervallet
        tInit = self.tInit                          #Henter ut start-tidspunkt
        normalSteps = int(np.floor(tInterval/self.ordinaryStepLen)) if not self.stepAmtBased else self.steps
                                                    #NormalSteps er antall steg hvilket vil utføres med
        endStep = not self.stepAmtBased             #'vanlig steglengde'. Dersom prosedyren er step-length-basert,
                                                    #legges det også til et sluttsteg hvis eksistens er beskrevet av bool endStep
        totalSteps = normalSteps + endStep          #Variabel hvilket beskriver totalt antall steg prosedyren baseres på
        totalPoints = totalSteps + 1                #Variabel hvilket beskriver totalt antall y-verdier som prosedyren gjennomgår
        stepLen = float(self.ordinaryStepLen if not self.stepAmtBased else tInterval/self.steps)
                                                    #stepLen angir lengden på de normale (ikke nødv. endstep) stegene under prosedyren
        methodYInitial = self.y0                    #Henter ut start-verdi

        if(self.methodName in stepMethods.__dict__): method = stepMethods.__dict__[self.methodName]
        else: raise Exception("No applicable method found for gived method-name")
                                                    #Henter ut den aktuelle numeriske utviklingsmetoden method fra stepMethods

        function = self.function                    #Henter funksjonen fra instansen


        saveTimeline = self.saveTimeline            #Henter div valgfrie funksjonalitets-parametre som er angitt i konstruktøren
        savePlaneCuts = self.savePlaneCuts


        methodHasRollover = stepMethods.hasRollover.get(self.methodName) # Henter om den aktuelle utviklingsmetoden inkluderer rollover (baserer seg på forrgige kjøring)
        rolloverInit = stepMethods.__dict__[self.methodName + "InitialRollover"](function, methodYInitial, tInit, stepLen) if methodHasRollover else None

        dimension = len(methodYInitial)             #i våre beregninger blir dette alltid 4
                                                    #men mulighet for vilkårlig-dimensjonelle systemer er inkludert
        planesAmt = len(self.planes)
        planes = self.planes
        planesLocations = np.array([pl.position for pl in planes]) #Henter ut liste av alle plan-posisjoner
        planesNormals = np.array([pl.normal for pl in planes]) #Henter ut liste av alle plan-normaler
        planesBasis = np.array([pl.basis for pl in planes]) #Henter ut liste av alle plan-basiser

        jumpLen = self.timelineJumps                #Tallfester hopp mellom y-verdiene som lagres for return-y.
        totTimelinePts = totalPoints // jumpLen     #Tallfester antall y-verdier som lagres for return-y-list

        global normalize, siz, dot
        jitNormalize = jit(nopython=True)(normalize)    #Definerer ekvivalente vektor-algebra uttrykk som i
        jitSiz = jit(nopython=True)(siz)                #vectorAlgebra.py, men som er jit-akselerert
        jitDot = jit(nopython=True)(dot)

        '''
        jitRegisterCuts er en en jit-akselerert metode hvilket for en gitt
        input-parameter inputPosition angir hvilken side av alle relevante plan (fra instance.planes)
        det gitte punktet ligger på. Den returnerer altså en liste av bools. 
        I videre beregninger vil et 'kutt' registreres når man over ett tidssteg oppdager
        en endring i en slik bool korresponderende til et plan.
        '''
        @jit(nopython=True)
        def jitRegisterCuts(inputPosition):             #
            nonlocal planesLocations, planesAmt, planesNormals #Henter posisjoner, antall plan og plan-retninger
            def jitHalfPlane(inputPosition, planePosition, normal): #Finner hvilken side ett punkt ligger i forhold til ett gitt plan
                return np.sum((inputPosition - planePosition) * normal) >= 0
            newCutState = [jitHalfPlane(inputPosition, planesLocations[enum], planesNormals[enum])
                           for enum in range(planesAmt)]
                    #cutState vil kun tilsvare en liste (med rekkefølge som i rekkefølgen av
                    #instans.planes) av bools hvor hver bool forteller om inputPusition er på oppsiden
                    # (siden indikert av hyperplanets normal) (True) eller nedsiden (False)
            return np.array(newCutState) #Returnerer newCutState men castet til numpy-array

        '''
        planesCutCoordinates er en jit-akselerert funksjon hvilket tar inn 
        en ndim-array inputposition, samt en ndim-array direction
        sammen med en ortogonal-basis og planePosition tilhørende et hyperplan.
        Metoden vil finne punktet i planet (målt etter planets egen basis)
        hvilket også ligger på aksen utspent av direction fra startpunktet inputPosition
        Gitt direction er en vektor parallell med vektoren fra punkt før skjæring til punkt etter skæring
        vil man finne den lineære interpolasjonen mellom disse punktene hvilket svarer til
        skjæring.
        '''
        @jit(nopython=True)
        def planeCutCoordinates(direction, inputPosition, basis, planePosition):
            nonlocal dimension, methodHasRollover, function
            planeNormal = basis[0]              #Henter ut normalvektoren til hyperplanet
            planeDistance = jitDot((inputPosition-planePosition),planeNormal) #Finner avstand fra inputposition til plan
            requiredTravel = -planeDistance/jitDot(planeNormal,direction) #Finner antall ganger man må legge
                                                #direction til inputPosition for å havne i det aktuelle hyperplan
            position = inputPosition + requiredTravel*direction #finner posisjonen denne skjæringen korresponderer til
            crd = np.zeros(dimension-1)
            for i in range(dimension-1):        #Finner koordinatene for dette punktet angitt i hyperplanets
                crd[i] = np.sum(position*basis[i+1]) #n-1 utspennende vektorer
            return crd

        '''
        Definerer her execute-funksjonen som kjøres når man ønsker å simulere systemet.
        Først defineres en decorator, hvilket avhengig av om man satte nopyhtonExe til sann
        ved instansiering, enten vil jit-kompilere funksjonen, eller gjøre ingenting.
        Metoden henter methodYInitial, og integrerer stegvis med den angitte steg-prosedyren til
        den til slutt returnerer y-verdier, tidspunkter og/eller plane-cuts avhengig av hva man 
        tidligere ble spesifisert.
        '''
        decorator = jit(nopython=True) if self.nopython else emptyDecorator
        @decorator
        def execute():
            nonlocal methodHasRollover, rolloverInit, saveTimeline, methodYInitial, totalPoints, planesBasis
                        #Henter div variabler fra skop utenfor jit-akselerasjon
            y = methodYInitial                  #Setter y lik start-y
            prevstepY = np.zeros_like(y)*1.0    #Definerer previous-step
            arr = times = None                  #definerer arr og times, hvilket
                                                #forblir som none dersom saveTimeline er false
                                                #eller blir redefinert som en numpy-array av verdier
                                                #senere om saveTimeLine er true
            h = 0.0                             #Deklarerer steglengde-variabelen
            time = 0.0                          #Deklarerer current-time-variabelen
            index = 0                           #Definerer index-variabel som angir index for steg
                                                #hvilket stegvis økes opp til totalSteps under kjøring

            if(saveTimeline) :                  #Redefinerer arr og times om saveTimeLine er True, som nevnt
                arr = np.zeros((totTimelinePts,len(y),)) * 0.0; #For n-dimensjonelt-system vil arr bli
                arr[0,:] = y                                    #en array av totTimeLinePts antall n-dimensjonelle verdier
                times = np.zeros((totTimelinePts))    * 0.0;    #Times vil bli en totTimeLinePts lang skalar-liste
                times[0] = tInit                #Man definerer første element i arr og t som initial-verdier


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
                                    #Jeg skriver dette i frustrasjon

            cutState = jitRegisterCuts(y)
                                    #Lagrer cut-state for start-posisjonen

            '''
            checkCuts er en metode hvilket stegvis sjekker om cutState, slik beregnet
            i jitRegisterCuts, endres over et tidsteg for hvert plan. Dersom dette er tilfellet, 
            vil punktets 'nedfall' i det aktuelle planet som ble skjært i forrige tidssteg
            registreres. Metoden for 'nedfall' er lineær-regresjon med forrige y-verdi før skjæring
            Nedfallets punkt i det aktuelle planets egne koordinaet registreres i listen cutpoints[i],
            'i' er her indeksen til det gitte planet i instans.planes (rekkefølgen og indeks på planene bevares 
            i alle metoder de er involvert i)
            '''
            def checkCuts():
                nonlocal cutState, cutpoints, function
                newCutState = jitRegisterCuts(y) #Beregner ny cut-state
                if(planesAmt and savePlaneCuts): #if-sjekk hvilket må være med for jit-kompilasjon
                                                #enkelte metoder innenfor her er kun definert dersom en av disse
                                                #er false. Selv om disse metodene aldri kjøres i dette tilfellet
                                                #er if-setningen nødvendig for å få jit til å kompilere.
                    for i in range(planesAmt): #Looper gjennom alle plan
                        if newCutState[i]!=cutState[i]: #Sjekker om cutState endret med forrige
                            cutState[i]=newCutState[i]  #Ved endring, endres cutState til ny verdi
                            direction = y - prevstepY   #vektoren mellom det nåværende og det forrige punktet hentes
                            cutpt = planeCutCoordinates(direction, y, planesBasis[i], planesLocations[i])
                                                        #skjæringen mellom aksen utspent av denne vektoren fra
                                                        #den nåværende y-verdien med plan indeks i registreres
                                                        # og lagres i cutpoints[i]
                            cutpoints[i].append(cutpt)


            while(index < totalSteps):          #Looper gjennom alle steg som skal utføres
                time = tInit + index*stepLen    #finner nåværende tidssteg
                h = stepLen if index!=normalSteps else tInterval-normalSteps*stepLen
                                                #Bruker standard steglengde om man ikke
                                                #handler med enstep. Beregner endstep dersom
                                                #steget er et endstep (dvs index == (normalSteps-1)+1)
                                                #normalSteps-1 er siste index med kvalifikasjon som normal step

                prevstepY = y;                  #Setter forrige step til nåværende y
                if(methodHasRollover) : y, rollover = method(function, y, time, h, rollover)
                else : y = method(function, y, time, h)
                                                #Beregner neste y-verdi. Avhengig av om metoden brukt har
                                                #innebygget rollover, vil method-kallet reflektere dette
                index += 1;                     #Øker index
                if (saveTimeline and index%jumpLen==0): times[index//jumpLen] = time+h; arr[index//jumpLen, :] = y
                                                #Dersom savetimeline er sann, og index mod jumpLen er 0
                                                #vil det aktuelle elemenet lagres.
                                                #(dvs hvert (jumpLen)-te element) vil lagres i times og arr
                checkCuts()
            return ((arr, times, cutpoints) if savePlaneCuts else (arr, times)) \
                                       if saveTimeline else \
                            ((y, cutpoints) if savePlaneCuts else y)
            #Om saveTimeline & savePlaneCuts er
                    #True, True returneres arr, times, cutpoints
                    #False, True returneres y_slutt, cutpoints
                    #True, False returneres arr, times
                    #False, False returneres y_slutt
        self.execution = execute