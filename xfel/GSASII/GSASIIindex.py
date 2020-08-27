# -*- coding: utf-8 -*-


################################################################################ 
# NOTE ADDED TO CCTBX DISTRIBUTION 
# This file contains a modified version of code from GSAS-II (B. H. Toby, R. B.
# Von Dreele; https://doi.org/10.1107/S0021889813003531). GSAS-II is distributed
# under an open-source license which is reproduced at the end of this file. 
# This product includes software produced by UChicago Argonne, LLC
# under Contract No. DE-AC02-06CH11357 with the Department of Energy.
################################################################################

#GSASII cell indexing program: variation on that of A. Coehlo
#   includes cell refinement from peak positions (not zero as yet)
'''
*GSASIIindex: Cell Indexing Module*
===================================

Cell indexing program: variation on that of A. Coehlo
includes cell refinement from peak positions
'''

from __future__ import division, print_function
import math
import time
import numpy as np
from xfel.GSASII import GSASIIlattice as G2lat
import scipy.optimize as so
from cctbx import sgtbx


# This was originally set in GSASIIpwd. I have doubts about the wisdom of 
# keeping it permanently.
np.seterr(divide='ignore')

# trig functions in degrees
acosd = lambda x: 180.*math.acos(x)/math.pi
    
def scaleAbyV(A,V):
    'needs a doc string'
    v = G2lat.calc_V(A)
    scale = math.exp(math.log(v/V)/3.)**2
    for i in range(6):
        A[i] *= scale
    
def ranaxis(dmin,dmax):
    'needs a doc string'
    import random as rand
    return rand.random()*(dmax-dmin)+dmin
    
def ran2axis(k,N):
    'needs a doc string'
    import random as rand
    T = 1.5+0.49*k/N
#    B = 0.99-0.49*k/N
#    B = 0.99-0.049*k/N
    B = 0.99-0.149*k/N
    R = (T-B)*rand.random()+B
    return R
    
    
def ranAbyV(Bravais,dmin,dmax,V):
    'needs a doc string'
    cell = [0,0,0,0,0,0]
    bad = True
    while bad:
        bad = False
        cell = rancell(Bravais,dmin,dmax)
        G,g = G2lat.cell2Gmat(cell)
        A = G2lat.Gmat2A(G)
        if G2lat.calc_rVsq(A) < 1:
            scaleAbyV(A,V)
            cell = G2lat.A2cell(A)
            for i in range(3):
                bad |= cell[i] < dmin
    return A
    
def ranAbyR(Bravais,A,k,N,ranFunc):
    'needs a doc string'
    R = ranFunc(k,N)
    if Bravais in [0,1,2]:          #cubic - not used
        A[0] = A[1] = A[2] = A[0]*R
        A[3] = A[4] = A[5] = 0.
    elif Bravais in [3,4]:          #hexagonal/trigonal
        A[0] = A[1] = A[3] = A[0]*R
        A[2] *= R
        A[4] = A[5] = 0.        
    elif Bravais in [5,6]:          #tetragonal
        A[0] = A[1] = A[0]*R
        A[2] *= R
        A[3] = A[4] = A[5] = 0.        
    elif Bravais in [7,8,9,10,11,12]:     #orthorhombic
        A[0] *= R
        A[1] *= R
        A[2] *= R
        A[3] = A[4] = A[5] = 0.        
    elif Bravais in [13,14,15]:        #monoclinic
        A[0] *= R
        A[1] *= R
        A[2] *= R
        A[4] *= R
        A[3] = A[5] = 0.        
    else:                           #triclinic
        A[0] *= R
        A[1] *= R
        A[2] *= R
        A[3] *= R
        A[4] *= R
        A[5] *= R
    return A
    
def rancell(Bravais,dmin,dmax):
    'needs a doc string'
    if Bravais in [0,1,2]:          #cubic
        a = b = c = ranaxis(dmin,dmax)
        alp = bet = gam = 90
    elif Bravais in [3,4]:          #hexagonal/trigonal
        a = b = ranaxis(dmin,dmax)
        c = ranaxis(dmin,dmax)
        alp = bet =  90
        gam = 120
    elif Bravais in [5,6]:          #tetragonal
        a = b = ranaxis(dmin,dmax)
        c = ranaxis(dmin,dmax)
        alp = bet = gam = 90
    elif Bravais in [7,8,9,10,11,12]:       #orthorhombic - F,I,P - a<b<c convention
        abc = [ranaxis(dmin,dmax),ranaxis(dmin,dmax),ranaxis(dmin,dmax)]
        if Bravais in [7,8,12]:
            abc.sort()
        a = abc[0]
        b = abc[1]
        c = abc[2]
        alp = bet = gam = 90
    elif Bravais in [13,14,15]:        #monoclinic - C,P - a<c convention
        ac = [ranaxis(dmin,dmax),ranaxis(dmin,dmax)]
        if Bravais in [13,14]:
            ac.sort()
        a = ac[0]
        b = ranaxis(dmin,dmax)
        c = ac[1]
        alp = gam = 90
        bet = ranaxis(90.,140.)
    else:                           #triclinic - a<b<c convention
        abc = [ranaxis(dmin,dmax),ranaxis(dmin,dmax),ranaxis(dmin,dmax)]
        abc.sort()
        a = abc[0]
        b = abc[1]
        c = abc[2]
        r = 0.5*b/c
        alp = ranaxis(acosd(r),acosd(-r))
        r = 0.5*a/c
        bet = ranaxis(acosd(r),acosd(-r))
        r = 0.5*a/b
        gam = ranaxis(acosd(r),acosd(-r))  
    return [a,b,c,alp,bet,gam]
    
def calc_M20(peaks,HKL,ifX20=True):
    'needs a doc string'
    diff = 0
    X20 = 0
    for Nobs20,peak in enumerate(peaks):
        if peak[3]:
            Qobs = 1.0/peak[7]**2
            Qcalc = 1.0/peak[8]**2
            diff += abs(Qobs-Qcalc)
        elif peak[2]:
            X20 += 1
        if Nobs20 == 19: 
            d20 = peak[7]
            break
    else:
        d20 = peak[7]
        Nobs20 = len(peaks)
    for N20,hkl in enumerate(HKL):
        if hkl[3] < d20:
            break                
    Q20 = 1.0/d20**2
    if diff:
        M20 = Q20/(2.0*diff)
    else:
        M20 = 0
    if ifX20:
        M20 /= (1.+X20)
    return M20,X20
    
    
def sortM20(cells):
    'needs a doc string'
    #cells is M20,X20,Bravais,a,b,c,alp,bet,gam
    #sort highest M20 1st
    T = []
    for i,M in enumerate(cells):
        T.append((M[0],i))
    D = dict(zip(T,cells))
    T.sort()
    T.reverse()
    X = []
    for key in T:
        X.append(D[key])
    return X
                
    
                
def IndexPeaks(peaks,HKL):
    'needs a doc string'
    import bisect
    N = len(HKL)
    if N == 0: return False,peaks
    hklds = list(np.array(HKL).T[3])+[1000.0,0.0,]
    hklds.sort()                                        # ascending sort - upper bound at end
    hklmax = [0,0,0]
    for ipk,peak in enumerate(peaks):
        peak[4:7] = [0,0,0]                           #clear old indexing
        peak[8] = 0.
        if peak[2]:
            i = bisect.bisect_right(hklds,peak[7])          # find peak position in hkl list
            dm = peak[-2]-hklds[i-1]                         # peak to neighbor hkls in list
            dp = hklds[i]-peak[-2]
            pos = N-i                                       # reverse the order
            if dp > dm: pos += 1                            # closer to upper than lower
            if pos >= N:
                break
            hkl = HKL[pos]                                 # put in hkl
            if hkl[-1] >= 0:                                 # peak already assigned - test if this one better
                opeak = peaks[int(hkl[-1])]                 #hkl[-1] needs to be int here
                dold = abs(opeak[-2]-hkl[3])
                dnew = min(dm,dp)
                if dold > dnew:                             # new better - zero out old
                    opeak[4:7] = [0,0,0]
                    opeak[8] = 0.
                else:                                       # old better - do nothing
                    continue                
            hkl[-1] = ipk
            peak[4:7] = hkl[:3]
            peak[8] = hkl[3]                                # fill in d-calc
    for peak in peaks:
        peak[3] = False
        if peak[2]:
            if peak[-1] > 0.:
                for j in range(3):
                    if abs(peak[j+4]) > hklmax[j]: hklmax[j] = abs(peak[j+4])
                peak[3] = True
    if hklmax[0]*hklmax[1]*hklmax[2] > 0:
        return True,peaks
    else:
        return False,peaks  #nothing indexed!
        
def Values2A(ibrav,values):
    'needs a doc string'
    if ibrav in [0,1,2]:
        return [values[0],values[0],values[0],0,0,0]
    elif ibrav in [3,4]:
        return [values[0],values[0],values[1],values[0],0,0]
    elif ibrav in [5,6]:
        return [values[0],values[0],values[1],0,0,0]
    elif ibrav in [7,8,9,10,11,12]:
        return [values[0],values[1],values[2],0,0,0]
    elif ibrav in [13,14,15]:
        return [values[0],values[1],values[2],0,values[3],0]
    else:
        return list(values[:6])
        
def A2values(ibrav,A):
    'needs a doc string'
    if ibrav in [0,1,2]:
        return [A[0],]
    elif ibrav in [3,4,5,6]:
        return [A[0],A[2]]
    elif ibrav in [7,8,9,10,11,12]:
        return [A[0],A[1],A[2]]
    elif ibrav in [13,14,15]:
        return [A[0],A[1],A[2],A[4]]
    else:
        return A
        
def FitHKL(ibrav,peaks,A,Pwr):
    'needs a doc string'
                
    def errFit(values,ibrav,d,H,Pwr):
        A = Values2A(ibrav,values)
        Qo = 1./d**2
        Qc = G2lat.calc_rDsq(H,A)
        return (Qo-Qc)*d**Pwr
        
    def dervFit(values,ibrav,d,H,Pwr):
        if ibrav in [0,1,2]:
            derv = [H[0]*H[0]+H[1]*H[1]+H[2]*H[2],]
        elif ibrav in [3,4,]:
            derv = [H[0]*H[0]+H[1]*H[1]+H[0]*H[1],H[2]*H[2]]
        elif ibrav in [5,6]:
            derv = [H[0]*H[0]+H[1]*H[1],H[2]*H[2]]
        elif ibrav in [7,8,9,10,11,12]:
            derv = [H[0]*H[0],H[1]*H[1],H[2]*H[2]]
        elif ibrav in [13,14,15]:
            derv = [H[0]*H[0],H[1]*H[1],H[2]*H[2],H[0]*H[2]]
        else:
            derv = [H[0]*H[0],H[1]*H[1],H[2]*H[2],H[0]*H[1],H[0]*H[2],H[1]*H[2]]
        derv = -np.array(derv)
        return (derv*d**Pwr).T
    
    Peaks = np.array(peaks).T
    values = A2values(ibrav,A)
    result = so.leastsq(errFit,values,Dfun=dervFit,full_output=True,ftol=0.000001,
        args=(ibrav,Peaks[7],Peaks[4:7],Pwr))
    A = Values2A(ibrav,result[0])
    return True,np.sum(errFit(result[0],ibrav,Peaks[7],Peaks[4:7],Pwr)**2),A,result
               
def rotOrthoA(A):
    'needs a doc string'
    return [A[1],A[2],A[0],0,0,0]
    
def swapMonoA(A):
    'needs a doc string'
    return [A[2],A[1],A[0],0,A[4],0]
    
def oddPeak(indx,peaks):
    'needs a doc string'
    noOdd = True
    for peak in peaks:
        H = peak[4:7]
        if H[indx] % 2:
            noOdd = False
    return noOdd
    
def halfCell(ibrav,A,peaks):
    'needs a doc string'
    if ibrav in [0,1,2]:
        if oddPeak(0,peaks):
            A[0] *= 2
            A[1] = A[2] = A[0]
    elif ibrav in [3,4,5,6]:
        if oddPeak(0,peaks):
            A[0] *= 2
            A[1] = A[0]
        if oddPeak(2,peaks):
            A[2] *=2
    else:
        if oddPeak(0,peaks):
            A[0] *=2
        if oddPeak(1,peaks):
            A[1] *=2
        if oddPeak(2,peaks):
            A[2] *=2
    return A
    
def getDmin(peaks):
    'needs a doc string'
    return peaks[-1][-2]
    
def getDmax(peaks):
    'needs a doc string'
    return peaks[0][-2]
    
    
def refinePeaks(peaks,ibrav,A,ifX20=True):
    'needs a doc string'
    dmin = getDmin(peaks)
    smin = 1.0e10
    pwr = 8
    maxTries = 10
    OK = False
    tries = 0
    sgtype = G2lat.make_sgtype(ibrav)
    HKL = G2lat.GenHBravais(dmin,ibrav,A, sg_type=sgtype)
    while len(HKL) > 2 and IndexPeaks(peaks,HKL)[0]:
        Pwr = pwr - (tries % 2)
        HKL = []
        tries += 1
        osmin = smin
        oldA = A[:]
        Vold = G2lat.calc_V(oldA)
        OK,smin,A,result = FitHKL(ibrav,peaks,A,Pwr)
        Vnew = G2lat.calc_V(A)
        if Vnew > 2.0*Vold or Vnew < 2.:
            A = ranAbyR(ibrav,oldA,tries+1,maxTries,ran2axis)
            OK = False
            continue
        try:
            HKL = G2lat.GenHBravais(dmin,ibrav,A, sgtype)
        except FloatingPointError:
            A = oldA
            OK = False
            break
        if len(HKL) == 0: break                         #absurd cell obtained!
        rat = (osmin-smin)/smin
        if abs(rat) < 1.0e-5 or not OK: break
        if tries > maxTries: break
    if OK:
        OK,smin,A,result = FitHKL(ibrav,peaks,A,2)
        Peaks = np.array(peaks).T
        H = Peaks[4:7]
        try:
            Peaks[8] = 1./np.sqrt(G2lat.calc_rDsq(H,A))
            peaks = Peaks.T
        except FloatingPointError:
            A = oldA
        
    M20,X20 = calc_M20(peaks,HKL,ifX20)
    return len(HKL),M20,X20,A
        
def findBestCell(dlg,ncMax,A,Ntries,ibrav,peaks,V1,ifX20=True):
    'needs a doc string'
# dlg & ncMax are used for wx progress bar 
# A != 0 find the best A near input A,
# A = 0 for random cell, volume normalized to V1;
# returns number of generated hkls, M20, X20 & A for best found
    mHKL = [3,3,3, 5,5, 5,5, 7,7,7,7,7,7, 9,9,9, 10]
    dmin = getDmin(peaks)-0.05
    amin = 2.5
    amax = 5.*getDmax(peaks)
    Asave = []
    GoOn = True
    Skip = False
    if A:
        HKL = G2lat.GenHBravais(dmin,ibrav,A[:])
        if len(HKL) > mHKL[ibrav]:
            peaks = IndexPeaks(peaks,HKL)[1]
            Asave.append([calc_M20(peaks,HKL,ifX20),A[:]])
    tries = 0
    while tries < Ntries and GoOn:
        if A:
            Abeg = ranAbyR(ibrav,A,tries+1,Ntries,ran2axis)
            if ibrav > 12:         #monoclinic & triclinic
                Abeg = ranAbyR(ibrav,A,tries/10+1,Ntries,ran2axis)
        else:
            Abeg = ranAbyV(ibrav,amin,amax,V1)
        HKL = G2lat.GenHBravais(dmin,ibrav,Abeg)
        Nc = len(HKL)
        if Nc >= ncMax:
            GoOn = False
        else:
            if dlg:
                dlg.Raise()
                GoOn = dlg.Update(100*Nc/ncMax)[0]
                if Skip or not GoOn:
                    GoOn = False
                    break
        
        if IndexPeaks(peaks,HKL)[0] and len(HKL) > mHKL[ibrav]:
            Lhkl,M20,X20,Aref = refinePeaks(peaks,ibrav,Abeg,ifX20)
            Asave.append([calc_M20(peaks,HKL,ifX20),Aref[:]])
            if ibrav in [9,10,11]:                          #C-centered orthorhombic
                for i in range(2):
                    Abeg = rotOrthoA(Abeg[:])
                    Lhkl,M20,X20,Aref = refinePeaks(peaks,ibrav,Abeg,ifX20)
                    HKL = G2lat.GenHBravais(dmin,ibrav,Aref)
                    peaks = IndexPeaks(peaks,HKL)[1]
                    Asave.append([calc_M20(peaks,HKL,ifX20),Aref[:]])
            elif ibrav == 13:                      #C-centered monoclinic
                Abeg = swapMonoA(Abeg[:])
                Lhkl,M20,X20,Aref = refinePeaks(peaks,ibrav,Abeg,ifX20)
                HKL = G2lat.GenHBravais(dmin,ibrav,Aref)
                peaks = IndexPeaks(peaks,HKL)[1]
                Asave.append([calc_M20(peaks,HKL,ifX20),Aref[:]])
        else:
            break
        Nc = len(HKL)
        tries += 1
    X = sortM20(Asave)
    if X:
        Lhkl,M20,X20,A = refinePeaks(peaks,ibrav,X[0][1],ifX20)
        return GoOn,Skip,Lhkl,M20,X20,A        
    else:
        return GoOn,Skip,0,0,0,0
        
def monoCellReduce(ibrav,A):
    'needs a doc string'
    a,b,c,alp,bet,gam = G2lat.A2cell(A)
    G,g = G2lat.A2Gmat(A)
    if ibrav in [13]:
        u = [0,0,-1]
        v = [1,0,2]
        anew = math.sqrt(np.dot(np.dot(v,g),v))
        if anew < a:
            cang = np.dot(np.dot(u,g),v)/(anew*c)
            beta = acosd(-abs(cang))
            A = G2lat.cell2A([anew,b,c,90,beta,90])
    else:
        u = [-1,0,0]
        v = [1,0,1]
        cnew = math.sqrt(np.dot(np.dot(v,g),v))
        if cnew < c:
            cang = np.dot(np.dot(u,g),v)/(a*cnew)
            beta = acosd(-abs(cang))
            A = G2lat.cell2A([a,b,cnew,90,beta,90])
    return A

def DoIndexPeaks(peaks,controls,bravais,dlg,ifX20=True,timeout=None):
    'needs a doc string'
    
    delt = 0.005                                     #lowest d-spacing cushion - can be fixed?
    amin = 2.5
    amax = 5.0*getDmax(peaks)
    dmin = getDmin(peaks)-delt
    bravaisNames = ['Cubic-F','Cubic-I','Cubic-P','Trigonal-R','Trigonal/Hexagonal-P',
        'Tetragonal-I','Tetragonal-P','Orthorhombic-F','Orthorhombic-I','Orthorhombic-A',
        'Orthorhombic-B','Orthorhombic-C',
        'Orthorhombic-P','Monoclinic-I','Monoclinic-C','Monoclinic-P','Triclinic']
    tries = ['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th']
    N1s = [1,1,1,   5,5,  5,5, 50,50,50,50,50,50,  100,100,100, 200]
    N2s = [1,1,1,   2,2,  2,2,     2,2,2,2,2,2,   2,2,2,   4]
    Nm  = [1,1,1,   1,1,  1,1,     1,1,1,1,1,1,   2,2,2,   4]
    notUse = 0
    for peak in peaks:
        if not peak[2]:
            notUse += 1
    Nobs = len(peaks)-notUse
    zero,ncno = controls[1:3]
    ncMax = Nobs*ncno
    print ("%s %8.3f %8.3f" % ('lattice parameter range = ',amin,amax))
    print ("%s %.4f %s %d %s %d" % ('Zero =',zero,'Nc/No max =',ncno,' Max Nc =',ncno*Nobs))
    cells = []
    lastcell = np.zeros(7)
    for ibrav in range(17):
        begin = time.time()
        if bravais[ibrav]:
            print ('cell search for ',bravaisNames[ibrav])
            print ('      M20  X20  Nc       a          b          c        alpha       beta      gamma     volume      V-test')
            V1 = controls[3]
            bestM20 = 0
            topM20 = 0
            cycle = 0
            while cycle < 5:
                if dlg:
                    dlg.Raise()
                    dlg.Update(0,newmsg=tries[cycle]+" cell search for "+bravaisNames[ibrav])
                try:
                    GoOn = True
                    while GoOn:                                                 #Loop over increment of volume
                        N2 = 0
                        while N2 < N2s[ibrav]:                                  #Table 2 step (iii)               
                            if time.time() - begin > timeout: 
                                GoOn = False
                                break
                            if ibrav > 2:
                                if not N2:
                                    A = []
                                    GoOn,Skip,Nc,M20,X20,A = findBestCell(dlg,ncMax,A,Nm[ibrav]*N1s[ibrav],ibrav,peaks,V1,ifX20)
                                    if Skip:
                                        break
                                if A:
                                    GoOn,Skip,Nc,M20,X20,A = findBestCell(dlg,ncMax,A[:],N1s[ibrav],ibrav,peaks,0,ifX20)
                            else:
                                GoOn,Skip,Nc,M20,X20,A = findBestCell(dlg,ncMax,0,Nm[ibrav]*N1s[ibrav],ibrav,peaks,V1,ifX20)
                            if Skip:
                                break
                            elif Nc >= ncMax:
                                GoOn = False
                                break
                            elif 3*Nc < Nobs:
                                N2 = 10
                                break
                            else:
                                if not GoOn:
                                    break
                                if 1.e6 > M20 > 1.0:    #exclude nonsense
                                    bestM20 = max(bestM20,M20)
                                    A = halfCell(ibrav,A[:],peaks)
                                    if ibrav in [14,]:
                                        A = monoCellReduce(ibrav,A[:])
                                    HKL = G2lat.GenHBravais(dmin,ibrav,A)
                                    peaks = IndexPeaks(peaks,HKL)[1]
                                    a,b,c,alp,bet,gam = G2lat.A2cell(A)
                                    V = G2lat.calc_V(A)
                                    if M20 >= 10.0 and X20 <= 2:
                                        cell = [M20,X20,ibrav,a,b,c,alp,bet,gam,V,False,False, Nc]
                                        newcell = np.array(cell[3:10])
                                        if not np.allclose(newcell,lastcell):
                                            print ("%10.3f %3d %3d %10.5f %10.5f %10.5f %10.3f %10.3f %10.3f %10.2f %10.2f %s"  \
                                                %(M20,X20,Nc,a,b,c,alp,bet,gam,V,V1,bravaisNames[ibrav]))
                                            cells.append(cell)
                                        lastcell = np.array(cell[3:10])
                            if not GoOn:
                                break
                            N2 += 1
                        if Skip:
                            cycle = 10
                            GoOn = False
                            break
                        if ibrav < 13:
                            V1 *= 1.1
                        elif ibrav in range(13,17):
                            V1 *= 1.025
                        if not GoOn:
                            if bestM20 > topM20:
                                topM20 = bestM20
                                if cells:
                                    V1 = cells[0][9]
                                else:
                                    V1 = controls[3]
                                ncMax += Nobs
                                cycle += 1
                                print ('Restart search, new Max Nc = %d'%ncMax)
                            else:
                                cycle = 10
                finally:
                    pass
#                dlg.Destroy()
            print ('%s%s%s%s%s%d'%('finished cell search for ',bravaisNames[ibrav], \
                ', elapsed time = ',G2lat.sec2HMS(time.time()-begin),' Vfinal ',V1))
            
    if cells:
        return True,dmin,cells
    else:
        return False,0,[]
        
        


#            General Structure Analysis System - II (GSAS-II)
#                          OPEN SOURCE LICENSE
# 
# Copyright 2010, UChicago Argonne, LLC, Operator of Argonne National Laboratory
# All rights reserved.
# 
# GSAS-II may be used by anyone on a royalty-free basis. Use and
# redistribution, with or without modification, are permitted provided
# that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Software changes, modifications, or derivative works should be noted
#   with comments and the author and organization's name.
# * Distribution of changed, modified or derivative works based on
#   GSAS-II grants the GSAS-II copyright holder unrestricted permission
#   to include any, or all, new and changed code in future GSAS-II
#   releases.
# * Redistributions that include binary forms must include all relevant
#   source code and reproduce the above copyright notice, this list of
#   conditions and the following disclaimers in the documentation and/or
#   other materials provided with the distribution.
# * Neither the names of UChicago Argonne, LLC or the Department of
#   Energy nor the names of its contributors may be used to endorse or
#   promote products derived from this software without specific prior
#   written permission.
# * The software and the end-user documentation included with the
#   redistribution, if any, must include the following acknowledgment:
#   "This product includes software produced by UChicago Argonne, LLC
#   under Contract No. DE-AC02-06CH11357 with the Department of Energy."
# 
# *****************************************************************************
# WARRANTY DISCLAIMER: THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY
# OF ANY KIND. THE COPYRIGHT HOLDERS, THEIR THIRD PARTY LICENSORS, THE
# UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND THEIR
# EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT, (2) DO NOT ASSUME
# ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
# OR USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF THE
# SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4) DO NOT WARRANT
# THAT THE SOFTWARE WILL FUNCTION UNINTERRUPTED, THAT IT IS ERROR-FREE
# OR THAT ANY ERRORS WILL BE CORRECTED.
# 
# LIMITATION OF LIABILITY: IN NO EVENT WILL THE COPYRIGHT HOLDERS, THEIR
# THIRD PARTY LICENSORS, THE UNITED STATES, THE UNITED STATES DEPARTMENT
# OF ENERGY, OR THEIR EMPLOYEES: BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
# CONSEQUENTIAL, SPECIAL OR PUNITIVE DAMAGES OF ANY KIND OR NATURE,
# INCLUDING BUT NOT LIMITED TO LOSS OF PROFITS OR LOSS OF DATA, FOR ANY
# REASON WHATSOEVER, WHETHER SUCH LIABILITY IS ASSERTED ON THE BASIS OF
# CONTRACT, TORT (INCLUDING NEGLIGENCE OR STRICT LIABILITY), OR
# OTHERWISE, EVEN IF ANY OF SAID PARTIES HAS BEEN WARNED OF THE
# POSSIBILITY OF SUCH LOSS OR DAMAGES.
# ******************************************************************************
