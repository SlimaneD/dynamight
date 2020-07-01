# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:22:02 2020

@author: Benjamin Giraudon
"""

import math
import numpy as np
from scipy.integrate import odeint
from sympy import Matrix
from sympy.abc import x, y

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import equationsolver as eqsol
import dynamics


def arrow_dyn3(xStart, xEnd, fig, ax, arrow_size, arrow_width, arrow_color, zOrder):
    cf=arrow_width
    af=arrow_size
    x0= xStart
    xA= xEnd
    xB= [0,0,0]
    xF= [0,0,0]
    if(x0[0]==xA[0]):
        xB[0]=xA[0]
        xF[0]=xA[0]
        if(x0[1]>=xA[1]):
            xF[1]=af+xA[1]
            xB[1]=-cf+xF[1]
        else:
            xF[1]=-af+xA[1]
            xB[1]=cf+xF[1]
        xC = [xF[0]-cf,xF[1], 0]
        xD = [xF[0]+cf,xF[1], 0]
    elif(x0[1]==xA[1]):
        xF[1]=xA[1]
        xB[1]=xA[1]
        if(x0[0]>=xA[0]):
            xF[0]=af+xA[0]
            xB[0]=-cf+xF[0]
        else:
            xF[0]=-af+xA[0]
            xB[0]=cf+xF[0]
        xC = [xF[0],xF[1]-cf, 0]
        xD = [xF[0],xF[1]+cf, 0]
    elif(xA[0]>x0[0]):
        sf = (xA[1]-x0[1])/(xA[0]-x0[0])
        xF = [eqsol.solF(xA[0], xA[1], sf, af)[0][0], eqsol.solF(xA[0], xA[1], sf, af)[0][1], 0]
        xB = [eqsol.solB(xF[0], xF[1], sf, cf)[1][0], eqsol.solB(xF[0], xF[1], sf, cf)[1][1], 0]
        xC = [eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[0][0], eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[0][1], 0]
        xD = [eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[1][0], eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[1][1], 0]
    elif(xA[0]<x0[0]):
        sf = (xA[1]-x0[1])/(xA[0]-x0[0])
        xF = [eqsol.solF(xA[0], xA[1], sf, af)[1][0], eqsol.solF(xA[0], xA[1], sf, af)[1][1], 0]
        xB = [eqsol.solB(xF[0], xF[1], sf, cf)[0][0], eqsol.solB(xF[0], xF[1], sf, cf)[0][1], 0]
        xC = [eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[0][0], eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[0][1], 0]
        xD = [eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[1][0], eqsol.solC(xF[0], xF[1], (1/sf)*xF[0]+xF[1], sf, cf)[1][1], 0] 
    xs = [x0[0], xA[0]]
    ys = [x0[1], xA[1]]
    zs = [x0[2], xA[2]]
    arrLine = plt.plot(xs, ys, zs, color=arrow_color, zorder=zOrder)
    #kek = ax.quiver(x0[0], x0[1], x0[2], xA[0], xA[1], xA[2], length = 0.2, color = arrow_color, zorder = zOrder, normalize = True)
    testx = []
    testy = []
    testz = []
    arrow = [xA, xC, xB, xD]
    for pt in arrow:
        testx.append(pt[0])
        testy.append(pt[1])
        testz.append(pt[2])
    verts = [list(zip(testx, testy, testz))]
    arrHead = Poly3DCollection(verts, facecolor=arrow_color, edgecolor=arrow_color, alpha = 1, zorder=zOrder)
    ax.add_collection3d(arrHead)
    return arrLine+[arrHead]


#def arrow_dyn3(xStart, xEnd, fig, ax, arrow_size, arrow_width, arrow_color, zOrder):
#    u = xEnd[0] - xStart[0]
#    v = xEnd[1] - xStart[1]
#    w = xEnd[2] - xStart[2]
#    quiv = ax.quiver(xStart[0], xStart[1], xStart[2], u, v, w, length=0.05, color = arrow_color, zorder=zOrder, normalize = True)
#    return [quiv]

def setSimplex(strat, payMtx, ax, fontSize, zOrder):
    if payMtx[0].shape == (3,):
        pt1 = eqsol.sim_to_p(0,0)
        pt2 = eqsol.sim_to_p(1,0)
        pt3 = eqsol.sim_to_p(0,1)
        lbl1 = ax.text(pt1[0] + 0.03, pt1[1] - 0.022, pt1[2], strat[0], fontsize=fontSize, zorder = zOrder)
        lbl2 = ax.text(pt2[0] - 0.02, pt2[1] + 0.080, 0, strat[1], fontsize=fontSize, zorder = zOrder)
        lbl3 = ax.text(pt3[0] - 0.1, pt3[1] - 0.022, 0, strat[2], fontsize=fontSize, zorder = zOrder)
        xs = [[pt1[0], pt2[0]], [pt1[0], pt3[0]], [pt2[0], pt3[0]]]
        ys = [[pt1[1], pt2[1]], [pt1[1], pt3[1]], [pt2[1], pt3[1]]]
        bdr1 = plt.plot(xs[0], ys[0], 0, color='black', zorder=zOrder, alpha=1)
        bdr2 = plt.plot(xs[1], ys[1], 0, color='black', zorder=zOrder, alpha=1)
        bdr3 = plt.plot(xs[2], ys[2], 0, color='black', zorder=zOrder, alpha=1)
        return bdr1+bdr2+bdr3 + [lbl1] + [lbl2] + [lbl3]
    elif payMtx[0].shape == (2, 2):
        lbl1 = ax.text(0.5 , - 0.12, 0, strat[0], fontsize=fontSize, zorder = zOrder)
        lbl2 = ax.text(-0.12, 0.5, 0, strat[1], fontsize=fontSize, zorder = zOrder)
        bdr1 = plt.plot([0, 1], [0, 0], [0, 0], color='black', zorder=zOrder, alpha=1)
        bdr2 = plt.plot([1, 1], [0, 1], [0, 0], color='black', zorder=zOrder, alpha=1)
        bdr3 = plt.plot([1, 0], [1, 1], [0, 0], color='black', zorder=zOrder, alpha=1)
        bdr4 = plt.plot([0, 0], [1, 0], [0, 0], color='black', zorder=zOrder, alpha=1)
        return bdr1+bdr2+bdr3+bdr4
        

def numSdeSimplexGen3(x0, y0, payMtx, step, parr, Tmax, fig, ax, col, arrSize, arrWidth, zd):
    t = np.linspace(0, Tmax, int(Tmax/step))
    if payMtx[0].shape == (3,): #S_2P3S
        sol = odeint(dynamics.repDyn3, [x0, y0], t, (payMtx,))
        solRev = odeint(dynamics.repDyn3Rev, [x0, y0], t, (payMtx,))
        solX=[]
        solY=[]
        solXrev=[]
        solYrev=[]
        for pt in sol:
            cPt = eqsol.sim_to_p(pt[0], pt[1])
            solX.append(cPt[0])
            solY.append(cPt[1])
        for pt in solRev:
            cPt = eqsol.sim_to_p(pt[0],pt[1])
            solXrev.append(cPt[0])
            solYrev.append(cPt[1])
        psol = plt.plot(solX, solY, color=col, zorder=zd)
        psolRev = plt.plot(solXrev, solYrev, color=col, zorder=zd)
        dirs = arrow_dyn3([solX[math.floor(parr[0]*len(solX))], solY[math.floor(parr[0]*len(solX))], 0], [solX[math.floor(parr[0]*len(solX))+1], solY[math.floor(parr[0]*len(solX))+1], 0], fig, ax, arrow_width=arrWidth, arrow_size=arrSize, arrow_color='k', zOrder=zd)
        #dirsRev = arrow_dyn3([solXrev[math.floor(parr[0]*len(solXrev))], solYrev[math.floor(parr[0]*len(solXrev))], 0],[solXrev[math.floor(parr[0]*len(solXrev))+1], solYrev[math.floor(parr[0]*len(solXrev))+1], 0],fig, ax, arrow_width=arrWidth, arrow_size=arrSize, arrow_color='k', zOrder=zd)
        for i in range(1, len(parr)):
            dirs = dirs + arrow_dyn3([solX[math.floor(parr[i]*len(solX))], solY[math.floor(parr[i]*len(solX))], 0],[solX[math.floor(parr[i]*len(solX))+1], solY[math.floor(parr[i]*len(solX))+1], 0], fig, ax, arrow_width=arrWidth, arrow_size=arrSize, arrow_color='k', zOrder=zd)
            #dirsRev = dirsRev + arrow_dyn3([solXrev[math.floor(parr[i]*len(solXrev))+1], solYrev[math.floor(parr[i]*len(solXrev))+1], 0], [solXrev[math.floor(parr[i]*len(solXrev))], solYrev[math.floor(parr[i]*len(solXrev))], 0], fig, ax, arrow_width=arrWidth, arrow_size=arrSize, arrow_color='k', zOrder=zd)
        #return (psol + psolRev + dirs + dirsRev)
        return (psol + psolRev + dirs)
    elif payMtx[0].shape == (2, 2): #AS_2P2S
        sol = odeint(dynamics.testrep, [x0, y0], t, (payMtx,))
        solRev = odeint(dynamics.testrepRev, [x0, y0], t, (payMtx,))
        solX=sol[:,0]
        solY=sol[:,1]
        #solZ = [0 for i in range(len(solX))]
        solXrev=solRev[:,0]
        solYrev=solRev[:,1]
        psol = plt.plot(solX,solY,color=col,zorder=zd)
        psolRev = plt.plot(solXrev,solYrev,color=col,zorder=zd)
        dirs = arrow_dyn3([solX[math.floor(parr[0]*len(solX))],solY[math.floor(parr[0]*len(solX))], 0], [solX[math.floor(parr[0]*len(solX))+1],solY[math.floor(parr[0]*len(solX))+1], 0], fig, ax, arrow_width=arrWidth, arrow_size=arrSize, arrow_color=col, zOrder=zd)
        #dirsRev = arrow_dyn3([solXrev[math.floor(parr[0]*len(solXrev))],solYrev[math.floor(parr[0]*len(solXrev))]], [solXrev[math.floor(parr[0]*len(solXrev))+1],solYrev[math.floor(parr[0]*len(solXrev))+1], 0], fig, ax, arrow_width=arrWidth, arrow_size=arrSize,arrow_color=col,zOrder=zd)
        for i in range(1, len(parr)):
            dirs = dirs+arrow_dyn3([solX[math.floor(parr[i]*len(solX))],solY[math.floor(parr[i]*len(solX))], 0], [solX[math.floor(parr[i]*len(solX))+1],solY[math.floor(parr[i]*len(solX))+1], 0], fig, ax, arrow_width=arrWidth, arrow_size=arrSize,arrow_color=col,zOrder=zd)
            #dirsRev = dirsRev + arrow_dyn3([solXrev[math.floor(parr[i]*len(solXrev))+1],solYrev[math.floor(parr[i]*len(solXrev))+1]], [solXrev[math.floor(parr[i]*len(solXrev))],solYrev[math.floor(parr[i]*len(solXrev))], 0], fig, ax, arrow_width=arrWidth, arrow_size=arrSize, arrow_color=col, zOrder=zd)
        return(psol+psolRev+dirs)

def eqShowCompGen(payMtx, ax, colSnk, colSdl, colSce, ptSize, zd):
    source = [] #list of sources (both eigenvalues are positive)
    sink = []   #list of sinks (both eigenvalues are negative)
    saddle = [] #list of saddles (one neg., one pos. eig.)
    centre = []
    undet = []  #list of equilibria with both 0 eigenvalues
    numEqs = []
    numEig = []
    #Compute equilibria of the replicator dynamics
    nuEqsRaw = eqsol.solGame(payMtx)
    #Compute eigenvalues of Jacobian evaluated at each equilibrium
    if payMtx[0].shape == (3,):
        for i in range(len(nuEqsRaw)): #Check that all equilibria are within the simplex
            if (0 <=  nuEqsRaw[i][0] <= 1 and 0 <=  nuEqsRaw[i][1] <= 1 and nuEqsRaw[i][0] + nuEqsRaw[i][1] <= 1):
                numEqs += [[nuEqsRaw[i][0],nuEqsRaw[i][1]]]
        for i in range(len(numEqs)): #Check that equilibria are real
            if (numEqs[i][0].imag !=0 or numEqs[i][1].imag != 0):
                numEqs[i][0] = 99 #attributing unrealistic values in case there are complex eigenvalues, to draw attention
                numEqs[i][1] = 99
        for i in range(len(numEqs)):
            t = 0
            X = Matrix(dynamics.repDyn3([x,y], t, payMtx))
            Y = Matrix([x, y])
            JC = X.jacobian(Y)
            valuedJC = np.array(JC.subs([(x, numEqs[i][0]), (y, numEqs[i][1])]))
            M = np.zeros(valuedJC.shape)
            for i in range(len(valuedJC)):
                for j in range(len(valuedJC)):
                    M[i][j] = valuedJC[i][j]
            w, v = np.linalg.eig(M)
            if w[0] or w[1] != 0:
                numEig.append(w)
                
    elif payMtx[0].shape == (2, 2):
        for i in range(len(nuEqsRaw)):
            if (0 <=  nuEqsRaw[i][0] <= 1 and 0 <=  nuEqsRaw[i][1] <= 1):
                numEqs += [[nuEqsRaw[i][0],nuEqsRaw[i][1]]]
        for i in range(len(numEqs)): #Check that equilibria coords are real
            if (numEqs[i][0].imag !=0 or numEqs[i][1].imag != 0):
                numEqs[i][0] = 99 #attributing unrealistic values in case there are complex eigenvalues, to draw attention
                numEqs[i][1] = 99
        for i in range(len(numEqs)):
            t = 0
            X = Matrix(dynamics.testrep([x, y], t, payMtx))
            Y = Matrix([x, y])
            JC = X.jacobian(Y)
            valuedJC = np.array(JC.subs([(x, numEqs[i][0]), (y, numEqs[i][1])]))
            M = np.zeros(valuedJC.shape)
            for i in range(len(valuedJC)):
                for j in range(len(valuedJC)):
                    M[i][j] = valuedJC[i][j]
            w, v = np.linalg.eig(M)
            numEig.append(w)
    for i in range(len(numEqs)): #Classify equilibria into sinks, saddles, sources, degenerate
        print("equilibrium", numEqs[i], "eigenvalue", numEig[i])
        if payMtx[0].shape == (2, 2): point_to_plot = np.array([numEqs[i][0], numEqs[i][1], 0])
        elif payMtx[0].shape == (3,): point_to_plot = np.array(eqsol.sim_to_p(numEqs[i][0] , numEqs[i][1]))
        if (0<=numEqs[i][0]<=1 and 0<=numEqs[i][1]<=1):
            l1, l2 = numEig[i][0], numEig[i][1]
            suml, prodl = l1+l2, l1*l2
            if prodl<0: saddle.append(point_to_plot)
            else:
                if suml>0: source.append(point_to_plot)
                elif suml<0: sink.append(point_to_plot)
                else:
                    if l1.imag !=0:
                        centre.append(point_to_plot)
                    else: undet.append(point_to_plot)
                
    #Plot equilibria
    saddlexs, saddleys, saddlezs = [], [], []
    sinkxs, sinkys, sinkzs = [], [], []
    sourcexs, sourceys, sourcezs = [], [], []
    centrexs, centreys, centrezs = [], [], []
    undetxs, undetys, undetzs = [], [], []
    for pt in source:
        sourcexs.append(pt[0])
        sourceys.append(pt[1])
        sourcezs.append(pt[2])
    for pt in saddle:
        saddlexs.append(pt[0])
        saddleys.append(pt[1])
        saddlezs.append(pt[2])
    for pt in centre :
        centrexs.append(pt[0])
        centreys.append(pt[1])
        centrezs.append(pt[2])
    for pt in sink:
        sinkxs.append(pt[0])
        sinkys.append(pt[1])
        sinkzs.append(pt[2])
    for pt in undet:
        undetxs.append(pt[0])
        undetys.append(pt[1])
        undetzs.append(pt[2])
    pSink = ax.scatter(sinkxs, sinkys, sinkzs, s=ptSize, color=colSnk, marker='o', edgecolors='black', alpha=1, depthshade=False, zorder=zd)
    pSource = ax.scatter(sourcexs, sourceys, sourcezs, s=ptSize, color=colSce, marker='o', edgecolors='black', alpha=1, depthshade=False, zorder=zd)
    pSaddle = ax.scatter(saddlexs, saddleys, saddlezs, s=ptSize, color=colSdl, marker='o', edgecolors='black', alpha=1, depthshade=False, zorder=zd)
    pCentre = ax.scatter(centrexs, centreys, centrezs, s=ptSize, color='orange', marker='*', edgecolors='black', alpha=1, depthshade=False, zorder=zd)
    pUndet = ax.scatter(undetxs, undetys, undetzs, s=ptSize, color='red', marker='x', edgecolors='black', alpha=1, depthshade=False, zorder=zd)
    #return [pSink] + [pSource] + [pSaddle] + [pUndet]
    return [source] + [saddle] + [sink] + [centre] + [undet]
        

def matrix_to_colors(matrix, cmap):
    color_dimension = matrix # It must be in 2D - as for "X, Y, Z".
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap = cmap)
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)
    return fcolors, m

def speed_plot(x_region, y_region, z_region, step, payMtx, ax, cmap, levels, zorder):
    x = np.linspace(x_region[0], x_region[1], step)
    #y = np.linspace(0, np.sqrt(3/4), 150)
    y = np.linspace(y_region[0], y_region[1], step)
    #z = np.linspace(z_region[0], z_region[1], step)
    X, Y = np.meshgrid(x, y)
    #Z = np.zeros(X.shape)
    #remove_pts = eqsol.simplexboundaries_bool(X, Y)
    X, Y = eqsol.outofbounds_reproject(X, Y)
    C = eqsol.speedGrid(X, Y, payMtx)
    surf = ax.contourf(X, Y, C, zdir='z', offset=0, levels=levels, cmap=cmap, corner_mask = False, alpha=0.9)
    return surf
