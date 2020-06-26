# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:25:04 2020

@author: Benjamin Giraudon
"""

import random
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import drawer

#test = "arrow"
test = "2P3S"

fig = plt.figure()
ax = fig.gca(projection = '3d', xlabel='x axis', ylabel = 'y axis', zlabel = 'z axis')

if test == "arrow":
    example = 1
    Ot = [-4, 3, 0]
    At = [5.4, 3, 0]
    
    Ot2 = [-4, 3, 0]
    At2 = [5.4, 4.1, 0]
    
    Ot3 = [6.5, 2, 0]
    At3 = [6.5, 3.7, 0]
    
    res1 = drawer.arrow_dyn3(Ot, At, fig, ax, 1, 0.33, 'purple', zOrder=3)
    res2 = drawer.arrow_dyn3(Ot2, At2, fig, ax, 1, 0.33, 'orange', zOrder=3)
    res3 = drawer.arrow_dyn3(Ot3, At3, fig, ax, 1, 0.33, 'black', zOrder=3)
    
    N=10
    res = [res1, res2, res3]
    for i in range(N):
        color = (random.random(), random.random(), random.random())
        res.append(drawer.arrow_dyn3([random.randint(-5,5),random.randint(-5,5), 0],[random.randint(-5,5),random.randint(-5,5), 0], fig, ax, 1,0.33,color,zOrder=3))

elif test == "2P3S":
    #example = 1
    example = 2
    print("testing : {} -- example n°{}".format(test, example))
    
    arrowSize= 1/25.0
    arrowWidth= (1/2)*arrowSize
    step = 0.01
    Tmax = 50
    ax.set_axis_off()
    ax.azim = -90
    ax.elev = 91
    
    if example == 1:
        pMrps = np.array([[0,-1,2],[2,0,-1],[-1,2,0]])
        #pMrps = np.array([[0,-1,1],[1,0,-1],[-1,1,0]]) ##division par 0...
        #pMrps = np.array([[1,0,0],[0,2,0],[0,0,3]])
        #pMrps = np.array([[1,0,0],[0,1,0],[0,0,1]])
        fSim  = drawer.setSimplex("$R$","$P$","$S$", ax, 13, 53)
        traj1 = drawer.numSdeSimplexGen3(0.9, 0.05, pMrps, step, [0.01,0.06,0.12,0.2], 50, fig, ax, 'black', arrowSize, arrowWidth, 53)
        traj2 = drawer.numSdeSimplexGen3(0.5, 0, pMrps, step, [0.0001], 10, fig, ax, 'black', arrowSize, arrowWidth, 53)
        traj3 = drawer.numSdeSimplexGen3(0,0.5, pMrps, step, [0.0001], 10, fig, ax, 'black', arrowSize, arrowWidth, 53)
        traj4 = drawer.numSdeSimplexGen3(0.5, 0.5, pMrps, step, [0.0001], 10, fig, ax, 'black', arrowSize, arrowWidth, 53)
        csp   = drawer.speed_plot([0, 1], [0, np.sqrt(3/4)], [0, 0], 50, pMrps, ax, cm.coolwarm, levels = 50, zorder=50)
        eqs   = drawer.eqShowCompGen(pMrps, ax, 'black', 'gray', 'white', 80, 54)
        print("------------------------------")
        print("EQUILIBRIA :")
        print("{} SOURCES".format(len(eqs[0])))
        print("{} SADDLES".format(len(eqs[1])))
        print("{} SINKS".format(len(eqs[2])))
        print("{} UNDETERMINED".format(len(eqs[3])))
    elif example == 2:
        pMrps = np.array([[0,6,-4],[-3,0,5],[-1,3,0]])
        fSim  = drawer.setSimplex("$1$","$2$","$3$", ax, 13, 53)
        traj1 = drawer.numSdeSimplexGen3(0.9, 0.05, pMrps, step, [0.01,0.06,0.12,0.2], 50, fig, ax, 'black', arrowSize, arrowWidth, 53)
        traj2 = drawer.numSdeSimplexGen3(0.5, 0, pMrps, step, [0.0001], 10, fig, ax, 'black', arrowSize, arrowWidth, 53)
        traj3 = drawer.numSdeSimplexGen3(0,0.5, pMrps, step, [0.0001], 10, fig, ax, 'black', arrowSize, arrowWidth, 53)
        traj4 = drawer.numSdeSimplexGen3(0.5, 0.5, pMrps, step, [0.0001], 10, fig, ax, 'black', arrowSize, arrowWidth, 53)
        traj5 = drawer.numSdeSimplexGen3(0.3, 0.3, pMrps, step, [0.0001], 10, fig, ax, 'black',arrowSize,arrowWidth, 53)
        csp   = drawer.speed_plot([0, 1], [0, np.sqrt(3/4)], [0, 0], 50, pMrps, ax, cm.coolwarm, levels = 50, zorder=50)
        eqs   = drawer.eqShowCompGen(pMrps, ax, 'black', 'gray', 'white', 80, 54)
        print("------------------------------")
        print("EQUILIBRIA :")
        print("{} SOURCES".format(len(eqs[0])))
        print("{} SADDLES".format(len(eqs[1])))
        print("{} SINKS".format(len(eqs[2])))
        print("{} UNDETERMINED".format(len(eqs[3])))
        