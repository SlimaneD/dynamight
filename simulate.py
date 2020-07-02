# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:25:04 2020

@author: Benjamin Giraudon
"""

import time
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import drawer
import parameters as param



def exec_sim():
    print("TEST :", param.dict_test)
    test = param.dict_test[int(input("-> Please enter the desired test ID :"))]
    print("---------------------------------------------------")
    if test == "arrow":
        fig = plt.figure()
        ax = fig.gca(projection = '3d', xlabel='x axis', ylabel = 'y axis', zlabel = 'z axis')
        print("Testing : {}".format(test))
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
        print("2P3S :", param.dict_2P3S)
        example = abs(int(input("-> Please enter the desired example ID :")))
        print("----------------------------------------------------")
        pMrps = param.PAYMTX_2P3S[example - 1]
        print("Testing : {} -- {}".format(test, param.dict_2P3S[example]))
        fig = plt.figure()
        ax = fig.gca(projection = '3d', xlabel='x axis', ylabel = 'y axis', zlabel = 'z axis')
        ax.set_axis_off()
        ax.w_zaxis.line.set_lw(0.)
        ax.set_zticks([])
        ax.azim = -90
        ax.elev = 91
        start_time = time.time()
        if example == 1:
            fSim  = drawer.setSimplex(["$R$","$P$","$S$"], pMrps, ax, 13, 53)
            traj1 = drawer.numSdeSimplexGen3(0.9, 0.05, pMrps, param.step, [0.01,0.06,0.12,0.2], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            traj2 = drawer.numSdeSimplexGen3(0.5, 0, pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            traj3 = drawer.numSdeSimplexGen3(0,0.5, pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            traj4 = drawer.numSdeSimplexGen3(0.5, 0.5, pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            csp   = drawer.speed_plot([0, 1], [0, np.sqrt(3/4)], [0, 0], 50, pMrps, ax, cm.coolwarm, levels = 50, zorder=50)
            eqs   = drawer.eqShowCompGen(pMrps, ax, 'black', 'gray', 'white', 80, 54)
        elif example == 2:
            fSim  = drawer.setSimplex(["$1$","$2$","$3$"], pMrps, ax, 13, 53)
            traj1 = drawer.numSdeSimplexGen3(0.9, 0.05, pMrps, param.step, [0.01,0.06,0.12,0.2], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            traj2 = drawer.numSdeSimplexGen3(0.5, 0, pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            traj3 = drawer.numSdeSimplexGen3(0,0.5, pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            traj4 = drawer.numSdeSimplexGen3(0.5, 0.5, pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            traj5 = drawer.numSdeSimplexGen3(0.3, 0.3, pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
            csp   = drawer.speed_plot([0, 1], [0, np.sqrt(3/4)], [0, 0], 50, pMrps, ax, cm.coolwarm, levels = 50, zorder=50)
            eqs   = drawer.eqShowCompGen(pMrps, ax, 'black', 'gray', 'white', 80, 54)
        else:
            print(" /!\ No trajectory has been set for this example /!\ ")
            fSim  = drawer.setSimplex(["$1$","$2$","$3$"], pMrps, ax, 13, 53)
            csp   = drawer.speed_plot([0, 1], [0, np.sqrt(3/4)], [0, 0], 50, pMrps, ax, cm.coolwarm, levels = 50, zorder=50)
            eqs   = drawer.eqShowCompGen(pMrps, ax, 'black', 'gray', 'white', 80, 54)
    
        
    elif test == "2P2S":
        print("2P2S :", param.dict_2P2S)
        example = abs(int(input("-> Please enter the desired example ID :")))
        print("----------------------------------------------------")
        pMrps = param.PAYMTX_2P2S[example - 1]
        print("Testing : {} -- {}".format(test, param.dict_2P2S[example]))
        fig = plt.figure()
        ax = fig.gca(projection = '3d', xlabel='x axis', ylabel = 'y axis', zlabel = 'z axis')
        ax.set_axis_on()
        ax.w_zaxis.line.set_lw(0.)
        ax.set_zticks([])
        ax.azim = -90
        ax.elev = 91
        start_time = time.time()
        if example == 1:
            fSim  = drawer.setSimplex(["$p1$", "$p2$"], pMrps, ax, 13, 53)
            traj1 = drawer.numSdeSimplexGen3(0.6,0.2, pMrps, param.step, [0.0001], 10, fig, ax, 'blue', param.arrowSize, param.arrowWidth, 20)
            traj2 = drawer.numSdeSimplexGen3(0.8,0.1, pMrps, param.step, [0.01], 10, fig, ax, 'blue', param.arrowSize, param.arrowWidth, 20)
            eqs = drawer.eqShowCompGen(pMrps, ax, 'black', 'gray','white', 80, 2);
        if example == 2:
            fSim  = drawer.setSimplex(["$p1$", "$p2$"], pMrps, ax, 13, 53)
            traj1 = drawer.numSdeSimplexGen3(0.6,0.2, pMrps, param.step,[0.0001],30,fig, ax,'blue', param.arrowSize, param.arrowWidth, 20);
            traj2 = drawer.numSdeSimplexGen3(0.8,0.1, pMrps, param.step,[0.01],30,fig, ax,'blue', param.arrowSize, param.arrowWidth,20);
            traj3 = drawer.numSdeSimplexGen3(0.5,0.5, pMrps, param.step,[0.01],10,fig, ax,'blue', param.arrowSize, param.arrowWidth,20);
            traj4 = drawer.numSdeSimplexGen3(0.9,0.9, pMrps, param.step,[0.01],10,fig, ax,'blue', param.arrowSize, param.arrowWidth,20);
            traj5 = drawer.numSdeSimplexGen3(0.2,0.6, pMrps, param.step,[0.0001],30,fig, ax,'blue', param.arrowSize, param.arrowWidth,20);
            traj6 = drawer.numSdeSimplexGen3(0.1,0.8, pMrps, param.step,[0.01],30,fig, ax,'blue', param.arrowSize, param.arrowWidth,20);
            traj7 = drawer.numSdeSimplexGen3(0.05,0.05, pMrps, param.step,[0.01],30,fig, ax,'blue', param.arrowSize, param.arrowWidth,20);
            eqs = drawer.eqShowCompGen(pMrps, ax, 'black', 'gray','white', 80, 2);
    
    if test != "arrow":
        print("-----------------------------------------------------")
        print("EQUILIBRIA :")
        print("{} SOURCES".format(len(eqs[0])))
        print("{} SADDLES".format(len(eqs[1])))
        print("{} SINKS".format(len(eqs[2])))
        print("{} CENTRES".format(len(eqs[3])))
        print("{} UNDETERMINED".format(len(eqs[4])))
        print("-----------------------------------------------------")
        print("Execution time : %s seconds" % round((time.time() - start_time), 3))
    return None

    