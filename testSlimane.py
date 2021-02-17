import time
import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

import drawer
import parameters as param

import dynamics as dyn

payoffMtx = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])

dyn.w4(0.1, 0.1, 0.4, 0.1, 0.1, 0.4, payoffMtx)

dyn.repDyn22([0.4,0.5],7,payoffMtx)


example = 1
pMrps = param.PAYMTX_2P3S[example - 1]

drawer.setSimplex(['1','2','3'], pMrps, ax, 13, 53)
drawer.trajectory([0.9, 0.05], pMrps, param.step, [0.0001], 50, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
drawer.trajectory([0.5, 0], pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
drawer.trajectory([0,0.5], pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
drawer.trajectory([0.5, 0.5], pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
drawer.trajectory([0.3, 0.3], pMrps, param.step, [0.0001], 10, fig, ax, 'black', param.arrowSize, param.arrowWidth, 53)
drawer.speed_plot([0, 1], [0, np.sqrt(3/4)], 50, pMrps, ax, cm.coolwarm, levels = 50, zorder=50)
eqs = drawer.equilibria(pMrps, ax, 'black', 'gray', 'white', 80, 54)
