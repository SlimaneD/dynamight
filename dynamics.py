# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:18:00 2020

@author: Benjamin Giraudon
"""

import numpy as np

def w3(x_1, x_2, y_1, y_2, payMtx):
    return x_1*(y_1*payMtx[0][0] + y_2*payMtx[0][1] + (1-y_1-y_2)*payMtx[0][2]) + x_2*(y_1*payMtx[1][0] + y_2*payMtx[1][1] + (1-y_1-y_2)*payMtx[1][2]) + (1-x_1-x_2)*(y_1*payMtx[2][0] + y_2*payMtx[2][1] + (1-y_1-y_2)*payMtx[2][2])

def repDyn3(X, t, payMtx):
    x_1, x_2 = X
    return np.array([x_1*(w3(1, 0, x_1, x_2, payMtx) - w3(x_1, x_2, x_1, x_2, payMtx)), x_2*(w3(0, 1, x_1, x_2, payMtx) - w3(x_1, x_2, x_1, x_2, payMtx))])

def repDyn3Speed(x_1, x_2, payMtx):
    return np.array([x_1*(w3(1, 0, x_1, x_2, payMtx) - w3(x_1, x_2, x_1, x_2, payMtx)), x_2*(w3(0, 1, x_1, x_2, payMtx) - w3(x_1, x_2, x_1, x_2, payMtx))])

def repDyn3Rev(X, t, payMtx):
    x_1, x_2 = X
    return np.array([-x_1*(w3(1, 0, x_1, x_2, payMtx) - w3(x_1, x_2, x_1, x_2, payMtx)), -x_2*(w3(0, 1, x_1, x_2, payMtx) - w3(x_1, x_2, x_1, x_2, payMtx))])

#Replicator dynamics for an asymmetric 2x2 game
def repDyn22(X, t, payMtx):
    x, y = X
    return x*(1-x)*(x*(y*payMtx[0][0] + (1 - y)*payMtx[0][1]) - (1-x)*(y*payMtx[1][0] + (1 - y)*payMtx[1][1]))

def repDyn22Rev(X, t, payMtx):
    x, y = X
    return -(x*(1-x)*(x*(y*payMtx[0][0] + (1 - y)*payMtx[0][1]) - (1-x)*(y*payMtx[1][0] + (1 - y)*payMtx[1][1])))

def testrep(X, t, payMtx):
    x, y = X[0], X[1]
    return [repDyn22([x, y], t, payMtx[0]), repDyn22([y, x], t, payMtx[1])]

def testrepRev(X, t, payMtx):
    x, y = X[0], X[1]
    return [repDyn22Rev([x, y], t, payMtx[0]), repDyn22Rev([y, x], t, payMtx[1])]
