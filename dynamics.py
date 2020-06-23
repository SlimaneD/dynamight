# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:18:00 2020

@author: Benjamin Giraudon
"""

import numpy as np

#def w3(x_1, x_2, y_1, y_2, payMtx):
#    return x_1*(y_1*payMtx[0][0] + y_2*payMtx[0][1] + (1-y_1-y_2)*payMtx[0][2]) + x_2*(y_1*payMtx[1][0] + y_2*payMtx[1][1] + (1-y_1-y_2)*payMtx[1][2]) + (1-x_1-x_2)*(y_1*payMtx[2][0] + y_2*payMtx[2][1] + (1-y_1-y_2)*payMtx[2][2])

#def repDyn3(x_1, x_2, payMtx):
#    return [x_1*(w3(1,0,x_1,x_2,payMtx) - w3(x_1,x_2,x_1,x_2,payMtx)), x_2*(w3(0,1,x_1,x_2,payMtx) - w3(x_1,x_2,x_1,x_2,payMtx))]

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