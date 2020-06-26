# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:19:22 2020

@author: Benjamin Giraudon
"""

import numpy as np
import dynamics as dyn

    
def solF(x_A, y_A, s, a):
    return [[((s**2 + 1)*x_A - np.sqrt(s**2 + 1)*a)/(s**2 + 1), -(np.sqrt(s**2 + 1)*a*s - (s**2 + 1)*y_A)/(s**2 + 1)], [((s**2 + 1)*x_A + np.sqrt(s**2 + 1)*a)/(s**2 + 1), (np.sqrt(s**2 + 1)*a*s + (s**2 + 1)*y_A)/(s**2 + 1)]]

def solB(x_F, y_F, s, c):
    return [[((s**2 + 1)*x_F - np.sqrt(s**2 + 1)*c)/(s**2 + 1), -(np.sqrt(s**2 + 1)*c*s - (s**2 + 1)*y_F)/(s**2 + 1)], [((s**2 + 1)*x_F + np.sqrt(s**2 + 1)*c)/(s**2 + 1),(np.sqrt(s**2 + 1)*c*s + (s**2 + 1)*y_F)/(s**2 + 1)]]

def solC(x_F, y_F, i_p, s, c):
    return [[(s**2*x_F + i_p*s - s*y_F - np.sqrt(-s**2*y_F**2 + (c**2 - i_p**2)*s**2 + 2*i_p*s*x_F + c**2 - x_F**2 + 2*(i_p*s**2 - s*x_F)*y_F)*s)/(s**2 + 1),(i_p*s**2 - s*x_F + y_F + np.sqrt(-s**2*y_F**2 + (c**2 - i_p**2)*s**2 + 2*i_p*s*x_F + c**2 - x_F**2 + 2*(i_p*s**2 - s*x_F)*y_F))/(s**2 + 1)],[(s**2*x_F + i_p*s - s*y_F + np.sqrt(-s**2*y_F**2 + (c**2 - i_p**2)*s**2 + 2*i_p*s*x_F + c**2 - x_F**2 + 2*(i_p*s**2 - s*x_F)*y_F)*s)/(s**2 + 1),(i_p*s**2 - s*x_F + y_F - np.sqrt(-s**2*y_F**2 + (c**2 - i_p**2)*s**2 + 2*i_p*s*x_F + c**2 - x_F**2 + 2*(i_p*s**2 - s*x_F)*y_F))/(s**2 + 1)]]

def sim_to_p(x,y):
    return [-(1/2)*x+1-y, (np.sqrt(3)/2)*x, 0]
    
def p_to_sim(x,y):
    #print("xy p_to_sim", x, y)
    return [2/3*np.sqrt(3)*y, -1/3*np.sqrt(3)*y - x + 1, 0]

def solGame(payMtx):
    m_11, m_12, m_13 = payMtx[0][0], payMtx[0][1], payMtx[0][2]
    m_21, m_22, m_23 = payMtx[1][0], payMtx[1][1], payMtx[1][2]
    m_31, m_32, m_33 = payMtx[2][0], payMtx[2][1], payMtx[2][2]
    return [[ 0, 0], [-(m_13 - m_33)/(m_11 - m_13 - m_31 + m_33), 0], [ 0, -(m_23 - m_33)/(m_22 - m_23 - m_32 + m_33)], [ (m_13*m_22 - m_12*m_23 - (m_13 - m_23)*m_32 + (m_12 - m_22)*m_33)/((m_12 - m_13)*m_21 - (m_11 - m_13)*m_22 + (m_11 - m_12)*m_23 - (m_12 - m_13 - m_22 + m_23)*m_31 + (m_11 - m_13 - m_21 + m_23)*m_32 - (m_11 - m_12 - m_21 + m_22)*m_33), -(m_13*m_21 - m_11*m_23 - (m_13 - m_23)*m_31 + (m_11 - m_21)*m_33)/((m_12 - m_13)*m_21 - (m_11 - m_13)*m_22 + (m_11 - m_12)*m_23 - (m_12 - m_13 - m_22 + m_23)*m_31 + (m_11 - m_13 - m_21 + m_23)*m_32 - (m_11 - m_12 - m_21 + m_22)*m_33)], [1, 0], [0, 1], [-(m_12 - m_22)/(m_11 - m_12 - m_21 + m_22), (m_11 - m_21)/(m_11 - m_12 - m_21 + m_22)]]

def simplexboundaries2D(x, y, epsilon):
    return ((-(2*epsilon + np.sqrt(3))/(2*epsilon + 1))*x + (2*epsilon**2 + epsilon*(np.sqrt(3) + 2) + np.sqrt(3))/(2*epsilon + 1) - y)*(((2*epsilon + np.sqrt(3))/(2*epsilon + 1))*x + (2*epsilon**2 + np.sqrt(3)*epsilon)/(2*epsilon + 1) - y);

def simplexboundaries_bool(X, Y): # returns boolean matrix : True iif point is out of simplex.
    bool_mtx = np.zeros(X.shape)
    for i in range(len(X)):
        for j in range(len(X)):
            if X[i, j]<=0.5:
                if Y[i, j]>2*np.sqrt(3/4)*X[i, j]:
                    bool_mtx[i, j] = True
                else: bool_mtx[i, j] = False
            elif X[i, j]>0.5:
                if Y[i, j]>2*np.sqrt(3/4)*(1-X[i, j]):
                    bool_mtx[i, j] = True
                else: bool_mtx[i, j] = False
            else: bool_mtx[i, j] = False
    return bool_mtx

def outofbounds_reproject(X, Y): # applies orthogonal reprojection to the simplex for out of bounds points
    bool_mtx = simplexboundaries_bool(X, Y)
    for i in range(len(X)):
        for j in range(len(Y)):
            if bool_mtx[i][j]:
                if X[i, j] < 0.5:
                    xB, yB = 0, 0
                    xV, yV = 1, 2*np.sqrt(3/4)
                    BH = ((X[i, j] - xB)*xV + (Y[i, j] - yB)*yV)/(np.sqrt(xV**2 + yV**2))
                    X[i, j] = xB + (BH/np.sqrt(xV**2 + yV**2))*xV
                    Y[i, j] = 2*np.sqrt(3/4)*X[i, j]
                elif X[i, j] > 0.5:
                    xB, yB = 0.5, np.sqrt(3/4)
                    xV, yV = 1, -2*np.sqrt(3/4)
                    BH = ((X[i, j] - xB)*xV + (Y[i, j] - yB)*yV)/(np.sqrt(xV**2 + yV**2))
                    X[i, j] = xB + (BH/np.sqrt(xV**2 + yV**2))*xV
                    Y[i, j] = 2*np.sqrt(3/4)*(1 - X[i, j])
    return X, Y
    
    
def speedS(x, y, payMtx):
    calc = dyn.repDyn3Speed(p_to_sim(x, y)[0], p_to_sim(x, y)[1], payMtx)
    return np.linalg.norm(calc)

def speedGrid(X, Y, payMtx):
    CALC = np.zeros(X.shape)
    for i in range(len(X)):
        for j in range(len(Y)):
            CALC[i][j] = speedS(X[i][j] , Y[i][j] , payMtx)
    return CALC