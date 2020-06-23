# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:19:22 2020

@author: Benjamin Giraudon
"""

import math

def solF(x_A, y_A, s, a):
    return [[((s**2 + 1)*x_A - math.sqrt(s**2 + 1)*a)/(s**2 + 1), -(math.sqrt(s**2 + 1)*a*s - (s**2 + 1)*y_A)/(s**2 + 1)], [((s**2 + 1)*x_A + math.sqrt(s**2 + 1)*a)/(s**2 + 1), (math.sqrt(s**2 + 1)*a*s + (s**2 + 1)*y_A)/(s**2 + 1)]]

def solB(x_F, y_F, s, c):
    return [[((s**2 + 1)*x_F - math.sqrt(s**2 + 1)*c)/(s**2 + 1), -(math.sqrt(s**2 + 1)*c*s - (s**2 + 1)*y_F)/(s**2 + 1)], [((s**2 + 1)*x_F + math.sqrt(s**2 + 1)*c)/(s**2 + 1),(math.sqrt(s**2 + 1)*c*s + (s**2 + 1)*y_F)/(s**2 + 1)]]

def solC(x_F, y_F, i_p, s, c):
    return [[(s**2*x_F + i_p*s - s*y_F - math.sqrt(-s**2*y_F**2 + (c**2 - i_p**2)*s**2 + 2*i_p*s*x_F + c**2 - x_F**2 + 2*(i_p*s**2 - s*x_F)*y_F)*s)/(s**2 + 1),(i_p*s**2 - s*x_F + y_F + math.sqrt(-s**2*y_F**2 + (c**2 - i_p**2)*s**2 + 2*i_p*s*x_F + c**2 - x_F**2 + 2*(i_p*s**2 - s*x_F)*y_F))/(s**2 + 1)],[(s**2*x_F + i_p*s - s*y_F + math.sqrt(-s**2*y_F**2 + (c**2 - i_p**2)*s**2 + 2*i_p*s*x_F + c**2 - x_F**2 + 2*(i_p*s**2 - s*x_F)*y_F)*s)/(s**2 + 1),(i_p*s**2 - s*x_F + y_F - math.sqrt(-s**2*y_F**2 + (c**2 - i_p**2)*s**2 + 2*i_p*s*x_F + c**2 - x_F**2 + 2*(i_p*s**2 - s*x_F)*y_F))/(s**2 + 1)]]

def sim_to_p(x,y):
    return [-(1/2)*x+1-y, (math.sqrt(3)/2)*x, 0]
    
def p_to_sim(x,y):
    #print("xy p_to_sim", x, y)
    return [2/3*math.sqrt(3)*y, -1/3*math.sqrt(3)*y - x + 1, 0]

def solGame(payMtx):
    m_11, m_12, m_13 = payMtx[0][0], payMtx[0][1], payMtx[0][2]
    m_21, m_22, m_23 = payMtx[1][0], payMtx[1][1], payMtx[1][2]
    m_31, m_32, m_33 = payMtx[2][0], payMtx[2][1], payMtx[2][2]
    return [[ 0, 0], [-(m_13 - m_33)/(m_11 - m_13 - m_31 + m_33), 0], [ 0, -(m_23 - m_33)/(m_22 - m_23 - m_32 + m_33)], [ (m_13*m_22 - m_12*m_23 - (m_13 - m_23)*m_32 + (m_12 - m_22)*m_33)/((m_12 - m_13)*m_21 - (m_11 - m_13)*m_22 + (m_11 - m_12)*m_23 - (m_12 - m_13 - m_22 + m_23)*m_31 + (m_11 - m_13 - m_21 + m_23)*m_32 - (m_11 - m_12 - m_21 + m_22)*m_33), -(m_13*m_21 - m_11*m_23 - (m_13 - m_23)*m_31 + (m_11 - m_21)*m_33)/((m_12 - m_13)*m_21 - (m_11 - m_13)*m_22 + (m_11 - m_12)*m_23 - (m_12 - m_13 - m_22 + m_23)*m_31 + (m_11 - m_13 - m_21 + m_23)*m_32 - (m_11 - m_12 - m_21 + m_22)*m_33)], [1, 0], [0, 1], [-(m_12 - m_22)/(m_11 - m_12 - m_21 + m_22), (m_11 - m_21)/(m_11 - m_12 - m_21 + m_22)]]
