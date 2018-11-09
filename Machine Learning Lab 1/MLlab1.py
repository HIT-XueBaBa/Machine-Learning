# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:49:50 2018

@author: HIT 1160300207 Chong Liu
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lg
# =============================================================================
# 产生样本
# =============================================================================
points_num = 100
step = 1/points_num
x = np.arange(0,1,step)
y = np.sin(2*np.pi*x)
plt.plot(x,y,'g',label = "sin")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin")
y0 = y +  0.1*np.random.randn(points_num)
plt.plot(x,y0,'b',label = "noise")
plt.legend(loc='upper right')
plt.show()
# =============================================================================
# 方均根值函数 E_rms(W,Y,y)
# =============================================================================
def E_rms(W,Y,y):
    tem = 0
    for i in range(0,points_num):
        tem += np.power(Y[i] - y[i],2)/points_num
    return tem
# =============================================================================
# 产生训练集合 Y
# =============================================================================
sample_num = 50
poly_func_order = 3
X = np.zeros((sample_num,poly_func_order+1))
Y = np.zeros((sample_num,1))
for i in range(0,sample_num):
    Y[i] = y0[i*np.int(points_num/sample_num)]
    for j in range(0,poly_func_order+1):
        X[i,j] = np.power(x[i*np.int(points_num/sample_num)],poly_func_order-j)
    plt.plot(x[i*np.int(points_num/sample_num)],y0[i*np.int(points_num/sample_num)],'k^')
# =============================================================================
# 解析解 无正则项 y1
# =============================================================================
A = np.dot(np.dot(lg.inv(np.dot(X.T,X)),X.T),Y)
y1 = 0
for n in range(0,poly_func_order+1):
    y1 += A[poly_func_order-n] * np.power(x,n)
# =============================================================================
# 解析解 有正则项 y2
# =============================================================================
lmd = 1e-3
B = np.dot(np.dot(lg.inv(np.dot(X.T,X)+lmd*np.identity(poly_func_order+1)),X.T),Y)
y2 = 0
for n in range(0,poly_func_order+1):
     y2 += B[poly_func_order-n] * np.power(x,n)
# =============================================================================
# 梯度下降 有正则项 y3
# =============================================================================
lmd = 1e-3
alpha = 1e-3
C = np.random.randn(poly_func_order+1,1)
grad = np.dot(X.T,(np.dot(X,C)-Y)) 
while not np.all(np.absolute(grad) <= 1e-6):
    C = C - alpha * grad
    grad = np.dot(X.T,(np.dot(X,C)-Y)) 
y3 = 0
for n in range(0,poly_func_order+1):
    y3 += C[poly_func_order-n] * np.power(x,n)
# =============================================================================
# 共轭梯度 有正则项 y4   
# =============================================================================
lmd = 1e-3
D = np.random.randn(poly_func_order+1,1)
def CG(X,Y,W):
    r = Y - np.dot(X,W)
    p = r
    while True:
        if np.all(np.abs(p) < 1e-5):
            return W
        else:
            a = np.dot(r.T,r)/np.dot(np.dot(p.T,X),p)
            W = W + a*p
            tem = r
            r = r - a*np.dot(X,p)
            b = np.dot(r.T,r)/np.dot(tem.T,tem)
            p = r + b*p
D = CG(np.dot(X.T,X)+lmd*np.identity(poly_func_order+1),np.dot(X.T,Y),D)
y4 = 0
for n in range(0,poly_func_order+1):     
    y4 += D[poly_func_order-n] * np.power(x,n)
# =============================================================================
# 显示拟合图像
# =============================================================================
plt.plot(x,y,'g',label = "sin")
plt.plot(x,y0,'b.',label = "noise")
plt.plot(x,y1,'r',label = "y1")
plt.plot(x,y2,'m',label = "y2")
plt.plot(x,y3,'c',label = "y3")
plt.plot(x,y4,'y',label = "y4")
plt.legend(loc='upper right')
plt.xlabel("x")
plt.ylabel("y")
plt.title("M="+'%d'%(poly_func_order)+" fitting image")
plt.show()