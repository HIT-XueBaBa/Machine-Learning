# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 20:28:59 2018

@author: t
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 读取数据
# =============================================================================
def loadDataSet(filename):   #读取数据
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[1]), float(lineArr[2]), float(lineArr[3]), float(lineArr[4]), float(lineArr[5]), float(lineArr[6])])  
        labelMat.append(int(lineArr[0]))
    return dataMat,labelMat
# =============================================================================
# sigmoid函数
# =============================================================================
def sigmoid(inX):  #sigmoid函数
    return 1.0/(1+np.exp(-inX))
# =============================================================================
# 梯度下降法（无正则项）
# =============================================================================
def gradDescent(dataMat,labelMat): 
    dataMatrix=np.mat(dataMat)
    classLabels=np.mat(labelMat).transpose() 
    m,n = np.shape(dataMatrix)
    alpha = 0.001  
    weights = np.zeros((n,1)) 
    grad = -np.dot(dataMatrix.T, (classLabels - sigmoid(np.dot(dataMatrix, weights))))
    clock = 0
    while not np.all(np.absolute(grad) <= 1e-3):
        clock = clock + 1
        weights = weights - alpha * grad 
        grad = -np.dot(dataMatrix.T,(classLabels - sigmoid(np.dot(dataMatrix, weights))))
    print(clock)
    return weights
# =============================================================================
# 梯度下降法（有正则项）
# =============================================================================
def gradDescentRegularization(dataMat,labelMat): 
    dataMatrix=np.mat(dataMat) 
    classLabels=np.mat(labelMat).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001  
    lmd = 1e-3
    weights = np.zeros((n,1)) 
    grad = -np.dot(dataMatrix.T, (classLabels - sigmoid(np.dot(dataMatrix, weights))))
    clock = 0
    while not np.all(np.absolute(grad) <= 1e-3):
        clock = clock + 1
        weights = weights + alpha * lmd * weights - alpha * grad 
        grad = -np.dot(dataMatrix.T,(classLabels - sigmoid(np.dot(dataMatrix, weights))))
    print(clock)
    return weights  
# =============================================================================
# 牛顿法（无正则项）
# =============================================================================
def newtonMethod(dataMat, labelMat):
    dataMatrix = np.mat(dataMat)
    classLabels = np.mat(labelMat).T
    m,n = np.shape(dataMatrix)
    weights = np.zeros((n,1))
    alpha = 0.001
    clock = 0
    while True:
        clock += 1
        logit = sigmoid(np.dot(dataMatrix,weights))
        gradient = np.dot(dataMatrix.T, logit - classLabels)
        temp = np.array(logit) * np.array(1-logit) * np.eye(m)
        hessian = (dataMatrix.T * temp * dataMatrix)
        delta = alpha * np.dot(hessian.I,gradient)
        weights = weights-delta
        if np.all(np.absolute(delta) <= 1e-6):
            break
    print(clock)
    return weights
# =============================================================================
# 牛顿法（有正则项）
# =============================================================================
def newtonMethodRegularization(dataMat, labelMat):
    dataMatrix = np.mat(dataMat)
    classLabels = np.mat(labelMat).T
    m,n = np.shape(dataMatrix)
    weights = np.zeros((n,1))
    alpha = 0.001
    lmd = 1e-3
    clock = 0
    while True:
        clock = clock + 1
        logit = sigmoid(np.dot(dataMatrix,weights))
        gradient = np.dot(dataMatrix.T, logit - classLabels)
        temp = np.array(logit) * np.array(1-logit) * np.eye(m)
        hessian = (dataMatrix.T * temp * dataMatrix)
        delta = alpha * (np.dot(hessian.I,gradient) + lmd * weights)
        weights = weights-delta
        if np.all(np.absolute(delta) <= 1e-6):
            break
    print(clock)
    return weights  
# =============================================================================
# 显示图像
# =============================================================================
def plotResult(weights):  
    dataMat,labelMat=loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    plt.scatter(xcord1, ycord1, c = '',edgecolors = 'r', marker = 'o')
    plt.scatter(xcord2, ycord2, c = 'b', marker = '+')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    plt.plot(x, y, 'k')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
# =============================================================================
# 评估函数
# =============================================================================
def correctRate(dataMat, labelMat, weights):
    dataMatrix = np.mat(dataMat)
    classLabels = np.mat(labelMat).T
    m,n = np.shape(dataMatrix)
    temp = dataMatrix * weights
    numOfAClass = 0
    for i in range(m):
        if temp[i] >= 0 and classLabels[i] == 1:
            numOfAClass += 1
        if temp[i] < 0 and classLabels[i] == 0:
            numOfAClass += 1
    return numOfAClass/m
# =============================================================================
# main函数
# =============================================================================
def main():
    dataMat, labelMat = loadDataSet('train1.txt')
    weights = gradDescent(dataMat, labelMat).getA()
    print(correctRate(dataMat,labelMat,weights))
    dataMat, labelMat = loadDataSet('test1.txt')
    print(correctRate(dataMat,labelMat,weights))
    print('------------------------------------')
    
    dataMat, labelMat = loadDataSet('train1.txt')
    weights = gradDescentRegularization(dataMat, labelMat).getA()
    print(correctRate(dataMat,labelMat,weights))
    dataMat, labelMat = loadDataSet('test1.txt')
    print(correctRate(dataMat,labelMat,weights))
    print('------------------------------------')
    
    dataMat, labelMat = loadDataSet('train1.txt')
    weights = newtonMethod(dataMat, labelMat).getA()
    print(correctRate(dataMat,labelMat,weights))
    dataMat, labelMat = loadDataSet('test1.txt')
    print(correctRate(dataMat,labelMat,weights))
    print('------------------------------------')
    
    dataMat, labelMat = loadDataSet('train1.txt')
    weights = newtonMethodRegularization(dataMat, labelMat).getA()
    print(correctRate(dataMat,labelMat,weights))
    dataMat, labelMat = loadDataSet('test1.txt')
    print(correctRate(dataMat,labelMat,weights))
# =============================================================================
# 主函数入口
# =============================================================================
if __name__=='__main__':
    main()