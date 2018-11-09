# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 01:11:59 2018

@author: t
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 文件目录
# =============================================================================
filename='MLlab2data1.txt' #文件目录
# =============================================================================
# 读取数据
# =============================================================================
def loadDataSet():   #读取数据
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])   
        labelMat.append(int(lineArr[2]))
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
    grad = np.dot(dataMatrix.T, (sigmoid(np.dot(dataMatrix, weights)) - classLabels))
    clock = 0
    while not np.all(np.absolute(grad) <= 1e-3):
        clock += 1
        weights = weights - alpha * grad 
        grad = np.dot(dataMatrix.T,(sigmoid(np.dot(dataMatrix, weights)) - classLabels))
    print(clock)
    return weights
# =============================================================================
# 梯度下降法（有正则项）
# =============================================================================
def gradDescentRegularization(dataMat,labelMat): #梯度下降法（又正则项）求最优参数
    dataMatrix=np.mat(dataMat) #将读取的数据转换为矩阵
    classLabels=np.mat(labelMat).transpose() #将读取的数据转换为矩阵
    m,n = np.shape(dataMatrix)
    alpha = 0.001  #设置梯度的步长，该值越大梯度上升幅度越大
    lmd = 0.001
    weights = np.zeros((n,1)) #设置初始的参数，并都赋默认值为1。注意这里权重以矩阵形式表示三个参数。
    grad = np.dot(dataMatrix.T, (sigmoid(np.dot(dataMatrix, weights)) - classLabels))
    clock = 0
    while not np.all(np.absolute(grad) <= 1e-3):
#    while (d<100000000):
        clock += 1
        weights = weights - alpha * grad + alpha * lmd * weights
#        grad = dataMatrix.T * (classLabels - sigmoid(dataMatrix * weights))
        grad = np.dot(dataMatrix.T,(sigmoid(np.dot(dataMatrix, weights)) - classLabels))
#        pri...nt(grad)
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
        clock += 1
        logit = sigmoid(np.dot(dataMatrix,weights))
        gradient = np.dot(dataMatrix.T, logit - classLabels)
        temp = np.array(logit) * np.array(1-logit) * np.eye(m)
        hessian = (dataMatrix.T * temp * dataMatrix)
        delta = alpha *(np.dot(hessian.I,gradient) + lmd * weights)
        weights = weights-delta
        if np.all(np.absolute(delta) <= 1e-6):
            break
    print(clock)
    return weights  
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
# 显示图像
# =============================================================================
def plotBestFit(weights):  #画出最终分类的图
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
# main函数
# =============================================================================
def main():
    dataMat, labelMat = loadDataSet()
    weights = gradDescent(dataMat, labelMat).getA()
    plotBestFit(weights)
#    print(correctRate(dataMat,labelMat,weights))
    weights = gradDescentRegularization(dataMat, labelMat).getA()
    plotBestFit(weights)
#    print(correctRate(dataMat,labelMat,weights))
    weights= newtonMethod(dataMat, labelMat).getA()
    plotBestFit(weights)
#    print(correctRate(dataMat,labelMat,weights))
    weights= newtonMethodRegularization(dataMat, labelMat).getA()
    plotBestFit(weights)
#    print(correctRate(dataMat,labelMat,weights))
# =============================================================================
# 主函数入口
# =============================================================================
if __name__=='__main__':
    main()