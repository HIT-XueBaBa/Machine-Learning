# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 14:02:43 2018

@author: HIT1160300207 ChongLiu
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 文件读入
# =============================================================================
def getData():
    return
# =============================================================================
# 生成数据
# =============================================================================
def generateData():
    mean1 = [0,0]
    cov1 = [[1,0], [0,10]]
    data = np.random.multivariate_normal(mean1, cov1, 70)
    
    mean2 = [5,5]
    cov2 = [[10,0], [0,1]]
    data = np.append(data, np.random.multivariate_normal(mean2, cov2, 80), 0)
    
    mean3 = [10,0]
    cov3 = [[3,0], [0,4]]
    data = np.append(data, np.random.multivariate_normal(mean3, cov3, 90), 0)
    
    np.random.shuffle(data)
    plt.scatter(data[:,0], data[:,1])
    plt.show()
    return np.round(data,4)
# =============================================================================
# 计算欧几里得距离
# =============================================================================
def getEuclideanDistance(vecA, vecB):
    return np.sqrt(np.sum(np.square(vecA - vecB))) 
# =============================================================================
# 构建聚簇中心,取k个随机质心
# =============================================================================
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))   
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids
# =============================================================================
# k-means聚类算法
# =============================================================================
def kMeans(dataSet, k, distMeans =getEuclideanDistance, createCent = randCent):
     m = np.shape(dataSet)[0]
     result = np.zeros((m), dtype = np.int)
     clusterAssment = np.mat(np.zeros((m,2)))   
     centroids = createCent(dataSet, k)
     clusterChanged = True 
     while clusterChanged:        
        clusterChanged = False;
        for i in range(m): 
            minDist = np.inf; minIndex = -1;
            for j in range(k):
                 distJI = distMeans(centroids[j,:], dataSet[i,:])
#                 print(type(distJI))
#                 print(distJI)
                 if distJI < minDist:
                    minDist = distJI; minIndex = j  
            if clusterAssment[i,0] != minIndex: clusterChanged = True;  
            clusterAssment[i,:] = minIndex,minDist**2   
            result[i] = minIndex
#            print(minIndex)
#            print(result[i])
#        print(centroids)
        for cent in range(k):   
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]  
            centroids[cent,:] = np.mean(ptsInClust, axis = 0)  
     return centroids, clusterAssment,result
# =============================================================================
# 主函数入口
# =============================================================================
k = 3
datMat = generateData()
myCentroids,clustAssing,result = kMeans(datMat,k)
#print(myCentroids)
#print(clustAssing)
#plt.colors.
#最多显示八种颜色
cValue = ['b','g','r','c','m','y','k','w']
for color in range(k): 
    for i in range(240):
        if result[i] == color:
            plt.scatter(datMat[i,0],datMat[i,1],color = cValue[color],marker = 'o')
#    if result[i] == 0:
#        plt.scatter(datMat[i,0],datMat[i,1],c='r',marker = 'o')
#    elif result[i] == 1:
#        plt.scatter(datMat[i,0],datMat[i,1],c='b',marker = 'o')
##    elif result[i] == 2:
##        plt.scatter(datMat[i,0],datMat[i,1],c='k',marker = 'o')
#    else:
#        plt.scatter(datMat[i,0],datMat[i,1],c='y',marker = 'o')
plt.show()