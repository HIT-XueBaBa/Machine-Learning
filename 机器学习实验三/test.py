# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 14:02:43 2018

@author: t
"""

import numpy as np
import matplotlib.pyplot as plt

#文件读入
def getData():
    return

#生成数据
def generateData():
    mean1 = [0,0]
    cov1 = [[1,0],[0,10]]
    data = np.random.multivariate_normal(mean1,cov1,100)
    
    mean2 = [10,10]
    cov2 = [[10,0],[0,1]]
    data = np.append(data,
                     np.random.multivariate_normal(mean2,cov2,100),
                     0)
    
    mean3 = [10,0]
    cov3 = [[3,0],[0,4]]
    data = np.append(data,
                     np.random.multivariate_normal(mean3,cov3,100),
                     0)
    np.random.shuffle(data)
    plt.scatter(data[:,0],data[:,1])
    plt.show()
    return np.round(data,4)

# 计算欧几里得距离
def getEuclideanDistance(vecA, vecB):
    return np.sqrt(np.sum(np.square(vecA - vecB))) # 求两个向量之间的距离

# 构建聚簇中心，取k个随机质心
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))   # 每个质心有n个坐标值，总共要k个质心
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids
 
 # k-means 聚类算法
def kMeans(dataSet, k, distMeans =getEuclideanDistance, createCent = randCent):
     m = np.shape(dataSet)[0]
     result = np.zeros((m), dtype = np.int)
     clusterAssment = np.mat(np.zeros((m,2)))    # 用于存放该样本属于哪类及质心距离
     # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
     centroids = createCent(dataSet, k)
     clusterChanged = True   # 用来判断聚类是否已经收敛
     while clusterChanged:        
        clusterChanged = False;
        for i in range(m):  # 把每一个数据点划分到离它最近的中心点
            minDist = np.inf; minIndex = -1;
            for j in range(k):
                 distJI = distMeans(centroids[j,:], dataSet[i,:])
#                 print(type(distJI))
#                 print(distJI)
                 if distJI < minDist:
                    minDist = distJI; minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
            if clusterAssment[i,0] != minIndex: clusterChanged = True;  # 如果分配发生变化，则需要继续迭代
            clusterAssment[i,:] = minIndex,minDist**2   # 并将第i个数据点的分配情况存入字典
            result[i] = minIndex
#            print(minIndex)
#            print(result[i])
#        print(centroids)
        for cent in range(k):   # 重新计算中心点
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]   # 取第一列等于cent的所有列
            centroids[cent,:] = np.mean(ptsInClust, axis = 0)  # 算出这些数据的中心点
     return centroids, clusterAssment,result
#--------------------测试----------------------------------------------------
 # 用测试数据及测试kmeans算法
#datMat = mat(loadDataSet('testSet.txt'))
datMat = generateData()
myCentroids,clustAssing,result = kMeans(datMat,3)
#print(myCentroids)
#print(clustAssing)
for i in range(300):
    if result[i] == 0:
#        print(True)
        plt.scatter(datMat[i,0],datMat[i,1],c='r',marker = 'o')
    elif result[i] == 1:
#        print(False)
        plt.scatter(datMat[i,0],datMat[i,1],c='b',marker = 'o')
    else:
        plt.scatter(datMat[i,0],datMat[i,1],c='y',marker = 'o')
#    print(type(result[i]))
plt.show()