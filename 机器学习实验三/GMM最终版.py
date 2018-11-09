# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:57:03 2018

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
#    mean1 = [0,0]
#    cov1 = [[1,0], [0,10]]
#    data = np.random.multivariate_normal(mean1, cov1, 70)
#    
#    mean2 = [5,5]
#    cov2 = [[10,0], [0,1]]
#    data = np.append(data, np.random.multivariate_normal(mean2, cov2, 80), 0)
#    
#    mean3 = [10,0]
#    cov3 = [[3,0], [0,4]]
#    data = np.append(data, np.random.multivariate_normal(mean3, cov3, 90), 0)
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
    plt.scatter(data[:,0], data[:,1])
    plt.show()
    return np.round(data,4)
# =============================================================================
# 高斯混合模型 GMM
# =============================================================================
class Gaussian_Mixture_Model:
    '''
    输入：
        样本集D
        高斯混合成分个数k
    '''
    def __init__(self,D,k):
        self.D = np.matrix(D)
        self.k = int(k)
        self.M, self.N = self.D.shape
    '''
    高斯混合聚类算法
    '''
    def Mixture_of_Gaussian(self):
        #初始化高斯混合分布的模型参数
        alpha = np.ones((self.k)) * (1. / self.k)
#        mu = np.ones((self.k, 2))
        mu = np.random.random((self.k, 2))
#        sigma = np.random.random((self.k, 2, 2)) 
        sigmaElem = [[30, 0.0],
                 [0.0, 30]]
        sigma = [sigmaElem, sigmaElem, sigmaElem]
#        print(sigma[4])
        #采用EM算法进行迭代优化求解
#        while True:
        for i in range(100):
            gammaJI = self.__EM_Expectation(alpha, mu, sigma)
            self.__EM_Maximization(gammaJI, alpha, mu, sigma)
#        return alpha, mu, sigma
        print(alpha)
        print(mu)
        return gammaJI
    '''
    正态分布公式
    '''
    def normalDistribution(self, x, mu, sigma):
#        return (1. / np.sqrt(2 * np.pi)) * (np.exp(-(x - mu) ** 2 / 2 * sigma ** 2))
#        print(sigma)
        return np.exp(-(x-mu)*np.linalg.inv(sigma)*np.transpose(x-mu))/np.sqrt(np.linalg.det(sigma))
    '''
    EM算法的E步
    '''
    def __EM_Expectation(self, alpha, mu, sigma):
        gammaJI = np.ones((self.M, self.k))
        for j in range(self.M):
            temp = 0
            for i in range(self.k):
                temp += alpha[i] * self.normalDistribution(self.D[j, :], mu[i, :], sigma[i])
            for i in range(self.k):
                gammaJI[j, i] = alpha[i] * self.normalDistribution(self.D[j, :], mu[i, :], sigma[i]) / float(temp)
        return gammaJI
    '''
    EM算法的M步
    '''
    def __EM_Maximization(self, gammaJI, alpha, mu, sigma):
#        gammaJI = self.__EM_Expectation()
        for i in range(self.k):
            tem1 = 0
            tem2 = 0
            tem3 = 0
            for j in range(self.M):
                tem1 += gammaJI[j, i]
                tem2 += gammaJI[j, i] * self.D[j, :]
            mu[i, :] = tem2 / tem1
            for j in range(self.M):
                temp = self.D[j, :] - mu[i, :]
                tem3 += gammaJI[j, i] * np.dot(temp.T, temp)
#                print(np.dot(temp.T, temp))
#                tem3 += gammaJI[j, i] * np.sqrt(np.linalg.det(self.D[j, :] - mu[i, :]))
            sigma[i] = tem3 / tem1
            alpha[i] = tem1 / self.M
#        return
    '''
    可视化结果
    '''
    def plotResult(self, gammaJI):
        order = np.ones(self.M)
        cValue = ['b','g','r','c','m','y','k','w']
        for j in range(self.M):
            for i in range(self.k):
                if gammaJI[j, i] == max(gammaJI[j, :]):
                    order[j] = i  
                plt.scatter(self.D[j, 0], self.D[j, 1], color = cValue[int(order[j])], marker = 'o')
        return order
# =============================================================================
# 主函数
# =============================================================================
#def main():
datMat = generateData()
GMM = Gaussian_Mixture_Model(datMat,3)
o = GMM.plotResult(GMM.Mixture_of_Gaussian())
print(type(o))
# =============================================================================
# 主函数入口    
# =============================================================================
#if __name__ == '__main__':
#    main()