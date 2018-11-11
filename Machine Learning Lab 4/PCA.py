# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 01:32:24 2018

@author: HIT1160300207 Chong Liu
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# THE MNIST DATABASE
# =============================================================================

class DataUtils(object):
    """MNIST数据集加载"""
    def __init__(self, filename=None, outpath=None):
        self._filename = filename
        self._outpath = outpath

        self._tag = '>'
        self._twoBytes = 'II'
        self._fourBytes = 'IIII'    
        self._pictureBytes = '784B'
        self._labelByte = '1B'
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes
        self._pictureBytes2 = self._tag + self._pictureBytes
        self._labelByte2 = self._tag + self._labelByte
    
    def getImage(self):
        """将MNIST的二进制文件转换成像素特征数据"""
        binfile = open(self._filename, 'rb') #以二进制方式打开文件
        buf = binfile.read() 
        binfile.close()
        index = 0
        numMagic,numImgs,numRows,numCols=struct.unpack_from(self._fourBytes2,buf,index)
        index += struct.calcsize(self._fourBytes)
        images = []
        for i in range(numImgs):
            imgVal = struct.unpack_from(self._pictureBytes2, buf, index)
            index += struct.calcsize(self._pictureBytes2)
            imgVal = list(imgVal)
            for j in range(len(imgVal)):
                if imgVal[j] > 1:
                    imgVal[j] = 1
            images.append(imgVal)
        return np.array(images)
    	
    def getLabel(self):
        """将MNIST中label二进制文件转换成对应的label数字特征"""
        binFile = open(self._filename,'rb')
        buf = binFile.read()
        binFile.close()
        index = 0
        magic, numItems= struct.unpack_from(self._twoBytes2, buf,index)
        index += struct.calcsize(self._twoBytes2)
        labels = [];
        for x in range(numItems):
            im = struct.unpack_from(self._labelByte2,buf,index)
            index += struct.calcsize(self._labelByte2)
            labels.append(im[0])
        return np.array(labels)

    def outImg(self, arrX, arrY):
        """根据生成的特征和数字标号，输出png的图像"""
        m, n = np.shape(arrX)
        #每张图是28*28=784Byte
        for i in range(1):
            img = np.array(arrX[i])
            img = img.reshape(28,28)
#            outfile = str(i) + "_" +  str(arrY[i]) + ".png"
            plt.figure()
            plt.imshow(img, cmap = 'binary') #将图像黑白显示
#            plt.savefig(self._outpath + "/" + outfile)
# =============================================================================
#             
# =============================================================================
            #def main():
trainfile_X = 'train-images.idx3-ubyte'
trainfile_y = 'train-labels.idx1-ubyte'
testfile_X = 't10k-images.idx3-ubyte'
testfile_y = 't10k-labels.idx1-ubyte'
        
train_X = DataUtils(trainfile_X).getImage()
train_y = DataUtils(trainfile_y).getLabel()
test_X = DataUtils(testfile_X).getImage()
test_y = DataUtils(testfile_y).getLabel()
# =============================================================================
# 生成数据
# =============================================================================
#D = np.zeros((100,3))
#D[:,0] += np.random.normal(0,0.1,100)
#D[:,1] += np.random.normal(0,50,100)
#D[:,2] += np.random.normal(0,100,100)
#ax = Axes3D(plt.figure())
#ax.scatter(D[:,0],D[:,1],D[:,2])
#plt.show()
#print(D)

D = train_X

#trainSet = []
#imgs = train_X
#labels = train_y
#for i in range(60000):
##       取前100个手写3
##      if labels[i] == 3 and len(trainSet) < 100:
#    if labels[i] == 3:
#          trainSet.append(imgs[i])
#D = trainSet
# =============================================================================
# PCA算法
# =============================================================================
'''对所有样本进行中心化'''
mu = 0
#for i in range(10000):
#    mu += D[i,:]
#mu = mu/10000
mu = np.mean(D,0)
#print(mu)
D = D - mu
#print(D)
#A = 0
#for i in range(100):
#    A += D[i,:]
#print(A)
'''计算样本的协方差矩阵'''
cov = np.cov(D,rowvar = 0)
'''对协方差矩阵做特征值分解'''
lmd , lmdV = np.linalg.eig(cov)
#print(lmd)
'''取最大的d'个特征值所对应的特征向量'''
temp = np.argsort(lmd)
W = (lmdV[temp[-1]],lmdV[temp[-2]])
W = np.matrix(W)
#print(W)
# =============================================================================
# 重构
# =============================================================================
result = np.dot(D,W.T)
#plt.scatter(result[:,0].getA(),result[:,1].getA())
#ax.scatter(result[:,0].getA(),result[:,1].getA(),c = 'g')
result = np.dot(result,W) + mu
#for i in range(60000):
#    result[i,:] += mu
#bx = Axes3D(plt.figure())
#bx.scatter(result[:,0],result[:,1],result[:,2],c = 'r')

   
#cx = comb_imgs(result,10,10,28,28,'L')
#cx.show()
result = np.real(result)
#tempSet = []
#imgs = train_X
#labels = train_y
#for i in range(60000):
#      if labels[i] == 7 and len(tempSet) < 100:
#          tempSet.append(imgs[i])
#result = tempSet
DataUtils().outImg(result,None)
