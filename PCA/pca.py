# encoding=utf8
'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]  # map(function, iterable, ...)
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # remove mean

    covMat = cov(meanRemoved, rowvar=0)  # 计算协方差
    eigVals, eigVects = linalg.eig(mat(covMat))  # 特征值,特征向量
    eigValInd = argsort(eigVals)  # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # cut off unwanted dimensions
    redEigVects = eigVects[:, eigValInd]  # reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 原始数据乘以topN特征值的结果跟原始矩阵相差不多
    return lowDDataMat, reconMat


def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])  # values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal  # set NaN values to mean
    return datMat


if __name__ == '__main__':
    # 简单的测试PCA
    # dataMat = loadDataSet('testSet.txt')
    # lowDDataMat, reconMat = pca(dataMat, 1)
    # print shape(lowDDataMat)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    # ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=5, c='red')
    # plt.show()

    # 半导体制造数据降维
    datMat = replaceNanWithMean()
    print datMat.shape
    lowDDataMat, reconMat = pca(datMat, 6)
    print lowDDataMat.shape
    print lowDDataMat[:4]