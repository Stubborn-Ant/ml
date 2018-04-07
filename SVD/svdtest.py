# encoding=utf8
from numpy import *


# 求矩阵的奇异值
# U, Sigma, VT = linalg.svd([[1, 1], [1, 7]])
# print U
# '''
# [[ 0.16018224  0.98708746]
#  [ 0.98708746 -0.16018224]]
# '''
# print Sigma
# '''[7.16227766 0.83772234]'''
# print VT
# '''
# [[ 0.16018224  0.98708746]
#  [ 0.98708746 -0.16018224]]
# '''

# 复杂矩阵的奇异值分解
def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


data = loadExData()
U, Sigma, VT = linalg.svd(data)  # m*n=m*m.m*n.n*n
Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
print U[:, :3] * Sig3 * VT[:3, :]
