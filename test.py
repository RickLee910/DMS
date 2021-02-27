from math import cos, pi


def Fiv(S):
    A = [[0] * 4 for _ in range(4)]
    for i in range(4):
        for v in range(4):
            for j in range(4):
                A[i][v] += a(v) * S[i][j] * cos(pi * (j + 0.5) * v / 4)
            A[i][v] = round(A[i][v], 2)
    return A


# def Suv(F):
#     A = [[0] * 4 for _ in range(4)]
#     for i in range(4):
#         for v in range(4):
#             for j in range(4):
#                 A[v][i] += a(v) * F[j][i] * cos(pi * (j + 0.5) * v / 4)
#             A[v][i] = round(A[v][i], 2)
#     return A
#
#
# def a(v):
#     return 0.5 if v == 0 else (1 / 2) ** 0.5
#
#
# S = [[20, 20, 10, 10],
#      [20, 20, 10, 10],
#      [0, 0, 0, 0],
#      [0, 0, 0, 0]]
#
# print(Fiv(S))
# print(Suv(Fiv(S)))
# #[[ 30.     9.24   0.    -3.83]
#  # [ 27.72   8.54   0.    -3.54]
#  # [  0.     0.     0.    -0.  ]
#  # [-11.48  -3.54   0.     1.46]]

# -*- coding: utf-8 -*-
"""
@author: 蔚蓝的天空Tom
Talk is cheap, show me the code
Aim：计算一个多维度样本的协方差矩阵covariance matrix
Note:协方差矩阵是计算的样本中每个特征之间的协方差，所以协方差矩阵是特征个数阶的对称阵
"""

import numpy as np

from numpy import array as matrix, zeros, matmul
class CCovMat(object):
    '''计算多维度样本集的协方差矩阵
    Note：请保证输入的样本集m×n，m行样例，每个样例n个特征
    '''

    def __init__(self, samples):
        # 样本集shpae=(m,n)，m是样本总数，n是样本的特征个数
        self.samples = samples
        self.covmat1 = []  # 保存方法1求得的协方差矩阵
        self.covmat2 = []  # 保存方法1求得的协方差矩阵

        # 用方法1计算协方差矩阵
        self._calc_covmat1()
        # 用方法2计算协方差矩阵
        self._calc_covmat2()

    def _covariance(self, X, Y):
        '''
        计算两个等长向量的协方差convariance
        '''
        n = np.shape(X)[0]
        X, Y = np.array(X), np.array(Y)
        meanX, meanY = np.mean(X), np.mean(Y)
        # 按照协方差公式计算协方差，Note:分母一定是n-1
        cov = sum(np.multiply(X - meanX, Y - meanY)) / (n - 1)
        return cov

    def _calc_covmat1(self):
        '''
        方法1：根据协方差公式和协方差矩阵的概念计算协方差矩阵
        '''
        S = self.samples  # 样本集
        na = np.shape(S)[1]  # 特征attr总数
        self.covmat1 = np.full((na, na), fill_value=0.)  # 保存协方差矩阵
        for i in range(na):
            for j in range(na):
                self.covmat1[i, j] = self._covariance(S[:, i], S[:, j])
        return self.covmat1

    def _calc_covmat2(self):
        '''
        方法2：先样本集中心化再求协方差矩阵
        '''
        S = self.samples  # 样本集
        ns = np.shape(S)[0]  # 样例总数
        mean = np.array([np.mean(attr) for attr in S.T])  # 样本集的特征均值
        print('样本集的特征均值:\n', mean)
        centrS = S - mean  ##样本集的中心化
        print('样本集的中心化(每个元素将去当前维度特征的均值):\n', centrS)
        # 求协方差矩阵
        self.covmat2 = np.dot(centrS.T, centrS) / (ns - 1)
        return self.covmat2

    def CovMat1(self):
        return self.covmat1

    def CovMat2(self):
        return self.covmat2


if __name__ == '__main__':
    '10样本3特征的样本集'
    samples = np.array([[0, 0, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 1, 2],
                        [1, 1, 1, 1],
                        [2, 2, 2, 4],
                        [1, 2, 4, 4],
                        [1, 2, 1, 2],
                        [2, 3, 3, 3],
                        [4, 4, 4, 4]])

    cm = CCovMat(samples.T)

    print('样本集(10行3列，10个样例，每个样例3个特征):\n', samples)
    print('按照协方差公式求得的协方差矩阵:\n', cm.CovMat1())
    print('按照样本集的中心化求得的协方差矩阵:\n', cm.CovMat1())
    print('numpy.cov()计算的协方差矩阵:\n', np.cov(samples.T))
