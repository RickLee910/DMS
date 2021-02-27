# -*- coding: utf-8 -*-
# @Time    : 2/3/2021 12:55 PM
# @Author  : RICK LEE
# @FileName: PCA.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from PIL import Image
import cv2 as cv

class DimensionValueError(ValueError):
    """定义异常类"""
    pass


class PCA(object):
    """定义PCA类"""

    def __init__(self, x, n_components=None):
        """x的数据结构应为ndarray"""
        self.x = x
        self.dimension = x.shape[1]

        if n_components and n_components >= self.dimension:
            raise DimensionValueError("n_components error")

        self.n_components = n_components

    def cov(self):
        """求x的协方差矩阵"""
        x_T = np.transpose(self.x)  # 矩阵转秩
        x_cov = np.cov(x_T)  # 协方差矩阵
        return x_cov

    def get_feature(self):
        """求协方差矩阵C的特征值和特征向量"""
        x_cov = self.cov()
        a, b = np.linalg.eig(x_cov)
        m = a.shape[0]
        c = np.hstack((a.reshape((m, 1)), b))
        c_df = pd.DataFrame(c)
        #c_df_sort = c_df.sort_values(by=m, ascending=False)  # 按照特征值大小降序排列特征向量
        return c_df

    def explained_variance_(self):
        c_df_sort = self.get_feature()
        return c_df_sort.values[:, 0]

    def paint_variance_(self):
        explained_variance_ = self.explained_variance_()
        plt.figure()
        plt.plot(explained_variance_, 'k')
        plt.xlabel('n_components', fontsize=16)
        plt.ylabel('explained_variance_', fontsize=16)
        plt.show()

    def reduce_dimension(self):
        """指定维度降维和根据方差贡献率自动降维"""
        c_df_sort = self.get_feature()
        variance = self.explained_variance_()

        if self.n_components:  # 指定降维维度
            p = c_df_sort.values[0:self.n_components, 1:]
            y = np.dot(p, np.transpose(self.x))  # 矩阵叉乘
            return np.transpose(y)

        variance_sum = sum(variance)  # 利用方差贡献度自动选择降维维度
        variance_radio = variance / variance_sum

        variance_contribution = 0
        for R in range(self.dimension):
            variance_contribution += variance_radio[R]  # 前R个方差贡献度之和
            if variance_contribution >= 0.99:
                break

        p = c_df_sort.values[0:R + 1, 1:]  # 取前R个特征向量
        y = np.dot(p, np.transpose(self.x))  # 矩阵叉乘
        return np.transpose(y)
#图片转矩阵
def image_matrix(filename):
    x = Image.open(filename)
    data = np.asarray(x)
    #print(data.shape)#输出图片尺寸
    #print(data)#输出图片矩阵
    return data
#矩阵转图片
def matrix_image(matrix):
    img = Image.fromarray(matrix)
    return img
im = cv.imread('Original-Lena-image-512-512-pixels.png')###导入图片
x = cv.cvtColor(im, cv.COLOR_BGR2GRAY)###将彩色图片灰度化
if __name__ == '__main__':
    pca = PCA(x,501)
    y = pca.reduce_dimension()
    plt.imshow(y,'gray')
    plt.show()
    # new_img = matrix_image(y)
    # new_img.show()
    # pca.paint_variance_()