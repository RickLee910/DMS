from numpy import array as matrix, zeros, matmul
import numpy as np
from math import sqrt, cos, pi

a = matrix([[20, 20, 10, 10],
            [20, 20, 10, 10],
            [0, 0, 0, 0],
            [0, 0, 0, 0]])
A = zeros((4,4))

shape=a.shape[1]
class TWOD_DCT:
    def a(self, k: int, N: int):
        return sqrt(1 / N) if k == 0 else sqrt(2 / N)

    def F(self):
        for i in range(4):
            for j in range(4):
                x = self.a(i,shape)
                A[i][j] = x * cos(pi * (j + 0.5) * i/shape)
        A_T = A.transpose()
        Y1 = matmul(A,a)
        Y = matmul(Y1,A_T)
        print(np.around(Y,2))

x = TWOD_DCT()
x.F()