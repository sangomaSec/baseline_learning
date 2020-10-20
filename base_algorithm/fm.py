import numpy as np
import sys
import math
import sklearn as sk

"""
 特征组合的优势在于对特征组合以及高维灾难两方面问题的处理
 其中对于特征组合，FM是通过两两特征组合，引入交叉项特征即二阶特征，提高模型得分
 对于高位灾难问题，是通过引入隐向量即对参数矩阵进行分解，来完成特征参数的估计
 数据集为稀疏矩阵的时候更适合使用FM来解决
"""

"""
 那我们代码中第一步是不是应该先进行矩阵的分解，即把每一行看作一个向量Vi
 然后这些向量再重新组合成为一个新的交互矩阵W，
 
 这段代码就让我们先写数据集的加载吧，可是数据集是加载的啥数据集啊。。。
"""


class FM(object):

    def __init__(self, w, lr):
        self.data = None
        self.w_0 = None
        self.w = w
        self.lr = lr

    def load_data(self, file_path):
        i_file = open(file_path)
        self.w_0 = 0
        results = []
        """
         * put tuples to results. after this method the results is (v1, v2, v3......)
        """
        for line in i_file:
            line_list = line.strip().split(',')  # do not know the real dataset use what char to split the data
            line_tuple = tuple(line_list)
            results.append(line_tuple)
        return results

    def _factor_machine(self, args, n, epochs):
        self.w = np.zeros((n, 1))
        self.w_0 = 0
        for epoch in epochs:
            print(epoch)

    @staticmethod
    def sig_function(inx):
        return 1.0 / (1 + math.exp(-inx))

    def train(self, data):
        """"""

    def predict(self, data):
        """"""


if __name__ == '__main__':
    # fm = FM()
    # dataset = fm.load_data('.\\old.txt')
    print()

