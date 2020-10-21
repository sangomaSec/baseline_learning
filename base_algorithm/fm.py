from typing import Any

import torch
import torch.nn as nn

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


class FM(nn.Module):

    def __init__(self, n=10, k=5):
        """

        :param n: 对象由几维特征进行表示
        :param k: 对象的特征由几维子特征表示
        """
        super(FM, self).__init__()
        self.n = n
        self.k = k
        self.linear = nn.Linear(self.n, 1)
        self.v = nn.Parameter(torch.randn(self.n, self.k))
        nn.init.uniform_(self.v, -0.1, 0.1)

    def fm_layer(self, x):
        linear_part = self.linear(x)
        interaction_part_1 = torch.mm(x, self.v)
        interaction_part_1 = torch.pow(interaction_part_1, 2)
        interaction_part_2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
        out = linear_part + 0.5 * torch.sum(interaction_part_1 - interaction_part_2, 1, keepdim=False)
        return out

    def forward(self, x):
        return self.fm_layer(x)


if __name__ == '__main__':
    fm = FM(n=10, k=5)
    x = torch.randn(1, 10)
    output = fm(x)
    print(output)


