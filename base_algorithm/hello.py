# 这种是注释
"""
这种还是注释
"""
import numpy as np
import random as rd
import sys


def this_is_a_method(this_parameter):
    print('print1  is (this_parameter)')
    print(f'print2  is {this_parameter}')


if __name__ == '__main__':

    u = np.random.randn(4, 3) + 0.5
    v = np.random.randn(4, 3) + 0.5
    i = 3
    j = 2
    print(f'u3 is {u[i]}')
    print(f'v2 is {v[j]}')
    print(f'u[i] * v[j] is {u[i] * v[j]}')
    print(f'sum(u[i] * v[j]) is {sum(u[i] * v[j])}')
