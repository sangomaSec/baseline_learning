import torch
import numpy as np
from math import log
from math import fsum


def compute_entropy(p_array, base):

    return torch.sum(-p_array * log())


if __name__ == '__main__':
    # emb_1 = torch.nn.Embedding(5, 3)
    # # print(emb_1.weight.data)
    # torch.nn.init.normal_(emb_1.weight, std=0.01)
    # print(emb_1.weight.data)

    result = -0.99 * log(0.99, 2) - 0.01 * log(0.01, 2)
    print(f'{result}')


