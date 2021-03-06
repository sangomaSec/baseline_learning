import torch
import numpy as np
from math import log
from math import fsum
import sys


def compute_entropy(p_array, base):

    return torch.sum(-p_array * log())


if __name__ == '__main__':
    # emb_1 = torch.nn.Embedding(5, 3)
    # # print(emb_1.weight.data)
    # torch.nn.init.normal_(emb_1.weight, std=0.01)
    # print(emb_1.weight.data)
    #

    # result = -0.99 * log(0.99, 2) - 0.01 * log(0.01, 2)
    # print(f'{result}')

    np.set_printoptions(threshold=sys.maxsize)

    boxes = np.load('..\\dataset\\amazon\\men.npy', allow_pickle=True).item()
    # print(boxes['train'])
    # print(boxes['val'])
    # print(boxes)
    train_set = boxes['train']
    train_list = []
    warm_start = set([])
    i = 0
    for user, items in enumerate(train_set):
        i += 1
        for item in items:
            train_list.append((user, item))
            warm_start.add(item)

        if i > 10:
            break

    print(train_list, warm_start)



