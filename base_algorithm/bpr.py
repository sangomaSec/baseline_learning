from typing import Any

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as fun
import numpy as np


class BPR(nn.Module):

    def _forward_unimplemented(self, *inp: Any) -> None:
        pass

    def __init__(self, dataset, args):
        super(BPR, self).__init__()
        self.dataset = dataset
        self.args = args

        self.user_matrix = nn.Embedding(self.dataset.usz, self.args.k)  # k is the dim of the latent factors of item
        # user_matrix 可以视为用户关于某个latent factor的权重
        nn.init.normal_(self.user_matrix.weight, std=0.01)  # 进行正则化操作

        self.item_matrix = nn.Embedding(self.dataset.isz, self.args.k)  # require_grads default value is True
        # item_matrix 可以视为item在某个latent factor的值的大小
        nn.init.normal_(self.item_matrix.weight, std=0.01)  # 进行正则化操作

    def predict(self, uid, iid):
        """
        uid of user_matrix
        iid of item_matrix
        :return:
        """
        return torch.sum(self.user_matrix(uid), self.item_matrix(iid), dim=1)

    def bpr_loss(self, uid, iid, jid):
        """
        bpr的算法是，对一个用户u求i和j两个item的分数，然后比较更喜欢哪个，
        所以这里需要进行两次预测，分别是第i个item的和第j个item的
        """
        pre_i = self.predict(uid, iid)
        pre_j = self.predict(uid, jid)
        dev = pre_i - pre_j
        return torch.sum(fun.softplus(-dev))

    def regs(self, uid, iid, jid):
        # regs: 我也不知道是啥啊 default value is 0
        reg = self.args.reg
        uid_v = self.user_matrix(uid)
        iid_v = self.item_matrix(iid)
        jid_v = self.item_matrix(jid)
        emb_regs = torch.sum(uid_v * uid_v) + torch.sum(iid_v * iid_v) + torch.sum(jid_v * jid_v)
        return reg * emb_regs

    def train(self):
        """
        lr: learning rate default value is 0.01
        :return:
        """
        learning_rate = self.args.lr
        optimizer = torch.optim.Adagrad([self.user_matrix.weight, self.item_matrix.weight],
                                        lr=learning_rate, weight_decay=0)
        epochs = 100
        for epoch in tqdm(range(epochs)):
            generator = self.dataset.sample()
            while True:
                optimizer.zero_grad()
                s = next(generator)
                if s is None:
                    break
                uid, iid, jid = s[:, 0], s[:, 1], s[:, 2]

                loss = self.bpr_loss(uid, iid, jid) + self.regs(uid, iid, jid)

                loss.backward()
                optimizer.step()

            print(epoch)


if __name__ == '__main__':
    dataset = np.load('..\\dataset\\amazon\\men.npy', allow_pickle=True).item()

    training_set = dataset['train']

    user_size = len(dataset['train'])
    item_size = len(dataset['feat'])


    print(dataset)
