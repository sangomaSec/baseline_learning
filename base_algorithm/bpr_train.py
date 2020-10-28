from typing import Any

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as fun
import numpy as np
from dataset.amazon_dataset import Amazon
from base_algorithm.model import  Model


class BPR(Model):

    def __init__(self, data):
        super(BPR, self).__init__()
        self.amazon = data
        self.usz = self.amazon.user_size
        self.k = 32
        self.isz = self.amazon.item_size
        self.train_list = self.amazon.train_list
        self.train_ = self.amazon.train
        self.sz = self.amazon.sz
        self.batch_size = self.amazon.batch_size

        self.user_matrix = nn.Embedding(self.usz, self.k)  # k default value is 32
        # user_matrix 可以视为用户关于某个latent factor的权重
        nn.init.normal_(self.user_matrix.weight, std=0.01)  # 进行正则化操作

        self.item_matrix = nn.Embedding(self.isz, self.k)  # k default value is 32
        # item_matrix 可以视为item在某个latent factor的值的大小
        nn.init.normal_(self.item_matrix.weight, std=0.01)  # 进行正则化操作

    def predict(self, uid, iid):
        """
        uid of user_matrix
        iid of item_matrix
        :return:
        """
        return torch.sum(self.user_matrix(uid) * self.item_matrix(iid), dim=1)

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
        # regs:  default value is 0
        reg = 0
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
        learning_rate = 0.01
        optimizer = torch.optim.Adagrad([self.user_matrix.weight, self.item_matrix.weight],
                                        lr=learning_rate, weight_decay=0)
        epochs = 100
        for epoch in tqdm(range(epochs)):
            print(epoch)
            generator = self.amazon.sample()
            while True:
                optimizer.zero_grad()
                s = next(generator)
                if s is None:
                    break
                uid, iid, jid = s[:, 0], s[:, 1], s[:, 2]

                loss = self.bpr_loss(uid, iid, jid) + self.regs(uid, iid, jid)

                loss.backward()
                optimizer.step()
            if epoch % 5 == 0:
                print(f'self.user.weight is {self.user_matrix.weight} \n self.item.weight is {self.item_matrix.weight}')
                self.val(), self.test(), self.test_warm(), self.test_cold()


if __name__ == '__main__':
    # dataset = np.load('..\\dataset\\amazon\\men.npy', allow_pickle=True).item()
    dataset = np.load('/home/share/liqi/amazon', allow_pickle=True).item()
    amazon_dataset = Amazon(dataset)

    bpr = BPR(amazon_dataset)
    bpr.train()




