from typing import Any

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as fun
import numpy as np


class BPR(nn.Module):

    def _forward_unimplemented(self, *inp: Any) -> None:
        pass

    def __init__(self, data_set, args):
        super(BPR, self).__init__()
        self.dataset = data_set
        self.args = args

        self.user_matrix = nn.Embedding(self.dataset.usz, self.args.k)  # k default value is 32
        # user_matrix 可以视为用户关于某个latent factor的权重
        nn.init.normal_(self.user_matrix.weight, std=0.01)  # 进行正则化操作

        self.item_matrix = nn.Embedding(self.dataset.isz, self.args.k)  # k default value is 32
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
        # regs:  default value is 0
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
        learning_rate = 0.01
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


def sample(train_list_i, train_i, sz_i, batch_size_i):
    np.random.shuffle(train_list_i)
    for i in tqdm(range(sz_i // batch_size_i)):
        pairs = []
        sub_train_list = train_list_i[i * batch_size_i:(i + 1) * batch_size_i, :]
        for i, j in sub_train_list:
            i_neg = j
            while i_neg in train_i[i]:
                i_neg = np.random.randint(item_size)
            pairs.append((i, j, i_neg))

        yield torch.LongTensor(pairs)
    yield None


if __name__ == '__main__':
    dataset = np.load('..\\dataset\\amazon\\men.npy', allow_pickle=True).item()

    batch_size = 512  # in mtp the variable is bsz
    test_bsz = 1000  # it seems that the variable is not test
    user_set = dataset['train']

    item_set = dataset['feat']  # items features set
    item_tensor = torch.Tensor(item_set)

    val_set = dataset['val']
    test_set = dataset['test']

    user_size = len(user_set)
    item_size = len(item_set)
    feat_size = 64
    train_list = []
    warm_start = set([])
    cold_start = set([])

    # 在train数据集里的item是有用户对应的item 所以是non-cold的item
    for user, items in tqdm(enumerate(user_set)):
        for item in items:
            train_list.append((user, item))
            warm_start.add(item)

    train = [1 for i in range(user_size)]
    for user in tqdm(range(user_size)):
        train[user] = set(user_set[user])

    print('train list is ready')
    pos_set = set(train_list)

    val_list = [[] for i in range(user_size)]
    val_gt = np.zeros((user_size, test_bsz))
    for user, items in tqdm(enumerate(val_set)):
        val_list[user].extend(items)
        sits = set(items)
        psz = len(items)
        cold_start.update(items)
        val_gt[user, :psz] = 1  # 切片选择

        for i in range(test_bsz - psz):
            ele = items[-1]
            while ele in sits or (user, ele) in pos_set:
                ele = np.random.randint(item_size)
            val_list[user].append(ele)
            sits.add(ele)

    val_samples = np.array(val_list)
    print(f'val list is ready')

    train_list = np.array(train_list)
    sz = train_list.shape[0]







