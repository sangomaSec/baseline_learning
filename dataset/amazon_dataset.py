import torch
import numpy as np


class Amazon:

    def __init__(self, dataset):
        self.dataset = dataset
        self.batch_size = 512  # in mtp the variable is bsz
        self.test_bsz = 1000  # it seems that the variable is not test
        self.user_set = self.dataset['train']

        self.item_set = self.dataset['feat']  # items features set
        self.item_tensor = torch.Tensor(self.item_set)

        self.val_set = self.dataset['val']
        self.test_set = self.dataset['test']

        self.user_size = len(self.user_set)
        self.item_size = len(self.item_set)
        self.feat_size = 64
        self.train_list = []
        self.warm_start = set([])
        self.cold_start = set([])

        # 在train数据集里的item是有用户对应的item 所以是non-cold的item
        for user, items in enumerate(self.user_set):
            for item in items:
                self.train_list.append((user, item))
                self.warm_start.add(item)

        self.train = [1 for i in range(self.user_size)]
        for user in range(self.user_size):
            self.train[user] = set(self.user_set[user])

        print('train list is ready')
        self.pos_set = set(self.train_list)

        val_list = [[] for i in range(self.user_size)]
        self.val_gt = np.zeros((self.user_size, self.test_bsz))
        for user, items in enumerate(self.val_set):
            val_list[user].extend(items)
            sits = set(items)
            psz = len(items)
            self.cold_start.update(items)
            self.val_gt[user, :psz] = 1  # 切片选择

            for i in range(self.test_bsz - psz):
                ele = items[-1]
                while ele in sits or (user, ele) in self.pos_set:
                    ele = np.random.randint(self.item_size)
                val_list[user].append(ele)
                sits.add(ele)

        self.val_samples = np.array(val_list)
        print(f'val list is ready')

        test_list = [[] for i in range(self.user_size)]
        self.test_gt = np.zeros((self.user_size, self.test_bsz))
        for user, items in enumerate(self.test_set):
            test_list[user].extend(items)
            sits = set(items)
            psz = len(items)
            self.cold_start.update(items)
            self.test_gt[user, :psz] = 1
            for i in range(self.test_bsz - psz):
                ele = items[-1]
                while ele in sits or (user, ele) in self.pos_set:
                    ele = np.random.randint(self.item_size)
                test_list[user].append(ele)
                sits.add(ele)
        self.test_samples = np.array(test_list)
        print('test list is ready')

        self.cold_start = self.cold_start - self.warm_start

        self.train_list = np.array(self.train_list)
        self.sz = self.train_list.shape[0]

        self.val_warm_u, \
        self.val_cold_u, \
        self.val_warm_samples, \
        self.val_cold_samples, \
        self.val_warm_gt, \
        self.val_cold_gt = self.separate(self.val_set)
        print('val warm/cold separated')

        self.test_warm_u, \
        self.test_cold_u, \
        self.test_warm_samples, \
        self.test_cold_samples, \
        self.test_warm_gt, \
        self.test_cold_gt = self.separate(self.test_set)
        print('test warm/cold separated')

    def sample(self):
        np.random.shuffle(self.train_list)
        for i in range(self.sz // self.batch_size):
            pairs = []
            sub_train_list = self.train_list[i * self.batch_size:(i + 1) * self.batch_size, :]
            for m, j in sub_train_list:
                m_neg = j
                while m_neg in self.train[m]:
                    m_neg = np.random.randint(self.item_size)
                pairs.append((m, j, m_neg))

            yield torch.LongTensor(pairs)
        yield None

    def separate(self, d_pos):
        warm_u = []
        cold_u = []
        warm_gt = []
        cold_gt = []
        warm_samples = []
        cold_samples = []

        for u in range(self.user_size):
            for iid in d_pos[u]:
                cs = []
                ws = []
                if iid in self.cold_start:
                    cs.append(iid)
                else:
                    ws.append(iid)

            p_csz = len(cs)

            if p_csz > 0:
                cold_u.append(u)  # 1
                cold_gt.append(np.zeros(self.test_bsz))
                cold_gt[-1][:p_csz] = 1  # 2

                sits = set(d_pos[u])  # The set of selected items.
                for i in range(self.test_bsz - p_csz):
                    ele = np.random.randint(self.item_size)
                    while ele in sits or (u, ele) in self.pos_set:
                        ele = np.random.randint(self.item_size)
                    cs.append(ele)
                    sits.add(ele)
                cold_samples.append(cs)  # 3

            p_wsz = len(ws)
            if p_wsz > 0:
                warm_u.append(u)  # 1
                warm_gt.append(np.zeros(self.test_bsz))
                warm_gt[-1][:p_wsz] = 1  # 2

                sits = set(d_pos[u])  # The set of selected items.
                for i in range(self.test_bsz - p_wsz):
                    ele = np.random.randint(self.item_size)
                    while ele in sits or (u, ele) in self.pos_set:
                        ele = np.random.randint(self.item_size)
                    ws.append(ele)
                    sits.add(ele)
                warm_samples.append(ws)  # 3

        return torch.tensor(warm_u), torch.tensor(cold_u),\
               torch.tensor(warm_samples), torch.tensor(cold_samples),\
               np.array(warm_gt), np.array(cold_gt)
