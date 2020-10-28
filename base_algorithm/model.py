import numpy as np
from base_algorithm.util import Metric
import torch


class Model:

    def compute_results(self, u, test_samples):
        rs = []
        for i in test_samples.T:
            rs.append(self.predict(u, torch.LongTensor(i)).detach().numpy())
        results = np.vstack(rs).T
        if np.isnan(results).any():
            raise Exception('nan')
        return results

    @staticmethod
    def compute_scores(gt, preds):
        ret = {
            'ndcg': Metric.ndcg(gt, preds),
            'auc': Metric.auc(gt, preds)
        }
        return ret

    def __logscore(self, scores):
        metrics = list(scores.keys())
        metrics.sort()
        print(' '.join(['%s: %s' % (m, str(scores[m])) for m in metrics]))
        # self.logging.info(' '.join(['%s: %s' % (m,str(scores[m])) for m in metrics]))

    def test(self):
        u = torch.LongTensor(range(self.amazon.user_size))
        results = self.compute_results(u, self.amazon.test_samples)
        scores = self.compute_scores(self.amazon.test_gt, results)
        print('----- test -----')
        self.__logscore(scores)

    def val(self):
        u = torch.LongTensor(range(self.amazon.user_size))
        results = self.compute_results(u, self.amazon.val_samples)
        scores = self.compute_scores(self.amazon.val_gt, results)
        print('----- val -----')
        self.__logscore(scores)

    def test_warm(self):
        u = self.amazon.test_warm_u
        results = self.compute_results(u, self.amazon.test_warm_samples)
        scores = self.compute_scores(self.amazon.test_warm_gt, results)
        print('----- test_warm -----')
        self.__logscore(scores)

    def test_cold(self):
        u = self.amazon.test_cold_u
        results = self.compute_results(u, self.amazon.test_cold_samples)
        scores = self.compute_scores(self.amazon.test_cold_gt, results)
        print('----- test_cold -----')
        self.__logscore(scores)

    def train(self):
        raise Exception('no implementation')

    def regs(self):
        raise Exception('no implementation')

    def predict(self):
        raise Exception('no implementation')

    def save(self):
        raise Exception('no implementation')