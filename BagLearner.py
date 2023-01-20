import RTLearner as rt
import numpy as np
from scipy import stats


def author():
    return 'jwilkins36'


class BagLearner(object):

    def __init__(self, learner=rt.RTLearner, kwargs={'leaf_size': 5}, bags=25, boost=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.models = [learner(**kwargs) for i in range(0, bags)]

    def add_evidence(self,  data_x, data_y):
        for model in self.models:
            idx = np.random.choice(len(data_x), len(data_x))
            model.add_evidence(data_x[idx], data_y[idx])

    def query(self, points):
        preds = np.ones([self.bags, points.shape[0]])
        for i in range(self.bags):
            preds[i] = self.models[i].query(points)
        preds_mode = stats.mode(preds)[0]
        return preds_mode

