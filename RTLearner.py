import numpy as np
from scipy import stats

def author():
    return 'jwilkins36'

class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def add_evidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        if data_y.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])

        randIdx = np.random.randint(0, data_x.shape[1])
        splitVal = np.median(data_x[:, randIdx])

        if (data_x[data_x[:, randIdx] <= splitVal]).shape[0] == data_x.shape[0]:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        if (data_x[data_x[:, randIdx] > splitVal]).shape[0] == data_x.shape[0]:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])

        left_tree = self.build_tree(data_x[data_x[:, randIdx] <= splitVal], data_y[data_x[:, randIdx] <= splitVal])
        right_tree = self.build_tree(data_x[data_x[:, randIdx] > splitVal], data_y[data_x[:, randIdx] > splitVal])

        root = np.array([[randIdx, splitVal, 1, left_tree.shape[0] + 1]])
        tree = (np.vstack((root, left_tree, right_tree)))
        return tree

    def query(self, points):
        preds = np.zeros(shape=(points.shape[0],))
        row = 0
        for point in points:
            i = 0
            node = self.tree[i]
            while node[0] != -1:
                featIdx = int(node[0])
                if point[featIdx] <= node[1]:
                    i += int(node[2])
                    node = self.tree[i]
                else:
                    i += int(node[3])
                    node = self.tree[i]
            preds[row] = (node[1])
            row = row + 1
        return preds

