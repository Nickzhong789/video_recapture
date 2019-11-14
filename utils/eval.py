import numpy as np
from sklearn import metrics


__all__ = ['cal_auc', 'accuracy', 'AverageMeter']


def cal_auc(pred, y):
    fpr, tpr, thresholds = metrics.roc_curve(np.array(y), np.array(pred), pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return auc


def accuracy(pred, y):
    return 1.0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += 1
        self.avg = self.sum / self.count
